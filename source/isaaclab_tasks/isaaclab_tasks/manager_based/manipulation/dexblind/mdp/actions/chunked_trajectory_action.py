# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory index selection action for residual RL.

핵심 개념: Trajectory를 "Action Library"로 사용
============================================

일반 Residual RL:
    - Trajectory를 시간 순서대로 재생
    - PPO는 미세 조정만 담당
    
이 방식 (Trajectory Index Selection):
    - Trajectory 전체를 "참조 동작 라이브러리"로 봄
    - PPO가 현재 상황에 가장 적합한 trajectory 위치를 선택
    - 선택된 위치의 joint position + residual = 최종 타겟

이점:
    - 상황에 맞는 동작을 즉시 선택 가능 (시간 순서 강제 없음)
    - 실패 시 이전 위치로 돌아가거나, 건너뛰기 가능
    - Trajectory가 "what to do" 라이브러리, PPO가 "when to use" 결정
"""

from __future__ import annotations

import os
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.joint_actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import chunked_trajectory_action_cfg


class ChunkedTrajectoryAction(JointAction):
    """Joint action term that selects trajectory index via scalar offset and applies residual.

    PPO Action Space (Scalar Index Offset + Residual):
        - actions[:, 0]: index offset scalar (-1 ~ 1) → [current - max_offset, current + max_offset]
        - actions[:, 1:]: residual joint actions (num_joints)

    Processing:
        1. PPO outputs 1 scalar for index offset
        2. Scale from (-1, 1) to (-max_offset, +max_offset) and add to current index
        3. Clamp to [0, num_frames-1] and round to integer
        4. Get joint positions from trajectory[index]
        5. Apply optional smoothing
        6. Add residual * scale
        7. Set as joint position target
    """

    cfg: chunked_trajectory_action_cfg.ChunkedTrajectoryActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: chunked_trajectory_action_cfg.ChunkedTrajectoryActionCfg, env: ManagerBasedEnv):
        """Initialize the trajectory index selection action term.

        Args:
            cfg: The configuration parameters for the action term.
            env: The environment object.
        """
        # Store config before parent init (needed for action_dim property)
        self._cfg = cfg
        self._num_joints = None

        # Load trajectory first to determine num_joints
        self._load_trajectory(cfg)

        # Initialize the base class
        super().__init__(cfg, env)

        # Verify joint count matches
        if len(self._joint_names) != self.traj_num_joints:
            raise ValueError(
                f"Joint count mismatch: trajectory has {self.traj_num_joints} joints, "
                f"but action config specifies {len(self._joint_names)} joints.\n"
                f"Trajectory joints: {self.traj_joint_names}\n"
                f"Action joints: {self._joint_names}"
            )

        # Create reordering indices if joint order differs
        self._create_reorder_indices()

        # ========================================
        # Simple state tracking (no ensemble!)
        # ========================================
        
        # Current selected index per environment
        self.current_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Previous target for smoothing
        self._prev_target = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._prev_target_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Step counter for initial sequential forcing
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Flag to track which envs started from random position (skip initial_sequential_steps)
        self._random_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Observation buffer: [normalized_index, normalized_velocity]
        self._index_info = torch.zeros(self.num_envs, 2, device=self.device)

        # Move trajectory to device
        self.trajectory_positions = self.trajectory_positions.to(self.device)

        print(f"[TrajectoryIndexAction] Loaded trajectory: {self.traj_num_frames} states, {self.traj_num_joints} joints")
        print(f"[TrajectoryIndexAction] Mode: SCALAR OFFSET (±{cfg.max_index_offset} from current), "
              f"smoothing: {cfg.use_smoothing} (alpha={cfg.smoothing_alpha})")
        print(f"[TrajectoryIndexAction] Action dim: 1 (offset) + {self.traj_num_joints} (residuals) = {self.action_dim}")

    def _load_trajectory(self, cfg):
        """Load trajectory data from file."""
        trajectory_path = cfg.trajectory_file
        if not os.path.isabs(trajectory_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            trajectory_path = os.path.join(base_dir, "data", trajectory_path)

        if not os.path.isfile(trajectory_path):
            raise ValueError(f"Trajectory file not found: {trajectory_path}")

        data = np.load(trajectory_path)

        # Extract raw data
        positions_raw = torch.tensor(data["positions"], dtype=torch.float32)
        joint_names_raw = data["joint_names"].tolist() if isinstance(data["joint_names"], np.ndarray) else data["joint_names"]

        # Filter excluded joints
        excluded_joints = cfg.excluded_joints or [
            "R_Index_Roll_Joint", "R_Middle_Roll_Joint",
            "R_Ring_Roll_Joint", "R_Little_Roll_Joint"
        ]
        keep_indices = [i for i, name in enumerate(joint_names_raw) if name not in excluded_joints]

        # Store filtered trajectory
        self.trajectory_positions = positions_raw[:, keep_indices]
        self.traj_joint_names = [joint_names_raw[i] for i in keep_indices]
        self.traj_num_frames = self.trajectory_positions.shape[0]
        self.traj_num_joints = self.trajectory_positions.shape[1]
        self._num_joints = self.traj_num_joints

    def _create_reorder_indices(self):
        """Create indices to reorder trajectory to match action joint order."""
        self._reorder_indices = None

        if self.traj_joint_names != list(self._joint_names):
            self._reorder_indices = torch.zeros(
                len(self._joint_names), dtype=torch.long, device=self.device
            )
            for i, action_joint in enumerate(self._joint_names):
                if action_joint not in self.traj_joint_names:
                    raise ValueError(
                        f"Joint '{action_joint}' not found in trajectory.\n"
                        f"Trajectory joints: {self.traj_joint_names}"
                    )
                self._reorder_indices[i] = self.traj_joint_names.index(action_joint)

    @property
    def action_dim(self) -> int:
        """Action dimension: 1 (index scalar) + num_joints (residuals)."""
        if self._num_joints is None:
            return 1
        return 1 + self._num_joints

    @property
    def index_info(self) -> torch.Tensor:
        """Get current index information for observation.

        Returns:
            Tensor of shape (num_envs, 2) containing:
            [normalized_index (0~1), index_velocity (normalized)]
        """
        return self._index_info

    @property
    def raw_index_scalar(self) -> torch.Tensor:
        """Get the raw index scalar from the last action for debugging.

        Returns:
            Tensor of shape (num_envs,) containing the raw index scalar (-1 ~ 1).
        """
        return self._raw_actions[:, 0]

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the action term for specified environments."""
        super().reset(env_ids)

        if env_ids is None:
            env_ids = slice(None)
            num_reset = self.num_envs
        else:
            num_reset = len(env_ids)

        # Reset step counter
        self._step_count[env_ids] = 0
        
        # Reset smoothing buffer validity
        self._prev_target_valid[env_ids] = False

        # ========================================
        # Determine start indices (random or zero)
        # ========================================
        start_indices = torch.zeros(num_reset, dtype=torch.long, device=self.device)
        random_mask = torch.zeros(num_reset, dtype=torch.bool, device=self.device)

        if self.cfg.random_start_prob > 0.0:
            random_mask = torch.rand(num_reset, device=self.device) < self.cfg.random_start_prob
            
            if random_mask.any():
                min_frac, max_frac = self.cfg.random_start_range
                min_idx = int(min_frac * (self.traj_num_frames - 1))
                max_idx = int(max_frac * (self.traj_num_frames - 1))
                
                random_indices = torch.randint(
                    min_idx, max_idx + 1,
                    (random_mask.sum().item(),),
                    dtype=torch.long,
                    device=self.device
                )
                start_indices[random_mask] = random_indices

        # Set current index
        if isinstance(env_ids, slice):
            self.current_index[:] = start_indices
            self._random_started[:] = random_mask
        else:
            self.current_index[env_ids] = start_indices
            self._random_started[env_ids] = random_mask

        # Initialize prev_target with trajectory at start index (with reorder!)
        init_positions = self.trajectory_positions[start_indices]
        if self._reorder_indices is not None:
            init_positions = init_positions[:, self._reorder_indices]
        
        if isinstance(env_ids, slice):
            self._prev_target[:] = init_positions
        else:
            self._prev_target[env_ids] = init_positions

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions from PPO.

        핵심 로직 (Scalar Offset + Residual):
        1. PPO output에서 index offset scalar와 residual 분리
        2. Scalar (-1 ~ 1) → current_index ± max_offset 범위로 변환
        3. Trajectory에서 해당 index의 position 가져오기
        4. (옵션) Smoothing 적용
        5. Residual 더하기

        Args:
            actions: Raw actions of shape (num_envs, action_dim)
                - actions[:, 0]: index offset scalar (-1 ~ 1)
                - actions[:, 1:]: residual actions (num_joints)
        """
        # Store raw actions
        self._raw_actions[:] = actions

        # ========================================
        # 1. PPO output 분리 (index offset + residuals)
        # ========================================
        offset_scalar = actions[:, 0]  # (num_envs,) 범위: 대략 -1 ~ 1
        residual_actions = actions[:, 1:]  # (num_envs, num_joints)
        
        # ========================================
        # 2. Scalar → Integer index 변환 (현재 기준 ±offset)
        # ========================================
        prev_index = self.current_index.clone()
        
        # offset_scalar (-1 ~ 1) → (-max_offset ~ +max_offset)
        max_offset = self.cfg.max_index_offset
        offset = offset_scalar * max_offset  # 연속값 offset
        
        # 현재 인덱스에 offset 더하기
        new_index_float = prev_index.float() + offset
        new_index = torch.round(new_index_float).long()  # 반올림하여 정수 변환
        new_index = torch.clamp(new_index, 0, self.traj_num_frames - 1)  # 범위 제한

        # ========================================
        # 3. Initial sequential forcing (optional)
        # Only for envs that started from index 0 (not random start)
        # ========================================
        if self.cfg.initial_sequential_steps > 0:
            # Only apply to envs that: (1) in initial phase AND (2) NOT random started
            in_initial_phase = (self._step_count < self.cfg.initial_sequential_steps) & (~self._random_started)
            if in_initial_phase.any():
                sequential_index = self._step_count[in_initial_phase].clamp(0, self.traj_num_frames - 1)
                new_index[in_initial_phase] = sequential_index

        # Update current index
        self.current_index = new_index

        # ========================================
        # 4. Trajectory에서 position 가져오기
        # ========================================
        # Direct indexing: trajectory_positions[new_index] → (num_envs, num_joints)
        target_positions = self.trajectory_positions[new_index]

        # Reorder if needed
        if self._reorder_indices is not None:
            target_positions = target_positions[:, self._reorder_indices]

        # ========================================
        # 5. Smoothing (optional)
        # ========================================
        if self.cfg.use_smoothing:
            alpha = self.cfg.smoothing_alpha
            
            # Only apply smoothing if prev_target is valid
            valid_mask = self._prev_target_valid
            if valid_mask.any():
                # Smooth only for valid environments
                target_positions[valid_mask] = (
                    alpha * target_positions[valid_mask] +
                    (1 - alpha) * self._prev_target[valid_mask]
                )
            
            # Update prev_target
            self._prev_target = target_positions.clone()
            self._prev_target_valid[:] = True

        # ========================================
        # 6. Residual 적용
        # ========================================
        scaled_residuals = residual_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            scaled_residuals = torch.clamp(
                scaled_residuals, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        # Final target = trajectory position + residual
        self._processed_actions = target_positions + scaled_residuals * self.cfg.residual_scale

        # ========================================
        # 7. Update counters and observation info
        # ========================================
        self._step_count += 1

        # Update index info for observation
        self._index_info[:, 0] = new_index.float() / (self.traj_num_frames - 1)  # Normalized index
        self._index_info[:, 1] = (new_index - prev_index).float() / max(self.traj_num_frames - 1, 1)  # Normalized velocity

    def apply_actions(self):
        """Apply the processed actions as joint position targets."""
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    # ========================================
    # Compatibility properties (for existing observation code)
    # ========================================
    
    @property
    def current_target_index(self) -> torch.Tensor:
        """Alias for current_index (backward compatibility)."""
        return self.current_index

    @property
    def chunk_info(self) -> torch.Tensor:
        """Compatibility property for existing observation code.
        
        Returns tensor of shape (num_envs, 4):
        [normalized_index, 0, 0, 0]
        """
        info = torch.zeros(self.num_envs, 4, device=self.device)
        info[:, 0] = self.current_index.float() / (self.traj_num_frames - 1)
        return info
