# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for trajectory index selection action (simplified chunked trajectory)."""

from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.utils import configclass

from . import chunked_trajectory_action


@configclass
class ChunkedTrajectoryActionCfg(JointActionCfg):
    """Configuration for trajectory index selection action.

    핵심 개념: Trajectory를 "Action Library"로 사용
    ============================================
    
    일반 Residual RL: 시간 순서대로 trajectory를 재생하며 미세 조정
    
    이 방식: Trajectory 전체를 "참조 동작 라이브러리"로 보고,
            PPO가 현재 상황에 가장 적합한 위치를 선택

    Action space (Scalar Index + Residual):
        - actions[:, 0]: index offset scalar (-1 ~ 1) → [current - max_offset, current + max_offset]
        - actions[:, 1:]: residual actions (num_joints)

    Total action_dim = 1 + num_joints
    
    예시 (20개 상태, 18개 관절, max_offset=5):
        - actions[:, 0]: index offset scalar (1차원), 현재 인덱스 기준 ±5 범위 내 선택
        - actions[:, 1:19]: 18개 관절 residual
        - Total: 19차원
    """

    class_type: type = chunked_trajectory_action.ChunkedTrajectoryAction

    # ========================================
    # Trajectory file
    # ========================================
    trajectory_file: str = MISSING
    """Path to the trajectory file (.npz format).

    Required keys:
        - `positions`: Joint positions (N, num_joints) in radians
        - `joint_names`: List of joint names (num_joints,)
    """

    # ========================================
    # Index selection (Scalar → Integer with offset)
    # ========================================
    # PPO outputs 1 scalar (-1 ~ 1) → [current - max_offset, current + max_offset] 범위로 변환
    # 현재 인덱스 기준 ±offset 내에서만 선택 가능 → 급격한 점프 방지
    
    max_index_offset: int = 5
    """Maximum offset from current index. Defaults to 5.
    
    PPO action (-1 ~ 1) maps to [current_index - max_offset, current_index + max_offset].
    Result is clamped to [0, num_frames-1].
    
    예시 (current_index=10, max_offset=5):
        - action = -1.0 → index = 5
        - action = 0.0  → index = 10 (현재 유지)
        - action = +1.0 → index = 15
    
    작은 값 → 더 부드러운 진행, 큰 값 → 더 자유로운 점프
    """

    # ========================================
    # Residual action
    # ========================================
    residual_scale: float = 0.1
    """Scaling factor for residual actions. Defaults to 0.1.

    Final target = trajectory[index] + residual * residual_scale
    """

    # ========================================
    # Smoothing (optional)
    # ========================================
    use_smoothing: bool = True
    """Whether to apply exponential smoothing to target positions. Defaults to True.
    
    If True: target = alpha * new_target + (1 - alpha) * prev_target
    This prevents jerky motion when PPO makes sudden index changes.
    """

    smoothing_alpha: float = 0.7
    """Smoothing factor (0~1). Defaults to 0.7.
    
    Higher = more responsive to new selections (less smooth)
    Lower = smoother transitions (more lag)
    
    Recommended: 0.5~0.8
    """

    # ========================================
    # Random start (exploration)
    # ========================================
    random_start_prob: float = 0.0
    """Probability of starting from a random index at episode reset. Defaults to 0.0.
    
    - 0.0: Always start from index 0
    - 0.5: 50% chance of random start
    - 1.0: Always start from random index
    """

    random_start_range: tuple[float, float] = (0.0, 0.7)
    """Range of random start position as fraction of trajectory. Defaults to (0.0, 0.7).
    
    (0.0, 0.7) means random start between 0% and 70% of trajectory.
    """

    # ========================================
    # Initial sequential forcing
    # ========================================
    initial_sequential_steps: int = 0
    """Number of steps to force sequential progression at episode start. Defaults to 0.

    During these initial steps, index = step_count (ignoring PPO's index output).
    This helps the agent experience the trajectory from the beginning.
    """

    # ========================================
    # Joint filtering
    # ========================================
    excluded_joints: list[str] | None = None
    """List of joint names to exclude from trajectory. Defaults to None.
    
    If None, uses default exclusion list:
    ["R_Index_Roll_Joint", "R_Middle_Roll_Joint", "R_Ring_Roll_Joint", "R_Little_Roll_Joint"]
    """

    # ========================================
    # DEPRECATED (kept for config compatibility)
    # ========================================
    chunk_length: int = 10
    """DEPRECATED: No longer used. Kept for config compatibility."""

    ensemble_k: int = 10
    """DEPRECATED: No longer used. Kept for config compatibility."""

    ensemble_m: float = 0.01
    """DEPRECATED: No longer used. Kept for config compatibility."""

    decision_interval: int = 1
    """DEPRECATED: No longer used. Kept for config compatibility."""

    force_reset_on_fail: bool = False
    """DEPRECATED: No longer used. Kept for config compatibility."""
    
    fail_progress_thresh: float = 0.85
    """DEPRECATED: No longer used. Kept for config compatibility."""
    
    fail_tau_thresh: float = 2.0
    """DEPRECATED: No longer used. Kept for config compatibility."""
