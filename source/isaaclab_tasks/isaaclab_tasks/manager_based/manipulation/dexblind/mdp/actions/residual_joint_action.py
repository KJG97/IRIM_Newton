# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Residual joint position action: reference_trajectory + residual * scale."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import residual_joint_action_cfg as residual_action_cfgs

logger = logging.getLogger(__name__)


class ResidualJointPositionAction(JointAction):
    r"""참조 궤적 위에 잔여 관절 위치를 더하는 행동.

    final_action = reference_trajectory + (scale * action + offset) * residual_scale
    참조는 command_manager의 reference_trajectory command에서 가져옴.
    """

    cfg: residual_action_cfgs.ResidualJointPositionActionCfg

    def __init__(self, cfg: residual_action_cfgs.ResidualJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.command_name = cfg.command_name

        try:
            self._command_term = env.command_manager._terms[self.command_name]
        except KeyError:
            raise ValueError(
                f"Command '{self.command_name}' not found. "
                f"Available: {list(env.command_manager._terms.keys())}"
            )

        self._dim_checked = False
        self._reorder: torch.Tensor | None = None

    # 학습 시 env0의 action을 10스텝마다 출력하려면 True로 변경
    _PRINT_ACTION_DEBUG: bool = False
    _debug_step: int = 0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        ref = self._env.command_manager.get_command(self.command_name)

        if not self._dim_checked:
            if ref.shape[1] != actions.shape[1]:
                raise ValueError(
                    f"Trajectory joints {ref.shape[1]} != action joints {actions.shape[1]}. "
                    "Match trajectory file to action joint_names."
                )
            if hasattr(self._command_term, "joint_names"):
                traj_names = self._command_term.joint_names
                action_names = self._joint_names
                if traj_names != action_names:
                    self._reorder = torch.tensor(
                        [traj_names.index(n) for n in action_names],
                        dtype=torch.long,
                        device=self.device,
                    )
                    logger.warning(
                        "Trajectory joint order differs from action; reordering. "
                        "Traj: %s | Action: %s",
                        traj_names,
                        action_names,
                    )
            self._dim_checked = True

        if self._reorder is not None:
            ref = ref[:, self._reorder]

        residual = actions * self._scale + self._offset
        if self.cfg.clip is not None:
            residual = torch.clamp(residual, self._clip[:, :, 0], self._clip[:, :, 1])

        self._processed_actions = ref + residual * self.cfg.residual_scale

        if self._PRINT_ACTION_DEBUG:
            ResidualJointPositionAction._debug_step += 1
            if ResidualJointPositionAction._debug_step % 10 == 1:
                def _fmt(t: torch.Tensor) -> str:
                    return " ".join(f"{x:+.4f}" for x in t.cpu().tolist())
                # env0의 현재 관절 위치 및 토크 읽기
                robot = self._asset
                jp = robot.data.joint_pos[0, self._joint_ids]
                pos_err = self._processed_actions[0] - jp
                # applied torque (PD drive 출력, effort limit 클리핑 후) — 19개 관절
                try:
                    torque_all = robot.data.applied_torque[0]
                    _TORQUE_NAMES = [
                        "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
                        "R_Elbow_Joint",
                        "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
                        "R_Index_Roll_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint",
                        "R_Middle_Roll_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
                        "R_Ring_Roll_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
                        "R_Little_Roll_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint",
                    ]
                    if not hasattr(ResidualJointPositionAction, "_torque_ids"):
                        ids, _ = robot.find_joints(_TORQUE_NAMES)
                        ResidualJointPositionAction._torque_ids = torch.tensor(ids, dtype=torch.long, device=jp.device)
                    torque_19 = torque_all[ResidualJointPositionAction._torque_ids]
                    torque_str = f"  torque   (19): {_fmt(torque_19)}"
                except Exception:
                    torque_str = "  torque: N/A"
                print(
                    f"[train_action] step={ResidualJointPositionAction._debug_step}\n"
                    f"  raw_policy(18): {_fmt(actions[0])}\n"
                    f"  ref_traj  (18): {_fmt(ref[0])}\n"
                    f"  final_act (18): {_fmt(self._processed_actions[0])}\n"
                    f"  joint_pos (18): {_fmt(jp)}\n"
                    f"  pos_error (18): {_fmt(pos_err)}  (max={pos_err.abs().max().item():.4f})\n"
                    f"{torque_str}"
                )

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self.processed_actions, joint_ids=self._joint_ids
        )
