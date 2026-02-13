# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# PLAY 시 터미널에 right_hand_joint_torque(19개) 출력하려면 True로 변경
_PRINT_JOINT_TORQUE_DEBUG = False

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_joint_ids(robot: Articulation, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_names: list[str] | None):
    """asset_cfg와 joint_names로 joint_ids 또는 slice(None) 반환."""
    if joint_names is not None:
        ids, _ = robot.find_joints(joint_names)
        return torch.tensor(ids, dtype=torch.long, device=env.device)
    if getattr(asset_cfg, "joint_names", None) is not None:
        ids, _ = robot.find_joints(asset_cfg.joint_names)
        return torch.tensor(ids, dtype=torch.long, device=env.device)
    if asset_cfg.joint_ids != slice(None):
        return (
            torch.tensor(asset_cfg.joint_ids, dtype=torch.long, device=env.device)
            if isinstance(asset_cfg.joint_ids, (list, tuple))
            else asset_cfg.joint_ids
        )
    return slice(None)


def joint_applied_torque(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Applied torque (PD drive 출력, effort limit 클리핑 후) 반환.

    ImplicitActuator: Kp*(target-pos) + Kd*(0-vel) → effort_limit 클리핑.
    Isaac Lab 기본 joint_torques_l2 와 동일한 물리량 (asset.data.applied_torque).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    result = robot.data.applied_torque[:, asset_cfg.joint_ids]

    if _PRINT_JOINT_TORQUE_DEBUG and result.shape[1] == 19:
        if not hasattr(joint_applied_torque, "_debug_step"):
            joint_applied_torque._debug_step = 0
        joint_applied_torque._debug_step += 1
        if joint_applied_torque._debug_step % 10 == 1:
            v = result[0].cpu().numpy()
            print(f"[right_hand_joint_torque] env0 step={joint_applied_torque._debug_step} dim=19 | " + " ".join(f"{x:.4f}" for x in v))

    return result


def right_hand_base_pos_b(
    env: ManagerBasedRLEnv,
    body_name: str = "R_Hand_Pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Origin_Body 기준 R_Hand_Pose의 위치+쿼터니언 [pos(3), quat(4)] 반환."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_ids, _ = robot.find_bodies(body_name)
    origin_ids, _ = robot.find_bodies("Origin_Body")
    if not body_ids or not origin_ids:
        raise ValueError(
            f"Body '{body_name}' or 'Origin_Body' not found. Available: {robot.body_names}"
        )
    body_id, origin_id = body_ids[0], origin_ids[0]

    pos_origin, quat_origin = subtract_frame_transforms(
        robot.data.body_pos_w[:, origin_id],
        robot.data.body_quat_w[:, origin_id],
        robot.data.body_pos_w[:, body_id],
        robot.data.body_quat_w[:, body_id],
    )
    return torch.cat([pos_origin, quat_unique(quat_origin)], dim=-1)


def reference_joint_pos(
    env: ManagerBasedRLEnv,
    command_name: str = "reference_trajectory",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """참조 궤적의 관절 목표 위치 (num_envs, num_joints). joint_names 순서."""
    try:
        ref = env.command_manager.get_command(command_name)
        term = env.command_manager.get_term(command_name)
        names = joint_names if joint_names is not None else getattr(asset_cfg, "joint_names", None)
        if names is not None and hasattr(term, "joint_names") and list(term.joint_names) != list(names):
            traj_names = term.joint_names
            idx = [traj_names.index(n) for n in names]
            ref = ref[:, idx]
    except (KeyError, AttributeError):
        ref = torch.zeros(env.num_envs, len(joint_names) if joint_names else 18, device=env.device)
    return ref


def grasp_progress_chunked(
    env: ManagerBasedRLEnv,
    action_name: str = "action",
) -> torch.Tensor:
    """ChunkedTrajectoryAction의 현재 state index (0, 1, 2, ...)를 (num_envs, 1) float로 반환."""
    try:
        term = env.action_manager._terms[action_name]
        if hasattr(term, "current_index"):
            return term.current_index.float().unsqueeze(1)
    except (KeyError, AttributeError):
        pass
    return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)
