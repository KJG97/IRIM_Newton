# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for dexblind_newton. Newton-compatible wrappers where needed."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.manipulation.dexblind.mdp.observations import (  # noqa: F401
    reference_joint_pos,
)

def joint_applied_torque(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """MuJoCo qfrc_actuator를 Newton joint 순서로 반환.

    Implicit actuator에서 PD 토크는 MuJoCo 내부에서 계산되므로
    Control.joint_f (= robot.data.joint_effort)는 항상 0.
    실제 액추에이터 토크는 mjw_data.qfrc_actuator에 있다.
    """
    from isaaclab.sim._impl.newton_manager import NewtonManager

    solver = NewtonManager._solver
    model = NewtonManager._model
    mjw_data = solver.mjw_data
    dof_map = solver.mjc_dof_to_newton_dof  # (nworld, nv): mjc_dof → global newton dof

    qfrc_act = wp.to_torch(mjw_data.qfrc_actuator)   # (nworld, nv)
    dof_map_t = wp.to_torch(dof_map).long()           # (nworld, nv)

    nworld = qfrc_act.shape[0]
    dofs_per_world = model.joint_dof_count // model.num_worlds
    device = qfrc_act.device

    torque_flat = torch.zeros(nworld * dofs_per_world, dtype=qfrc_act.dtype, device=device)
    valid = dof_map_t >= 0
    torque_flat.scatter_(0, dof_map_t[valid], qfrc_act[valid])
    torque_newton = torque_flat.view(nworld, dofs_per_world)

    robot: Articulation = env.scene[asset_cfg.name]
    rv = robot.root_view
    dof_begin = rv._arti_joint_dof_begin % dofs_per_world
    dof_end = rv._arti_joint_dof_end % dofs_per_world
    robot_torque = torque_newton[:, dof_begin:dof_end]

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, (list, tuple)):
        joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=device)
    return robot_torque[:, joint_ids]


def hammer_relative_pose(
    env: ManagerBasedRLEnv,
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
) -> torch.Tensor:
    """Origin_Body 기준 Hammer의 상대좌표 (pos3 + quat_xyzw4) = 7-dim."""
    from .utils import get_body_poses_batched, relative_pose_batched

    n, dev = env.num_envs, env.device
    o_pos, o_quat = get_body_poses_batched("Origin_Body", n, dev)
    h_pos, h_quat = get_body_poses_batched("hammer", n, dev)
    rel_pos, rel_quat = relative_pose_batched(o_pos, o_quat, h_pos, h_quat)
    return torch.cat([rel_pos, rel_quat], dim=-1)


def right_hand_relative_pose(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Origin_Body 기준 Right_Hand_base의 상대좌표 (pos3 + quat_xyzw4) = 7-dim."""
    from .utils import get_body_poses_batched, relative_pose_batched

    n, dev = env.num_envs, env.device
    o_pos, o_quat = get_body_poses_batched("Origin_Body", n, dev)
    h_pos, h_quat = get_body_poses_batched("Right_Hand_base", n, dev)
    rel_pos, rel_quat = relative_pose_batched(o_pos, o_quat, h_pos, h_quat)
    return torch.cat([rel_pos, rel_quat], dim=-1)


# ---------------------------------------------------------------------------
#  Privileged observations for Asymmetric Critic (SimToolReal Appendix D.4)
# ---------------------------------------------------------------------------

_FINGERTIP_BODIES = [
    "R_Hand_Thumb_Distal",
    "R_Hand_Index_Distal",
    "R_Hand_Middle_Distal",
    "R_Hand_Ring_Distal",
    "R_Hand_Little_Distal",
]


def _get_body_vel_batched(
    suffix: str, num_envs: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (lin_vel (N,3), ang_vel (N,3)) for a named body across all envs."""
    from isaaclab.sim._impl.newton_manager import NewtonManager
    from .utils import _get_body_indices

    state = NewtonManager._state_0
    body_qd = wp.to_torch(state.body_qd)  # (total_bodies, 6)
    indices = _get_body_indices(suffix, num_envs, device)

    safe_idx = indices.clamp(min=0)
    vel = body_qd[safe_idx]  # (N, 6): [lin_vel(3), ang_vel(3)]

    valid = (indices >= 0).unsqueeze(-1)
    vel = vel * valid.float()

    return vel[:, :3], vel[:, 3:6]


def object_lin_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
) -> torch.Tensor:
    """Ground-truth object linear velocity (N, 3)."""
    lin_vel, _ = _get_body_vel_batched(asset_cfg.name, env.num_envs, env.device)
    return lin_vel


def object_ang_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
) -> torch.Tensor:
    """Ground-truth object angular velocity (N, 3)."""
    _, ang_vel = _get_body_vel_batched(asset_cfg.name, env.num_envs, env.device)
    return ang_vel


def palm_lin_vel(
    env: ManagerBasedRLEnv,
    body_name: str = "Right_Hand_base",
) -> torch.Tensor:
    """Ground-truth palm linear velocity (N, 3)."""
    lin_vel, _ = _get_body_vel_batched(body_name, env.num_envs, env.device)
    return lin_vel


def palm_ang_vel(
    env: ManagerBasedRLEnv,
    body_name: str = "Right_Hand_base",
) -> torch.Tensor:
    """Ground-truth palm angular velocity (N, 3)."""
    _, ang_vel = _get_body_vel_batched(body_name, env.num_envs, env.device)
    return ang_vel


def min_fingertip_object_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
) -> torch.Tensor:
    """Minimum distance from any fingertip to the object (N, 1).

    Uses a running minimum that resets each episode (stored in env).
    """
    from .utils import get_body_poses_batched

    n, dev = env.num_envs, env.device
    obj_pos, _ = get_body_poses_batched(asset_cfg.name, n, dev)

    dists = []
    for tip_name in _FINGERTIP_BODIES:
        tip_pos, _ = get_body_poses_batched(tip_name, n, dev)
        dists.append(torch.norm(tip_pos - obj_pos, dim=-1))
    min_dist = torch.stack(dists, dim=-1).min(dim=-1).values  # (N,)

    buf_key = "_priv_min_ft_obj_dist"
    if not hasattr(env, buf_key):
        setattr(env, buf_key, min_dist.clone())
    else:
        buf = getattr(env, buf_key)
        buf[:] = torch.min(buf, min_dist)
        min_dist = buf

    return min_dist.unsqueeze(-1)


def episode_step_count(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Normalized episode step count (N, 1). Divided by max_episode_length for stability."""
    max_len = env.max_episode_length if env.max_episode_length > 0 else 1.0
    return (env.episode_length_buf.float() / max_len).unsqueeze(-1)


def object_is_grasped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    lift_threshold: float = 0.91,
) -> torch.Tensor:
    """Binary grasped indicator I_grasped (N, 1). True once object z >= threshold."""
    from .utils import root_pos_w_z

    z = root_pos_w_z(env, asset_cfg)
    currently_lifted = z >= lift_threshold

    buf_key = "_priv_grasped_flag"
    if not hasattr(env, buf_key):
        setattr(env, buf_key, currently_lifted.clone())
    else:
        buf = getattr(env, buf_key)
        buf[:] = buf | currently_lifted

    return getattr(env, buf_key).float().unsqueeze(-1)


def instantaneous_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Most recent per-env reward r_t (N, 1)."""
    if hasattr(env, "reward_buf"):
        return env.reward_buf.unsqueeze(-1)
    return torch.zeros(env.num_envs, 1, dtype=torch.float32, device=env.device)


def cumulative_successes(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    lift_threshold: float = 0.91,
) -> torch.Tensor:
    """Cumulative number of times the object was lifted this episode (N, 1).

    Increments each time object transitions from below to above threshold.
    """
    from .utils import root_pos_w_z

    z = root_pos_w_z(env, asset_cfg)
    currently_lifted = z >= lift_threshold

    prev_key = "_priv_prev_lifted"
    count_key = "_priv_cum_successes"
    if not hasattr(env, prev_key):
        setattr(env, prev_key, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        setattr(env, count_key, torch.zeros(env.num_envs, dtype=torch.float32, device=env.device))

    prev = getattr(env, prev_key)
    count = getattr(env, count_key)
    new_lift = currently_lifted & ~prev
    count[:] = count + new_lift.float()
    prev[:] = currently_lifted

    return count.unsqueeze(-1)
