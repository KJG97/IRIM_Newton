# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for dexblind_newton (Newton-compatible)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.manipulation.dexblind.mdp.observations import (  # noqa: F401
    reference_joint_pos,
)

from ..config.constants import FINGERTIP_BODIES, SEGMENT_TO_IDX

# =============================================================================
# Joint torque (Newton-specific)
# =============================================================================


def joint_applied_torque(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return MuJoCo qfrc_actuator mapped to Newton joint order."""
    from isaaclab.sim._impl.newton_manager import NewtonManager

    solver = NewtonManager._solver
    model = NewtonManager._model
    qfrc_act = wp.to_torch(solver.mjw_data.qfrc_actuator)
    dof_map_t = wp.to_torch(solver.mjc_dof_to_newton_dof).long()

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


# =============================================================================
# Relative poses (Origin_Body frame)
# =============================================================================


def _relative_pose_to_origin(body_suffix: str, env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N, 7): pos3 + quat_xyzw4 relative to Origin_Body."""
    from .utils import get_body_poses_batched, relative_pose_batched

    n, dev = env.num_envs, env.device
    o_pos, o_quat = get_body_poses_batched("Origin_Body", n, dev)
    t_pos, t_quat = get_body_poses_batched(body_suffix, n, dev)
    rel_pos, rel_quat = relative_pose_batched(o_pos, o_quat, t_pos, t_quat)
    return torch.cat([rel_pos, rel_quat], dim=-1)


def hammer_relative_pose(
    env: ManagerBasedRLEnv,
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
) -> torch.Tensor:
    return _relative_pose_to_origin("hammer", env)


def hammer_initial_relative_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hammer pose relative to Origin_Body at reset/first creation only (N, 7).

    Reads from env._hammer_initial_relative_pose, updated in DexblindNewtonEnv
    on reset; does not change during the episode.
    """
    return env._hammer_initial_relative_pose


def right_hand_relative_pose(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _relative_pose_to_origin("Right_Hand_base", env)


# =============================================================================
# Body velocities
# =============================================================================


def _get_body_vel_batched(
    suffix: str, num_envs: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from isaaclab.sim._impl.newton_manager import NewtonManager
    from .utils import _get_body_indices

    body_qd = wp.to_torch(NewtonManager._state_0.body_qd)
    idx = _get_body_indices(suffix, num_envs, device).clamp(min=0)
    vel = body_qd[idx]
    valid = (_get_body_indices(suffix, num_envs, device) >= 0).unsqueeze(-1)
    vel = vel * valid.float()
    return vel[:, :3], vel[:, 3:6]


def object_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer")) -> torch.Tensor:
    lin, _ = _get_body_vel_batched(asset_cfg.name, env.num_envs, env.device)
    return lin


def object_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer")) -> torch.Tensor:
    _, ang = _get_body_vel_batched(asset_cfg.name, env.num_envs, env.device)
    return ang


def palm_lin_vel(env: ManagerBasedRLEnv, body_name: str = "Right_Hand_base") -> torch.Tensor:
    lin, _ = _get_body_vel_batched(body_name, env.num_envs, env.device)
    return lin


def palm_ang_vel(env: ManagerBasedRLEnv, body_name: str = "Right_Hand_base") -> torch.Tensor:
    _, ang = _get_body_vel_batched(body_name, env.num_envs, env.device)
    return ang


# =============================================================================
# Fingertip–object distance
# =============================================================================


def min_fingertip_object_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
) -> torch.Tensor:
    """Running-minimum fingertip-to-object distance (N, 1). Resets each episode."""
    from .utils import get_body_poses_batched

    n, dev = env.num_envs, env.device
    obj_pos, _ = get_body_poses_batched(asset_cfg.name, n, dev)

    dists = []
    for tip in FINGERTIP_BODIES:
        tip_pos, _ = get_body_poses_batched(tip, n, dev)
        dists.append(torch.norm(tip_pos - obj_pos, dim=-1))
    min_dist = torch.stack(dists, dim=-1).min(dim=-1).values

    buf_key = "_priv_min_ft_obj_dist"
    if not hasattr(env, buf_key):
        setattr(env, buf_key, min_dist.clone())
    else:
        buf = getattr(env, buf_key)
        buf[:] = torch.min(buf, min_dist)
        min_dist = buf
    return min_dist.unsqueeze(-1)


# =============================================================================
# Dynamic goal pose
# =============================================================================


def current_goal_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current dynamic goal pose (N, 7): pos(3) + quat_xyzw(4).

    Reads from env._dynamic_goal_pos / _dynamic_goal_rot which are managed by
    DexblindNewtonEnv (reset to default on episode start, resampled when close).
    """
    return torch.cat([env._dynamic_goal_pos, env._dynamic_goal_rot], dim=-1)


# =============================================================================
# Scalar privileged signals
# =============================================================================


def episode_step_count(env: ManagerBasedRLEnv) -> torch.Tensor:
    max_len = env.max_episode_length if env.max_episode_length > 0 else 1.0
    return (env.episode_length_buf.float() / max_len).unsqueeze(-1)


def object_is_grasped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    lift_threshold: float = 0.91,
) -> torch.Tensor:
    """1 while object z >= threshold, 0 when below."""
    from .utils import root_pos_w_z
    return (root_pos_w_z(env, asset_cfg) >= lift_threshold).float().unsqueeze(-1)


def instantaneous_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "reward_buf"):
        return env.reward_buf.unsqueeze(-1)
    return torch.zeros(env.num_envs, 1, dtype=torch.float32, device=env.device)


# =============================================================================
# Hand contact observations
# =============================================================================


def _build_finger_segment_map(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    finger_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Cached (num_bodies,) finger-index and segment-index tensors. -1 = skip."""
    cache_key = f"_hand_contact_fs_map_{sensor_name}"
    if hasattr(env, cache_key):
        return getattr(env, cache_key)

    body_names = getattr(env.scene.sensors[sensor_name], "body_names", None)
    if body_names is None:
        return None

    finger_map, segment_map = [], []
    for name in body_names:
        parts = name.split("_")
        fi, si = -1, -1
        if len(parts) >= 4 and parts[0] == "R" and parts[1] == "Hand":
            finger_part, seg_part = parts[-2], parts[-1]
            if finger_part in finger_names and seg_part in SEGMENT_TO_IDX:
                fi = finger_names.index(finger_part)
                si = SEGMENT_TO_IDX[seg_part]
        finger_map.append(fi)
        segment_map.append(si)

    result = (
        torch.tensor(finger_map, dtype=torch.long, device=env.device),
        torch.tensor(segment_map, dtype=torch.long, device=env.device),
    )
    setattr(env, cache_key, result)
    return result


def _get_contact_flags(env: ManagerBasedRLEnv, sensor_name: str, threshold: float) -> torch.Tensor | None:
    """(N, B) bool: per-body contact flags, or None if sensor has no data."""
    net_forces = env.scene.sensors[sensor_name].data.net_forces_w
    if net_forces is None:
        return None
    return net_forces.norm(dim=-1) > threshold


def hand_has_contact(
    env: ManagerBasedRLEnv,
    sensor_name: str = "hand_contact_sensor",
    force_threshold: float = 1e-6,
) -> torch.Tensor:
    """Binary (N, 1): any hand link has contact."""
    flags = _get_contact_flags(env, sensor_name, force_threshold)
    if flags is None:
        return torch.zeros(env.num_envs, 1, dtype=torch.float32, device=env.device)
    return flags.any(dim=1).float().unsqueeze(-1)


def hand_contact_per_finger(
    env: ManagerBasedRLEnv,
    sensor_name: str = "hand_contact_sensor",
    finger_names: tuple[str, ...] = ("Index", "Middle", "Ring", "Little", "Thumb"),
    segments_per_finger: int = 3,
    force_threshold: float = 1e-6,
) -> torch.Tensor:
    """Per-finger per-segment contact (N, 5×3=15). Order: Index, Middle, Ring, Little, Thumb × (Prox, Mid, Dist)."""
    nf, ns = len(finger_names), segments_per_finger
    flags = _get_contact_flags(env, sensor_name, force_threshold)
    if flags is None:
        return torch.zeros(env.num_envs, nf * ns, dtype=torch.float32, device=env.device)

    parsed = _build_finger_segment_map(env, sensor_name, finger_names)
    if parsed is None:
        return torch.zeros(env.num_envs, nf * ns, dtype=torch.float32, device=env.device)

    body_to_finger, body_to_segment = parsed
    n = flags.shape[0]
    out = torch.zeros(n, nf * ns, dtype=torch.float32, device=env.device)
    for f in range(nf):
        for s in range(ns):
            mask = (body_to_finger == f) & (body_to_segment == s)
            if mask.any():
                out[:, f * ns + s] = flags[:, mask].any(dim=1).float()
    return out
