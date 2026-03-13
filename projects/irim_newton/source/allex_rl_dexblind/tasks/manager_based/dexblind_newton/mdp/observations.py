# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for dexblind_newton (Newton-compatible)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from allex_rl_dexblind.tasks.manager_based.dexblind_newton.mdp.dexblind_compat import (  # noqa: F401
    reference_joint_pos,
)

from allex_rl_dexblind.tasks.manager_based.dexblind_newton.config.constants import (
    FINGERTIP_BODIES,
    FINGER_NAMES,
    FINGERTIP_PAD_SHAPE_TO_FINGER_IDX,
    SENSOR_ORDER_WHEN_PAD_NAMES_AMBIGUOUS,
)

NFINGERS = len(FINGER_NAMES)

# =============================================================================
# Relative poses (Origin_Body frame)
# =============================================================================


def _relative_pose_to_origin(body_suffix: str, env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N, 7): pos3 + quat_xyzw4 relative to Origin_Body."""
    from allex_rl_dexblind.tasks.manager_based.dexblind_newton.mdp.utils import (
        get_body_poses_batched,
        relative_pose_batched,
    )

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
    from isaaclab_newton.physics import NewtonManager
    from allex_rl_dexblind.tasks.manager_based.dexblind_newton.mdp.utils import (
        _get_body_indices,
    )

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
    from allex_rl_dexblind.tasks.manager_based.dexblind_newton.mdp.utils import (
        get_body_poses_batched,
    )

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
    from allex_rl_dexblind.tasks.manager_based.dexblind_newton.mdp.utils import (
        root_pos_w_z,
    )
    return (root_pos_w_z(env, asset_cfg) >= lift_threshold).float().unsqueeze(-1)


def instantaneous_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "reward_buf"):
        return env.reward_buf.unsqueeze(-1)
    return torch.zeros(env.num_envs, 1, dtype=torch.float32, device=env.device)


# =============================================================================
# Fingertip–hammer contact (hand_contact_sensor: fingertip bodies × hammer)
# =============================================================================

# FINGER_NAMES order: Index=0, Middle=1, Ring=2, Little=3, Thumb=4 (body names)
_FINGERTIP_BODY_TO_FINGER_IDX = {
    "R_Hand_Index_Distal": 0,
    "R_Hand_Middle_Distal": 1,
    "R_Hand_Ring_Distal": 2,
    "R_Hand_Little_Distal": 3,
    "R_Hand_Thumb_Distal": 4,
}


def _sensor_names_to_finger_indices(names: list[str], nf: int = NFINGERS) -> list[int]:
    """Map sensor names to finger indices. Shape map → body map → fixed order when 5 sensors and any unmapped."""
    s2f = [
        FINGERTIP_PAD_SHAPE_TO_FINGER_IDX.get(n, _FINGERTIP_BODY_TO_FINGER_IDX.get(n, -1))
        for n in names
    ]
    if len(s2f) == nf and -1 in s2f:
        return list(SENSOR_ORDER_WHEN_PAD_NAMES_AMBIGUOUS)
    return s2f


def _get_contact_flags(env: ManagerBasedRLEnv, sensor_name: str, threshold: float) -> torch.Tensor | None:
    """(N, S) bool: per-sensor contact flags (hand–hammer only when filter_prim_paths_expr is set)."""
    data = env.scene.sensors[sensor_name].data
    # Prefer force_matrix_w (SensorContact counterpart = hammer) like dexsuite
    if data.force_matrix_w is not None:
        forces_t = wp.to_torch(data.force_matrix_w)  # (N, S, F, 3)
        return (forces_t.norm(dim=-1) > threshold).any(dim=2)  # (N, S): any counterpart (hammer)
    net_forces = data.net_forces_w
    if net_forces is None:
        return None
    forces_t = wp.to_torch(net_forces)
    return forces_t.norm(dim=-1) > threshold


def fingertip_hammer_contact(
    env: ManagerBasedRLEnv,
    sensor_name: str = "hand_contact_sensor",
    force_threshold: float = 1e-6,
) -> torch.Tensor:
    """Fingertip–hammer contact (N, 5). Order: Index, Middle, Ring, Little, Thumb. 1 = contact, 0 = no contact."""
    n, dev = env.num_envs, env.device
    flags = _get_contact_flags(env, sensor_name, force_threshold)
    if flags is None:
        return torch.zeros(n, NFINGERS, dtype=torch.float32, device=dev)

    cache_key = f"_ft_hammer_s2f_{sensor_name}"
    if not hasattr(env, cache_key):
        sensor = env.scene.sensors[sensor_name]
        names = getattr(sensor, "sensor_names", None) or []
        setattr(env, cache_key, _sensor_names_to_finger_indices(names))
    s2f = getattr(env, cache_key)

    out = torch.zeros(n, NFINGERS, dtype=torch.float32, device=dev)
    n_sensors = flags.shape[1]
    for s_idx, fi in enumerate(s2f):
        if s_idx < n_sensors and fi >= 0:
            out[:, fi] = flags[:, s_idx].float()
    return out
