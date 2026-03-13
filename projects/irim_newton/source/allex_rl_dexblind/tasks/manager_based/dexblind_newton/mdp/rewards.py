# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for dexblind_newton.

Only division-by-zero and acos domain protection (eps, clamp) are applied so that
formula-induced NaN is avoided. No nan_to_num; if state is corrupted, NaN will
propagate so the cause can be identified.

NaN origin (when hammer pose/velocity become NaN in one step):
- Reward/observation code does not introduce NaN from valid inputs (no 0/0, safe acos).
- NaN almost always originates in the Newton physics step (contact/constraint solver
  or integration): e.g. degenerate contact normal, near-singular constraint Jacobian,
  or a single large impulse. Velocity need not be "abnormally high" for the solver
  to output NaN. The env then zeros reward and forces reset for affected envs.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from allex_rl_dexblind.tasks.manager_based.dexblind_newton.mdp.utils import (
    root_pos_w_z,
    get_body_poses_batched,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Avoid 0/0 or division-by-zero when std params are (erroneously) zero.
_EPS = 1e-6


def hammer_goal_proximity_reward(
    env: ManagerBasedRLEnv,
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    pos_std: float = 0.05,
    rot_std: float = 0.2,
    pos_weight: float = 0.5,
    rot_weight: float = 0.5,
    lift_threshold: float = 0.92,
) -> torch.Tensor:
    """Dense reward for bringing the hammer close to the dynamic goal pose.

    Reads per-env goal from env._dynamic_goal_pos / _dynamic_goal_rot
    (managed by DexblindNewtonEnv). Gated by hammer lift threshold.
    """
    n, dev = env.num_envs, env.device
    lifted = root_pos_w_z(env, hammer_cfg) >= lift_threshold

    h_pos, h_quat = get_body_poses_batched("hammer", n, dev)
    local_pos = h_pos - env.scene.env_origins

    g_pos = env._dynamic_goal_pos  # (N, 3)
    g_quat = env._dynamic_goal_rot  # (N, 4)

    pos_err = torch.norm(local_pos - g_pos, dim=-1)
    safe_pos_std = max(pos_std, _EPS)
    pos_rew = torch.exp(-pos_err / safe_pos_std)

    dot = torch.sum(h_quat * g_quat, dim=-1).abs().clamp(min=0.0, max=1.0)
    rot_err = torch.acos(dot)
    safe_rot_std = max(rot_std, _EPS)
    rot_rew = torch.exp(-rot_err / safe_rot_std)

    return (pos_weight * pos_rew + rot_weight * rot_rew) * lifted.float()


_GRASP_POINT_LOCAL_OFFSET = (-0.005, -0.13, 0.0)


def grasp_point_proximity_reward(
    env: ManagerBasedRLEnv,
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    pos_std: float = 0.05,
    grasp_offset: tuple[float, float, float] = _GRASP_POINT_LOCAL_OFFSET,
) -> torch.Tensor:
    """Dense reward for bringing Right_Hand_base close to the hammer's grasp_point.

    Always active (no lift gate) so the agent gets reward for approaching the
    grasp point even before lifting, encouraging retry after a failed grasp.

    grasp_point is a non-rigid Xform child of the hammer body. We compute its
    world position by applying the local offset to the hammer body pose.

    reward = exp(-||hand_pos - grasp_world_pos|| / pos_std)
    """
    from isaaclab.utils.math import quat_apply

    n, dev = env.num_envs, env.device
    hammer_pos, hammer_quat = get_body_poses_batched("hammer", n, dev)
    hand_pos, _ = get_body_poses_batched("Right_Hand_base", n, dev)

    offset = torch.tensor(grasp_offset, dtype=torch.float32, device=dev).expand(n, 3)
    grasp_world_pos = hammer_pos + quat_apply(hammer_quat, offset)

    dist = torch.norm(hand_pos - grasp_world_pos, dim=-1)
    safe_pos_std = max(pos_std, _EPS)
    return torch.exp(-dist / safe_pos_std)


def fingertip_hammer_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_name: str = "hand_contact_sensor",
    force_threshold: float = 1e-6,
) -> torch.Tensor:
    from allex_rl_dexblind.tasks.manager_based.dexblind_newton.mdp.observations import (
        _get_contact_flags,
    )

    flags = _get_contact_flags(env, sensor_name, force_threshold)
    if flags is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return flags.float().sum(dim=1)  


def hammer_lift_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    threshold: float = 0.9,
    late_scale: float = 1.0,
) -> torch.Tensor:
    """Reward when hammer z >= threshold; scaled linearly by episode progress.

    time_weight = 1.0 + (late_scale - 1.0) * progress, so later in the episode
    the same lift gives more reward when late_scale > 1.0.

    Args:
        asset_cfg: Hammer scene entity.
        threshold: Minimum hammer z to count as lifted.
        late_scale: Multiplier at episode end (progress=1). At start (progress=0) weight is 1.0. Default 1.0 (no scaling).
    """
    z = root_pos_w_z(env, asset_cfg)
    lifted = (z >= threshold).float()

    if late_scale == 1.0:
        return lifted

    max_len = env.max_episode_length if env.max_episode_length > 0 else 1
    progress = env.episode_length_buf.float().to(env.device) / max_len
    time_weight = 1.0 + (late_scale - 1.0) * progress
    return lifted * time_weight


def late_lift_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    lift_threshold: float = 0.92,
    mid_episode_ratio: float = 0.5,
) -> torch.Tensor:
    """Returns 1.0 when past mid-episode and hammer is still not lifted (for use as penalty with negative weight).

    Use with weight < 0 so that the agent is penalized for not having lifted
    the hammer by the middle of the episode.
    """
    max_len = env.max_episode_length if env.max_episode_length > 0 else 1
    progress = env.episode_length_buf.float().to(env.device) / max_len
    past_mid = progress >= mid_episode_ratio
    z = root_pos_w_z(env, asset_cfg)
    not_lifted = z < lift_threshold
    return (past_mid & not_lifted).float()


def reference_trajectory_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "reference_trajectory",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
    joint_std: float = 0.1,
    hammer_cfg: SceneEntityCfg | None = None,
    lift_threshold: float | None = None,
) -> torch.Tensor:
    """Dense reward for tracking the current reference trajectory (joint positions).

    Compares current robot joint positions to the reference trajectory's current
    target at this time step. reward = mean over joints of exp(-|q - q_ref| / joint_std).
    Only joints present in both the robot and the trajectory are used (trajectory
    may exclude e.g. roll joints).

    Optional: when hammer_cfg and lift_threshold are set, reward is zero until
    hammer z >= lift_threshold (only reward ref tracking after grasping the hammer).
    """
    n, dev = env.num_envs, env.device
    if joint_names is None:
        return torch.zeros(n, device=dev)

    cmd = env.command_manager.get_term(command_name)
    ref = cmd.command  # (N, num_traj_joints)
    common = [j for j in joint_names if j in cmd.joint_names]
    if not common:
        return torch.zeros(n, device=dev)

    traj_indices = [cmd.joint_names.index(j) for j in common]
    ref_sub = ref[:, traj_indices]  # (N, len(common))

    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = wp.to_torch(robot.data.joint_pos)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, (list, tuple)):
        joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=dev)
    current_full = joint_pos[:, joint_ids]  # (N, len(joint_names))
    name_to_col = {j: i for i, j in enumerate(joint_names)}
    robot_cols = [name_to_col[j] for j in common]
    current_sub = current_full[:, robot_cols]  # (N, len(common))

    err = torch.abs(current_sub - ref_sub)
    safe_joint_std = max(joint_std, _EPS)
    per_joint_rew = torch.exp(-err / safe_joint_std)
    rew = per_joint_rew.mean(dim=-1)

    if hammer_cfg is not None and lift_threshold is not None:
        lifted = root_pos_w_z(env, hammer_cfg) >= lift_threshold
        rew = rew * lifted.float()
    return rew


def hand_final_pose_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "reference_trajectory",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    hand_joint_names: list[str] | None = None,
    lift_threshold: float = 1.0,
    joint_std: float = 0.1,
) -> torch.Tensor:
    """Reward hand joints for matching the trajectory's final frame pose.

    Gated by hammer lift: only active when hammer z >= lift_threshold.

    Uses 0.5*mean(r_i) + 0.5*min(r_i) where r_i = exp(-|e_i|/joint_std): mean gives
    dense gradient to all joints, min penalises any lagging joint; convex combination
    keeps reward scale in [0,1] (product would squash scale).

    Args:
        command_name: name of the ReferenceTrajectoryCommand in command_manager.
        asset_cfg: robot articulation (with hand_joint_names resolved).
        hammer_cfg: hammer scene entity for lift gating.
        hand_joint_names: hand joint names to track. Must match trajectory joint names.
        lift_threshold: minimum hammer z to activate this reward.
        joint_std: std for exp(-error/std) shaping (smaller = tighter constraint).
    """
    n, dev = env.num_envs, env.device

    gate = (root_pos_w_z(env, hammer_cfg) >= lift_threshold).float()

    cmd = env.command_manager.get_term(command_name)
    final_positions = cmd.positions[-1]  # (num_traj_joints,)

    if hand_joint_names is None:
        return torch.zeros(n, device=dev)

    traj_indices = []
    for jn in hand_joint_names:
        if jn in cmd.joint_names:
            traj_indices.append(cmd.joint_names.index(jn))
        else:
            return torch.zeros(n, device=dev)
    target = final_positions[traj_indices].unsqueeze(0).expand(n, -1)  # (N, num_hand_joints)

    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = wp.to_torch(robot.data.joint_pos)  # (N, total_joints)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, (list, tuple)):
        joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=dev)
    current = joint_pos[:, joint_ids]  # (N, num_hand_joints)

    err = torch.abs(current - target)
    safe_joint_std = max(joint_std, _EPS)
    per_joint_rew = torch.exp(-err / safe_joint_std)  # (N, num_hand_joints)

    r_mean = per_joint_rew.mean(dim=-1)
    r_min = per_joint_rew.min(dim=-1).values
    # Convex combination: scale stays in [0,1]; mean gives gradient, min penalises lagging joints.
    rew = 0.1 * r_mean + 0.9 * r_min
    return rew * gate