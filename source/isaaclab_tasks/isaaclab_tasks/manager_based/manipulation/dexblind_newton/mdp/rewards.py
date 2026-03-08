# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for dexblind_newton.

Only division-by-zero and acos domain protection (eps, clamp) are applied so that
formula-induced NaN is avoided. No nan_to_num; if state is corrupted, NaN will
propagate so the cause can be identified.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .utils import root_pos_w_z, get_body_poses_batched

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Avoid 0/0 or division-by-zero when std params are (erroneously) zero.
_EPS = 1e-6


def hammer_goal_proximity_reward(
    env: ManagerBasedRLEnv,
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    goal_pos: tuple[float, float, float] = (0.5, -0.2, 1.15),
    goal_rot: tuple[float, float, float, float] = (0.0, -0.70711, -0.70711, 0.0),
    pos_std: float = 0.05,
    rot_std: float = 0.2,
    pos_weight: float = 0.5,
    rot_weight: float = 0.5,
    lift_threshold: float = 0.92,
) -> torch.Tensor:
    """Dense reward for bringing the hammer close to the goal marker pose.

    goal_pos/goal_rot are in env-local coordinates (same frame as GoalMarkerCfg).
    The hammer world pose is converted to env-local by subtracting env_origins.

    Gated by hammer lift: reward is zero until hammer z >= lift_threshold.
    """
    from isaaclab.utils.math import quat_mul, quat_inv

    n, dev = env.num_envs, env.device
    lifted = root_pos_w_z(env, hammer_cfg) >= lift_threshold

    h_pos, h_quat = get_body_poses_batched("hammer", n, dev)

    env_origins = env.scene.env_origins  # (N, 3)
    local_pos = h_pos - env_origins

    g_pos = torch.tensor(goal_pos, dtype=torch.float32, device=dev).unsqueeze(0).expand(n, 3)
    g_quat = torch.tensor(goal_rot, dtype=torch.float32, device=dev).unsqueeze(0).expand(n, 4)

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
    lift_threshold: float = 0.92,
    grasp_offset: tuple[float, float, float] = _GRASP_POINT_LOCAL_OFFSET,
) -> torch.Tensor:
    """Dense reward for bringing Right_Hand_base close to the hammer's grasp_point.

    Gated by hammer lift: reward is zero until hammer z >= lift_threshold.

    grasp_point is a non-rigid Xform child of the hammer body, so it does not
    appear in Newton body_q. We compute its world position by applying the
    local offset to the hammer body pose via quaternion rotation.

    reward = exp(-||hand_pos - grasp_world_pos|| / pos_std) * lifted
    """
    from isaaclab.utils.math import quat_apply

    n, dev = env.num_envs, env.device
    lifted = root_pos_w_z(env, hammer_cfg) >= lift_threshold

    hammer_pos, hammer_quat = get_body_poses_batched("hammer", n, dev)
    hand_pos, _ = get_body_poses_batched("Right_Hand_base", n, dev)

    offset = torch.tensor(grasp_offset, dtype=torch.float32, device=dev).expand(n, 3)
    grasp_world_pos = hammer_pos + quat_apply(hammer_quat, offset)

    dist = torch.norm(hand_pos - grasp_world_pos, dim=-1)
    safe_pos_std = max(pos_std, _EPS)
    return torch.exp(-dist / safe_pos_std) * lifted.float()


def hammer_lift_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    threshold: float = 0.9,
) -> torch.Tensor:
    """Reward 1.0 when hammer z >= threshold, else 0.0."""
    z = root_pos_w_z(env, asset_cfg)
    return (z >= threshold).float()


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
    goal_pos: tuple[float, float, float] = (0.65, -0.2, 1.2),
    goal_rot: tuple[float, float, float, float] = (0.0, -0.70711, -0.70711, 0.0),
    proximity_threshold: float = 0.8,
    joint_std: float = 0.1,
) -> torch.Tensor:
    """Reward hand joints for matching the trajectory's final frame pose.

    Gated by hammer-goal proximity: only active when the combined
    pos+rot proximity score exceeds proximity_threshold.

    Args:
        command_name: name of the ReferenceTrajectoryCommand in command_manager.
        asset_cfg: robot articulation (with hand_joint_names resolved).
        hammer_cfg: hammer scene entity for proximity gating.
        hand_joint_names: hand joint names to track. Must match trajectory joint names.
        goal_pos: goal position for proximity gating (env-local).
        goal_rot: goal quaternion for proximity gating (env-local, xyzw).
        proximity_threshold: minimum proximity score to activate this reward.
        joint_std: std for exp(-error/std) shaping.
    """
    n, dev = env.num_envs, env.device

    # --- proximity gate ---
    h_pos, h_quat = get_body_poses_batched("hammer", n, dev)
    local_pos = h_pos - env.scene.env_origins
    g_pos = torch.tensor(goal_pos, dtype=torch.float32, device=dev).unsqueeze(0)
    g_quat = torch.tensor(goal_rot, dtype=torch.float32, device=dev).unsqueeze(0)

    pos_err = torch.norm(local_pos - g_pos, dim=-1)
    pos_score = torch.exp(-pos_err / max(0.05, _EPS))
    dot = torch.sum(h_quat * g_quat, dim=-1).abs().clamp(min=0.0, max=1.0)
    rot_score = torch.exp(-torch.acos(dot) / 0.2)
    proximity = 0.5 * pos_score + 0.5 * rot_score
    gate = (proximity >= proximity_threshold).float()

    # --- get final frame hand joint targets from trajectory ---
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

    # --- current hand joint positions ---
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = wp.to_torch(robot.data.joint_pos)  # (N, total_joints)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, (list, tuple)):
        joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=dev)
    current = joint_pos[:, joint_ids]  # (N, num_hand_joints)

    # --- per-joint error → exp reward ---
    err = torch.abs(current - target)
    safe_joint_std = max(joint_std, _EPS)
    per_joint_rew = torch.exp(-err / safe_joint_std)
    mean_rew = per_joint_rew.mean(dim=-1)  # (N,)

    return mean_rew * gate
