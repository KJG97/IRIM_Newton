# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_ACTION_RATE_CLAMP = 1000.0

# -----------------------------------------------------------------------------
# Action / Regularization
# -----------------------------------------------------------------------------


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """행동 변화율에 대한 L2 패널티 (clamped)."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta), dim=1).clamp(-_ACTION_RATE_CLAMP, _ACTION_RATE_CLAMP)


# -----------------------------------------------------------------------------
# Task Rewards (Privileged: obs 미사용, reward만 사용)
# -----------------------------------------------------------------------------


def hammer_lift_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    threshold: float = 0.55,
) -> torch.Tensor:
    """망치가 threshold 이상 높이에 있으면 1, 아니면 0."""
    hammer: RigidObject = env.scene[asset_cfg.name]
    return (hammer.data.root_pos_w[:, 2] >= threshold).float()


def hammer_fallen_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    z_threshold: float = 0.4,
) -> torch.Tensor:
    """망치 높이가 z_threshold 미만이면 1.0 (패널티), 이상이면 0.0. weight를 음수로 주면 낮은 높이만 패널티."""
    hammer: RigidObject = env.scene[asset_cfg.name]
    return (hammer.data.root_pos_w[:, 2] < z_threshold).float()


def _hand_handle_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    hammer_cfg: SceneEntityCfg,
    hand_body_name: str,
    handle_offset: tuple[float, float, float],
) -> torch.Tensor:
    """손(R_Hand_Pose)과 망치 손잡이(handle_offset 적용) 사이 거리. shape (num_envs,)."""
    robot: Articulation = env.scene[robot_cfg.name]
    hammer: RigidObject = env.scene[hammer_cfg.name]
    device = env.device
    n = env.num_envs

    body_ids, _ = robot.find_bodies(hand_body_name)
    if not body_ids:
        return torch.full((n,), float("inf"), device=device)

    hand_pos_w = robot.data.body_pos_w[:, body_ids[0]]
    hammer_pos_w = hammer.data.root_pos_w
    hammer_quat_w = hammer.data.root_quat_w
    offset = torch.tensor(handle_offset, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    handle_pos_w = hammer_pos_w + quat_apply(hammer_quat_w, offset)
    return torch.norm(hand_pos_w - handle_pos_w, dim=1)


def hand_hammer_distance_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    hand_body_name: str = "R_Hand_Pose",
    std: float = 0.05,
    handle_offset: tuple[float, float, float] = (0.0, -0.13, 0.0),
) -> torch.Tensor:
    """손-망치(손잡이) 거리 기반 보상. 거리 ↓ → 보상 ↑ (exp 커널)."""
    distance = _hand_handle_distance(env, robot_cfg, hammer_cfg, hand_body_name, handle_offset)
    return torch.exp(-distance / std)


def hammer_goal_pos_reward(
    env: ManagerBasedRLEnv,
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_pos: tuple[float, float, float] = (0.5, -0.2, 0.7),
    pos_std: float = 0.1,
) -> torch.Tensor:
    """망치 목표 위치(로봇 base 기준)에 대한 dense 보상. 위치 오차 ↓ → 보상 ↑ (exp 커널)."""
    hammer: RigidObject = env.scene[hammer_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device
    n = env.num_envs

    hammer_pos_w = hammer.data.root_pos_w
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w

    target_pos_w = torch.tensor(target_pos, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    target_quat_w = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    target_pos_world, _ = combine_frame_transforms(root_pos, root_quat, target_pos_w, target_quat_w)

    pos_error = torch.norm(hammer_pos_w - target_pos_world, dim=1)
    return torch.exp(-pos_error / pos_std)


def hammer_goal_quat_reward(
    env: ManagerBasedRLEnv,
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    quat_std: float = 0.5,
) -> torch.Tensor:
    """망치 목표 자세(로봇 base 기준)에 대한 dense 보상. 자세 오차 ↓ → 보상 ↑ (exp 커널)."""
    hammer: RigidObject = env.scene[hammer_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device
    n = env.num_envs

    hammer_quat_w = hammer.data.root_quat_w
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w

    target_pos_w = torch.zeros(n, 3, device=device, dtype=torch.float32)
    target_quat_w = torch.tensor(target_quat, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    _, target_quat_world = combine_frame_transforms(root_pos, root_quat, target_pos_w, target_quat_w)

    quat_error = quat_error_magnitude(hammer_quat_w, target_quat_world)
    return torch.exp(-quat_error / quat_std)


def grasp_success_reward(
    env: ManagerBasedRLEnv,
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_name: str = "R_Hand_Pose",
    handle_offset: tuple[float, float, float] = (0.0, -0.13, 0.0),
    hand_handle_std: float = 0.02,
    progress_thresh: float = 0.99,
    height_thresh: float = 0.6,
    command_name: str = "reference_trajectory",
    sensor_cfgs: list[SceneEntityCfg] | None = None,
    min_contact_force: float = 0.1,
    hammer_filter_idx: int = 0,
) -> torch.Tensor:
    """강한 제약: 진행률·높이·접촉·손이 손잡이 2cm 이내 모두 충족 시 1, 하나라도 아니면 0."""
    device = env.device
    n = env.num_envs

    # 1. trajectory_progress
    try:
        ref_term = env.command_manager.get_term(command_name)
        progress = ref_term.metrics.get("trajectory_progress", None)
    except (KeyError, AttributeError):
        progress = None
    if progress is None:
        progress_ok = torch.zeros(n, device=device, dtype=torch.bool)
    else:
        progress_ok = progress >= progress_thresh

    # 2. hammer height
    hammer: RigidObject = env.scene[hammer_cfg.name]
    height_ok = hammer.data.root_pos_w[:, 2] >= height_thresh

    # 3. hand-hammer contact
    if sensor_cfgs is None:
        contact_ok = torch.zeros(n, device=device, dtype=torch.bool)
    else:
        count = _count_contact_bodies(env, sensor_cfgs, hammer_filter_idx, min_contact_force)
        contact_ok = count >= 1

    # 4. 손-손잡이 거리: 2cm(hand_handle_std) 이내만 성공 (강한 제약, 0/1)
    distance = _hand_handle_distance(env, robot_cfg, hammer_cfg, hand_body_name, handle_offset)
    near_handle = distance <= hand_handle_std

    success = progress_ok & height_ok & contact_ok & near_handle
    return success.float()


def grasp_failure_penalty(
    env: ManagerBasedRLEnv,
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_name: str = "R_Hand_Pose",
    handle_offset: tuple[float, float, float] = (0.0, -0.13, 0.0),
    hand_handle_std: float = 0.02,
    progress_thresh: float = 0.99,
    height_thresh: float = 0.6,
    command_name: str = "reference_trajectory",
    sensor_cfgs: list[SceneEntityCfg] | None = None,
    min_contact_force: float = 0.1,
    hammer_filter_idx: int = 0,
) -> torch.Tensor:
    """실패 시 1.0, 성공 시 0.0. weight를 음수로 주면 실패할 때만 패널티."""
    success = grasp_success_reward(
        env,
        hammer_cfg=hammer_cfg,
        robot_cfg=robot_cfg,
        hand_body_name=hand_body_name,
        handle_offset=handle_offset,
        hand_handle_std=hand_handle_std,
        progress_thresh=progress_thresh,
        height_thresh=height_thresh,
        command_name=command_name,
        sensor_cfgs=sensor_cfgs,
        min_contact_force=min_contact_force,
        hammer_filter_idx=hammer_filter_idx,
    )
    return 1.0 - success


def grasp_final_pose_tracking_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_name: str = "R_Hand_Pose",
    handle_offset: tuple[float, float, float] = (0.0, -0.13, 0.0),
    hand_handle_max: float = 0.02,
    command_name: str = "reference_trajectory",
    joint_names: list[str] | None = None,
    finger_joint_names: list[str] | None = None,
    progress_thresh: float = 0.99,
    height_thresh: float = 0.6,
    sensor_cfgs: list[SceneEntityCfg] | None = None,
    min_contact_force: float = 0.1,
    hammer_filter_idx: int = 0,
    arm_std: float = 0.3,
    finger_std: float = 0.15,
    finger_weight: float = 3.0,
    arm_weight: float = 1.0,
) -> torch.Tensor:
    """성공 조건 + 손이 손잡이 2cm 이내일 때만 궤적 마지막 자세 추적 보상 적용.

    hand_handle_max 이내에 손이 들어와야 grasp_final_pose_tracking 보상이 적용됨.
    """
    device = env.device
    n = env.num_envs

    # ---- 1. 성공 조건 평가 (진행률·높이·접촉·손-손잡이 거리) ----
    success = grasp_success_reward(
        env,
        hammer_cfg=hammer_cfg,
        robot_cfg=robot_cfg,
        hand_body_name=hand_body_name,
        handle_offset=handle_offset,
        hand_handle_std=hand_handle_max,
        progress_thresh=progress_thresh,
        height_thresh=height_thresh,
        command_name=command_name,
        sensor_cfgs=sensor_cfgs,
        min_contact_force=min_contact_force,
        hammer_filter_idx=hammer_filter_idx,
    )
    success_mask = success > 0.5  # bool

    # 손이 손잡이 hand_handle_max(2cm) 이내일 때만 추적 보상 적용
    distance = _hand_handle_distance(env, robot_cfg, hammer_cfg, hand_body_name, handle_offset)
    near_handle_gate = (distance <= hand_handle_max).float()

    # 성공 조건 미충족 시 early return
    if not success_mask.any():
        return torch.zeros(n, device=device)

    # ---- 2. 현재 관절 위치 vs 궤적 마지막 자세 (command) ----
    robot: Articulation = env.scene[asset_cfg.name]
    try:
        ref_term = env.command_manager.get_term(command_name)
        # loop=False이므로 progress >= 1.0이면 command_buffer는 마지막 프레임 자세
        target_all = ref_term.command  # (num_envs, num_traj_joints)
        traj_names = ref_term.joint_names
    except (KeyError, AttributeError):
        return torch.zeros(n, device=device)

    # joint_names 가 지정되면 해당 순서대로 인덱싱
    if joint_names is not None:
        try:
            traj_idx = [traj_names.index(jn) for jn in joint_names]
        except ValueError:
            return torch.zeros(n, device=device)
        target = target_all[:, traj_idx]
        robot_ids, _ = robot.find_joints(joint_names)
        robot_ids_t = torch.tensor(robot_ids, dtype=torch.long, device=device)
        current = robot.data.joint_pos[:, robot_ids_t]
    else:
        target = target_all
        current = robot.data.joint_pos[:, :target_all.shape[1]]

    # ---- 3. 손가락 / 팔 분리 및 가중 보상 계산 ----
    if finger_joint_names is not None and joint_names is not None:
        # joint_names 내에서 finger index 마스크
        finger_mask = torch.zeros(len(joint_names), dtype=torch.bool, device=device)
        for fn in finger_joint_names:
            if fn in joint_names:
                finger_mask[joint_names.index(fn)] = True
        arm_mask = ~finger_mask

        error = torch.abs(current - target)  # (n, num_joints)

        # 손가락 보상: base(exp) — exp(-|e|/σ) per joint → 평균
        if finger_mask.any():
            finger_err = error[:, finger_mask]  # (n, num_finger_joints)
            finger_reward = torch.exp(-finger_err / finger_std).mean(dim=1)
        else:
            finger_reward = torch.ones(n, device=device)

        # 팔 보상: per-joint exp → 평균 (기존 유지)
        if arm_mask.any():
            arm_err = error[:, arm_mask]
            arm_reward = torch.exp(-arm_err / arm_std).mean(dim=1)
        else:
            arm_reward = torch.ones(n, device=device)

        w_sum = finger_weight + arm_weight
        reward = (finger_weight * finger_reward + arm_weight * arm_reward) / w_sum
    else:
        # 분리 없이 전체 uniform
        error = torch.abs(current - target)
        reward = torch.exp(-error / finger_std).mean(dim=1)

    # 성공 조건 미충족 또는 손이 손잡이 2cm 밖이면 0
    return torch.where(success_mask, reward * near_handle_gate, torch.zeros_like(reward))


# -----------------------------------------------------------------------------
# Contact 기반 보상 / 패널티 (공통 헬퍼)
# -----------------------------------------------------------------------------


def _count_contact_bodies(
    env: ManagerBasedRLEnv,
    sensor_cfgs: list[SceneEntityCfg],
    filter_idx: int,
    min_force: float,
) -> torch.Tensor:
    """각 env별로, 주어진 filter 인덱스에 대해 min_force 이상 접촉한 센서 수를 반환."""
    n = env.num_envs
    device = env.device
    count = torch.zeros(n, device=device)
    for cfg in sensor_cfgs:
        try:
            sensor: ContactSensor = env.scene[cfg.name]
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is None:
                continue
            force = force_matrix[:, :, filter_idx, :]
            mag = torch.norm(force, dim=-1)
            count += (mag > min_force).any(dim=-1).float()
        except (KeyError, AttributeError):
            continue
    return count


def hammer_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfgs: list[SceneEntityCfg],
    min_contact_force: float = 1.0,
    min_contact_bodies: int = 2,
    hammer_filter_idx: int = 0,
) -> torch.Tensor:
    """손가락-망치 접촉 비율 보상. min_contact_bodies 미만이면 0."""
    count = _count_contact_bodies(env, sensor_cfgs, hammer_filter_idx, min_contact_force)
    ratio = count / len(sensor_cfgs)
    return torch.where(count >= min_contact_bodies, ratio, torch.zeros_like(ratio))


def finger_table_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfgs: list[SceneEntityCfg],
    min_contact_force: float = 0.1,
    table_filter_idx: int = 1,
) -> torch.Tensor:
    """손가락-테이블 접촉 비율 패널티 (weight 음수로 사용)."""
    count = _count_contact_bodies(env, sensor_cfgs, table_filter_idx, min_contact_force)
    return count/len(sensor_cfgs)
