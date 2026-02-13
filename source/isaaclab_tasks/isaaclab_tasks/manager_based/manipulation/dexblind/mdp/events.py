# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for Dexblind manipulation task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.envs.mdp import events as mdp_events
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_table_and_hammer_height_linked(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    z_offset_range: tuple[float, float] = (-0.03, 0.03),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
) -> None:
    """테이블과 해머의 z 높이를 동일한 오프셋으로 함께 랜덤화.

    - 각 env 마다 U(z_offset_range) 에서 하나 샘플한 offset_z 를
      table, hammer 두 개에 동시에 더해 줍니다.
    - x, y, roll, pitch, yaw 는 건드리지 않습니다.
    - reset_root_state_uniform과 동일하게 env_origins를 고려하고 write_root_pose_to_sim을 사용.
    """
    # asset 가져오기
    table: RigidObject = env.scene[table_cfg.name]
    hammer: RigidObject = env.scene[hammer_cfg.name]

    # env_ids 해석
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=table.device)

    # 기본 root state 가져오기 (reset_root_state_uniform과 동일한 방식)
    table_root_states = table.data.default_root_state[env_ids].clone()
    hammer_root_states = hammer.data.default_root_state[env_ids].clone()

    # env 별로 z offset 샘플
    z_min, z_max = z_offset_range
    offsets = torch.empty(len(env_ids), device=table.device).uniform_(z_min, z_max)

    # positions = default + env_origins → z += offset
    # reset_root_state_uniform과 동일하게 env_origins를 더함
    table_positions = table_root_states[:, 0:3] + env.scene.env_origins[env_ids]
    hammer_positions = hammer_root_states[:, 0:3] + env.scene.env_origins[env_ids]
    
    # z에만 offset 적용
    table_positions[:, 2] += offsets
    hammer_positions[:, 2] += offsets

    # orientations는 기본값 그대로 사용
    table_orientations = table_root_states[:, 3:7]
    hammer_orientations = hammer_root_states[:, 3:7]

    # write_root_pose_to_sim 사용 (reset_root_state_uniform과 동일)
    table.write_root_pose_to_sim(torch.cat([table_positions, table_orientations], dim=-1), env_ids=env_ids)
    hammer.write_root_pose_to_sim(torch.cat([hammer_positions, hammer_orientations], dim=-1), env_ids=env_ids)

def apply_hammer_force_when_lifted(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float] | list[float],
    torque_range: tuple[float, float] | list[float],
    height_threshold: float = 0.52,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
):
    """Apply random external force to hammer when its height is above threshold.
    
    This function checks if the hammer's z-position (height) is above the specified threshold.
    If so, it applies a random external force and torque to the hammer to encourage the policy
    to maintain a firm grip.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to apply the force to. If None, applies to all environments.
        force_range: Tuple of (min, max) force magnitude in Newtons for each axis (x, y, z).
        torque_range: Tuple of (min, max) torque magnitude in N⋅m for each axis (x, y, z).
        height_threshold: Height threshold in meters. Defaults to 0.52.
        asset_cfg: Scene entity configuration for the hammer (for checking height and applying force).
    """
    # Extract hammer asset
    hammer: RigidObject = env.scene[asset_cfg.name]
    
    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=hammer.device)
    
    # Get hammer height (z-position in world frame)
    hammer_z = hammer.data.root_pos_w[:, 2]  # (num_envs,)
    
    # Find environments where hammer height is above threshold
    above_threshold = hammer_z[env_ids] >= height_threshold
    
    if not above_threshold.any():
        # No environments meet the condition, skip applying force
        return
    
    # Filter env_ids to only those above threshold
    valid_env_ids = env_ids[above_threshold]
    
    # Convert list to tuple if needed (for compatibility with apply_external_force_torque)
    if isinstance(force_range, list):
        force_range = tuple(force_range)
    if isinstance(torque_range, list):
        torque_range = tuple(torque_range)
    
    # Apply external force using the base function
    mdp_events.apply_external_force_torque(
        env=env,
        env_ids=valid_env_ids,
        force_range=force_range,
        torque_range=torque_range,
        asset_cfg=asset_cfg,
    )


