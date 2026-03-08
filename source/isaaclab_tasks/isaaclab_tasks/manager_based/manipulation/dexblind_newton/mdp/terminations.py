# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for dexblind_newton lift task."""

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from .utils import root_pos_w_z

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hammer_fallen(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    z_threshold: float = 0.4,
) -> torch.Tensor:
    """True when hammer z falls below threshold."""
    z = root_pos_w_z(env, asset_cfg)
    return z < z_threshold


def hammer_velocity_exceeded(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    max_lin_vel: float = 15.0,
    max_ang_vel: float = 30.0,
) -> torch.Tensor:
    """True when hammer linear or angular speed exceeds threshold (reset to avoid physics explosion)."""
    hammer = env.scene[asset_cfg.name]
    lin_vel = wp.to_torch(hammer.data.root_lin_vel_w)
    ang_vel = wp.to_torch(hammer.data.root_ang_vel_w)
    if lin_vel.dim() == 3:
        lin_vel = lin_vel.squeeze(1)
    if ang_vel.dim() == 3:
        ang_vel = ang_vel.squeeze(1)
    lin_speed = lin_vel.norm(dim=-1)
    ang_speed = ang_vel.norm(dim=-1)
    return (lin_speed > max_lin_vel) | (ang_speed > max_ang_vel)


def hammer_too_far_from_table(
    env: "ManagerBasedRLEnv",
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    max_distance: float = 0.8,
) -> torch.Tensor:
    """True when hammer is farther than max_distance from table center (per env)."""
    hammer = env.scene[hammer_cfg.name]
    table = env.scene[table_cfg.name]
    hammer_pos = wp.to_torch(hammer.data.root_pos_w)
    table_pos = wp.to_torch(table.data.root_pos_w)
    if hammer_pos.dim() == 3:
        hammer_pos = hammer_pos.squeeze(1)
    if table_pos.dim() == 3:
        table_pos = table_pos.squeeze(1)
    dist = (hammer_pos - table_pos).norm(dim=-1)
    return dist > max_distance
