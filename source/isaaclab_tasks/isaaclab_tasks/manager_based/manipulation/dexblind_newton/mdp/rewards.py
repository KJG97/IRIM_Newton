# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal reward for dexblind_newton: hammer lift only."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from .utils import root_pos_w_z

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hammer_lift_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    threshold: float = 0.55,
) -> torch.Tensor:
    """Reward 1.0 when hammer z >= threshold, else 0.0."""
    z = root_pos_w_z(env, asset_cfg)
    return (z >= threshold).float()
