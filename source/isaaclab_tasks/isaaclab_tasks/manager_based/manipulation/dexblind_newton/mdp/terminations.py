# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for dexblind_newton lift task."""

from __future__ import annotations

import torch
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
