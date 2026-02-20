# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP helpers for dexblind_newton (e.g. world position for cloned XFormPrim)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.prims import resolve_prim_pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_pos_w_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World z position (N,) for scene entity. Supports RigidObject and cloned XFormPrim."""
    entity = env.scene[asset_cfg.name]
    if hasattr(entity, "data") and hasattr(entity.data, "root_pos_w"):
        return entity.data.root_pos_w[:, 2]

    # Cloned XFormPrim: _prim_paths point at template (invalid at runtime). Use _regex_prim_paths.
    pattern = None
    if hasattr(entity, "_regex_prim_paths") and entity._regex_prim_paths:
        pattern = entity._regex_prim_paths[0]
    if not pattern:
        pos_w, _ = entity.get_world_poses()
        if not pos_w.is_cuda and env.device.type == "cuda":
            pos_w = pos_w.to(env.device)
        return pos_w[:, 2]

    stage = env.scene.stage
    paths = sim_utils.find_matching_prim_paths(pattern, stage)
    # Deterministic order: env_0, env_1, ...
    paths = sorted(paths, key=lambda p: _env_index_from_path(p))

    positions = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            pos, _ = resolve_prim_pose(prim, ref_prim=None)
            positions.append(pos)
    if not positions:
        return torch.zeros(env.scene.num_envs, device=env.device)
    pos_w = torch.tensor(positions, device=env.device, dtype=torch.float32)
    return pos_w[:, 2]


def _env_index_from_path(path: str) -> int:
    """Extract env index from path like /World/envs/env_2/hammer for stable sort."""
    m = re.search(r"env_(\d+)", path)
    return int(m.group(1)) if m else 0
