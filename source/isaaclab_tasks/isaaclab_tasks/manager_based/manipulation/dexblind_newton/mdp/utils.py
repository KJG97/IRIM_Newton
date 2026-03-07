# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP helpers for dexblind_newton (e.g. world position for cloned XFormPrim)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.prims import resolve_prim_pose
from isaaclab.utils.math import quat_inv, quat_mul, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
#  Cached body-index lookup
# ---------------------------------------------------------------------------
_body_index_cache: dict[str, torch.Tensor] = {}


def _get_body_indices(suffix: str, num_envs: int, device: torch.device) -> torch.Tensor:
    """Return a (num_envs,) int64 tensor mapping env_id -> body index for the
    given body suffix (e.g. 'Origin_Body', 'hammer', 'grasp_point').

    The result is computed once and cached for the lifetime of the process.
    """
    key = f"{suffix}_{num_envs}"
    if key in _body_index_cache:
        return _body_index_cache[key]

    from isaaclab.sim._impl.newton_manager import NewtonManager

    model = NewtonManager._model
    body_keys: list[str] = model.body_key
    body_world = model.body_world.numpy()

    indices = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    for b in range(model.body_count):
        eid = int(body_world[b])
        if eid >= num_envs:
            continue
        bk = body_keys[b]
        if bk.endswith(suffix) or bk.endswith(f"/{suffix}"):
            indices[eid] = b

    _body_index_cache[key] = indices
    return indices


def get_body_poses_batched(
    suffix: str, num_envs: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world (pos, quat_xyzw) for the named body across all envs.

    Returns (num_envs, 3) and (num_envs, 4). Envs where the body was not
    found get zeros (pos) and identity quat (0,0,0,1).
    """
    from isaaclab.sim._impl.newton_manager import NewtonManager

    state = NewtonManager._state_0
    body_q = wp.to_torch(state.body_q)  # (total_bodies, 7)
    indices = _get_body_indices(suffix, num_envs, device)

    valid = indices >= 0
    safe_idx = indices.clamp(min=0)

    all_poses = body_q[safe_idx]  # (num_envs, 7)
    pos = all_poses[:, :3]
    quat = all_poses[:, 3:7]

    if not valid.all():
        inv = ~valid
        pos[inv] = 0.0
        quat[inv] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)

    return pos, quat


def relative_pose_batched(
    origin_pos: torch.Tensor, origin_quat: torch.Tensor,
    target_pos: torch.Tensor, target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched relative pose: target in origin frame. All (N, 3/4) xyzw.

    Canonicalises output quaternion so w >= 0.
    """
    q_inv = quat_inv(origin_quat)                               # (N, 4)
    rel_pos = quat_apply(q_inv, target_pos - origin_pos)        # (N, 3)
    rel_quat = quat_mul(q_inv, target_quat)                     # (N, 4)
    neg_w = rel_quat[:, 3] < 0
    rel_quat[neg_w] = -rel_quat[neg_w]
    return rel_pos, rel_quat


def root_pos_w_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World z position (N,) for scene entity. Supports RigidObject and cloned XFormPrim."""
    entity = env.scene[asset_cfg.name]
    if hasattr(entity, "data") and hasattr(entity.data, "root_pos_w"):
        pos = entity.data.root_pos_w
        if isinstance(pos, wp.array):
            pos = wp.to_torch(pos)
        if pos.dim() == 3:
            pos = pos.squeeze(1)
        return pos[:, 2]

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
