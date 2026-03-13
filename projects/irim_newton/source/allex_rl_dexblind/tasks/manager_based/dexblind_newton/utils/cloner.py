# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Newton replicate wrapper: applies equality_constraints per prototype via add_equality_constraint_joint."""

from __future__ import annotations

import torch
import warp as wp
from isaaclab_newton.cloner.newton_replicate import get_inverse_env_xform


def _joint_index(builder, name: str) -> int:
    for k, key in enumerate(builder.joint_key):
        if key == name or key.endswith("/" + name) or key.endswith(name):
            return k
    raise KeyError(f"Joint {name!r} not in builder.joint_key")


def _apply_equality_constraints(p, equality_constraints: list | None):
    for follower, driver, polycoef in equality_constraints or []:
        coef = (list(polycoef) + [0.0] * 5)[:5]
        p.add_equality_constraint_joint(
            joint1=_joint_index(p, follower),
            joint2=_joint_index(p, driver),
            polycoef=coef,
        )


def _approximate_meshes_per_asset(
    builder,
    simplify_meshes: bool | str | dict,
) -> None:
    """Apply approximate_meshes with per-asset granularity (backup cloner_utils logic).

    simplify_meshes:
        True — convex_hull for all mesh shapes.
        str — that method for all.
        dict — keys are name fragments matched against shape_key; \"*\" is fallback.
              Values: str (method) or (method, kwargs) e.g. (\"coacd\", {\"threshold\": 0.1}).
    """
    import newton

    mesh_collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    eligible = [
        i for i in range(builder.shape_count)
        if builder.shape_type[i] == newton.GeoType.MESH and builder.shape_flags[i] & mesh_collide
    ]
    if not eligible:
        return

    def _parse_entry(entry) -> tuple[str, dict]:
        if isinstance(entry, tuple):
            return entry[0], entry[1] if len(entry) > 1 else {}
        return entry, {}

    if isinstance(simplify_meshes, dict):
        default_entry = simplify_meshes.get("*", "convex_hull")
        groups: dict[tuple[str, frozenset], list[int]] = {}
        entry_map: dict[tuple[str, frozenset], dict] = {}
        for idx in eligible:
            key = builder.shape_key[idx]
            method, kwargs = _parse_entry(default_entry)
            for pattern, val in simplify_meshes.items():
                if pattern != "*" and pattern in key:
                    method, kwargs = _parse_entry(val)
                    break
            gkey = (method, frozenset(kwargs.items()))
            groups.setdefault(gkey, []).append(idx)
            entry_map[gkey] = kwargs
        for gkey, indices in groups.items():
            method = gkey[0]
            builder.approximate_meshes(
                method, keep_visual_shapes=True, shape_indices=indices, **entry_map[gkey]
            )
    else:
        method = simplify_meshes if isinstance(simplify_meshes, str) else "convex_hull"
        builder.approximate_meshes(method, keep_visual_shapes=True)


def dexblind_newton_replicate(
    stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    up_axis: str = "Z",
    simplify_meshes: bool | dict = True,
    *,
    equality_constraints: list | None = None,
    **kwargs,
):
    """Replicate into Newton ModelBuilder and apply equality_constraints per prototype.

    simplify_meshes: If True, use convex_hull for all. If a dict, pass through to
    approximate_meshes for per-asset methods (e.g. {"hammer": ("coacd", {"threshold": 0.1}), "*": "convex_hull"}).
    """
    from isaaclab_newton.physics import NewtonManager
    from newton import ModelBuilder, solvers

    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    builder = ModelBuilder(up_axis=up_axis)
    builder.add_usd(stage, ignore_paths=["/World/envs"] + sources)

    load_visual = kwargs.get("load_visual_shapes", True)
    protos = {}
    for src_path in sources:
        p = ModelBuilder(up_axis=up_axis)
        solvers.SolverMuJoCo.register_custom_attributes(p)
        p.add_usd(
            stage,
            root_path=src_path,
            load_visual_shapes=load_visual,
            skip_mesh_approximation=bool(simplify_meshes),
            xform=get_inverse_env_xform(stage, src_path),
        )
        _apply_equality_constraints(p, equality_constraints)
        if simplify_meshes:
            _approximate_meshes_per_asset(p, simplify_meshes)
        protos[src_path] = p

    for col, env_id in enumerate(env_ids.tolist()):
        builder.begin_world()
        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            builder.add_builder(
                protos[sources[row]],
                xform=wp.transform(positions[col].tolist(), quaternions[col].tolist()),
            )
        builder.end_world()

    for i, src_path in enumerate(sources):
        pl = len(src_path.rstrip("/"))
        swap = lambda n, r, _pl=pl: r + n[_pl:]  # noqa: E731
        world_cols = torch.nonzero(mapping[i], as_tuple=True)[0].tolist()
        roots = {int(env_ids[c]): destinations[i].format(int(env_ids[c])) for c in world_cols}
        for t in ("body", "joint", "shape", "articulation"):
            keys, worlds = getattr(builder, f"{t}_key"), getattr(builder, f"{t}_world")
            for k, w in enumerate(worlds):
                if w in roots and keys[k].startswith(src_path):
                    keys[k] = swap(keys[k], roots[w])

    NewtonManager.set_builder(builder)
    NewtonManager._num_envs = mapping.size(1)
    return builder
