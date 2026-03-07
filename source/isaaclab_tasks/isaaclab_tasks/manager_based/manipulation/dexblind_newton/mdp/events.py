# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for dexblind_newton (Newton backend).

Newton's actuator/articulation API differs from IsaacSim's class-based event
terms (``randomize_actuator_gains``, ``randomize_joint_parameters``,
``randomize_rigid_body_mass``).  The functions below provide Newton-compatible
replacements that operate directly on the articulation data and write back
through the Newton write-to-sim methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp import events as mdp_events
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Actuator gain randomization (Newton-compatible)
# ---------------------------------------------------------------------------

def randomize_actuator_gains_newton(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    operation: str = "scale",
):
    """Randomize joint stiffness / damping via Newton's write-to-sim API.

    Unlike the class-based ``randomize_actuator_gains`` this is a plain
    function that reads the *current* sim values, applies a random scale /
    offset, and writes back.  It works with Newton's ``ImplicitActuator``
    which lacks the ``stiffness`` / ``joint_indices`` properties expected
    by the upstream implementation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    joint_ids = asset_cfg.joint_ids
    if joint_ids == slice(None):
        joint_ids = None

    if stiffness_distribution_params is not None:
        default = wp.to_torch(asset.data.joint_stiffness)
        stiffness = _apply_op(default, env_ids, joint_ids, stiffness_distribution_params, operation, asset.device)
        asset.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids, env_ids=env_ids)

    if damping_distribution_params is not None:
        default = wp.to_torch(asset.data.joint_damping)
        damping = _apply_op(default, env_ids, joint_ids, damping_distribution_params, operation, asset.device)
        asset.write_joint_damping_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Joint friction randomization (Newton-compatible)
# ---------------------------------------------------------------------------

def randomize_joint_friction_newton(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    operation: str = "scale",
):
    """Randomize joint friction coefficient via Newton's write-to-sim API."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    joint_ids = asset_cfg.joint_ids
    if joint_ids == slice(None):
        joint_ids = None

    if friction_distribution_params is not None:
        default = wp.to_torch(asset.data.joint_friction_coeff)
        friction = _apply_op(default, env_ids, joint_ids, friction_distribution_params, operation, asset.device)
        friction = torch.clamp(friction, min=0.0)
        asset.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Rigid body mass randomization (Newton-compatible)
# ---------------------------------------------------------------------------

# Cache initial body_mass / body_inertia per asset so we always scale from USD default,
# not from the previous reset (asset.data.body_mass is sim-bound and changes after set_masses).
_default_mass_cache: dict[str, torch.Tensor] = {}
_default_inertia_cache: dict[str, torch.Tensor] = {}

# Newton does not apply USD mass_props; override default mass for known assets (kg).
_DEFAULT_MASS_OVERRIDE: dict[str, float] = {"hammer": 0.55}


def randomize_rigid_body_mass_newton(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float] = (0.8, 1.2),
    operation: str = "scale",
    recompute_inertia: bool = True,
):
    """Randomize body mass (and optionally inertia) via Newton's write-to-sim API.

    Uses cached initial mass from the first call so each reset scales from the
    intended default, not from the previous reset value. For assets in
    _DEFAULT_MASS_OVERRIDE (e.g. hammer), the cache is set to that value because
    Newton does not apply USD mass_props.
    """
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    name = asset_cfg.name
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    body_ids = asset_cfg.body_ids
    if body_ids == slice(None):
        body_ids = None

    if name not in _default_mass_cache:
        sim_mass = wp.to_torch(asset.data.body_mass).clone()
        sim_inertia = wp.to_torch(asset.data.body_inertia).clone()
        override = _DEFAULT_MASS_OVERRIDE.get(name)
        if override is not None:
            # Use override as default mass; scale inertia so it matches that mass.
            _default_mass_cache[name] = torch.full_like(sim_mass, override, device=asset.device)
            mass_ratio = (override / sim_mass.clamp(min=1e-9)).unsqueeze(-1).unsqueeze(-1)  # (N, B, 1, 1)
            _default_inertia_cache[name] = (sim_inertia * mass_ratio).clone()
        else:
            _default_mass_cache[name] = sim_mass
            _default_inertia_cache[name] = sim_inertia
    default_mass = _default_mass_cache[name]
    default_inertia = _default_inertia_cache[name]

    masses = _apply_op(default_mass, env_ids, body_ids, mass_distribution_params, operation, asset.device)
    masses = torch.clamp(masses, min=1e-6)

    asset.set_masses(masses, body_ids=body_ids, env_ids=env_ids)

    if recompute_inertia:
        n_bodies = default_inertia.shape[1]
        default_inertia_flat = default_inertia.reshape(asset.num_instances, n_bodies, 9)

        ratios = masses / default_mass[env_ids]
        if body_ids is not None:
            ratios = ratios[:, body_ids] if isinstance(body_ids, (list, torch.Tensor)) else ratios
        scaled = default_inertia_flat[env_ids].clone()
        if body_ids is not None:
            scaled = scaled[:, body_ids] * ratios[..., None]
        else:
            scaled = scaled * ratios[..., None]
        scaled_33 = scaled.reshape(scaled.shape[0], scaled.shape[1], 3, 3)
        asset.set_inertias(scaled_33, body_ids=body_ids, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Existing custom events
# ---------------------------------------------------------------------------

def reset_table_and_hammer_height_linked(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    z_offset_range: tuple[float, float] = (-0.03, 0.03),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    hammer_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
) -> None:
    """Randomize table/hammer z-height together with a shared offset."""
    table: RigidObject | Articulation = env.scene[table_cfg.name]
    hammer: RigidObject | Articulation = env.scene[hammer_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    table_root_pose = wp.to_torch(table.data.default_root_pose)[env_ids].clone()
    hammer_root_pose = wp.to_torch(hammer.data.default_root_pose)[env_ids].clone()

    z_min, z_max = z_offset_range
    offsets = torch.empty(len(env_ids), device=env.device).uniform_(z_min, z_max)

    table_pos = table_root_pose[:, 0:3] + env.scene.env_origins[env_ids]
    hammer_pos = hammer_root_pose[:, 0:3] + env.scene.env_origins[env_ids]
    table_pos[:, 2] += offsets
    hammer_pos[:, 2] += offsets

    table_pose = torch.cat([table_pos, table_root_pose[:, 3:7]], dim=-1)
    hammer_pose = torch.cat([hammer_pos, hammer_root_pose[:, 3:7]], dim=-1)
    table.write_root_pose_to_sim(table_pose, env_ids=env_ids)
    hammer.write_root_pose_to_sim(hammer_pose, env_ids=env_ids)


def apply_hammer_force_when_lifted(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float] | list[float],
    torque_range: tuple[float, float] | list[float],
    height_threshold: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
):
    """Apply random external wrench only when hammer height is above threshold.

    Clears wrench for envs below threshold so that when the hammer is put down
    the disturbance stops; otherwise the same force would persist every step.
    """
    hammer: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    hammer_z = wp.to_torch(hammer.data.root_pos_w)[:, 2]
    above = hammer_z[env_ids] >= height_threshold
    valid_env_ids = env_ids[above]
    below_env_ids = env_ids[~above]

    num_bodies = hammer.num_bodies
    body_ids = getattr(asset_cfg, "body_ids", None)

    # Clear wrench for envs where hammer is below threshold
    if below_env_ids.numel() > 0:
        zeros_f = torch.zeros(len(below_env_ids), num_bodies, 3, device=hammer.device)
        zeros_t = torch.zeros(len(below_env_ids), num_bodies, 3, device=hammer.device)
        hammer.set_external_force_and_torque(zeros_f, zeros_t, env_ids=below_env_ids, body_ids=body_ids)

    if valid_env_ids.numel() == 0:
        return

    if isinstance(force_range, list):
        force_range = tuple(force_range)
    if isinstance(torque_range, list):
        torque_range = tuple(torque_range)

    mdp_events.apply_external_force_torque(
        env=env,
        env_ids=valid_env_ids,
        force_range=force_range,
        torque_range=torque_range,
        asset_cfg=asset_cfg,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_op(
    default: torch.Tensor,
    env_ids: torch.Tensor,
    dim1_ids: list[int] | torch.Tensor | None,
    params: tuple[float, float],
    operation: str,
    device: str,
) -> torch.Tensor:
    """Sample and apply add/scale/abs to a 2-D property tensor.

    Returns the *partial* tensor indexed by ``[env_ids, dim1_ids]`` ready to
    be written back to the simulation.
    """
    if dim1_ids is not None:
        if isinstance(dim1_ids, list):
            dim1_ids = torch.tensor(dim1_ids, dtype=torch.long, device=device)
        sub = default[env_ids][:, dim1_ids].clone()
    else:
        sub = default[env_ids].clone()

    lo, hi = params
    samples = math_utils.sample_uniform(lo, hi, sub.shape, device)

    if operation == "scale":
        sub *= samples
    elif operation == "add":
        sub += samples
    elif operation == "abs":
        sub = samples
    else:
        raise ValueError(f"Unknown operation '{operation}'")
    return sub
