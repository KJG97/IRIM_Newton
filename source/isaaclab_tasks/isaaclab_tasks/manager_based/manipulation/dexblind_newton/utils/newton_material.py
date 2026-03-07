"""Newton shape material utilities for event-based friction control."""

from __future__ import annotations

import torch
import warp as wp

import newton as nw
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim._impl.newton_manager import NewtonManager
from newton.solvers import SolverNotifyFlags

_COLLIDE = int(nw.ShapeFlags.COLLIDE_SHAPES)


def set_shape_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mu: float = 1.0,
    mu_range: tuple[float, float] | None = None,
):
    """Set Newton shape friction (mu) for **all** collision shapes of an asset.

    Writes directly to the global ``shape_material_mu`` array so that every
    collision shape owned by the asset's bodies is updated, including shapes
    that may not be indexed by the ArticulationView (e.g. the original mesh
    retained alongside CoACD convex decompositions).

    When *mu_range* is provided, a uniform-random value is sampled **per
    environment** on each call, overriding the fixed *mu* value.

    Args:
        env: The environment instance.
        env_ids: Environment indices to update. ``None`` means all environments.
        asset_cfg: Scene entity configuration identifying the target asset.
        mu: Slide friction coefficient to set. Defaults to 1.0.
            Ignored when *mu_range* is provided.
        mu_range: Optional (low, high) range for uniform-random sampling of mu
            per environment. When set, *mu* is ignored.
    """
    model = NewtonManager._model
    if model is None:
        return

    asset = env.scene[asset_cfg.name]
    prim_path: str = asset.cfg.prim_path
    asset_suffix = prim_path.split("/")[-1]

    body_keys: list[str] = model.body_key
    body_world_np = model.body_world.numpy()
    shape_body_np = model.shape_body.numpy()
    shape_flags_np = model.shape_flags.numpy()

    mu_tensor = wp.to_torch(model.shape_material_mu)

    target_envs: set[int] | None = None
    if env_ids is not None:
        target_envs = set(int(e) for e in env_ids)

    env_to_bodies: dict[int, list[int]] = {}
    for b in range(model.body_count):
        w = int(body_world_np[b])
        if target_envs is not None and w not in target_envs:
            continue
        bk = body_keys[b]
        if bk.endswith(f"/{asset_suffix}") or f"/{asset_suffix}/" in bk:
            env_to_bodies.setdefault(w, []).append(b)

    target_bodies_all = set()
    for bodies in env_to_bodies.values():
        target_bodies_all.update(bodies)

    env_to_shapes: dict[int, list[int]] = {}
    all_indices: list[int] = []
    for s in range(model.shape_count):
        sb = int(shape_body_np[s])
        if sb not in target_bodies_all:
            continue
        if not (int(shape_flags_np[s]) & _COLLIDE):
            continue
        all_indices.append(s)
        w = int(body_world_np[sb])
        env_to_shapes.setdefault(w, []).append(s)

    if not all_indices:
        return

    if mu_range is not None:
        lo, hi = mu_range
        for w, shape_ids in env_to_shapes.items():
            sampled_mu = lo + (hi - lo) * torch.rand(1, device=mu_tensor.device).item()
            idx = torch.tensor(shape_ids, device=mu_tensor.device, dtype=torch.long)
            mu_tensor[idx] = sampled_mu
    else:
        idx = torch.tensor(all_indices, device=mu_tensor.device, dtype=torch.long)
        mu_tensor[idx] = mu

    NewtonManager._solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)


def set_shape_contact_stiffness(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    ke: float = 100_000.0,
    kd: float = 1_000.0,
):
    """Set Newton contact stiffness (ke/kd) for all collision shapes of an asset.

    In MuJoCo these map to ``geom.solref = (-ke, -kd)`` (impedance convention).
    Higher *ke* makes the surface harder; higher *kd* adds contact damping.

    Args:
        env: The environment instance.
        env_ids: Environment indices to update. ``None`` means all environments.
        asset_cfg: Scene entity configuration identifying the target asset.
        ke: Contact stiffness. Default 100 000.
        kd: Contact damping. Default 1 000.
    """
    model = NewtonManager._model
    if model is None:
        return

    asset = env.scene[asset_cfg.name]
    prim_path: str = asset.cfg.prim_path
    asset_suffix = prim_path.split("/")[-1]

    body_keys: list[str] = model.body_key
    body_world_np = model.body_world.numpy()
    shape_body_np = model.shape_body.numpy()
    shape_flags_np = model.shape_flags.numpy()

    ke_tensor = wp.to_torch(model.shape_material_ke)
    kd_tensor = wp.to_torch(model.shape_material_kd)

    target_envs: set[int] | None = None
    if env_ids is not None:
        target_envs = set(int(e) for e in env_ids)

    target_bodies: set[int] = set()
    for b in range(model.body_count):
        w = int(body_world_np[b])
        if target_envs is not None and w not in target_envs:
            continue
        bk = body_keys[b]
        if bk.endswith(f"/{asset_suffix}") or f"/{asset_suffix}/" in bk:
            target_bodies.add(b)

    indices: list[int] = []
    for s in range(model.shape_count):
        sb = int(shape_body_np[s])
        if sb not in target_bodies:
            continue
        if not (int(shape_flags_np[s]) & _COLLIDE):
            continue
        indices.append(s)

    if not indices:
        return

    idx = torch.tensor(indices, device=ke_tensor.device, dtype=torch.long)
    ke_tensor[idx] = ke
    kd_tensor[idx] = kd

    NewtonManager._solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)


def randomize_object_pose_xy_yaw(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.05, 0.05),
    y_range: tuple[float, float] = (-0.05, 0.05),
    yaw_range: tuple[float, float] = (-0.5, 0.5),
    randomize_scale: float = 1.0,
):
    """Randomize an object's x, y position and yaw rotation on reset.

    Offsets are applied relative to the asset's default root state.
    Yaw is in radians and applied around the world Z axis.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Scene entity identifying the target object.
        x_range: (min, max) uniform range for x offset at full scale.
        y_range: (min, max) uniform range for y offset at full scale.
        yaw_range: (min, max) uniform range for yaw offset in radians at full scale.
        randomize_scale: Multiplier [0, 1] applied to all ranges.
            Schedule via curriculum from 0 (no randomization) to 1 (full range).
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    s = max(0.0, min(1.0, randomize_scale))
    n = len(env_ids)
    default_state = wp.to_torch(asset.data.default_root_state)[env_ids].clone()
    pose = default_state[:, :7]  # (x, y, z, qx, qy, qz, qw)
    pose[:, 0:3] += env.scene.env_origins[env_ids]

    # position offsets (scaled)
    pose[:, 0] += torch.empty(n, device=env.device).uniform_(x_range[0] * s, x_range[1] * s)
    pose[:, 1] += torch.empty(n, device=env.device).uniform_(y_range[0] * s, y_range[1] * s)

    # yaw offsets: compose a Z-rotation quaternion with the existing orientation
    yaw = torch.empty(n, device=env.device).uniform_(yaw_range[0] * s, yaw_range[1] * s)
    half = yaw * 0.5
    dq = torch.zeros(n, 4, device=env.device)
    dq[:, 2] = torch.sin(half)   # qz
    dq[:, 3] = torch.cos(half)   # qw

    # quaternion multiply: q_new = dq * q_old  (Hamilton, xyzw layout)
    q = pose[:, 3:7].clone()
    pose[:, 3] = dq[:, 3] * q[:, 0] + dq[:, 0] * q[:, 3] + dq[:, 1] * q[:, 2] - dq[:, 2] * q[:, 1]
    pose[:, 4] = dq[:, 3] * q[:, 1] - dq[:, 0] * q[:, 2] + dq[:, 1] * q[:, 3] + dq[:, 2] * q[:, 0]
    pose[:, 5] = dq[:, 3] * q[:, 2] + dq[:, 0] * q[:, 1] - dq[:, 1] * q[:, 0] + dq[:, 2] * q[:, 3]
    pose[:, 6] = dq[:, 3] * q[:, 3] - dq[:, 0] * q[:, 0] - dq[:, 1] * q[:, 1] - dq[:, 2] * q[:, 2]

    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(default_state[:, 7:], env_ids=env_ids)

    NewtonManager._solver.notify_model_changed(
        SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.JOINT_PROPERTIES
    )


def print_dynamics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """EventTerm-compatible wrapper that prints all dynamics for an asset."""
    print_asset_dynamics(env, asset_cfg.name)


def print_asset_dynamics(env: ManagerBasedEnv, asset_name: str) -> None:
    """Print all dynamics properties of an asset's shapes and bodies (env_0 only)."""
    import newton

    asset = env.scene[asset_name]
    model = asset.root_newton_model
    view = asset.root_view

    def _read(attr: str):
        return wp.to_torch(view.get_attribute(attr, model))[0].tolist()

    geo_names = {v: k for k, v in vars(newton.GeoType).items() if isinstance(v, int)}

    print(f"\n{'='*60}")
    print(f" Asset: {asset_name}")
    print(f"{'='*60}")

    shape_keys = model.shape_key
    shape_types = model.shape_type.numpy()
    shape_flags = model.shape_flags.numpy()
    body_keys = model.body_key
    body_mass = model.body_mass.numpy()
    body_inertia = model.body_inertia.numpy()
    body_com = model.body_com.numpy()
    shape_body = model.shape_body.numpy()

    mu = _read("shape_material_mu")
    torsional = _read("shape_material_torsional_friction")
    rolling = _read("shape_material_rolling_friction")
    restitution = _read("shape_material_restitution")
    ke = _read("shape_material_ke")
    kd = _read("shape_material_kd")

    # Body info
    env0_bodies = set()
    print(f"\n--- Shapes (env_0) ---")
    for i, (key, stype, flags) in enumerate(zip(shape_keys, shape_types, shape_flags)):
        body_idx = shape_body[i]
        env0_bodies.add(body_idx)
        collide = bool(flags & int(newton.ShapeFlags.COLLIDE_SHAPES))
        visible = bool(flags & int(newton.ShapeFlags.VISIBLE))
        m = mu[i] if isinstance(mu, list) else mu
        t = torsional[i] if isinstance(torsional, list) else torsional
        r = rolling[i] if isinstance(rolling, list) else rolling
        rest = restitution[i] if isinstance(restitution, list) else restitution
        k_e = ke[i] if isinstance(ke, list) else ke
        k_d = kd[i] if isinstance(kd, list) else kd
        print(
            f"  [{i}] {key}\n"
            f"       type={geo_names.get(int(stype), stype)}  collide={collide}  visible={visible}\n"
            f"       mu={m:.4f}  torsional={t:.6f}  rolling={r:.6f}\n"
            f"       restitution={rest:.4f}  ke={k_e:.1f}  kd={k_d:.1f}"
        )

    print(f"\n--- Bodies (env_0) ---")
    for i in sorted(env0_bodies):
        if i < 0 or i >= len(body_keys):
            continue
        inertia = body_inertia[i]
        print(
            f"  [{i}] {body_keys[i]}\n"
            f"       mass={body_mass[i]:.6f}  com={body_com[i].tolist()}\n"
            f"       inertia=diag({inertia[0][0]:.6f}, {inertia[1][1]:.6f}, {inertia[2][2]:.6f})"
        )
    print(f"{'='*60}\n", flush=True)
