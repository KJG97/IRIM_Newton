# source/isaaclab/isaaclab — Change log

Changes in this fork touch the **cloner**, **scene**, and **sim/_impl** packages so that Newton replication can receive **equality_constraints** (mimic joints) and so that MuJoCo-based Newton runs with a **dense Jacobian** and configurable **mj_data_memory**.

---

## 1. cloner/cloner_utils.py

### 1.1 Passing kwargs into the physics clone function (clone_from_template)

- **Original (upstream):** `template_clone_cfg.physics_clone_fn(stage, *replicate_args, positions=positions)` and similarly without `positions` in the heterogeneous path.
- **Modified (commit 297caefa):**
  - **~lines 74–76 (template path):** Before calling `physics_clone_fn`, add:
    - `clone_kwargs = getattr(cfg, "physics_clone_fn_kwargs", None) or {}`
    - Call: `template_clone_cfg.physics_clone_fn(stage, *replicate_args, positions=positions, **clone_kwargs)`.
  - **~lines 81–83 (heterogeneous path):** Same pattern: `clone_kwargs = getattr(cfg, "physics_clone_fn_kwargs", None) or {}`, then `physics_clone_fn(..., **clone_kwargs)`.
- **Reason:** So that `InteractiveSceneCfg.newton_replicate_kwargs` (e.g. `equality_constraints`) can be passed through to `newton_replicate` when the cloner runs.

### 1.2 Helper _newton_joint_index_by_name (new function)

- **Added (commit 297caefa):** ~lines 276–286.
- **Behavior:** Given a Newton `ModelBuilder` and a joint name string, returns the index in `builder.joint_key` that equals the name, ends with `"/" + name`, or has the same last path component. Returns `-1` if not found.
- **Reason:** Needed to resolve (mimic_name, driver_name) from `equality_constraints` to (joint1_index, joint2_index) for `p.add_equality_constraint_joint(...)`.

### 1.3 newton_replicate signature and docstring

- **Added (commit 297caefa):** Parameter `equality_constraints: list[tuple[str, str, tuple[float, ...]]] | None = None` and docstring describing it.
  - **Docstring:** Optional list of `(mimic_joint_name, driver_joint_name, (c0,c1,c2,c3,c4))` for MuJoCo joint equality `q_mimic = c0 + c1*q_driver + ...`; names are matched against `builder.joint_key` (exact or path suffix).
- **Reason:** ALLEX mimic joints (e.g. DIP following PIP) must be enforced by the solver; this list is the only way to inject them at clone time.

### 1.4 Newton add_usd scale note (comment)

- **Added (commit 297caefa):** Comment block before building prototypes (~lines 322–327 in current file).
  - **Content:** Newton’s `add_usd()` does not apply USD xform scale to mesh geometry (unlike Omniverse). Assets with a parent scale (e.g. 0.001 for mm→m) can appear at the wrong scale in the Newton visualizer; workarounds are baking scale into mesh geometry or fixing the Newton USD importer.
- **Reason:** Document why ALLEX USD had to be prepared (e.g. mesh in meters) for correct scale in Newton.

### 1.5 Injecting equality constraints in the prototype loop

- **Added (commit 297caefa):** After `p.approximate_meshes("convex_hull")` in the `for src_path in sources:` loop (~lines 336–357).
  - If `equality_constraints` is non-empty:
    - For each `(mimic_name, driver_name, polycoef)`:
      - Resolve indices with `_newton_joint_index_by_name(p, mimic_name)` and `_newton_joint_index_by_name(p, driver_name)`.
      - If both >= 0, call `p.add_equality_constraint_joint(joint1=idx1, joint2=idx2, polycoef=...)` (polycoef padded to length 5 with zeros if needed).
    - If no constraint was added but the list was non-empty, log a warning with a sample of `joint_key`.
- **Reason:** So ALLEX (and any other asset) can register mimic joints as MuJoCo equality constraints during replication.

---

## 2. cloner/cloner_cfg.py

- **Added (commit 297caefa):** After `clone_in_fabric` (~lines 87–88):
  - `physics_clone_fn_kwargs: dict | None = None`
  - Docstring: optional kwargs passed to `physics_clone_fn` when called (e.g. `equality_constraints` for Newton).
- **Reason:** Template clone config must hold the kwargs that are later passed to `newton_replicate`; `InteractiveScene` copies `newton_replicate_kwargs` from scene cfg into `cloner_cfg.physics_clone_fn_kwargs`.

---

## 3. scene/interactive_scene_cfg.py

- **Added (commit 297caefa):** At the end of the class (~lines 127–130):
  - `newton_replicate_kwargs: dict | None = None`
  - Docstring: optional kwargs passed to `newton_replicate` when cloning (e.g. `equality_constraints` for joint mimic / polynomial equality); used only in the direct clone path (single env source).
- **Reason:** Env configs (e.g. `AllexEnvCfg`, `AllexEnvNoLeftCfg`) set `scene = InteractiveSceneCfg(..., newton_replicate_kwargs={"equality_constraints": ALLEX_FULL_MIMIC_SPEC})` so that the scene passes those constraints into the cloner.

---

## 4. scene/interactive_scene.py

- **Added (commit 297caefa):**
  - **~lines 133–134:** After building `cloner_cfg`, if `getattr(self.cfg, "newton_replicate_kwargs", None)` is truthy, set `self.cloner_cfg.physics_clone_fn_kwargs = self.cfg.newton_replicate_kwargs`.
  - **~lines 183–185:** When calling `cloner.newton_replicate(...)`, replace the single call with:
    - `newton_kwargs = getattr(self.cfg, "newton_replicate_kwargs", None) or {}`
    - `cloner.newton_replicate(self.stage, *replicate_args, positions=self._default_env_origins, **newton_kwargs)`.
- **Reason:** So the scene actually forwards `newton_replicate_kwargs` (e.g. `equality_constraints`) into `newton_replicate` when cloning environments.

---

## 5. sim/_impl/newton_manager.py

### 5.1 Import and module-level patch

- **Added (commit e47f65e):**
  - **Line 8:** `import threading`.
  - **Lines 23–40:** Dense Jacobian patch:
    - Comment: with `mjJAC_AUTO`, nv>32 leads to sparse Jacobian and `mujoco_warp` `put_data` fails on unsupported attributes (e.g. `mjd.flexedge_J_rownnz`). Patch forces dense by setting `mjJAC_AUTO = mjJAC_DENSE`.
    - `_apply_newton_dense_jacobian_patch()`: if not already applied, set `mujoco.mjtJacobian.mjJAC_AUTO = mujoco.mjtJacobian.mjJAC_DENSE` (or via `__dict__`), then set `_isaaclab_dense_patch_applied = True`.
    - **Line 40:** Call `_apply_newton_dense_jacobian_patch()` at import time.
  - **Lines 42–61:** mj_data_memory patch:
    - `_mj_spec_memory_requested = threading.local()` to hold the requested memory size per thread.
    - `_apply_mj_spec_memory_patch(memory_bytes)`: set `_mj_spec_memory_requested.value = memory_bytes`; if not already patched, wrap `mujoco.MjSpec.compile` so that on the next `compile()` the spec’s `memory` is set to that value (then clear the request). So the **next** compile uses the given arena+stack size and avoids `mj_stackAlloc` overflow for large models (e.g. full ALLEX).
- **Reason:** Full ALLEX (60 DOF) and similar models need dense Jacobian and larger mjData; these patches make that possible without changing Newton/MuJoCo internals.

### 5.2 Call dense patch before building solver

- **Added (commit e47f65e):** ~line 233, right after `print(NewtonManager._model.gravity)` and before `NewtonManager._solver = NewtonManager._get_solver(...)`:
  - `_apply_newton_dense_jacobian_patch()`.
- **Reason:** Ensure the Jacobian is dense at the moment the solver (and thus the MuJoCo spec) is built.

### 5.3 _get_solver: mj_data_memory and memory patch

- **Modified (commit e47f65e):** In `_get_solver`, after `solver_cfg.pop("solver_type")` (~lines 403–408):
  - `mj_data_memory = solver_cfg.pop("mj_data_memory", None)`.
  - If `NewtonManager._solver_type == "mujoco_warp"` and `mj_data_memory is not None`, call `_apply_mj_spec_memory_patch(mj_data_memory)` before `return SolverMuJoCo(model, **solver_cfg)`.
- **Reason:** So `MJWarpSolverCfg(mj_data_memory=64*1024*1024)` (or similar) from the env config is consumed here and applied to the next MuJoCo compile, avoiding stack overflow for full ALLEX.

---

## 6. sim/_impl/solvers_cfg.py

- **Added (commit e47f65e):** In `MJWarpSolverCfg`, after `use_mujoco_contacts` (~lines 92–94):
  - `mj_data_memory: int | None = None`
  - Docstring: bytes for MuJoCo mjData arena+stack (mjSpec.memory). If None, MuJoCo default is used; increase (e.g. 16*1024*1024) when you see `mj_stackAlloc` overflow with large contact/constraint counts.
- **Reason:** Env config sets `ALLEX_SOLVER_CFG = MJWarpSolverCfg(..., mj_data_memory=64*1024*1024)`; this field must exist and be read by `newton_manager._get_solver`.

---

## 7. Summary table

| File | Location | Change |
|------|----------|--------|
| cloner_utils.py | clone_from_template | Pass `physics_clone_fn_kwargs` into `physics_clone_fn` (both template and heterogeneous paths). |
| cloner_utils.py | — | New `_newton_joint_index_by_name(builder, name)`. |
| cloner_utils.py | newton_replicate | New parameter `equality_constraints`; docstring; comment on add_usd scale; loop to call `p.add_equality_constraint_joint` and optional warning. |
| cloner_cfg.py | TemplateCloneCfg | New `physics_clone_fn_kwargs: dict \| None = None`. |
| interactive_scene_cfg.py | InteractiveSceneCfg | New `newton_replicate_kwargs: dict \| None = None`. |
| interactive_scene.py | clone_environments setup | Set `cloner_cfg.physics_clone_fn_kwargs` from `cfg.newton_replicate_kwargs`. |
| interactive_scene.py | clone_environments call | Pass `**newton_kwargs` into `newton_replicate(...)`. |
| newton_manager.py | top | `import threading`; dense Jacobian patch; mj_data_memory patch; call dense patch at import. |
| newton_manager.py | after model build | Call `_apply_newton_dense_jacobian_patch()` before `_get_solver`. |
| newton_manager.py | _get_solver | Pop `mj_data_memory`; if mujoco_warp and set, call `_apply_mj_spec_memory_patch(mj_data_memory)`. |
| solvers_cfg.py | MJWarpSolverCfg | New `mj_data_memory: int \| None = None` with docstring. |

No other files under `source/isaaclab/isaaclab/` were modified in this fork for the above behavior.
