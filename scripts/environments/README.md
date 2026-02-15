# scripts/environments — Change log

This directory contains **added** scripts for running ALLEX with the Newton visualizer. Only `joint_slider_agent.py` was added in this fork; the rest are from upstream.

---

## 1. Added file: `joint_slider_agent.py`

**Purpose:** PySide GUI to control ALLEX joint position targets in the same process as the simulation. Only **active (driver)** joints get sliders; mimic joints are constrained by Newton/MuJoCo equality and are not shown.

### 1.1 Docstring and run examples (lines 1–13)

- **Lines 4–12:** Module docstring.
  - **Added (commit 4142dc):** Two run examples — full ALLEX (60 DOF) with `Isaac-Allex-Direct-v0`, and No-Left (31 DOF) with `Isaac-Allex-Direct-NoLeft-v0`. Previously only the No-Left command was documented.
  - **Reason:** Support both tasks from the same script; user needs to know which `--task` to pass.

### 1.2 Dense Jacobian patch (lines 33–47)

- **Added (commit 4142dc):** Function `_apply_newton_dense_jacobian_patch()` and its call right after `simulation_app = app_launcher.app`.
  - **Lines 33–35:** Comment: when Newton builds with `mjJAC_AUTO`, `nv>32` uses sparse Jacobian and `mujoco_warp` `put_data` hits unsupported attributes (e.g. `mjd.flexedge_J_rownnz`). Patch forces dense by setting `mjJAC_AUTO = mjJAC_DENSE`.
  - **Lines 36–44:** Try/except: set `mujoco.mjtJacobian.mjJAC_AUTO = mjJAC_DENSE` (or via `__dict__` if the enum is read-only).
  - **Line 47:** Call `_apply_newton_dense_jacobian_patch()` so the patch is applied before the env (and thus Newton) is created.
  - **Reason:** This script launches the env in the same process; the patch must run before any MuJoCo model is built so full ALLEX (60 DOF) works.

### 1.3 Imports (lines 61–62)

- **Removed (commit 4142dc):** `from isaaclab_tasks.direct.allex.allex_env_cfg import ALLEX_MIMIC_SPEC`.
  - **Reason:** Mimic list is no longer hardcoded; it is read from the env config so both No-Left and full-body tasks are supported.

### 1.4 Driver vs mimic from config (lines 327–338)

- **Lines 327–329 (was 308–309):** Build the set of mimic joint names and the list of driver indices.
  - **Changed:** Instead of `mimic_names = {m[0] for m in ALLEX_MIMIC_SPEC}`, use `mimic_spec = getattr(unwrapped.cfg, "mimic_spec", None) or []` and `mimic_names = {m[0] for m in mimic_spec}`.
  - **Reason:** `AllexEnvCfg` uses `ALLEX_FULL_MIMIC_SPEC` (12 entries), `AllexEnvNoLeftCfg` uses `ALLEX_MIMIC_SPEC` (7 entries). Reading from `unwrapped.cfg.mimic_spec` makes the GUI show the correct driver-only set for the chosen task.
- **Lines 338 (was 318):** Loop for `driver_mimics_info`: `for mimic_name, driver_name, polycoef in mimic_spec:` (was `ALLEX_MIMIC_SPEC`).
  - **Reason:** Same as above; passive rows under drivers use the same spec as the env.

---

## 2. Summary

| Location              | Change                                                                 |
|-----------------------|------------------------------------------------------------------------|
| Docstring             | Run examples for full (60 DOF) and No-Left (31 DOF)                   |
| After line 31         | Add dense Jacobian patch and call it before other imports/env creation |
| Imports               | Remove `ALLEX_MIMIC_SPEC`; use cfg instead                              |
| main() driver/mimic   | Use `unwrapped.cfg.mimic_spec` for mimic names and mimic info loop     |

No other files in `scripts/environments/` were modified by this fork.
