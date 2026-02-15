# source/isaaclab_tasks/isaaclab_tasks — Change log

This fork adds and modifies the **direct/allex** task: two direct RL envs (full body and No-Left) that use Newton with joint equality for mimic joints. Only **direct/allex** and its registration are described here; other tasks (e.g. manager_based, other direct envs) are upstream.

---

## 1. direct/allex/__init__.py

### 1.1 Registered envs

- **Added (commit 297caefa, then adjusted):**
  - **Isaac-Allex-Direct-v0:** `entry_point` → `allex_env:AllexEnv`, `env_cfg_entry_point` → `allex_env_cfg:AllexEnvCfg`. Full ALLEX, 60 DOF.
  - **Isaac-Allex-Direct-NoLeft-v0:** same env class, `env_cfg_entry_point` → `allex_env_cfg:AllexEnvNoLeftCfg`. No-Left, 31 DOF.
- **Reason:** One env class (`AllexEnv`) with two configs; the config supplies `mimic_spec`, `robot`, and `newton_replicate_kwargs`.

---

## 2. direct/allex/allex_env_cfg.py

### 2.1 Constants and mimic specs (lines 17–44)

- **ALLEX_NUM_DOF = 60**, **ALLEX_NO_LEFT_NUM_DOF = 31:** Match the actual DOF of the USD/Newton articulation (full vs No-Left).
- **ALLEX_MIMIC_SPEC (lines 26–34):** List of `(mimic_joint_name, driver_joint_name, (c0,c1,c2,c3,c4))` for **No-Left**: waist pitch dummy/upper, right-hand DIP and Thumb_IP. Coefficients from `allex_contact_sensor.xml` `<equality>` `polycoef`.
- **ALLEX_FULL_MIMIC_SPEC (lines 36–44):** **Added (commit b9e4f579):** `ALLEX_MIMIC_SPEC` plus left-hand mimic entries (L_Thumb_IP, L_Index_DIP, L_Middle_DIP, L_Ring_DIP, L_Little_DIP) with the same polynomial form as the right hand.
- **Reason:** Full body needs 12 equality constraints; No-Left needs 7. Both are passed to `newton_replicate(equality_constraints=...)` via `newton_replicate_kwargs`.

### 2.2 ALLEX_SOLVER_CFG (lines 45–59)

- **Modified (commit b9e4f579):**
  - **nconmax:** 3000 → **6000** (comment: full ALLEX ~295 shapes, broadphase needs ~5850).
  - **mj_data_memory:** **64*1024*1024** (64 MiB). Comment: full ALLEX triggers mjData stack overflow without this.
- **Reason:** Full-body model has many contacts/constraints; default MuJoCo stack is too small; solver config is shared and must support the larger case.

### 2.3 AllexEnvCfg (lines 62–101)

- **Class docstring:** Updated to “full 60 DOF: waist + both arms/hands”.
- **mimic_spec:** Set to **ALLEX_FULL_MIMIC_SPEC** (so env and joint_slider_agent use the full mimic list).
- **scene:** `newton_replicate_kwargs={"equality_constraints": ALLEX_FULL_MIMIC_SPEC}`.
- **robot:** `ALLEX_CFG.replace(prim_path="/World/envs/env_.*/Robot")` (ALLEX_CFG uses `allex_test.usd`).

### 2.4 AllexEnvNoLeftCfg (lines 104–142)

- **mimic_spec:** **ALLEX_MIMIC_SPEC** (7 entries).
- **newton_cfg:** e.g. `num_substeps=4`, `use_cuda_graph=True` (optional; can be False for debugging).
- **scene:** `newton_replicate_kwargs={"equality_constraints": ALLEX_MIMIC_SPEC}`.
- **robot:** `ALLEX_NO_LEFT_CFG.replace(prim_path=...)`.

---

## 3. direct/allex/allex_env.py

### 3.1 Imports (line 22)

- **Imports:** `ALLEX_MIMIC_SPEC`, `AllexEnvCfg` from `allex_env_cfg`. `ALLEX_MIMIC_SPEC` is the default when a config does not define `mimic_spec`.

### 3.2 _ensure_joint_dof_idx (lines 43–51)

- **Modified (commit b9e4f579):** **Lines 47–50:** Instead of iterating over a hardcoded `ALLEX_MIMIC_SPEC`, use `mimic_spec = getattr(self.cfg, "mimic_spec", ALLEX_MIMIC_SPEC)` and build `_mimic_overrides` from that.
- **Reason:** Full body uses `ALLEX_FULL_MIMIC_SPEC`, No-Left uses `ALLEX_MIMIC_SPEC`; the env must use whichever list the config provides so driver-only targets are correct for both tasks.

### 3.3 _apply_action (lines 56–69)

- **Original (earlier):** Had `scale = 0.5`, `target = current + scale * self.actions`, and a branch on `use_newton_equality_for_mimic`: if True, set position target only for driver joints; else set for all.
- **Modified (commits 7108939, b9e4f579):**
  - **Removed:** `scale` and `use_newton_equality_for_mimic`; action is now a direct offset: `target = current + self.actions`.
  - **Lines 59–68:** If `_mimic_overrides` is non-empty: collect mimic joint indices, compute driver indices, then call `self.robot.set_joint_position_target(target_driver, joint_ids=driver_joint_ids)`. Else: `self.robot.set_joint_position_target(target, joint_ids=self._joint_dof_idx)`.
- **Reason:** Mimic joints are always enforced by Newton equality; the env must only set targets for driver joints so it does not fight the constraint. No config flag is needed.

---

## 4. Summary table

| File | Location | Change |
|------|----------|--------|
| __init__.py | — | Register Isaac-Allex-Direct-v0 (AllexEnvCfg) and Isaac-Allex-Direct-NoLeft-v0 (AllexEnvNoLeftCfg). |
| allex_env_cfg.py | 36–44 | Add ALLEX_FULL_MIMIC_SPEC (ALLEX_MIMIC_SPEC + left-hand mimic). |
| allex_env_cfg.py | 45–59 | ALLEX_SOLVER_CFG: nconmax=6000, mj_data_memory=64 MiB. |
| allex_env_cfg.py | 62–101 | AllexEnvCfg: mimic_spec=ALLEX_FULL_MIMIC_SPEC, newton_replicate_kwargs with FULL spec, robot=ALLEX_CFG. |
| allex_env_cfg.py | 104–142 | AllexEnvNoLeftCfg: mimic_spec=ALLEX_MIMIC_SPEC, newton_replicate_kwargs with ALLEX_MIMIC_SPEC, robot=ALLEX_NO_LEFT_CFG. |
| allex_env.py | 47–50 | _ensure_joint_dof_idx: use getattr(self.cfg, "mimic_spec", ALLEX_MIMIC_SPEC) instead of hardcoded list. |
| allex_env.py | 57–68 | _apply_action: target = current + actions; if _mimic_overrides, set position target for driver joints only; else all. Remove scale and use_newton_equality_for_mimic. |

Other subpackages under `source/isaaclab_tasks/isaaclab_tasks/` (e.g. manager_based, other direct tasks, utils) were not modified in this fork for the ALLEX Newton direct envs.
