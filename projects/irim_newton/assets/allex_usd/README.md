# source/isaaclab_assets/allex_usd — Change log

This directory holds **ALLEX robot assets** used by this fork with the Newton physics backend. The code references only a subset of these files; the rest are legacy or alternate variants.

---

## 1. Files actually used by the fork

| File / directory | Used by | Role |
|------------------|--------|------|
| **allex_test.usd** | `ALLEX_CFG` in `isaaclab_assets/robots/allex.py` | Full ALLEX (60 DOF): waist, neck, both arms and hands. Newton replication and visualization. Mesh geometry is expected to be in **meters** (no parent scale 0.001) because Newton’s `add_usd()` does not apply USD xform scale to mesh vertices. |
| **ALLEX_newton_no_left.usd** | `ALLEX_NO_LEFT_CFG` in `isaaclab_assets/robots/allex.py` | No-Left variant (31 DOF): waist, right arm and right hand only; left arm/hand removed; neck typically fixed. Same meter-unit expectation for Newton. |
| **allex_model_mjcf_250903/allex_contact_sensor.xml** | Reference only (equality coefficients) | MJCF export of ALLEX with contact sensors; contains `<equality>` joint definitions and `polycoef` used to build `ALLEX_MIMIC_SPEC` / `ALLEX_FULL_MIMIC_SPEC` in `allex_env_cfg.py`. Not loaded by Isaac Lab; only the polynomial coefficients are copied into the env config for `newton_replicate(equality_constraints=...)`. |

---

## 2. Additions and modifications (asset side)

- **allex_test.usd**  
  - **Origin:** Derived from full ALLEX USD (e.g. ALLEX_newton or ALLEX) for use as the **full-body** Newton asset.
  - **Changes made outside this repo (referred to in past work):**
    - Instanceable prims were uninstanced so that xform/scale could be edited.
    - xform scale 0.001 (mm→m) on mesh parents was set to (1,1,1) so that in USD the hierarchy has no scale.
    - Mesh vertices were scaled by 0.001 (mm→m) and extent updated so that geometry is in **meters**. That way Newton’s importer (which does not apply USD scale to mesh geometry) shows the robot at the same size as ALLEX_newton_no_left.usd.
  - **No in-repo script** in this fork modifies this file; the above was done by one-off scripts (since removed).

- **ALLEX_newton_no_left.usd**  
  - **Origin:** Newton-ready No-Left variant (left arm/hand removed, structure compatible with Newton replication).
  - **Usage:** Loaded as-is; no code changes in this fork to the file itself.

- **allex_model_mjcf_250903/**  
  - **allex_contact_sensor.xml:** Defines joint equalities (mimic) with `polycoef`. The env configs in `isaaclab_tasks` copy these coefficients into Python lists (`ALLEX_MIMIC_SPEC`, `ALLEX_FULL_MIMIC_SPEC`) for `equality_constraints`. No automatic loading of this XML in code; it is a reference for the numbers.

---

## 3. Other files in this directory (not used by fork code)

- **ALLEX.usd**, **ALLEX_backup.usd**, **ALLEX_newton.usd**, **ALLEX_newton_backup.usd**, **ALLEX_Right_Arm.usd**, **ALLEX_XML_test.usd**  
  Other variants or backups; the fork’s `allex.py` does **not** reference them.

- **URDF_ALLEX_RightArm/**  
  URDF and meshes for the right arm; not used by the Newton ALLEX envs in this fork.

- **configuration/**, **allex_contact_sensor** (if present), **mesh/** (if at top level)  
  Supporting or legacy data; not referenced by the fork’s Python or env configs.

---

## 4. Summary

| What | Where / how |
|------|-------------|
| Full-body (60 DOF) | `allex_test.usd` — used by `ALLEX_CFG`. Geometry in meters; no scale 0.001 on mesh parents in USD. |
| No-Left (31 DOF) | `ALLEX_newton_no_left.usd` — used by `ALLEX_NO_LEFT_CFG`. |
| Mimic (equality) coefficients | Taken from `allex_model_mjcf_250903/allex_contact_sensor.xml` and hardcoded in `allex_env_cfg.py` as `ALLEX_MIMIC_SPEC` and `ALLEX_FULL_MIMIC_SPEC`. |
| Scale / units | Newton does not apply USD xform scale to mesh; assets used with Newton should have mesh geometry in meters (e.g. after vertex * 0.001 and extent update). |

No Python or config files under `source/isaaclab_assets/allex_usd/` were added or modified in this fork; only the usage and expectations of the above USD/MJCF assets are documented here.
