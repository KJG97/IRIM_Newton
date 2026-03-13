# Project assets (ALLEX + objects)

This directory is used when running from `projects/irim_newton` so that the task does not depend on `source/isaaclab_assets`.

## Current layout (restored)

- **object/**  
  - `Hammer.usd`, `Hammer_goal_pose.usd`, `Hammer_yup_backup.usd`
- **allex_usd/**  
  - `URDF_ALLEX_RightArm/` (urdf, package.xml, meshes), `allex_model_mjcf_250903/` (xml, mesh/)
  - **ALLEX_newton_no_left.usd** — if missing, copy from your backup or build from the URDF; the robot config expects it here.

## How to populate

1. Copy from your existing `isaaclab_assets` (or IsaacLab `source/isaaclab_assets` before upstream revert):
   - `object/Hammer.usd`, `object/Hammer_goal_pose.usd`
   - `allex_usd/ALLEX_newton_no_left.usd` and the same `allex_usd` mesh/model layout that USD references.

2. Or set **ISAACLAB_ASSETS** to a directory that already has this layout; then `_resolve_assets_dir()` will use that instead of this folder.

## Resolution order

`_resolve_assets_dir()` in `dexblind_newton_env_cfg.py` uses, in order:

1. `projects/irim_newton/assets` if it exists and contains `object/`
2. `ISAACLAB_ASSETS` if set
3. `ISAACLAB_SOURCE/isaaclab_assets` if set
4. IsaacLab repo `source/isaaclab_assets`
