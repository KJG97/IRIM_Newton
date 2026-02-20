# scripts/environments

This directory contains scripts for running ALLEX with the Newton visualizer.  
**The only file added in this fork is `joint_slider_agent.py`**; all other files are unchanged from upstream.

---

## joint_slider_agent.py

**Purpose:** Controls ALLEX joint **position targets** via a PySide GUI in the same process as the simulation.  
Sliders are shown only for **driver (active) joints**. Mimic joints are constrained by Newton/MuJoCo equality constraints and are displayed without sliders (current/target values are computed from the driver polynomial).

### How to run

```bash
# Full ALLEX (60 DOF)
./isaaclab.sh -p scripts/environments/joint_slider_agent.py --task Isaac-Allex-Direct-v0 --visualizer newton

# Right arm only (31 DOF)
./isaaclab.sh -p scripts/environments/joint_slider_agent.py --task Isaac-Allex-Direct-NoLeft-v0 --visualizer newton
```

### Implementation summary

| Item | Description |
|------|--------------|
| **Dense Jacobian patch** | When `nv>32`, `mjJAC_AUTO` selects a sparse Jacobian and mujoco_warp’s `put_data` fails on unsupported fields. The patch overrides `mjJAC_AUTO` with `mjJAC_DENSE` so the model is always built with a dense Jacobian. Applied at script startup before any env (and thus Newton) is created. |
| **Driver vs mimic** | Driver and mimic sets are read from `unwrapped.cfg.mimic_spec`. Full body has 12 mimic entries, No-Left has 7, so the GUI builds the correct slider set for the chosen `--task` without hardcoding. |

No other files under `scripts/environments/` were modified in this fork.
