<div align="center">

# Isaac Lab (Newton + ALLEX)

**Isaac Lab with Newton physics: ALLEX humanoid direct RL environments and joint slider control**

[Overview](#-overview) •
[Installation](#-installation) •
[Project structure](#-project-structure) •
[Changes summary](#-changes-summary) •
[Run](#-run)

</div>

---

## 📖 Overview

This repository is a fork of **Isaac Lab**’s **Newton physics** branch, extended with custom environments and assets for the **ALLEX humanoid** robot.

### Main changes

- **Direct RL environments (ALLEX)**  
  Two variants: **full body** (60 DOF, `allex_test.usd`) and **No-Left** (31 DOF, `ALLEX_newton_no_left.usd`). Mimic joints are constrained via Newton/MuJoCo joint equality; only driver joints receive actions.
- **Joint Slider Agent**  
  PySide GUI to control active (driver) joints with degree sliders in the same process as the simulation. Works for both full and No-Left; driver vs mimic is determined by the env config `mimic_spec`.
- **Mimic constraints**  
  `equality_constraints` are passed into `newton_replicate(...)` and injected as MuJoCo joint equality. Mimic actuators use stiffness=0 and damping only.
- **Newton/MuJoCo integration**  
  Dense Jacobian patch for nv>32 to avoid sparse Jacobian issues; increased `mj_data_memory` and `nconmax` for the full-body model.

---

## 🔧 Installation

To reproduce the environment:

### 1. Isaac Sim

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### 2. Build dependencies (Linux)

```bash
sudo apt install cmake build-essential
```

### 3. Isaac Lab

From the repository root:

```bash
./isaaclab.sh --install
# or
./isaaclab.sh -i
```

Then use the [Run](#-run) commands below.

---

## 📁 Project structure

Only paths that were added or modified in this fork are listed.

```
IsaacLab/
├── scripts/
│   └── environments/
│       └── joint_slider_agent.py
│
└── source/
    ├── isaaclab/
    │   └── isaaclab/
    │       ├── cloner/
    │       │   └── cloner_utils.py
    │       ├── scene/
    │       │   └── interactive_scene_cfg.py
    │       └── sim/
    │           └── _impl/
    │               ├── newton_manager.py
    │               ├── newton_manager_cfg.py
    │               └── solvers_cfg.py
    │
    ├── isaaclab_assets/
    │   ├── isaaclab_assets/
    │   │   └── robots/
    │   │       └── allex.py
    │   └── allex_usd/
    │       ├── allex_test.usd           # Full ALLEX (60 DOF)
    │       ├── ALLEX_newton_no_left.usd # No-Left (31 DOF)
    │       └── allex_model_mjcf_250903/
    │           └── allex_contact_sensor.xml
    │
    └── isaaclab_tasks/
        └── isaaclab_tasks/
            └── direct/
                └── allex/
                    ├── allex_env.py
                    └── allex_env_cfg.py
```

---

## 📋 Changes summary

| Path | Role |
|------|------|
| **scripts/environments/joint_slider_agent.py** | PySide GUI: degree sliders for active (driver) joints only. Uses cfg `mimic_spec` for both full and No-Left; drives `env.step(actions)` in the same process. |
| **isaaclab/cloner/cloner_utils.py** | `newton_replicate(..., equality_constraints=...)` to inject MuJoCo joint equality for mimic joints. |
| **isaaclab/scene/interactive_scene_cfg.py** | Passes `newton_replicate_kwargs` (e.g. `equality_constraints`) into the cloner when replicating. |
| **isaaclab/sim/_impl/newton_manager.py, solvers_cfg.py** | Newton: dense Jacobian patch (nv>32), `mj_data_memory` and solver config; substeps and CUDA Graph options. |
| **isaaclab_assets/robots/allex.py** | ALLEX_CFG (allex_test.usd, 60 DOF) and ALLEX_NO_LEFT_CFG. Mimic actuators: stiffness=0, damping only. |
| **isaaclab_assets/allex_usd/** | Full-body and No-Left USD assets; MJCF equality definitions in `allex_contact_sensor.xml`. |
| **isaaclab_tasks/direct/allex/** | Direct RL env: sets position targets only for driver joints (from cfg `mimic_spec`); mimic motion is enforced by Newton equality. |

---

## 🚀 Run

### Joint Slider

**Full ALLEX (60 DOF)**

```bash
./isaaclab.sh -p scripts/environments/joint_slider_agent.py --task Isaac-Allex-Direct-v0 --visualizer newton
```

**No-Left (31 DOF, right arm and hand only)**

```bash
./isaaclab.sh -p scripts/environments/joint_slider_agent.py --task Isaac-Allex-Direct-NoLeft-v0 --visualizer newton
```

In both tasks, only active (driver) joints appear as sliders; mimic joints follow via equality constraints.

---

## 📚 References

- [Isaac Lab – Newton Physics Integration](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html)
