# IRIM Newton — Isaac Lab Extension

## Overview

This project contains the **allex_rl_dexblind** extension for Isaac Lab: ALLEX robot + Dexblind manipulation task with Newton physics.

**Key features:**

- Isolated from the core Isaac Lab repository
- Package name: `allex_rl_dexblind` (single package under `source/`)

## Installation

1. Install Isaac Lab (see [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)).

2. From the **Isaac Lab repository root**, install this project in editable mode:

   ```bash
   # From IsaacLab repo root
   python -m pip install -e projects/irim_newton
   ```

   Or from this project directory:

   ```bash
   cd projects/irim_newton
   python -m pip install -e .
   ```

3. Verify:

   ```bash
   python projects/irim_newton/scripts/list_envs.py
   ```

## Structure

```
projects/irim_newton/
├── README.md
├── setup.py
├── config/
├── scripts/
└── source/
    └── allex_rl_dexblind/    # Python package
        ├── __init__.py
        └── tasks/
```

## References

- [Create new project or task](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html)
