# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""List environments registered by allex_rl_dexblind."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from prettytable import PrettyTable

import allex_rl_dexblind.tasks  # noqa: F401

def main():
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments (allex_rl_dexblind)"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"
    index = 0
    # Project envs: Isaac-Dexblind-Newton-Allex-*
    for task_spec in gym.registry.values():
        if "Isaac-Dexblind" in task_spec.id:
            kwargs = task_spec.kwargs or {}
            config = kwargs.get("env_cfg_entry_point", "")
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, config])
            index += 1
    print(table)

if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
