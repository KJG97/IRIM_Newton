# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import warp as wp

from isaaclab.utils import close_simulation, is_simulation_running
from isaaclab.utils.timer import Timer

Timer.enable = False
Timer.enable_display_output = False

import isaaclab_tasks_experimental  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

# Set True to print table, robot base, and env coordinates every DEBUG_POSITIONS_INTERVAL steps.
DEBUG_POSITIONS = True
DEBUG_POSITIONS_INTERVAL = 100


def _print_scene_positions(env, step: int):
    """Print env origins, robot base pos, and table pos (if present) for debugging."""
    scene = env.unwrapped.scene
    print(f"\n[DEBUG positions] step={step}")
    # Env origins (num_envs, 3)
    origins = scene.env_origins
    if origins is not None and origins.numel():
        print(f"  env_origins (world): {origins.cpu().tolist()}")
    # Robot base position (world) — use scene.keys() to avoid __getitem__(index) from "in scene"
    if "robot" in scene.keys():
        robot = scene["robot"]
        try:
            root_pos = wp.to_torch(robot.data.root_link_pos_w)
            if root_pos is not None and root_pos.numel():
                print(f"  robot_base_pos_w: {root_pos.cpu().tolist()}")
        except Exception as e:
            print(f"  robot_base_pos_w: (error) {e}")
    # Table position (world) - from USD Xform if present
    if "table" in scene.keys():
        try:
            table = scene["table"]
            positions, _ = table.get_world_poses()
            if positions is not None and positions.numel():
                print(f"  table_pos_w: {positions.cpu().tolist()}")
        except Exception as e:
            print(f"  table_pos_w: (error) {e}")
    print(flush=True)


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    step_count = 0
    while is_simulation_running(simulation_app, env.unwrapped.sim):
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)
            step_count += 1
            if DEBUG_POSITIONS and step_count % DEBUG_POSITIONS_INTERVAL == 0:
                _print_scene_positions(env, step_count)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    close_simulation(simulation_app)
