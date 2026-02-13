# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from . import reference_trajectory_command as ref_cmd


@configclass
class ReferenceTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for reference trajectory command generator."""

    class_type: type = ref_cmd.ReferenceTrajectoryCommand

    trajectory_file: str = MISSING
    """Path to the trajectory file (.npz format).

    The file should contain:
        - `times`: Time array (N,) in seconds
        - `positions`: Joint positions (N, num_joints) in radians
        - `joint_names`: List of joint names (num_joints,)
        - `sample_rate`: Sampling rate in Hz

    If a relative path is provided, it will be resolved relative to the data directory.
    """

    loop: bool = True
    """Whether to loop the trajectory playback. Defaults to True.

    If True, the trajectory will restart from the beginning when it reaches the end.
    If False, the trajectory will stop at the last frame.
    """

    loop_start_time: float | None = None
    """Start time for loop playback after initial playback [s]. Defaults to None.

    If set, the trajectory will play once from 0 to the end, then loop from
    `loop_start_time` to the end. This is useful when the initial part of the
    trajectory causes collisions (e.g., moving from table to workspace).
    
    Example: If trajectory duration is 6.0s and loop_start_time=2.5, then:
        - Initial playback: 0.0s -> 6.0s (once)
        - Loop playback: 2.5s -> 6.0s (repeated)
    """

    space: str = "joint"
    """Space type for the trajectory. Defaults to "joint".

    Options:
        - "joint": Joint space trajectory (positions are joint angles)
        - "task": Task space trajectory (positions are end-effector poses)
    """

    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    """Time range before commands are resampled [s]. Defaults to (10.0, 10.0).

    For reference trajectories, resampling restarts the trajectory playback.
    Set to a large value if you want the trajectory to play continuously.
    """

    playback_speed: float = 1.0
    """Playback speed multiplier for the trajectory. Defaults to 1.0.
    
    This allows controlling the trajectory playback speed without changing decimation.
    For example:
        - playback_speed=1.0: Normal speed (1x)
        - playback_speed=2.0: Double speed (2x)
        - playback_speed=0.5: Half speed (0.5x)
    
    This is useful when you want to speed up or slow down the trajectory playback
    without affecting the simulation timestep or decimation settings.
    """

