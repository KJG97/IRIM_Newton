# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
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
    loop: bool = True
    loop_start_time: float | None = None
    space: str = "joint"
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    playback_speed: float = 1.0
