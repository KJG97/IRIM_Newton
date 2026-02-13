# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.utils import configclass

from . import residual_joint_action as residual_action


@configclass
class ResidualJointPositionActionCfg(JointActionCfg):
    """Configuration for residual joint position action term.

    This action term combines a reference trajectory (from command manager) with
    residual actions from the RL agent. The final action is:
        final_action = reference_trajectory + residual_action * residual_scale
    """

    class_type: type = residual_action.ResidualJointPositionAction

    command_name: str = MISSING
    """Name of the command term that provides the reference trajectory.

    This should match the name used in CommandsCfg (e.g., "reference_trajectory").
    The command must output joint positions with shape (num_envs, num_joints).
    """

    residual_scale: float = 0.1
    """Scaling factor for residual actions. Defaults to 0.1.

    This controls how much the RL agent can modify the reference trajectory.
    Smaller values (e.g., 0.1) mean the agent makes smaller corrections.
    Larger values (e.g., 1.0) allow larger modifications.
    """
