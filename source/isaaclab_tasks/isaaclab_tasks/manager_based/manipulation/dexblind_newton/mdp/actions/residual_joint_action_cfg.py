# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.utils import configclass

from . import residual_joint_action as residual_action


@configclass
class ResidualJointPositionActionCfg(JointActionCfg):
    """Reference trajectory + residual * residual_scale."""

    class_type: type = residual_action.ResidualJointPositionAction

    command_name: str = MISSING
    residual_scale: float = 0.1
