# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.utils import configclass

from . import residual_joint_action as residual_action


@configclass
class ResidualJointPositionActionCfg(JointActionCfg):
    """Blended action: smoothly transitions from residual-on-reference to relative joint control.

    The final target sent to the actuator is:

        target = w * ref + (1 - w) * q_current + residual * residual_scale

    where ``w`` = :attr:`reference_weight` (schedulable via curriculum).

    * **w = 1** (bootstrap): pure Residual RL, ``target = ref + residual * residual_scale``
    * **w = 0** (independent): pure relative control, ``target = q_current + residual * residual_scale``
    """

    class_type: type = residual_action.ResidualJointPositionAction

    command_name: str = MISSING
    residual_scale: float = 0.1

    reference_weight: float = 1.0
    """Blending weight for the reference trajectory (0.0 ~ 1.0).

    Schedule this from 1 -> 0 via curriculum to transition from
    Residual RL to independent policy control.
    """
