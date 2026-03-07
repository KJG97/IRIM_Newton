# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ALLEX No-Left robot config for dexblind_newton: Newton physics + Residual RL action."""

from isaaclab_assets.robots import ALLEX_NO_LEFT_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import dexblind_newton_env_cfg as newton_cfg
from ...dexblind_newton_env_cfg import RIGHT_ARM_HAND_JOINT_NAMES


@configclass
class AllexNewtonMixinCfg:
    """Set robot to ALLEX No-Left and wire observations/actions to joint names."""

    def __post_init__(self: newton_cfg.DexblindNewtonLiftEnvCfg):
        super().__post_init__()
        self.scene.robot = ALLEX_NO_LEFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        joint_cfg = SceneEntityCfg("robot", joint_names=RIGHT_ARM_HAND_JOINT_NAMES)
        if hasattr(self.observations.proprio, "joint_pos"):
            self.observations.proprio.joint_pos.params["asset_cfg"] = joint_cfg
        if hasattr(self.observations.proprio, "reference_joint_pos"):
            self.observations.proprio.reference_joint_pos.params["joint_names"] = (
                self.actions.action.joint_names
            )


@configclass
class DexblindNewtonAllexLiftEnvCfg(AllexNewtonMixinCfg, newton_cfg.DexblindNewtonLiftEnvCfg):
    pass


@configclass
class DexblindNewtonAllexLiftEnvCfg_PLAY(AllexNewtonMixinCfg, newton_cfg.DexblindNewtonLiftEnvCfg_PLAY):
    pass
