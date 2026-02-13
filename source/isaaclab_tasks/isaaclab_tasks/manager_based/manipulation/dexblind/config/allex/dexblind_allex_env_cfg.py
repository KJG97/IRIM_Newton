# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots import ALLEX_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import dexblind_env_cfg as dexblind
from ... import mdp

# 오른팔(7) + 오른손(15, Roll 포함) — joint_effort 관측용
RIGHT_ARM_HAND_TORQUE_JOINTS = [
    "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
    "R_Elbow_Joint",
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
    "R_Index_Roll_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_Roll_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_Roll_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_Roll_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint",
]


@configclass
class AllexResidualJointPosActionCfg:
    """Residual RL: final = reference_trajectory + residual × residual_scale. 궤적 18관절(팔7+손11, Roll 4개 제외)."""
    action = mdp.ResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
            "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
            "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
            "R_Index_MCP_Joint", "R_Index_PIP_Joint",
            "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
            "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
            "R_Little_MCP_Joint", "R_Little_PIP_Joint",
        ],
        preserve_order=True,
        command_name="reference_trajectory",
        residual_scale=0.1,
        scale=1.0,
    )


@configclass
class AllexMixinCfg:
    rewards: dexblind.RewardsCfg = dexblind.RewardsCfg()
    actions: AllexResidualJointPosActionCfg = AllexResidualJointPosActionCfg()

    def __post_init__(self: dexblind.DexblindLiftEnvCfg):
        super().__post_init__()
        self.scene.robot = ALLEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True

        joint_cfg = SceneEntityCfg("robot", joint_names=self.actions.action.joint_names)
        if hasattr(self.observations.proprio, "joint_pos"):
            self.observations.proprio.joint_pos.params["asset_cfg"] = joint_cfg
        if hasattr(self.observations.proprio, "reference_joint_pos"):
            self.observations.proprio.reference_joint_pos.params["joint_names"] = self.actions.action.joint_names

        # right_hand_joint_torque: base에서 이미 19개(_RIGHT_ARM_HAND_TORQUE_JOINT_NAMES) + preserve_order=True 사용


@configclass
class DexblindAllexLiftEnvCfg(AllexMixinCfg, dexblind.DexblindLiftEnvCfg):
    pass


@configclass
class DexblindAllexLiftEnvCfg_PLAY(AllexMixinCfg, dexblind.DexblindLiftEnvCfg_PLAY):
    pass
