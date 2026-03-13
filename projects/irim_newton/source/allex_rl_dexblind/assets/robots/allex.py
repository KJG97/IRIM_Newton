# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ALLEX No-Left robot config for dexblind_newton (migrated from isaaclab_assets)."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Project assets: .../allex_rl_dexblind/assets/robots/allex.py -> .../projects/irim_newton
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_ALLEX_NO_LEFT_USD_PATH = _PROJECT_ROOT / "assets" / "allex_usd" / "ALLEX_newton_no_left.usd"

ALLEX_NO_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_ALLEX_NO_LEFT_USD_PATH.resolve()),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Waist_Yaw_Joint": 0.0,
            "Waist_Pitch_Lower_Joint": 0.0,
            "R_Shoulder_Pitch_Joint": 0.174533,
            "R_Shoulder_Roll_Joint": 0.0,
            "R_Shoulder_Yaw_Joint": -0.174533,
            "R_Elbow_Joint": -1.8,
            "R_Wrist_Yaw_Joint": 3.14159,
            "R_Wrist_Roll_Joint": -0.261799,
            "R_Wrist_Pitch_Joint": -0.436332,
            "R_Thumb_Yaw_Joint": -1.48353,
            "R_Thumb_CMC_Joint": 0.872665,
            "R_Thumb_MCP_Joint": 0.0,
            "R_Index_Roll_Joint": 0.0,
            "R_Index_MCP_Joint": 0.261799,
            "R_Index_PIP_Joint": 0.349066,
            "R_Middle_Roll_Joint": 0.0,
            "R_Middle_MCP_Joint": 0.0,
            "R_Middle_PIP_Joint": 0.0,
            "R_Ring_Roll_Joint": 0.0,
            "R_Ring_MCP_Joint": 0.0,
            "R_Ring_PIP_Joint": 0.0,
            "R_Little_Roll_Joint": 0.0,
            "R_Little_MCP_Joint": 0.0,
            "R_Little_PIP_Joint": 0.0,
        },
        pos=(0.0, 0.0, 0.68),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[
                "Waist_Yaw_Joint",
                "Waist_Pitch_Lower_Joint",
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=2.61,
            stiffness=100.0,
            damping=10.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "R_Shoulder_Pitch_Joint",
                "R_Shoulder_Roll_Joint",
                "R_Shoulder_Yaw_Joint",
                "R_Elbow_Joint",
                "R_Wrist_Yaw_Joint",
                "R_Wrist_Roll_Joint",
                "R_Wrist_Pitch_Joint",
            ],
            effort_limit_sim={
                "R_Shoulder_Pitch_Joint": 300.0,
                "R_Shoulder_Roll_Joint": 150.0,
                "R_Shoulder_Yaw_Joint": 150.0,
                "R_Elbow_Joint": 150.0,
                "R_Wrist_Yaw_Joint": 120.0,
                "R_Wrist_Roll_Joint": 120.0,
                "R_Wrist_Pitch_Joint": 120.0,
            },
            velocity_limit_sim=10,
            stiffness={
                "R_Shoulder_Pitch_Joint": 200.0,
                "R_Shoulder_Roll_Joint": 175.0,
                "R_Shoulder_Yaw_Joint": 175.0,
                "R_Elbow_Joint": 175.0,
                "R_Wrist_Yaw_Joint": 125.0,
                "R_Wrist_Roll_Joint": 125.0,
                "R_Wrist_Pitch_Joint": 125.0,
            },
            damping={
                "R_Shoulder_Pitch_Joint": 20.0,
                "R_Shoulder_Roll_Joint": 17.5,
                "R_Shoulder_Yaw_Joint": 17.5,
                "R_Elbow_Joint": 17.5,
                "R_Wrist_Yaw_Joint": 12.5,
                "R_Wrist_Roll_Joint": 12.5,
                "R_Wrist_Pitch_Joint": 12.5,
            },
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
                "R_Index_Roll_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint",
                "R_Middle_Roll_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
                "R_Ring_Roll_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
                "R_Little_Roll_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint",
            ],
            effort_limit_sim={
                "R_Thumb_Yaw_Joint": 1.5, "R_Thumb_CMC_Joint": 2.0, "R_Thumb_MCP_Joint": 1.0,
                "R_Index_Roll_Joint": 1.5, "R_Index_MCP_Joint": 2.0, "R_Index_PIP_Joint": 1.0,
                "R_Middle_Roll_Joint": 1.5, "R_Middle_MCP_Joint": 2.0, "R_Middle_PIP_Joint": 1.0,
                "R_Ring_Roll_Joint": 1.5, "R_Ring_MCP_Joint": 2.0, "R_Ring_PIP_Joint": 1.0,
                "R_Little_Roll_Joint": 1.5, "R_Little_MCP_Joint": 2.0, "R_Little_PIP_Joint": 1.0,
            },
            velocity_limit_sim=10.0,
            stiffness={
                "R_Thumb_Yaw_Joint": 10.0, "R_Thumb_CMC_Joint": 10.0, "R_Thumb_MCP_Joint": 10.0,
                "R_Index_Roll_Joint": 10.0, "R_Index_MCP_Joint": 10.0, "R_Index_PIP_Joint": 10.0,
                "R_Middle_Roll_Joint": 10.0, "R_Middle_MCP_Joint": 10.0, "R_Middle_PIP_Joint": 10.0,
                "R_Ring_Roll_Joint": 10.0, "R_Ring_MCP_Joint": 10.0, "R_Ring_PIP_Joint": 10.0,
                "R_Little_Roll_Joint": 10.0, "R_Little_MCP_Joint": 10.0, "R_Little_PIP_Joint": 10.0,
            },
            damping={
                "R_Thumb_Yaw_Joint": 1.0, "R_Thumb_CMC_Joint": 1.0, "R_Thumb_MCP_Joint": 1.0,
                "R_Index_Roll_Joint": 1.0, "R_Index_MCP_Joint": 1.0, "R_Index_PIP_Joint": 1.0,
                "R_Middle_Roll_Joint": 1.0, "R_Middle_MCP_Joint": 1.0, "R_Middle_PIP_Joint": 1.0,
                "R_Ring_Roll_Joint": 1.0, "R_Ring_MCP_Joint": 1.0, "R_Ring_PIP_Joint": 1.0,
                "R_Little_Roll_Joint": 1.0, "R_Little_MCP_Joint": 1.0, "R_Little_PIP_Joint": 1.0,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
