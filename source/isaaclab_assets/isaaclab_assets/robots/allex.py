# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# Get absolute path to USD file relative to this file
_ALLEX_USD_PATH = Path(__file__).parent.parent.parent / "allex_usd" / "allex_test.usd"
_ALLEX_NO_LEFT_USD_PATH = Path(__file__).parent.parent.parent / "allex_usd" / "ALLEX_newton_no_left.usd"

ALLEX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_ALLEX_USD_PATH.resolve()),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Waist joints
            "Waist_Yaw_Joint": 0.0,
            "Waist_Pitch_Dummy_Joint": 0.0,
            "Waist_Pitch_Lower_Joint": 0.0,
            "Waist_Pitch_Upper_Joint": 0.0,
            # Neck joints
            "Neck_Pitch_Joint": 0.0,
            "Neck_Yaw_Joint": 0.0,
            # Left arm joints
            "L_Shoulder_Pitch_Joint": 0.0,
            "L_Shoulder_Roll_Joint": 0.0,
            "L_Shoulder_Yaw_Joint": 0.0,
            "L_Elbow_Joint": 0.0,
            "L_Wrist_Yaw_Joint": 0.0,
            "L_Wrist_Roll_Joint": 0.0,
            "L_Wrist_Pitch_Joint": 0.0,
            # Right arm joints
            "R_Shoulder_Pitch_Joint": 0.0,
            "R_Shoulder_Roll_Joint": 0.0,
            "R_Shoulder_Yaw_Joint": 0.0,
            "R_Elbow_Joint": -1.5708,
            "R_Wrist_Yaw_Joint": 3.49066,
            "R_Wrist_Roll_Joint": 0.0,
            "R_Wrist_Pitch_Joint": 0.0,
            # Left hand joints (MCP/PIP active; DIP, Thumb_IP passive/mimic)
            "L_Thumb_Yaw_Joint": 0.0,
            "L_Thumb_CMC_Joint": 0.0,
            "L_Thumb_MCP_Joint": 0.0,
            "L_Thumb_IP_Joint": 0.0,
            "L_Index_Roll_Joint": 0.0,
            "L_Index_MCP_Joint": 0.0,
            "L_Index_PIP_Joint": 0.0,
            "L_Index_DIP_Joint": 0.0,
            "L_Middle_Roll_Joint": 0.0,
            "L_Middle_MCP_Joint": 0.0,
            "L_Middle_PIP_Joint": 0.0,
            "L_Middle_DIP_Joint": 0.0,
            "L_Ring_Roll_Joint": 0.0,
            "L_Ring_MCP_Joint": 0.0,
            "L_Ring_PIP_Joint": 0.0,
            "L_Ring_DIP_Joint": 0.0,
            "L_Little_Roll_Joint": 0.0,
            "L_Little_MCP_Joint": 0.0,
            "L_Little_PIP_Joint": 0.0,
            "L_Little_DIP_Joint": 0.0,
            # Right hand joints
            "R_Thumb_Yaw_Joint": 0.0,
            "R_Thumb_CMC_Joint": 0.872665,
            "R_Thumb_MCP_Joint": 0.349066,
            "R_Thumb_IP_Joint": 0.0,
            "R_Index_Roll_Joint": 0.0,
            "R_Index_MCP_Joint": 0.0,
            "R_Index_PIP_Joint": 0.0,
            "R_Index_DIP_Joint": 0.0,
            "R_Middle_Roll_Joint": 0.0,
            "R_Middle_MCP_Joint": 0.0,
            "R_Middle_PIP_Joint": 0.0,
            "R_Middle_DIP_Joint": 0.0,
            "R_Ring_Roll_Joint": 0.0,
            "R_Ring_MCP_Joint": 0.0,
            "R_Ring_PIP_Joint": 0.0,
            "R_Ring_DIP_Joint": 0.0,
            "R_Little_Roll_Joint": 0.0,
            "R_Little_MCP_Joint": 0.0,
            "R_Little_PIP_Joint": 0.0,
            "R_Little_DIP_Joint": 0.0,
        },
        pos=(0.0, 0.0, 1.0),
        rot=(0.0, 0.0, 0.0, 1.0),  # init pos of the articulation for teleop
    ),
    actuators={
        # Body lift and torso actuators
        "body": ImplicitActuatorCfg(
            joint_names_expr=["Waist_Yaw_Joint", "Waist_Pitch_Lower_Joint"],
            effort_limit_sim=300.0,
            velocity_limit_sim=2.61,
            stiffness=1000.0,
            damping=100.0,
        ),
        # Passive/mimic joints (12): Dummy, Upper waist; all DIP; Thumb IP. Newton/MuJoCo requires actfrcrange[0] < actfrcrange[1]
        "passive": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*Dummy.*",
                "Waist_Pitch_Upper_Joint",
                ".*_DIP_Joint",
                ".*_Thumb_IP_Joint",
            ],
            effort_limit_sim=1.0,
            velocity_limit_sim=0.0,
            stiffness=0.0,
            damping=1.0,
        ),
        # Head actuators
        "head": ImplicitActuatorCfg(
            joint_names_expr=["Neck_Pitch_Joint", "Neck_Yaw_Joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=1.0,
            stiffness=80.0,
            damping=4.0,
        ),
        # Left arm actuator
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["L_Shoulder_Pitch_Joint", "L_Shoulder_Roll_Joint", "L_Shoulder_Yaw_Joint", "L_Elbow_Joint", "L_Wrist_Yaw_Joint", "L_Wrist_Roll_Joint", "L_Wrist_Pitch_Joint"],
            effort_limit_sim={
                "L_Shoulder_Pitch_Joint": 2000.0,
                "L_Shoulder_Roll_Joint": 1000.0,
                "L_Shoulder_Yaw_Joint": 1000.0,
                "L_Elbow_Joint": 1000.0,
                "L_Wrist_Yaw_Joint": 1000.0,
                "L_Wrist_Roll_Joint": 1000.0,
                "L_Wrist_Pitch_Joint": 1000.0,
            },
            velocity_limit_sim=1.57,
            stiffness={
            "L_Shoulder_Pitch_Joint": 1000.0, 
            "L_Shoulder_Roll_Joint": 1000.0, 
            "L_Shoulder_Yaw_Joint": 1000.0, 
            "L_Elbow_Joint": 1000.0, 
            "L_Wrist_Yaw_Joint": 1000.0, 
            "L_Wrist_Roll_Joint": 1000.0, 
            "L_Wrist_Pitch_Joint": 1000.0},
            damping={
            "L_Shoulder_Pitch_Joint": 100.0, 
            "L_Shoulder_Roll_Joint": 100.0, 
            "L_Shoulder_Yaw_Joint": 100.0, 
            "L_Elbow_Joint": 100.0, 
            "L_Wrist_Yaw_Joint": 100.0, 
            "L_Wrist_Roll_Joint": 100.0, 
            "L_Wrist_Pitch_Joint": 100.0},
        ),
        # Right arm actuator
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint", "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint"],
            effort_limit_sim={
                "R_Shoulder_Pitch_Joint": 300.0,  # 가장 큰 토크 필요 (팔 전체 무게 지탱)
                "R_Shoulder_Roll_Joint": 150.0,
                "R_Shoulder_Yaw_Joint": 150.0,
                "R_Elbow_Joint": 150.0,
                "R_Wrist_Yaw_Joint": 120.0,  # Wrist는 작은 토크
                "R_Wrist_Roll_Joint": 120.0,
                "R_Wrist_Pitch_Joint": 120.0,
            },
            velocity_limit_sim=2,
            stiffness={
                "R_Shoulder_Pitch_Joint": 200.0,  # Nm/rad
                "R_Shoulder_Roll_Joint": 175.0,
                "R_Shoulder_Yaw_Joint": 175.0,
                "R_Elbow_Joint": 175.0,
                "R_Wrist_Yaw_Joint": 125.0,  # Wrist는 낮은 강성
                "R_Wrist_Roll_Joint": 125.0,
                "R_Wrist_Pitch_Joint": 125.0,
            },
            damping={
                "R_Shoulder_Pitch_Joint": 20.0,  # Nm·s/rad (stiffness의 약 10%)
                "R_Shoulder_Roll_Joint": 17.5,
                "R_Shoulder_Yaw_Joint": 17.5,
                "R_Elbow_Joint": 17.5,
                "R_Wrist_Yaw_Joint": 12.5,
                "R_Wrist_Roll_Joint": 12.5,
                "R_Wrist_Pitch_Joint": 12.5,
            },
        ),
        # Left hand actuator
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=[
            "L_Thumb_Yaw_Joint", "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint", 
            "L_Index_Roll_Joint", "L_Index_MCP_Joint", "L_Index_PIP_Joint", 
            "L_Middle_Roll_Joint", "L_Middle_MCP_Joint", "L_Middle_PIP_Joint", 
            "L_Ring_Roll_Joint", "L_Ring_MCP_Joint", "L_Ring_PIP_Joint", "L_Little_Roll_Joint", 
            "L_Little_MCP_Joint", "L_Little_PIP_Joint"],
            effort_limit_sim={
            "L_Thumb_Yaw_Joint": 100.0, "L_Thumb_CMC_Joint": 100.0, "L_Thumb_MCP_Joint": 100.0, 
            "L_Index_Roll_Joint": 100.0, "L_Index_MCP_Joint": 100.0, "L_Index_PIP_Joint": 100.0, 
            "L_Middle_Roll_Joint": 100.0, "L_Middle_MCP_Joint": 100.0, "L_Middle_PIP_Joint": 100.0, 
            "L_Ring_Roll_Joint": 100.0, "L_Ring_MCP_Joint": 100.0, "L_Ring_PIP_Joint": 100.0, 
            "L_Little_Roll_Joint": 100.0, "L_Little_MCP_Joint": 100.0, "L_Little_PIP_Joint": 100.0},
            velocity_limit_sim=10.0,
            stiffness={
            "L_Thumb_Yaw_Joint": 100.0, "L_Thumb_CMC_Joint": 100.0, "L_Thumb_MCP_Joint": 100.0, 
            "L_Index_Roll_Joint": 100.0, "L_Index_MCP_Joint": 100.0, "L_Index_PIP_Joint": 100.0, 
            "L_Middle_Roll_Joint": 100.0, "L_Middle_MCP_Joint": 100.0, "L_Middle_PIP_Joint": 100.0, 
            "L_Ring_Roll_Joint": 100.0, "L_Ring_MCP_Joint": 100.0, "L_Ring_PIP_Joint": 100.0, 
            "L_Little_Roll_Joint": 100.0, "L_Little_MCP_Joint": 100.0, "L_Little_PIP_Joint": 100.0},
            damping={
            "L_Thumb_Yaw_Joint": 10.0, "L_Thumb_CMC_Joint": 10.0, "L_Thumb_MCP_Joint": 10.0, 
            "L_Index_Roll_Joint": 10.0, "L_Index_MCP_Joint": 10.0, "L_Index_PIP_Joint": 10.0, 
            "L_Middle_Roll_Joint": 10.0, "L_Middle_MCP_Joint": 10.0, "L_Middle_PIP_Joint": 10.0, 
            "L_Ring_Roll_Joint": 10.0, "L_Ring_MCP_Joint": 10.0, "L_Ring_PIP_Joint": 10.0, 
            "L_Little_Roll_Joint": 10.0, "L_Little_MCP_Joint": 10.0, "L_Little_PIP_Joint": 10.0},
        ),
        # Right hand actuator
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=[
            "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint", 
            "R_Index_Roll_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint", 
            "R_Middle_Roll_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint", 
            "R_Ring_Roll_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint", 
            "R_Little_Roll_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint"],
            effort_limit_sim={
            "R_Thumb_Yaw_Joint": 1.5, "R_Thumb_CMC_Joint": 2.0, "R_Thumb_MCP_Joint": 1.0, 
            "R_Index_Roll_Joint": 1.5, "R_Index_MCP_Joint": 2.0, "R_Index_PIP_Joint": 1.0, 
            "R_Middle_Roll_Joint": 1.5, "R_Middle_MCP_Joint": 2.0, "R_Middle_PIP_Joint": 1.0, 
            "R_Ring_Roll_Joint": 1.5, "R_Ring_MCP_Joint": 2.0, "R_Ring_PIP_Joint": 1.0, 
            "R_Little_Roll_Joint": 1.5, "R_Little_MCP_Joint": 2.0, "R_Little_PIP_Joint": 1.0},
            velocity_limit_sim=10.0,
            stiffness={
            "R_Thumb_Yaw_Joint": 10.0, "R_Thumb_CMC_Joint": 10.0, "R_Thumb_MCP_Joint": 10.0, 
            "R_Index_Roll_Joint": 10.0, "R_Index_MCP_Joint": 10.0, "R_Index_PIP_Joint": 10.0, 
            "R_Middle_Roll_Joint": 10.0, "R_Middle_MCP_Joint": 10.0, "R_Middle_PIP_Joint": 10.0, 
            "R_Ring_Roll_Joint": 10.0, "R_Ring_MCP_Joint": 10.0, "R_Ring_PIP_Joint": 10.0, 
            "R_Little_Roll_Joint": 10.0, "R_Little_MCP_Joint": 10.0, "R_Little_PIP_Joint": 10.0},
            damping={
            "R_Thumb_Yaw_Joint": 1.0, "R_Thumb_CMC_Joint": 1.0, "R_Thumb_MCP_Joint": 1.0, 
            "R_Index_Roll_Joint": 1.0, "R_Index_MCP_Joint": 1.0, "R_Index_PIP_Joint": 1.0, 
            "R_Middle_Roll_Joint": 1.0, "R_Middle_MCP_Joint": 1.0, "R_Middle_PIP_Joint": 1.0, 
            "R_Ring_Roll_Joint": 1.0, "R_Ring_MCP_Joint": 1.0, "R_Ring_PIP_Joint": 1.0, 
            "R_Little_Roll_Joint": 1.0, "R_Little_MCP_Joint": 1.0, "R_Little_PIP_Joint": 1.0},
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# ALLEX_newton_no_left.usd: 왼팔/왼손 제거. 목 2개 Fixed → Revolute 31 DOF. 허리+오른팔+오른손.
# Active (driver) joints와 mimic joints를 분리: mimic는 env에서 poly(driver)로 target 설정 후
# 높은 stiffness/damping으로 구속 링크처럼 잘 따르게 함.
ALLEX_NO_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_ALLEX_NO_LEFT_USD_PATH.resolve()),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Waist_Yaw_Joint": 0.0,
            "Waist_Pitch_Dummy_Joint": 0.0,
            "Waist_Pitch_Lower_Joint": 0.0,
            "Waist_Pitch_Upper_Joint": 0.0,
            "R_Shoulder_Pitch_Joint": 0.0,
            "R_Shoulder_Roll_Joint": 0.0,
            "R_Shoulder_Yaw_Joint": 0.0,
            "R_Elbow_Joint": 0.0,
            "R_Wrist_Yaw_Joint": 0.0,
            "R_Wrist_Roll_Joint": 0.0,
            "R_Wrist_Pitch_Joint": 0.0,
            "R_Thumb_Yaw_Joint": 0.0,
            "R_Thumb_CMC_Joint": 0.0,
            "R_Thumb_MCP_Joint": 0.0,
            "R_Thumb_IP_Joint": 0.0,
            "R_Index_Roll_Joint": 0.0,
            "R_Index_MCP_Joint": 0.0,
            "R_Index_PIP_Joint": 0.0,
            "R_Index_DIP_Joint": 0.0,
            "R_Middle_Roll_Joint": 0.0,
            "R_Middle_MCP_Joint": 0.0,
            "R_Middle_PIP_Joint": 0.0,
            "R_Middle_DIP_Joint": 0.0,
            "R_Ring_Roll_Joint": 0.0,
            "R_Ring_MCP_Joint": 0.0,
            "R_Ring_PIP_Joint": 0.0,
            "R_Ring_DIP_Joint": 0.0,
            "R_Little_Roll_Joint": 0.0,
            "R_Little_MCP_Joint": 0.0,
            "R_Little_PIP_Joint": 0.0,
            "R_Little_DIP_Joint": 0.0,
        },
        pos=(0.0, 0.0, 1.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        # Active (driver) joints: 사용자/정책이 제어. MJCF equality에서 joint1으로 등장하지 않는 관절.
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "Waist_Yaw_Joint",
                "Waist_Pitch_Lower_Joint",
                "R_Shoulder_Pitch_Joint",
                "R_Shoulder_Roll_Joint",
                "R_Shoulder_Yaw_Joint",
                "R_Elbow_Joint",
                "R_Wrist_Yaw_Joint",
                "R_Wrist_Roll_Joint",
                "R_Wrist_Pitch_Joint",
                "R_Thumb_Yaw_Joint",
                "R_Thumb_CMC_Joint",
                "R_Thumb_MCP_Joint",
                "R_Index_Roll_Joint",
                "R_Index_MCP_Joint",
                "R_Index_PIP_Joint",
                "R_Middle_Roll_Joint",
                "R_Middle_MCP_Joint",
                "R_Middle_PIP_Joint",
                "R_Ring_Roll_Joint",
                "R_Ring_MCP_Joint",
                "R_Ring_PIP_Joint",
                "R_Little_Roll_Joint",
                "R_Little_MCP_Joint",
                "R_Little_PIP_Joint",
            ],
            effort_limit_sim=300.0,
            velocity_limit_sim=2.0,
            stiffness=100.0,
            damping=10.0,
        ),
        # Mimic joints: MuJoCo equality가 자세 강제. 액추에이터는 damping만 두어 constraint와 견제하지 않음.
        "mimic": ImplicitActuatorCfg(
            joint_names_expr=[
                "Waist_Pitch_Dummy_Joint",
                "Waist_Pitch_Upper_Joint",
                "R_Thumb_IP_Joint",
                "R_Index_DIP_Joint",
                "R_Middle_DIP_Joint",
                "R_Ring_DIP_Joint",
                "R_Little_DIP_Joint",
            ],
            effort_limit_sim=1.0,
            velocity_limit_sim=0.0,
            stiffness=0.0,  # position target 없음; equality가 강제. damping만 사용.
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
