# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Trajectory Index Selection 환경 설정.

PPO가 trajectory 인덱스를 선택하고 residual을 더해 망치를 잡는 동작 학습.
Action: [index_scalar(1), residuals(18)] = 19차원

index_scalar: -1~1 범위의 연속값 → [0, num_frames-1] 정수로 변환
"""

from isaaclab_assets.robots import ALLEX_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import dexblind_env_cfg as dexblind
from ... import mdp


# Action용 관절 (18개): 오른팔 7 + 오른손 11 (Roll 관절 제외)
ALLEX_RIGHT_ARM_HAND_JOINTS = [
    # 오른팔 (7)
    "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
    "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
    # 오른손 (11)
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
    "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_MCP_Joint", "R_Little_PIP_Joint",
]

# 토크 관측용 관절 (19개): 어깨 3 + 팔꿈치 1 + 손가락 15 (Roll 포함)
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
class AllexChunkedActionCfg:
    """Trajectory Index Selection Action 설정."""
    action = mdp.ChunkedTrajectoryActionCfg(
        asset_name="robot",
        joint_names=ALLEX_RIGHT_ARM_HAND_JOINTS,
        preserve_order=True,
        trajectory_file="20state_data.npz",
        use_smoothing=False,  # offset 제한 + residual로 충분, smoothing 불필요
        residual_scale=0.1,
        scale=1.0,
        random_start_prob=1.0,
        random_start_range=(0.0, 0.7),
        initial_sequential_steps=0,
        max_index_offset=5,  # 현재 인덱스 기준 ±5 범위 내에서만 선택 가능
    )


@configclass
class AllexChunkedRewardsCfg(dexblind.RewardsCfg):
    """Trajectory Index Selection용 보상 설정.
    
    RewardsCfg를 상속받아 공통 보상함수 재사용:
    - hand_joint_torque_penalty, hand_hammer_distance, hammer_goal_pos, hammer_goal_quat,
      hammer_contact, table_contact_penalty는 부모 클래스에서 상속
    
    Chunked 환경 전용 변경사항:
    - reference_tracking, hammer_height_shaping 비활성화 (시간 기반 reference_trajectory와 충돌 방지)
    - action_rate_l2 weight 조정
    - hammer_lift_success threshold 조정
    - trajectory_tracking 추가 (Chunked 환경 전용)
    """
    
    # 부모 클래스 보상 비활성화 (시간 기반 reference_trajectory와 충돌 방지)
    reference_tracking = None
    hammer_height_shaping = None

    # action_rate_l2 weight 조정 (부모 클래스 -0.01 → -0.005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)
    
    # 망치 들기 성공 보상 (Binary) - threshold 조정 (부모 클래스 0.6 → 0.53)
    hammer_lift_success = RewTerm(
        func=mdp.hammer_lift_reward,
        weight=2.5,
        params={
            "asset_cfg": SceneEntityCfg("hammer"),
            "threshold": 0.53,
        },
    )
    hammer_low_height_penalty = RewTerm(
        func=mdp.hammer_low_height_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("hammer"), "z_threshold": 0.4},
    )

    # Chunked 환경용 궤적 추종 보상 (선택한 index의 target 추종)
    trajectory_tracking = RewTerm(
        func=mdp.trajectory_target_tracking_reward,
        weight=1.0,
        params={
            "action_name": "action",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.1,
        },
    )


@configclass
class AllexChunkedMixinCfg:
    """Blind Grasping ALLEX Mixin.
    
    Observation (Proprioceptive only):
    - joint_pos(18) + right_hand_joint_torque(19)
    - right_hand_base_pos_b(7) + current_progress(1) + grasp_flag(1)
    """
    rewards: AllexChunkedRewardsCfg = AllexChunkedRewardsCfg()
    actions: AllexChunkedActionCfg = AllexChunkedActionCfg()

    def __post_init__(self: dexblind.DexblindLiftEnvCfg):
        super().__post_init__()
        
        # Robot 설정
        self.scene.robot = ALLEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True
        self.episode_length_s = 5.0
        
        # 불필요한 Commands 비활성화
        self.commands.reference_trajectory = None
          
        
        # Observation 설정
        joint_cfg = SceneEntityCfg("robot", joint_names=self.actions.action.joint_names)
        
        if hasattr(self.observations.proprio, "joint_pos"):
            self.observations.proprio.joint_pos.params["asset_cfg"] = joint_cfg

        # 불필요한 Observation 비활성화
        for attr in ["reference_joint_pos", "playback_speed", "grasp_flag", 
                     "hammer_pos", "target_right_hand_pose"]:
            if hasattr(self.observations.proprio, attr):
                setattr(self.observations.proprio, attr, None)
        
        # 현재 trajectory index (0~5)
        self.observations.proprio.current_progress = ObsTerm(
            func=mdp.grasp_progress_chunked,
            params={"action_name": "action"},
        )
        
        # 토크 관측 설정
        if hasattr(self.observations.proprio, "right_hand_joint_torque"):
            self.observations.proprio.right_hand_joint_torque.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=RIGHT_ARM_HAND_TORQUE_JOINTS
            )
        
        # Observation History (temporal context)
        self.observations.proprio.history_length = 5


@configclass
class DexblindAllexChunkedLiftEnvCfg(AllexChunkedMixinCfg, dexblind.DexblindLiftEnvCfg):
    pass


@configclass
class DexblindAllexChunkedLiftEnvCfg_PLAY(AllexChunkedMixinCfg, dexblind.DexblindLiftEnvCfg_PLAY):
    pass
