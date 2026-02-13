# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .adr_curriculum import CurriculumCfg

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

_HAMMER_USD_PATH = Path(__file__).resolve().parent / "data" / "Hammer.usd"

# Contact filter: index 0 = hammer, 1 = table (force_matrix_w[:, :, idx, :])
_CONTACT_FILTER_EXPR = ["{ENV_REGEX_NS}/hammer", "{ENV_REGEX_NS}/table"]

# Hand joint names (15): thumb 3 + index/middle/ring/little each 3
_HAND_JOINT_NAMES = [
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
    "R_Index_Roll_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_Roll_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_Roll_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_Roll_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint",
]

# Arm-only joints (7) for reset randomization
_ARM_JOINT_NAMES = [
    "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
    "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
]

# Right arm + right hand (18): arm 7 + hand 11 (Roll 4개 제외), Allex 행동 궤적 순서와 동일
_RIGHT_ARM_HAND_JOINT_NAMES = [
    "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
    "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
    "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_MCP_Joint", "R_Little_PIP_Joint",
]

# 오른팔(4) + 오른손(15, Roll 포함) 19개 — joint_applied_torque(토크) 관측용. preserve_order=True로 사용.
_RIGHT_ARM_HAND_TORQUE_JOINT_NAMES = [
    "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
    "R_Elbow_Joint",
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
    "R_Index_Roll_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_Roll_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_Roll_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_Roll_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint",
]

# Finger-only joints (Roll 제외, 궤적에 포함되는 11개: thumb 3 + index/middle/ring/little each 2)
_FINGER_JOINT_NAMES = [
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
    "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_MCP_Joint", "R_Little_PIP_Joint",
]

# SceneEntityCfg list for reward/penalty (finger + palm contact sensors)
FINGER_HAMMER_CONTACT_SENSORS = [
    SceneEntityCfg("finger_contact_sensor"),
    SceneEntityCfg("finger_contact_sensor_2"),
    SceneEntityCfg("finger_contact_sensor_3"),
    SceneEntityCfg("finger_contact_sensor_4"),
    SceneEntityCfg("finger_contact_sensor_5"),
    SceneEntityCfg("finger_contact_sensor_6"),
    SceneEntityCfg("finger_contact_sensor_7"),
    SceneEntityCfg("finger_contact_sensor_8"),
    SceneEntityCfg("finger_contact_sensor_9"),
    SceneEntityCfg("finger_contact_sensor_10"),
    SceneEntityCfg("palm_contact_sensor"),
]


def _finger_contact_sensor(prim_path: str) -> ContactSensorCfg:
    """Finger/palm contact sensor with hammer+table filter."""
    return ContactSensorCfg(
        prim_path=prim_path,
        update_period=0.0,
        debug_vis=False,
        filter_prim_paths_expr=_CONTACT_FILTER_EXPR,
    )


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Dexblind scene: robot, hammer, table, contact sensors (hammer/table filter)."""

    robot: ArticulationCfg = MISSING

    hammer: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/hammer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_HAMMER_USD_PATH),
            scale=(1.1, 1.1, 1.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.25, 0.0)),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.46, 0.1, 0.46), rot=(0.0, 1.0, 0.0, 0.0)),
    )

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.6, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.45, 0.1, 0.42434), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # Finger/palm contact sensors (hammer + table); force_matrix_w[:,:,0]=hammer, [:,:,1]=table
    finger_contact_sensor = _finger_contact_sensor("{ENV_REGEX_NS}/Robot/R_Hand_Little_Distal")
    finger_contact_sensor_2 = _finger_contact_sensor("{ENV_REGEX_NS}/Robot/R_Hand_Little_Proximal")
    finger_contact_sensor_3 = _finger_contact_sensor("{ENV_REGEX_NS}/Robot/R_Hand_Ring_Distal")
    finger_contact_sensor_4 = _finger_contact_sensor("{ENV_REGEX_NS}/Robot/R_Hand_Ring_Proximal")
    finger_contact_sensor_5 = _finger_contact_sensor("{ENV_REGEX_NS}/Robot/R_Hand_Middle_Distal")
    finger_contact_sensor_6 = _finger_contact_sensor("{ENV_REGEX_NS}/Robot/R_Hand_Middle_Proximal")
    finger_contact_sensor_7 = _finger_contact_sensor("{ENV_REGEX_NS}/Robot/R_Hand_Index_Distal")
    finger_contact_sensor_8 = _finger_contact_sensor("{ENV_REGEX_NS}/Robot/R_Hand_Index_Proximal")
    finger_contact_sensor_9 = _finger_contact_sensor("{ENV_REGEX_NS}/Robot/R_Hand_Thumb_Distal")
    finger_contact_sensor_10 = _finger_contact_sensor("{ENV_REGEX_NS}/Robot/R_Hand_Thumb_Proximal")
    palm_contact_sensor = _finger_contact_sensor("{ENV_REGEX_NS}/Robot/Right_Hand_Palm")

    # Required for GPU contact filtering (PhysxContactReportAPI)
    hammer_contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/hammer", update_period=0.0, debug_vis=False)
    table_contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/table", update_period=0.0, debug_vis=False)


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    reference_trajectory = mdp.ReferenceTrajectoryCommandCfg(
        trajectory_file="Hammer_right_arm_hand_trajectory.npz",
        loop=False,
        loop_start_time=None,
        space="joint",
        resampling_time_range=(60.0, 60.0),
        playback_speed=1.0,
    )


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy group: last action, history=3."""

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Proprio: joint_pos, reference_joint_pos(참조 궤적), right_hand_joint_torque, right_hand_base_pos.
        AllexMixinCfg sets joint_names for torque/ref; chunked env adds current_progress and disables ref terms.
        """

        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.05, n_max=0.05))

        reference_joint_pos = ObsTerm(
            func=mdp.reference_joint_pos,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={
                "command_name": "reference_trajectory",
                "asset_cfg": SceneEntityCfg("robot", joint_names=_RIGHT_ARM_HAND_JOINT_NAMES),
                "joint_names": _RIGHT_ARM_HAND_JOINT_NAMES,
            },
        )

        right_hand_joint_torque = ObsTerm(
            func=mdp.joint_applied_torque,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=_RIGHT_ARM_HAND_TORQUE_JOINT_NAMES,
                    preserve_order=True,
                ),
            },
        )

        right_hand_base_pos = ObsTerm(
            func=mdp.right_hand_base_pos_b,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"body_name": "R_Hand_Pose", "asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()


# -----------------------------------------------------------------------------
# Events (randomization & interval)
# -----------------------------------------------------------------------------


@configclass
class EventCfg:
    """Domain randomization and interval-based events (e.g. hammer force when lifted)."""

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.9, 1.1],
            "damping_distribution_params": [0.9, 1.1],
            "operation": "scale",
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [0.8, 1.3],
            "operation": "scale",
        },
    )
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0]},
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_robot_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_RIGHT_ARM_HAND_JOINT_NAMES),
            "position_range": [-0.05, 0.05],
            "velocity_range": [0.0, 0.0],
        },
    )
    hammer_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("hammer"),
            "mass_distribution_params": [0.8, 1.6],
            "operation": "scale",
        },
    )
    reset_hammer = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.05, 0.05],
                "y": [-0.05, 0.05],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-0.4, 0.4],
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("hammer"),
        },
    )
    hammer_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("hammer", body_names=".*"),
            "static_friction_range": [0.4, 0.6],
            "dynamic_friction_range": [0.3, 0.5],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )
    hammer_force_when_lifted = EventTerm(
        func=mdp.apply_hammer_force_when_lifted,
        mode="interval",
        interval_range_s=(0.1, 0.2),
        params={
            "asset_cfg": SceneEntityCfg("hammer"),
            "force_range": [-5.0, 5.0],
            "torque_range": [-0.5, 0.5],
            "height_threshold": 0.6,
        },
    )
    reset_table_hammer_height = EventTerm(
        func=mdp.reset_table_and_hammer_height_linked,
        mode="reset",
        params={
            "z_offset_range": (-0.03, 0.03),
            "table_cfg": SceneEntityCfg("table"),
            "hammer_cfg": SceneEntityCfg("hammer"),
        },
    )


# -----------------------------------------------------------------------------
# Actions (base: empty; overridden in allex / chunked configs)
# -----------------------------------------------------------------------------


@configclass
class ActionsCfg:
    pass


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    # applied torque L2 패널티 (Isaac Lab 기본 joint_torques_l2 = asset.data.applied_torque)
    arm_joint_torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINT_NAMES)},
    )
    hand_joint_torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_HAND_JOINT_NAMES)},
    )

    table_contact_penalty = RewTerm(
        func=mdp.finger_table_contact_penalty,
        weight=-1.0,
        params={
            "sensor_cfgs": FINGER_HAMMER_CONTACT_SENSORS,
            "min_contact_force": 10.0,
            "table_filter_idx": 1,
        },
    )

    hammer_lift_success = RewTerm(
        func=mdp.hammer_lift_reward,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("hammer"), "threshold": 0.5},
    )

    hammer_contact = RewTerm(
        func=mdp.hammer_contact_reward,
        weight=0.5,
        params={
            "sensor_cfgs": FINGER_HAMMER_CONTACT_SENSORS,
            "min_contact_force": 0.1,
            "min_contact_bodies": 1,
            "hammer_filter_idx": 0,
        },
    )

    # 목표 위치/자세 보상 분리 (로깅·가독성). 기존과 동일 효과: pos_weight=0, quat_weight=1
    hammer_goal_pos = RewTerm(
        func=mdp.hammer_goal_pos_reward,
        weight=1.0,
        params={
            "hammer_cfg": SceneEntityCfg("hammer"),
            "robot_cfg": SceneEntityCfg("robot"),
            "target_pos": (0.55, -0.2, 0.6),
            "pos_std": 0.08,
        },
    )
    hammer_goal_quat = RewTerm(
        func=mdp.hammer_goal_quat_reward,
        weight=3.0,
        params={
            "hammer_cfg": SceneEntityCfg("hammer"),
            "robot_cfg": SceneEntityCfg("robot"),
            "target_quat": (0.0, 0.0, -0.70711, -0.70711),
            "quat_std": 0.3,
        },
    )


    grasp_final_pose_tracking = RewTerm(
        func=mdp.grasp_final_pose_tracking_reward,
        weight=20.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_RIGHT_ARM_HAND_JOINT_NAMES),
            "hammer_cfg": SceneEntityCfg("hammer"),
            "robot_cfg": SceneEntityCfg("robot"),
            "hand_body_name": "R_Hand_Pose",
            "handle_offset": (0.0, -0.13, 0.0),
            "hand_handle_max": 0.05,
            "command_name": "reference_trajectory",
            "joint_names": _RIGHT_ARM_HAND_JOINT_NAMES,
            "finger_joint_names": _FINGER_JOINT_NAMES,
            "progress_thresh": 0.99,
            "height_thresh": 0.6,
            "sensor_cfgs": FINGER_HAMMER_CONTACT_SENSORS,
            "min_contact_force": 0.1,
            "hammer_filter_idx": 0,
            "arm_std": 0.3,
            "finger_std": 0.15,
            "finger_weight": 1.0,
            "arm_weight": 1.0,
        },
    )


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    hammer_fallen = DoneTerm(
        func=mdp.hammer_fallen,
        params={"asset_cfg": SceneEntityCfg("hammer"), "z_threshold": 0.38},
    )


# -----------------------------------------------------------------------------
# Env config
# -----------------------------------------------------------------------------


@configclass
class DexblindLiftEnvCfg(ManagerBasedEnvCfg):
    """Dexblind lift task: blind grasping with reference trajectory + rewards."""

    viewer: ViewerCfg = ViewerCfg(eye=(-2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env")
    scene: SceneCfg = SceneCfg(num_envs=8192, env_spacing=3, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 4.0
        self.is_finite_horizon = True
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation


@configclass
class DexblindLiftEnvCfg_PLAY(DexblindLiftEnvCfg):
    """Dexblind lift task for evaluation (no hammer force disturbance)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.hammer_force_when_lifted = None
