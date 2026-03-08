# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal dexblind_newton: ALLEX only (hammer/table are in robot USD)."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.visualizers import NewtonVisualizerCfg, RerunVisualizerCfg
from isaaclab.visualizers.newton_visualizer_cfg import GoalMarkerCfg

from isaaclab_tasks.direct.allex.allex_env_cfg import ALLEX_MIMIC_SPEC

from . import mdp
from .utils import randomize_object_pose_xy_yaw, set_shape_friction




NUM_ENVS = 4096

# Arm-only joints (7)
_ARM_JOINT_NAMES = [
    "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
    "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
]

# Hand joints (11, Roll excluded)
_HAND_JOINT_NAMES = [
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
    "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_MCP_Joint", "R_Little_PIP_Joint",
]

# Right arm + hand (18 joints): arm 7 + hand 11, Roll excluded. Single source for obs/action.
RIGHT_ARM_HAND_JOINT_NAMES = _ARM_JOINT_NAMES + _HAND_JOINT_NAMES

_ARM_TORQUE_JOINT_NAMES = [
    "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint","R_Elbow_Joint",
]
# Hand joints with Roll (15): Roll joints included for torque sensing.
_HAND_JOINT_NAMES_WITH_ROLL = [
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
    "R_Index_Roll_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_Roll_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_Roll_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_Roll_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint",
]

# Arm + hand with Roll (22 joints): for torque observation.
RIGHT_ARM_HAND_JOINT_NAMES_TORQUE = _ARM_TORQUE_JOINT_NAMES + _HAND_JOINT_NAMES_WITH_ROLL


DEXBLIND_NEWTON_SOLVER_CFG = MJWarpSolverCfg(
    solver="newton",
    integrator="implicit",
    nconmax_margin=100,
    njmax_multiply=2.0,
    impratio=10.0,
    cone="elliptic",
    update_data_interval=2,
    iterations=100,
    ls_iterations=15,
    ls_parallel=True,
    mj_data_memory=64 * 1024 * 1024,
)

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Minimal scene: table + robot (hammer in robot USD)."""

    hammer: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/hammer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).resolve().parents[5] / "isaaclab_assets" / "object" / "Hammer.usd"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.55, -0.1, 0.9), rot=(0.0, 1.0, 0.0, 0.0)),
        actuators={},
        articulation_root_prim_path="",
    )

    table: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.4, 0.6, 0.885),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visible=True
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.65, -0.1, 0.4425), rot=(1.0, 0.0, 0.0, 0.0)),
        actuators={},
        articulation_root_prim_path="",
    )

    robot: ArticulationCfg = MISSING  # set in config/allex

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )


@configclass
class SceneCfgNewton(SceneCfg):
    replicate_physics: bool = True
    clone_in_fabric: bool = True
    newton_replicate_kwargs: dict | None = None


@configclass
class CommandsCfg:
    """Reference trajectory for tracking reward (action is still RelativeJointPosition)."""

    reference_trajectory = mdp.ReferenceTrajectoryCommandCfg(
        trajectory_file="Hammer_Hammer_RL_Trajectory.npz",
        loop=False,
        loop_start_time=None,
        space="joint",
        resampling_time_range=(60.0, 60.0),
        playback_speed=1.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class ProprioObsCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_HAND_JOINT_NAMES)},
            noise=GaussianNoiseCfg(
                mean=0.0, 
                std=0.01,
            ),
        )

        reference_joint_pos = ObsTerm(
            func=mdp.reference_joint_pos,
            params={
                "command_name": "reference_trajectory",
                "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_HAND_JOINT_NAMES),
                "joint_names": RIGHT_ARM_HAND_JOINT_NAMES,
            },
        )

        right_hand_joint_torque = ObsTerm(
            func=mdp.joint_applied_torque,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=RIGHT_ARM_HAND_JOINT_NAMES_TORQUE,
                    preserve_order=True,
                ),
            },
            noise=GaussianNoiseCfg(
                mean=0.0, 
                std=0.1,
            ),
        )

        right_hand_relative_pose = ObsTerm(
            func=mdp.right_hand_relative_pose,
            params={"robot_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoiseCfg(
                mean=0.0,
                std=0.01,
            ),
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    @configclass
    class PrivilegedObsCfg(ObsGroup):
        """Privileged state for Asymmetric Critic (SimToolReal Appendix D.4).

        Contains all proprio terms (noise-free) plus additional privileged
        signals: ground-truth velocities, progress features, reward signals.
        """

        # --- proprio terms (clean, no noise) ---
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_HAND_JOINT_NAMES)},
        )
        reference_joint_pos = ObsTerm(
            func=mdp.reference_joint_pos,
            params={
                "command_name": "reference_trajectory",
                "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_HAND_JOINT_NAMES),
                "joint_names": RIGHT_ARM_HAND_JOINT_NAMES,
            },
        )
        right_hand_joint_torque = ObsTerm(
            func=mdp.joint_applied_torque,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=RIGHT_ARM_HAND_JOINT_NAMES_TORQUE,
                    preserve_order=True,
                ),
            },
        )
        right_hand_relative_pose = ObsTerm(
            func=mdp.right_hand_relative_pose,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        hammer_relative_pose = ObsTerm(
            func=mdp.hammer_relative_pose,
            params={"hammer_cfg": SceneEntityCfg("hammer")},
        )

        # --- privileged-only signals ---
        object_lin_vel = ObsTerm(
            func=mdp.object_lin_vel,
            params={"asset_cfg": SceneEntityCfg("hammer")},
        )
        object_ang_vel = ObsTerm(
            func=mdp.object_ang_vel,
            params={"asset_cfg": SceneEntityCfg("hammer")},
        )
        palm_lin_vel = ObsTerm(
            func=mdp.palm_lin_vel,
            params={"body_name": "Right_Hand_base"},
        )
        palm_ang_vel = ObsTerm(
            func=mdp.palm_ang_vel,
            params={"body_name": "Right_Hand_base"},
        )
        min_fingertip_object_distance = ObsTerm(
            func=mdp.min_fingertip_object_distance,
            params={"asset_cfg": SceneEntityCfg("hammer")},
        )
        episode_step_count = ObsTerm(func=mdp.episode_step_count)
        object_is_grasped = ObsTerm(
            func=mdp.object_is_grasped,
            params={"asset_cfg": SceneEntityCfg("hammer"), "lift_threshold": 0.91},
        )
        instantaneous_reward = ObsTerm(func=mdp.instantaneous_reward)
        cumulative_successes = ObsTerm(
            func=mdp.cumulative_successes,
            params={"asset_cfg": SceneEntityCfg("hammer"), "lift_threshold": 0.91},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


@configclass
class ActionsCfg:
    """Relative joint position: target = q_current + action * scale."""

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_HAND_JOINT_NAMES,
        preserve_order=True,
        scale=0.2,
        use_zero_offset=True,
    )


@configclass
class EventCfg:
    hammer_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass_newton,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("hammer"),
            "mass_distribution_params": (0.8, 1.6),
            "operation": "scale",
        },
    )
    hammer_friction = EventTerm(
        func=set_shape_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("hammer"),
            "mu_range": (0.3, 0.5),
        },
    )
    reset_hammer = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("hammer"),
            "pose_range": {"x": [-0.03, 0.03], "y": [-0.03, 0.03], "yaw": [-0.3, 0.3]},
            "velocity_range": {},
        },
    )
    hammer_force_when_lifted = EventTerm(
        func=mdp.apply_hammer_force_when_lifted,
        mode="interval",
        interval_range_s=(0.4, 0.5),
        params={
            "asset_cfg": SceneEntityCfg("hammer"),
            "force_range": (-3.0, 3.0),   # N, ~±7 m/s² for 0.55 kg
            "torque_range": (-0.5, 0.5),  # Nm, moderate rotation
            "height_threshold": 1.1,
        },
    )
    reset_robot_height = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"z": [-0.03, 0.03]},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    arm_joint_torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINT_NAMES)},
    )
    hand_joint_torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_HAND_JOINT_NAMES)},
    )
    arm_joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINT_NAMES)},
    )
    hand_joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_HAND_JOINT_NAMES)},
    )

    hammer_lift = RewTerm(
        func=mdp.hammer_lift_reward,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("hammer"), "threshold": 0.91},
    )

    hammer_goal_proximity = RewTerm(
        func=mdp.hammer_goal_proximity_reward,
        weight=2.0,
        params={
            "hammer_cfg": SceneEntityCfg("hammer"),
            "goal_pos": (0.65, -0.2, 1.2),
            "goal_rot": (0.0, -0.70711, -0.70711, 0.0),
            "pos_std": 0.05,
            "rot_std": 0.2,
            "pos_weight": 0.6,
            "rot_weight": 0.4,
            "lift_threshold": 0.91,
        },
    )

    reference_trajectory_hand_tracking = RewTerm(
        func=mdp.reference_trajectory_tracking_reward,
        weight=0.5,
        params={
            "command_name": "reference_trajectory",
            "asset_cfg": SceneEntityCfg("robot", joint_names=_HAND_JOINT_NAMES),
            "joint_names": _HAND_JOINT_NAMES,
            "joint_std": 0.2,
        },
    )

    reference_trajectory_arm_tracking = RewTerm(
        func=mdp.reference_trajectory_tracking_reward,
        weight=0.5,
        params={
            "command_name": "reference_trajectory",
            "asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINT_NAMES),
            "joint_names": _ARM_JOINT_NAMES,
            "joint_std": 0.1,
        },
    )

    grasp_point_proximity = RewTerm(
        func=mdp.grasp_point_proximity_reward,
        weight=0.1,
        params={"hammer_cfg": SceneEntityCfg("hammer"), "pos_std": 0.10, "lift_threshold": 0.85},
    )


    hand_final_pose = RewTerm(
        func=mdp.hand_final_pose_reward,
        weight=1.0,
        params={
            "command_name": "reference_trajectory",
            "asset_cfg": SceneEntityCfg("robot", joint_names=_HAND_JOINT_NAMES),
            "hammer_cfg": SceneEntityCfg("hammer"),
            "hand_joint_names": _HAND_JOINT_NAMES,
            "goal_pos": (0.65, -0.2, 1.2),
            "goal_rot": (0.0, -0.70711, -0.70711, 0.0),
            "proximity_threshold": 0.7,
            "joint_std": 0.1,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    hammer_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.8, "asset_cfg": SceneEntityCfg("hammer")},
    )
    hammer_velocity_exceeded = DoneTerm(
        func=mdp.hammer_velocity_exceeded,
        params={
            "asset_cfg": SceneEntityCfg("hammer"),
            "max_lin_vel": 30.0,
            "max_ang_vel": 50.0,
        },
    )
    hammer_too_far_from_table = DoneTerm(
        func=mdp.hammer_too_far_from_table,
        params={
            "hammer_cfg": SceneEntityCfg("hammer"),
            "table_cfg": SceneEntityCfg("table"),
            "max_distance": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum schedules:

    1. trajectory_tracking weight: 10 -> 1 over 0-5000 iterations
    2. hammer randomize_scale: 0 -> 1 over 500-6000 iterations (commented out)
    """


@configclass
class DexblindNewtonLiftEnvCfg(ManagerBasedRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg(eye=(0.0, 0.0, 0.0), lookat=(0.0, 0.0, 0.45), origin_type="env")

    scene: SceneCfgNewton = SceneCfgNewton(
        num_envs=NUM_ENVS,
        env_spacing=2.0,
        newton_replicate_kwargs={
            "equality_constraints": list(ALLEX_MIMIC_SPEC),
            "simplify_meshes": {
                "hammer": ("coacd", {"threshold": 0.1}),
                "*": "convex_hull",
            },
            "load_visual_shapes": False,
            "disable_collision_bodies": [
                "Waist_Base", "Waist_Yaw", "Waist_Pitch_Back",
                "Waist_Pitch_Lower", "Waist_Pitch_Upper",
                "Neck_Pitch", "Neck_Yaw", "Camera_Body",
            ],
            "disable_collision_shapes": [
                "ALLEX_Right_Shoulder_Yaw_Frame_Collision_1",
                "ALLEX_Right_Shoulder_Yaw_Frame_Collision_2",
                "ALLEX_Right_Upperarm_Cover_Collision1",
                "ALLEX_Right_Upperarm_Cover_Collision2",
                "ALLEX_Right_Upperarm_Cover_Collision3",
                "ALLEX_Right_Elbow_Frame",
                "ALLEX_Right_Forearm_Base_Cover",
                "ALLEX_Right_Forearm_Lower_Cover",
                "ALLEX_Right_Forearm_Middle_Frame",
                "ALLEX_Right_Forearm_Cover",
                
                "ALLEX_Hand_Proximal_Frame2",
                "ALLEX_Hand_Proximal_Cover",
                "ALLEX_Hand_Proximal_Pad",
                "ALLEX_Hand_Middle_Frame2",
                "ALLEX_Hand_Middle_Pad",
                "ALLEX_Hand_Distal_Frame1",
                "ALLEX_Right_Hand_Thumb_Proximal_Cover",
                "ALLEX_Right_Hand_Thumb_Proximal_Pad",
                "ALLEX_Right_Hand_Thumb_Proximal_Link1",
                "ALLEX_Right_Hand_Thumb_Proximal_Link2",
                "ALLEX_Right_Hand_Thumb_Middle_Pad",
                "ALLEX_Right_Hand_Thumb_Middle_Link",
                "ALLEX_Right_Hand_Thumb_Distal_Frame",
            ],
            "lock_joints": [
                "Waist_Yaw_Joint",
                "Waist_Pitch_Lower_Joint",
            ],
        },
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        newton_cfg=NewtonCfg(
            solver_cfg=DEXBLIND_NEWTON_SOLVER_CFG,
            num_substeps=2,
            debug_mode=False,
            use_cuda_graph=True,
        ),
        visualizer_cfgs=[
            NewtonVisualizerCfg(
                goal_markers=[
                    GoalMarkerCfg(
                        pos=(0.65, -0.2, 1.2),
                        rot=(0.0, -0.70711, -0.70711, 0.0),
                        scale=(1.0, 1.0, 1.0),
                        color=(0.2, 0.85, 0.3),
                        usd_path=str(Path(__file__).resolve().parents[5] / "isaaclab_assets" / "object" / "Hammer_goal_pose.usd"),
                    ),
                ],
            ),
            RerunVisualizerCfg(
                record_to_rrd="logs/rsl_rl/record/training_record.rrd",
                keep_historical_data=True,
            ),
        ],
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2      
        self.sim.render_interval = self.decimation
        self.episode_length_s = 6.0
        self.is_finite_horizon = True


@configclass
class DexblindNewtonLiftEnvCfg_PLAY(DexblindNewtonLiftEnvCfg):
    pass