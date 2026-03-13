# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based env config for dexblind_newton hammer-lift task."""

from __future__ import annotations

import os
import sys
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.visualizers import RerunVisualizerCfg
from allex_rl_dexblind.tasks.manager_based.dexblind_newton import mdp
from allex_rl_dexblind.tasks.manager_based.dexblind_newton.utils.dexblind_visualizer_cfg import (
    DexblindNewtonVisualizerCfg,
    GoalMarkerCfg,
)
from allex_rl_dexblind.tasks.manager_based.dexblind_newton.config.constants import (
    ALLEX_JOINT_EQUALITY_CONSTRAINTS,
    ARM_HAND_JOINT_NAMES,
    ARM_HAND_TORQUE_JOINT_NAMES,
    ARM_JOINT_NAMES,
    FINGERTIP_CONTACT_PRIM_PATH_PLACEHOLDER,
    FINGERTIP_CONTACT_SHAPE_PAD_ONLY,
    GOAL_POS,
    GOAL_ROT,
    HAMMER_CONTACT_FILTER_EXPR,
    HAND_JOINT_NAMES,
    LOG_SOLVER_CONVERGENCE_INTERVAL,
    LOCK_JOINTS,
    NUM_ENVS,
)
from allex_rl_dexblind.tasks.manager_based.dexblind_newton.utils import (
    randomize_object_pose_xy_yaw,
    set_shape_friction,
)

# Re-export for backward compatibility (allex_env_cfg imports these).
RIGHT_ARM_HAND_JOINT_NAMES = ARM_HAND_JOINT_NAMES
RIGHT_ARM_HAND_JOINT_NAMES_TORQUE = ARM_HAND_TORQUE_JOINT_NAMES

# Phase 6.1: assets path — project assets first, then env, then repo isaaclab_assets
def _resolve_assets_dir() -> Path:
    """Resolve path to assets (object/Hammer.usd, allex_usd/ etc.). Prefer project assets, then env, then in-repo."""
    # 1) Project assets: .../dexblind_newton_env_cfg.py -> .../projects/irim_newton
    project_root = Path(__file__).resolve().parents[7]
    project_assets = project_root / "assets"
    if project_assets.is_dir() and (project_assets / "object").is_dir():
        return project_assets
    # 2) Env var
    env_assets = os.environ.get("ISAACLAB_ASSETS")
    if env_assets and Path(env_assets).is_dir():
        return Path(env_assets)
    env_source = os.environ.get("ISAACLAB_SOURCE")
    if env_source:
        candidate = Path(env_source) / "isaaclab_assets"
        if candidate.is_dir():
            return candidate
    # 3) In-repo isaaclab_assets (IsaacLab root: .../projects/irim_newton -> .../IsaacLab)
    isaaclab_root = project_root.parent
    return isaaclab_root / "source" / "isaaclab_assets"


_ASSETS_DIR = _resolve_assets_dir()


# =============================================================================
# Solver (Phase 4.1: DexblindNewtonSolverCfg, DEXBLIND_NEWTON_SOLVER_CFG)
# =============================================================================
@configclass
class DexblindNewtonSolverCfg(MJWarpSolverCfg):
    """Solver config with ccd_iterations and ccd_tolerance for MuJoCo CCD."""
    ccd_iterations: int | None = 50

DEXBLIND_NEWTON_SOLVER_CFG = DexblindNewtonSolverCfg(
    solver="newton",
    integrator="implicitfast",
    impratio=5.0,
    cone="elliptic",
    iterations=100,
    ls_iterations=15,
    ls_parallel=True,
    use_mujoco_contacts=True,
    ccd_iterations=10, 
)

# =============================================================================
# Scene
# =============================================================================
@configclass
class SceneCfg(InteractiveSceneCfg):
    """Scene: robot + hammer + table + hand contact sensor."""

    hammer: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/hammer",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_ASSETS_DIR / "object" / "Hammer.usd")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.1, 0.91), rot=(0.0, 1.0, 0.0, 0.0)),
    )

    table: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.6, 0.2),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                fix_root_link=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        actuators={},
        articulation_root_prim_path="/FixedJoint",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.65, -0.1, 0.8), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = MISSING

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    hand_contact_sensor = ContactSensorCfg(
        prim_path=FINGERTIP_CONTACT_PRIM_PATH_PLACEHOLDER,
        shape_path=FINGERTIP_CONTACT_SHAPE_PAD_ONLY,
        filter_prim_paths_expr=[HAMMER_CONTACT_FILTER_EXPR],
        update_period=0.0,
    )


@configclass
class SceneCfgNewton(SceneCfg):
    replicate_physics: bool = True
    clone_in_fabric: bool = True
    newton_replicate_kwargs: dict | None = None


# =============================================================================
# Commands
# =============================================================================
@configclass
class CommandsCfg:
    reference_trajectory = mdp.ReferenceTrajectoryCommandCfg(
        trajectory_file="Hammer_Hammer_RL_Trajectory.npz",
        loop=False,
        loop_start_time=None,
        space="joint",
        resampling_time_range=(60.0, 60.0),
        playback_speed=1.0,
    )


# =============================================================================
# Observations
# =============================================================================

_ROBOT_CFG = SceneEntityCfg("robot", joint_names=ARM_HAND_JOINT_NAMES)
_TORQUE_CFG = SceneEntityCfg("robot", joint_names=ARM_HAND_TORQUE_JOINT_NAMES, preserve_order=True)
_HAMMER_CFG = SceneEntityCfg("hammer")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": _ROBOT_CFG},
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )
        right_hand_joint_torque = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": _TORQUE_CFG},
            noise=GaussianNoiseCfg(mean=0.0, std=0.1),
        )
        right_hand_relative_pose = ObsTerm(
            func=mdp.right_hand_relative_pose,
            params={"robot_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True
            self.history_length = 5

    @configclass
    class PrivilegedObsCfg(ObsGroup):
        """Asymmetric Critic privileged state (SimToolReal Appendix D.4)."""

        # Proprio (noise-free)
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": _ROBOT_CFG})

        right_hand_joint_torque = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": _TORQUE_CFG},
        )
        right_hand_relative_pose = ObsTerm(
            func=mdp.right_hand_relative_pose,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        hammer_relative_pose = ObsTerm(
            func=mdp.hammer_relative_pose,
            params={"hammer_cfg": _HAMMER_CFG},
        )

        # Privileged-only
        object_lin_vel = ObsTerm(func=mdp.object_lin_vel, params={"asset_cfg": _HAMMER_CFG})
        object_ang_vel = ObsTerm(func=mdp.object_ang_vel, params={"asset_cfg": _HAMMER_CFG})
        palm_lin_vel = ObsTerm(func=mdp.palm_lin_vel, params={"body_name": "Right_Hand_base"})
        palm_ang_vel = ObsTerm(func=mdp.palm_ang_vel, params={"body_name": "Right_Hand_base"})
        min_fingertip_object_distance = ObsTerm(
            func=mdp.min_fingertip_object_distance, params={"asset_cfg": _HAMMER_CFG},
        )
        fingertip_hammer_contact = ObsTerm(
            func=mdp.fingertip_hammer_contact,
            params={"sensor_name": "hand_contact_sensor"},
        )
        episode_step_count = ObsTerm(func=mdp.episode_step_count)
        object_is_grasped = ObsTerm(
            func=mdp.object_is_grasped,
            params={"asset_cfg": _HAMMER_CFG, "lift_threshold": 1.0},
        )
        instantaneous_reward = ObsTerm(func=mdp.instantaneous_reward)

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class PolicyCriticCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    policy_critic: PolicyCriticCfg = PolicyCriticCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


# =============================================================================
# Actions
# =============================================================================


# Relative position action scale: arm (larger motion) vs hand (finer control).
_ARM_ACTION_SCALE: float = 0.1
_HAND_ACTION_SCALE: float = 0.5


@configclass
class ActionsCfg:
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        scale=_ARM_ACTION_SCALE,
        use_zero_offset=True,
    )
    hand_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=HAND_JOINT_NAMES,
        preserve_order=True,
        scale=_HAND_ACTION_SCALE,
        use_zero_offset=True,
    )


# =============================================================================
# Events
# =============================================================================
@configclass
class EventCfg:
    hammer_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass_newton,
        mode="reset",
        params={"asset_cfg": _HAMMER_CFG, "mass_distribution_params": (0.8, 1.6), "operation": "scale"},
    )
    hammer_friction = EventTerm(
        func=set_shape_friction,
        mode="reset",
        params={"asset_cfg": _HAMMER_CFG, "mu_range": (0.3, 0.5)},
    )
    reset_hammer = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": _HAMMER_CFG,
            "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "yaw": [-0.0, 0.0]},
            "velocity_range": {},
        },
    )
    hammer_force_when_lifted = EventTerm(
        func=mdp.apply_hammer_force_when_lifted,
        mode="interval",
        interval_range_s=(0.1, 0.2),
        params={
            "asset_cfg": _HAMMER_CFG,
            "force_range": (-5.0, 5.0),
            "torque_range": (-1.0, 1.0),
            "height_threshold": 1.1,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset_arm_hand,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "arm_joint_names": ARM_JOINT_NAMES,
            "hand_joint_names": HAND_JOINT_NAMES,
            "position_range_arm": (-0.10, 0.10),
            "velocity_range_arm": (0.0, 0.0),
            "position_range_hand": (-0.10, 0.10),
            "velocity_range_hand": (0.0, 0.0),
        },
    )

# =============================================================================
# Rewards
# =============================================================================

_ARM_CFG = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
_HAND_CFG = SceneEntityCfg("robot", joint_names=HAND_JOINT_NAMES)

@configclass
class RewardsCfg:
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    arm_joint_torque_penalty = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4, params={"asset_cfg": _ARM_CFG})
    hand_joint_torque_penalty = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4, params={"asset_cfg": _HAND_CFG})
    arm_joint_vel_penalty = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": _ARM_CFG})
    hand_joint_vel_penalty = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": _HAND_CFG})
    late_lift_penalty = RewTerm(func=mdp.late_lift_penalty,
        weight=-0.3,
        params={"asset_cfg": _HAMMER_CFG, "lift_threshold": 0.93, "mid_episode_ratio": 0.5},
    )

    hammer_lift = RewTerm(
        func=mdp.hammer_lift_reward,
        weight=0.5,
        params={
            "asset_cfg": _HAMMER_CFG, 
            "threshold": 0.93, 
            "late_scale": 1.5,
        },
    )
    hammer_goal_proximity = RewTerm(
        func=mdp.hammer_goal_proximity_reward,
        weight=5.0,
        params={
            "hammer_cfg": _HAMMER_CFG,
            "pos_std": 0.1,
            "rot_std": 0.1,
            "pos_weight": 0.5,
            "rot_weight": 0.5,
            "lift_threshold": 0.93,
        },
    )
    reference_trajectory_hand_tracking = RewTerm(
        func=mdp.reference_trajectory_tracking_reward,
        weight=0.3,
        params={
            "command_name": "reference_trajectory",
            "asset_cfg": _HAND_CFG,
            "joint_names": HAND_JOINT_NAMES,
            "joint_std": 0.1,
        },
    )
    reference_trajectory_arm_tracking = RewTerm(
        func=mdp.reference_trajectory_tracking_reward,
        weight=0.3,
        params={
            "command_name": "reference_trajectory",
            "asset_cfg": _ARM_CFG,
            "joint_names": ARM_JOINT_NAMES,
            "joint_std": 0.1,
        },
    )
    grasp_point_proximity = RewTerm(
        func=mdp.grasp_point_proximity_reward,
        weight=0.3,
        params={"hammer_cfg": _HAMMER_CFG, "pos_std": 0.10},
    )
    fingertip_hammer_contact = RewTerm(
        func=mdp.fingertip_hammer_contact_reward,
        weight=0.1,
        params={"sensor_name": "hand_contact_sensor"},
    )
    hand_final_pose = RewTerm(
        func=mdp.hand_final_pose_reward,
        weight=5.0,
        params={
            "command_name": "reference_trajectory",
            "asset_cfg": _HAND_CFG,
            "hammer_cfg": _HAMMER_CFG,
            "hand_joint_names": HAND_JOINT_NAMES,
            "lift_threshold": 0.93,
            "joint_std": 0.1,
        },
    )



# =============================================================================
# Terminations
# =============================================================================


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    hammer_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.8, "asset_cfg": _HAMMER_CFG},
    )
    hammer_velocity_exceeded = DoneTerm(
        func=mdp.hammer_velocity_exceeded,
        params={"asset_cfg": _HAMMER_CFG, "max_lin_vel": 30.0, "max_ang_vel": 80.0},
    )
    hammer_too_far_from_table = DoneTerm(
        func=mdp.hammer_too_far_from_table,
        params={"hammer_cfg": _HAMMER_CFG, "table_cfg": SceneEntityCfg("table"), "max_distance": 1.0},
    )


# =============================================================================
# Curriculum
# =============================================================================

_STEPS_PER_ENV = 16


@configclass
class CurriculumCfg:
    """Curriculum: ramp hammer reset pose range over training."""

    reset_hammer_pose_x = CurrTerm(
        func=mdp.modify_term_cfg_with_logging,
        params={
            "address": "events.reset_hammer.params.pose_range.x",
            "modify_fn": mdp.step_based_interpolate_fn,
            "modify_params": {
                "initial_value": [0.0, 0.0], "final_value": [-0.03, 0.03],
                "start_step": 100, "end_step": 1000, "num_steps_per_env": _STEPS_PER_ENV,
            },
        },
    )
    reset_hammer_pose_y = CurrTerm(
        func=mdp.modify_term_cfg_with_logging,
        params={
            "address": "events.reset_hammer.params.pose_range.y",
            "modify_fn": mdp.step_based_interpolate_fn,
            "modify_params": {
                "initial_value": [0.0, 0.0], "final_value": [-0.03, 0.03],
                "start_step": 100, "end_step": 1000, "num_steps_per_env": _STEPS_PER_ENV,
            },
        },
    )
    reset_hammer_pose_yaw = CurrTerm(
        func=mdp.modify_term_cfg_with_logging,
        params={
            "address": "events.reset_hammer.params.pose_range.yaw",
            "modify_fn": mdp.step_based_interpolate_fn,
            "modify_params": {
                "initial_value": [0.0, 0.0], "final_value": [-0.3, 0.3],
                "start_step": 100, "end_step": 1000, "num_steps_per_env": _STEPS_PER_ENV,
            },
        },
    )


# =============================================================================
# Top-level env config
# =============================================================================


@configclass
class DexblindNewtonLiftEnvCfg(ManagerBasedRLEnvCfg):
    # viewport(Isaac Sim) + Newton visualizer 둘 다 이 eye/lookat 사용 (__post_init__에서 Newton에 전달)
    viewer: ViewerCfg = ViewerCfg(eye=(3.0, 3.0, 2.0), lookat=(0.0, 0.0, 0.45), origin_type="env")

    log_solver_convergence_interval: int = LOG_SOLVER_CONVERGENCE_INTERVAL
    """Solver 수렴 로깅 주기. 0=비활성, N=매 N env step마다 max/mean/limit 출력. (constants.LOG_SOLVER_CONVERGENCE_INTERVAL)"""

    scene: SceneCfgNewton = SceneCfgNewton(
        num_envs=NUM_ENVS,
        env_spacing=2.0,
        newton_replicate_kwargs={
            "equality_constraints": list(ALLEX_JOINT_EQUALITY_CONSTRAINTS),
            "simplify_meshes": {
                "hammer": ("coacd", {"threshold": 0.12}),
                "*": "convex_hull",
            },
            "load_visual_shapes": False,
            "lock_joints": LOCK_JOINTS,
        },
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
        self.decimation = 4

        # Headless일 때는 visualizer를 생성하지 않음 (upstream은 CLI가 비어도 config의 visualizer_cfgs를 전부 쓰므로, 우리가 비워줌).
        _headless = "--headless" in sys.argv or os.environ.get("HEADLESS", "0") == "1"
        _visualizer_cfgs: list = []
        if not _headless:
            _visualizer_cfgs = [
                DexblindNewtonVisualizerCfg(
                    camera_position=self.viewer.eye,
                    camera_target=self.viewer.lookat,
                    goal_markers=[
                        GoalMarkerCfg(
                            pos=GOAL_POS,
                            rot=GOAL_ROT,
                            scale=(1.0, 1.0, 1.0),
                            color=(0.2, 0.85, 0.3),
                            usd_path=str(_ASSETS_DIR / "object" / "Hammer_goal_pose.usd"),
                        ),
                    ],
                ),
                RerunVisualizerCfg(record_to_rrd="logs/rsl_rl/record/training_record.rrd", keep_historical_data=True),
            ]
        # Phase 4.2: ManagerBasedRLEnvCfg.sim.physics.solver_cfg = DEXBLIND_NEWTON_SOLVER_CFG
        self.sim: SimulationCfg = SimulationCfg(
            physics=NewtonCfg(
                solver_cfg=DEXBLIND_NEWTON_SOLVER_CFG,
                num_substeps=4,
                debug_mode=False,
                use_cuda_graph=True,
            ),
            dt=1 / 200,
            gravity=(0.0, 0.0, -9.81),
            visualizer_cfgs=_visualizer_cfgs,
        )
        self.sim.render_interval = self.decimation
        self.episode_length_s = 4.0
        self.is_finite_horizon = True


@configclass
class DexblindNewtonLiftEnvCfg_PLAY(DexblindNewtonLiftEnvCfg):
    pass
