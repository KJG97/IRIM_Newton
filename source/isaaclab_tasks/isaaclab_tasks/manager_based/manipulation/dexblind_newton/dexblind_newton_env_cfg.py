# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal dexblind_newton: ALLEX only (hammer/table are in robot USD)."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from newton.solvers import SolverNotifyFlags

from isaaclab.envs.mdp.actions import actions_cfg as mdp_actions_cfg
from isaaclab_tasks.direct.allex.allex_env_cfg import ALLEX_MIMIC_SPEC

from . import mdp


def set_shape_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mu: float = 1.0,
):
    """Set Newton shape friction (mu) for all shapes of an asset."""
    asset = env.scene[asset_cfg.name]
    current_mu = wp.to_torch(
        asset.root_view.get_attribute("shape_material_mu", asset.root_newton_model)
    ).clone()
    if env_ids is None:
        current_mu[:] = mu
    else:
        current_mu[env_ids] = mu
    mask = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)
    if env_ids is None:
        mask[:] = True
    else:
        mask[env_ids] = True
    asset.root_view.set_attribute(
        "shape_material_mu", asset.root_newton_model, wp.from_torch(current_mu), mask=mask
    )
    NewtonManager._solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

NUM_ENVS = 2

# Right arm + hand (18 joints): arm 7 + hand 11, Roll excluded. Single source for obs/action.
RIGHT_ARM_HAND_JOINT_NAMES = [
    "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
    "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
    "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_MCP_Joint", "R_Little_PIP_Joint",
]

DEXBLIND_NEWTON_SOLVER_CFG = MJWarpSolverCfg(
    solver="newton",
    integrator="implicit",
    njmax=600,
    nconmax=6000,
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.65, -0.1, 0.95), rot=(0.0, 1.0, 0.0, 0.0)),
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
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.65, -0.1, 0.45), rot=(1.0, 0.0, 0.0, 0.0)),
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

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class SceneCfgNewton(SceneCfg):
    replicate_physics: bool = True
    clone_in_fabric: bool = True
    newton_replicate_kwargs: dict | None = None


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
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()


@configclass
class ActionsCfg:
    joint_pos = mdp_actions_cfg.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_HAND_JOINT_NAMES,
        preserve_order=True,
        scale=0.1,
    )


@configclass
class EventCfg:
    hammer_friction = EventTerm(
        func=set_shape_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("hammer"),
            "mu": 1.0,
        },
    )
    table_friction = EventTerm(
        func=set_shape_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "mu": 1.0,
        },
    )


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class DexblindNewtonLiftEnvCfg(ManagerBasedRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg(eye=(0.0, 0.0, 0.0), lookat=(0.0, 0.0, 0.45), origin_type="env")

    scene: SceneCfgNewton = SceneCfgNewton(
        num_envs=NUM_ENVS,
        env_spacing=3.0,
        newton_replicate_kwargs={
            "equality_constraints": list(ALLEX_MIMIC_SPEC),
            "simplify_meshes": {
                "hammer": "convex_hull", #("coacd", {"threshold": 0.15}),
                "table": "bounding_box",
                "*": "convex_hull",
            },
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
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: object | None = None
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 4.0
        self.is_finite_horizon = True


@configclass
class DexblindNewtonLiftEnvCfg_PLAY(DexblindNewtonLiftEnvCfg):
    pass
