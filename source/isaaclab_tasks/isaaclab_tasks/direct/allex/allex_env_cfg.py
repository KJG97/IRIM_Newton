# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal direct env config: ALLEX robot + ground only (Newton)."""

from isaaclab_assets.robots.allex import ALLEX_CFG, ALLEX_NO_LEFT_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass


# ALLEX total DOF: USD/Newton articulation 실제 DOF 수와 일치해야 함 (현재 60).
ALLEX_NUM_DOF = 60
# ALLEX_newton_no_left.usd: 왼팔/왼손 제거. 목 2개 Fixed → Revolute 31 DOF (허리+오른팔+오른손, passive 포함).
ALLEX_NO_LEFT_NUM_DOF = 31

NUM_ENVS = 1

# Newton joint equality (mimic): (mimic_joint_name, driver_joint_name, (c0,c1,c2,c3,c4)) with q_mimic = c0 + c1*q_driver + ...
# From allex_contact_sensor.xml <equality><joint ... polycoef="...">. Used by newton_replicate(equality_constraints=...).
ALLEX_MIMIC_SPEC: list[tuple[str, str, tuple[float, ...]]] = [
    ("Waist_Pitch_Upper_Joint", "Waist_Pitch_Lower_Joint", (0.0, -1.0, 0.0, 0.0, 0.0)),
    ("Waist_Pitch_Dummy_Joint", "Waist_Pitch_Lower_Joint", (0.0, 1.0, 0.0, 0.0, 0.0)),
    ("R_Thumb_IP_Joint", "R_Thumb_MCP_Joint", (-0.0015, 0.6651, 0.0186, 0.1224, -0.0696)),
    ("R_Index_DIP_Joint", "R_Index_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("R_Middle_DIP_Joint", "R_Middle_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("R_Ring_DIP_Joint", "R_Ring_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("R_Little_DIP_Joint", "R_Little_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
]

# Newton solver 전용 설정
ALLEX_SOLVER_CFG = MJWarpSolverCfg(
    solver="newton",
    integrator="implicit",
    njmax=600*NUM_ENVS,
    nconmax=3000,
    impratio=10.0,
    cone="elliptic",
    update_data_interval=2,
    iterations=100,
    ls_iterations=15,
    ls_parallel=True,
)


@configclass
class AllexEnvCfg(DirectRLEnvCfg):
    """Minimal config: ALLEX on a plane with Newton. No objects, no task."""

    episode_length_s = 30.0
    decimation = 2
    action_space = ALLEX_NUM_DOF
    observation_space = ALLEX_NUM_DOF
    state_space = 0

    solver_cfg = ALLEX_SOLVER_CFG
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        newton_cfg=newton_cfg,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot: ArticulationCfg = ALLEX_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class AllexEnvNoLeftCfg(DirectRLEnvCfg):
    """ALLEX_newton_no_left.usd 전용: 왼팔/왼손 제거, 목 Fixed → nv=31, 허리+오른팔+오른손 (Newton)."""

    episode_length_s = 30.0
    decimation = 2
    action_space = ALLEX_NO_LEFT_NUM_DOF
    observation_space = ALLEX_NO_LEFT_NUM_DOF
    state_space = 0

    use_newton_equality_for_mimic: bool = True
    """When True, Newton joint equality constraints are used for mimic joints (injected at build time).
    Only driver joint targets are set in _apply_action. When False, mimic positions are written manually each step."""

    solver_cfg = ALLEX_SOLVER_CFG
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=2,
        debug_mode=False,
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        newton_cfg=newton_cfg,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=NUM_ENVS,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
        newton_replicate_kwargs={"equality_constraints": ALLEX_MIMIC_SPEC},
    )

    robot: ArticulationCfg = ALLEX_NO_LEFT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
