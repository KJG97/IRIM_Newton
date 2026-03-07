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

# Full ALLEX (60 DOF): same as ALLEX_MIMIC_SPEC + left hand mimic (symmetric poly coefs).
ALLEX_FULL_MIMIC_SPEC: list[tuple[str, str, tuple[float, ...]]] = list(ALLEX_MIMIC_SPEC) + [
    ("L_Thumb_IP_Joint", "L_Thumb_MCP_Joint", (-0.0015, 0.6651, 0.0186, 0.1224, -0.0696)),
    ("L_Index_DIP_Joint", "L_Index_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("L_Middle_DIP_Joint", "L_Middle_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("L_Ring_DIP_Joint", "L_Ring_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("L_Little_DIP_Joint", "L_Little_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
]

# Newton solver 전용 설정. Full ALLEX(60 DOF, 295 shapes)는 mj_forward 시 mjData 스택 부족으로
# mj_stackAlloc overflow 발생 → mj_data_memory로 arena+stack 크기 확대 필요.
ALLEX_SOLVER_CFG = MJWarpSolverCfg(
    solver="newton",
    integrator="implicit",
    njmax=600 * NUM_ENVS,
    nconmax=6000,  # Full ALLEX ~295 shapes → broadphase needs ~5850
    impratio=10.0,
    cone="elliptic",
    update_data_interval=2,
    iterations=100,
    ls_iterations=15,
    ls_parallel=True,
    mj_data_memory=64 * 1024 * 1024,  # 64 MiB (Full ALLEX: MuJoCo requests >16 MiB for nefc/ncon)
)


@configclass
class AllexEnvCfg(DirectRLEnvCfg):
    """Minimal config: ALLEX on a plane with Newton (full 60 DOF: waist + both arms/hands). No objects, no task."""

    episode_length_s = 30.0
    decimation = 2
    action_space = ALLEX_NUM_DOF
    observation_space = ALLEX_NUM_DOF
    state_space = 0

    mimic_spec: list = ALLEX_FULL_MIMIC_SPEC
    """(mimic_name, driver_name, (c0,c1,c2,c3,c4)) for env and joint_slider_agent."""

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
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
        newton_replicate_kwargs={
            "equality_constraints": ALLEX_FULL_MIMIC_SPEC,
            "simplify_meshes": {"*": "convex_hull"},
            "load_visual_shapes": True,
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

    robot: ArticulationCfg = ALLEX_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class AllexEnvNoLeftCfg(DirectRLEnvCfg):
    """ALLEX_newton_no_left.usd 전용: 왼팔/왼손 제거, 목 Fixed → nv=31, 허리+오른팔+오른손 (Newton)."""

    episode_length_s = 30.0
    decimation = 2
    action_space = ALLEX_NO_LEFT_NUM_DOF
    observation_space = ALLEX_NO_LEFT_NUM_DOF
    state_space = 0

    mimic_spec: list = ALLEX_MIMIC_SPEC

    solver_cfg = ALLEX_SOLVER_CFG
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=4,
        debug_mode=False,
        use_cuda_graph=True,  # True면 [solver.substep] print는 캡처 시 1회만 찍힘. 매 스텝 찍으려면 False (성능 저하)
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
        newton_replicate_kwargs={
            "equality_constraints": ALLEX_MIMIC_SPEC,
            "simplify_meshes": {"*": "convex_hull"},
            "load_visual_shapes": True,
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

    robot: ArticulationCfg = ALLEX_NO_LEFT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
