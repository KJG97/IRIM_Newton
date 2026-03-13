# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared constants for dexblind_newton environments.

Robot joint names, contact/collision settings, Newton solver defaults, and
logging options live here so env_cfg / allex_cfg / obs / rewards import
from a single source of truth.
"""

from __future__ import annotations

# =============================================================================
# Environment & logging
# =============================================================================

NUM_ENVS: int = 4096

LOG_SOLVER_CONVERGENCE_INTERVAL: int = 10
"""Solver 수렴 로깅 주기 (env step 단위). 0=비활성, N=매 N env step마다 max/mean/limit 출력."""

# =============================================================================
# Joint name groups
# =============================================================================

ARM_JOINT_NAMES: list[str] = [
    "R_Shoulder_Pitch_Joint",
    "R_Shoulder_Roll_Joint",
    "R_Shoulder_Yaw_Joint",
    "R_Elbow_Joint",
    "R_Wrist_Yaw_Joint",
    "R_Wrist_Roll_Joint",
    "R_Wrist_Pitch_Joint",
]

HAND_JOINT_NAMES: list[str] = [
    "R_Thumb_Yaw_Joint",
    "R_Thumb_CMC_Joint",
    "R_Thumb_MCP_Joint",
    "R_Index_MCP_Joint",
    "R_Index_PIP_Joint",
    "R_Middle_MCP_Joint",
    "R_Middle_PIP_Joint",
    "R_Ring_MCP_Joint",
    "R_Ring_PIP_Joint",
    "R_Little_MCP_Joint",
    "R_Little_PIP_Joint",
]

ARM_HAND_JOINT_NAMES: list[str] = ARM_JOINT_NAMES + HAND_JOINT_NAMES
"""Arm 7 + Hand 11 = 18 joints (Roll excluded). Used for obs/action."""

ARM_TORQUE_JOINT_NAMES: list[str] = [
    "R_Shoulder_Pitch_Joint",
    "R_Shoulder_Roll_Joint",
    "R_Shoulder_Yaw_Joint",
    "R_Elbow_Joint",
]

HAND_JOINT_NAMES_WITH_ROLL: list[str] = [
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
]

ARM_HAND_TORQUE_JOINT_NAMES: list[str] = ARM_TORQUE_JOINT_NAMES + HAND_JOINT_NAMES_WITH_ROLL
"""Arm 4 + Hand 15 (with Roll) = 19 joints. Used for torque observation."""

# =============================================================================
# Fingertip body names (distance-based observations)
# =============================================================================

FINGERTIP_BODIES: list[str] = [
    "R_Hand_Thumb_Distal",
    "R_Hand_Index_Distal",
    "R_Hand_Middle_Distal",
    "R_Hand_Ring_Distal",
    "R_Hand_Little_Distal",
]

# =============================================================================
# Contact sensor
# =============================================================================

HAMMER_CONTACT_FILTER_EXPR: str = "{ENV_REGEX_NS}/hammer"
"""ContactSensorCfg.filter_prim_paths_expr — counterpart_bodies for SensorContact (hand–hammer only)."""

# 옵션 A: shape만 사용 — Pad만 감지 (Frame 제외). prim_path는 미사용(placeholder).
FINGERTIP_CONTACT_PRIM_PATH_PLACEHOLDER: str = "{ENV_REGEX_NS}/Robot"
"""ContactSensorCfg.prim_path placeholder when using shape_path only (not sent to Newton)."""

# Pad shape만 감지 (Distal_Frame 제외). Newton에 shape_names_expr만 전달됨.
# Isaac Sim에서 손가락별로 구분하려면 Pad 이름을 바꾸면 됨:
#   ALLEX_Hand_Index_Distal_Pad, ALLEX_Hand_Middle_Distal_Pad, ALLEX_Hand_Ring_Distal_Pad,
#   ALLEX_Hand_Little_Distal_Pad, ALLEX_Right_Hand_Thumb_Distal_Pad
FINGERTIP_CONTACT_SHAPE_PAD_ONLY: list[str] = [
    "ALLEX_Hand_Distal_Pad",
    "ALLEX_Right_Hand_Thumb_Distal_Pad",
]
"""ContactSensorCfg.shape_path — sensing objects = 이 shape만 (5개 Pad, Frame 제외)."""

# shape_path 사용 시 sensor_names가 shape 이름이 됨. 이름이 손가락별로 다르면 이 map으로 finger index 매핑.
# 이름 변경 후 여기에 추가: "ALLEX_Hand_Index_Distal_Pad": 0, "ALLEX_Hand_Middle_Distal_Pad": 1, ...
FINGERTIP_PAD_SHAPE_TO_FINGER_IDX: dict[str, int] = {
    "ALLEX_Right_Hand_Thumb_Distal_Pad": 4,
}
"""Shape 이름 → finger index (Index=0, Middle=1, Ring=2, Little=3, Thumb=4)."""

# shape_path 사용 시 4손가락이 동일 이름이면 Newton 반환 순서( body_key 순 )로 매핑. Link 순서: Index, Little, Middle, Ring, Thumb.
SENSOR_ORDER_WHEN_PAD_NAMES_AMBIGUOUS: tuple[int, ...] = (0, 3, 1, 2, 4)
"""5 sensors 순서가 Index, Little, Middle, Ring, Thumb일 때의 finger index."""

FINGER_NAMES: tuple[str, ...] = ("Index", "Middle", "Ring", "Little", "Thumb")
"""Per-finger contact order. 5 fingers × 3 segments = 15 dims."""

SEGMENTS_PER_FINGER: int = 3
"""Proximal/Roll/Yaw → 0, Middle → 1, Distal → 2."""

SEGMENT_TO_IDX: dict[str, int] = {
    "Proximal": 0,
    "Roll": 0,
    "Middle": 1,
    "Distal": 2,
    "Yaw": 0,
}

# =============================================================================
# Solver buffer (nconmax / njmax, 런타임 동적 설정용)
# =============================================================================

# nefc overflow 시 솔버가 NaN 반환. 로그 "increase njmax to N" 나오면
# (CON_PER_ENV + NCON_MARGIN) * NJMAX_MULTIPLY >= N 이 되도록 조정. (실제 91~100 보고됨)
CON_PER_ENV: int = 80
"""환경(로봇 1대+물체)당 contact 수. nefc overflow 나면 늘릴 것."""

NCON_MARGIN: int = 20
"""nconmax에 더하는 마진. 접촉 수 변동·OOM 방지."""

NJMAX_MULTIPLY: float = 2.0
"""njmax = nconmax * NJMAX_MULTIPLY. nconmax=65 → njmax=130."""

# =============================================================================
# Newton joint equality constraints (mimic, add_equality_constraint_joint)
# =============================================================================
# Waist: linear coupling q_follower = c0 + c1*q_driver → polycoef (c0, c1, 0, 0, 0)
# Hand:  quartic DIP/Thumb coupling via polycoef (c0, c1, c2, c3, c4)

ALLEX_JOINT_EQUALITY_CONSTRAINTS: list[tuple[str, str, tuple[float, ...]]] = [
    ("Waist_Pitch_Upper_Joint", "Waist_Pitch_Lower_Joint", (0.0, -1.0, 0.0, 0.0, 0.0)),
    ("Waist_Pitch_Dummy_Joint", "Waist_Pitch_Lower_Joint", (0.0, 1.0, 0.0, 0.0, 0.0)),
    ("R_Thumb_IP_Joint", "R_Thumb_MCP_Joint", (-0.0015, 0.6651, 0.0186, 0.1224, -0.0696)),
    ("R_Index_DIP_Joint", "R_Index_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("R_Middle_DIP_Joint", "R_Middle_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("R_Ring_DIP_Joint", "R_Ring_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("R_Little_DIP_Joint", "R_Little_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
]
"""Waist + right-hand. Applied via add_equality_constraint_joint (polycoef)."""

ALLEX_MIMIC_SPEC: list[tuple[str, str, tuple[float, ...]]] = list(ALLEX_JOINT_EQUALITY_CONSTRAINTS)
"""Alias for ALLEX_JOINT_EQUALITY_CONSTRAINTS. Used when a single list is expected."""

ALLEX_FULL_MIMIC_SPEC: list[tuple[str, str, tuple[float, ...]]] = list(ALLEX_MIMIC_SPEC) + [
    ("L_Thumb_IP_Joint", "L_Thumb_MCP_Joint", (-0.0015, 0.6651, 0.0186, 0.1224, -0.0696)),
    ("L_Index_DIP_Joint", "L_Index_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("L_Middle_DIP_Joint", "L_Middle_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("L_Ring_DIP_Joint", "L_Ring_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
    ("L_Little_DIP_Joint", "L_Little_PIP_Joint", (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)),
]
"""ALLEX_MIMIC_SPEC + left-hand mimic. For full-body tasks if needed."""

# =============================================================================
# Newton replicate options
# =============================================================================

LOCK_JOINTS: list[str] = [
    "Waist_Yaw_Joint",
    "Waist_Pitch_Lower_Joint",
]

# =============================================================================
# Goal pose (env-local coordinates)
# =============================================================================

GOAL_POS: tuple[float, float, float] = (0.65, -0.2, 1.2)
GOAL_ROT: tuple[float, float, float, float] = (0.0, -0.70711, -0.70711, 0.0)

# =============================================================================
# Contact stiffness (physics, first step after reset)
# =============================================================================
# 단위: ke [N/m], kd [N·s/m] → 접촉력 F = ke·d + kd·v 는 [N]. (newton_material.set_shape_contact_stiffness)

TABLE_CONTACT_KE: float = 2_500.0
TABLE_CONTACT_KD: float = 100.0
"""Table: 단단하게 (망치/손가락이 파고들지 않도록)."""

ROBOT_CONTACT_KE: float = 2_500.0
ROBOT_CONTACT_KD: float = 100.0
"""Robot (hand/arm): 덜 단단하게."""

HAMMER_CONTACT_KE: float = 2_500.0
HAMMER_CONTACT_KD: float = 100.0
"""Hammer: 덜 단단하게."""
