# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared constants for dexblind_newton environments.

All robot-specific joint names, collision patterns, contact sensor settings,
and Newton solver defaults live here so that env_cfg / allex_cfg / obs / rewards
import from a single source of truth.
"""

from __future__ import annotations

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
# Contact sensor
# =============================================================================

HAND_CONTACT_BODY_REGEX: str = (
    "{ENV_REGEX_NS}/Robot/.*(Right_Hand_Palm|R_Hand_Thumb"
    "|R_Hand_Index|R_Hand_Middle|R_Hand_Ring|R_Hand_Little).*"
)
"""Regex for ContactSensorCfg.prim_path — matches all hand links by body name."""

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
# Fingertip body names (for distance-based observations)
# =============================================================================

FINGERTIP_BODIES: list[str] = [
    "R_Hand_Thumb_Distal",
    "R_Hand_Index_Distal",
    "R_Hand_Middle_Distal",
    "R_Hand_Ring_Distal",
    "R_Hand_Little_Distal",
]

# =============================================================================
# Newton collision filtering
# =============================================================================

DISABLE_COLLISION_BODIES: list[str] = [
    "Waist_Base",
    "Waist_Yaw",
    "Waist_Pitch_Back",
    "Waist_Pitch_Lower",
    "Waist_Pitch_Upper",
    "Neck_Pitch",
    "Neck_Yaw",
    "Camera_Body",
]

DISABLE_COLLISION_SHAPES: list[str] = [
    # Arm
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
    # Hand (redundant collision geometry)
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
]

LOCK_JOINTS: list[str] = [
    "Waist_Yaw_Joint",
    "Waist_Pitch_Lower_Joint",
]

# =============================================================================
# Goal pose (env-local coordinates)
# =============================================================================

GOAL_POS: tuple[float, float, float] = (0.65, -0.2, 1.2)
GOAL_ROT: tuple[float, float, float, float] = (0.0, -0.70711, -0.70711, 0.0)

# Dynamic goal resampling (when hammer gets close to current goal)
GOAL_POS_RANGE: float = 0.03  # cm from GOAL_POS per axis
GOAL_ROT_RANGE_RAD: float = 0.3  # radians in radians
GOAL_PROXIMITY_THRESHOLD: float = 0.05  # pos distance to trigger goal resample

# =============================================================================
# Physics defaults
# =============================================================================

NUM_ENVS: int = 4096
HARD_CONTACT_KE: float = 500_000.0
HARD_CONTACT_KD: float = 1000.0
