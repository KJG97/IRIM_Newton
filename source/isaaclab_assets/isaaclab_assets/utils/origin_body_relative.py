# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task-space pose relative to Origin_Body prim.

Provides functions to express hammer, table, and other objects' poses in the
frame of the robot's Origin_Body, for consistent task-space coordinates across
the scene.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject


def get_origin_body_world_pose(
    robot: Articulation,
    origin_body_name: str = "Origin_Body",
    env_indices: torch.Tensor | slice | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Origin_Body world pose (position and quaternion).

    Args:
        robot: Articulation that has a body named ``origin_body_name`` (e.g. Origin_Body).
        origin_body_name: Name of the reference body on the robot.
        env_indices: Optional env indices to subset. If None, all envs are used.

    Returns:
        Tuple of (pos_w, quat_w):
        - pos_w: (N, 3) position in world frame.
        - quat_w: (N, 4) quaternion in (x, y, z, w) in world frame.

    Raises:
        ValueError: If ``origin_body_name`` is not found on the robot.
    """
    body_ids, _ = robot.find_bodies(origin_body_name)
    if not body_ids:
        raise ValueError(
            f"Body '{origin_body_name}' not found on robot. Available: {robot.body_names}"
        )
    body_id = body_ids[0]

    pos_w = wp.to_torch(robot.data.body_pos_w[:, body_id].view(robot.num_instances, 3))
    quat_w = wp.to_torch(robot.data.body_quat_w[:, body_id].view(robot.num_instances, 4))

    if env_indices is not None:
        if isinstance(env_indices, slice):
            pos_w = pos_w[env_indices]
            quat_w = quat_w[env_indices]
        else:
            pos_w = pos_w[env_indices]
            quat_w = quat_w[env_indices]
    return pos_w, quat_w


def _get_root_body_world_pose(asset: Articulation | RigidObject) -> tuple[torch.Tensor, torch.Tensor]:
    """Root body world pose (pos, quat) for an articulation or rigid object. (N, 3), (N, 4)."""
    pos_w = wp.to_torch(asset.data.body_pos_w[:, 0, :].view(asset.num_instances, 3))
    quat_w = wp.to_torch(asset.data.body_quat_w[:, 0, :].view(asset.num_instances, 4))
    return pos_w, quat_w


def get_pose_in_origin_body_frame(
    robot: Articulation,
    object_asset: Articulation | RigidObject,
    origin_body_name: str = "Origin_Body",
    env_indices: torch.Tensor | slice | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose of the object's root body in the Origin_Body frame (task-space relative pose).

    Use this for hammer, table, or any other scene object to get position and
    orientation relative to the robot's Origin_Body prim.

    Args:
        robot: Articulation that has ``origin_body_name`` (e.g. Origin_Body).
        object_asset: Articulation or RigidObject (hammer, table, etc.) whose root pose is desired.
        origin_body_name: Name of the reference body on the robot.
        env_indices: Optional env indices. If None, all envs are used. Both robot and
            object_asset are indexed with the same env_indices (same batch size).

    Returns:
        Tuple of (pos_rel, quat_rel):
        - pos_rel: (N, 3) position of object root in Origin_Body frame.
        - quat_rel: (N, 4) quaternion (x, y, z, w) of object root in Origin_Body frame.
    """
    origin_pos, origin_quat = get_origin_body_world_pose(
        robot, origin_body_name=origin_body_name, env_indices=env_indices
    )
    obj_pos, obj_quat = _get_root_body_world_pose(object_asset)
    if env_indices is not None:
        if isinstance(env_indices, slice):
            obj_pos = obj_pos[env_indices]
            obj_quat = obj_quat[env_indices]
        else:
            obj_pos = obj_pos[env_indices]
            obj_quat = obj_quat[env_indices]

    pos_rel, quat_rel = subtract_frame_transforms(origin_pos, origin_quat, obj_pos, obj_quat)
    return pos_rel, quat_rel
