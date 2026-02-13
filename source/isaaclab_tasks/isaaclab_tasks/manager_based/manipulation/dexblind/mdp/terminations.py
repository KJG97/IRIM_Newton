# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the dexsuite task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def hammer_fallen(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hammer"),
    z_threshold: float = 0.4,
) -> torch.Tensor:
    """Termination condition for when the hammer falls below a certain height threshold.

    Args:
        env: The environment.
        asset_cfg: The hammer configuration. Defaults to SceneEntityCfg("hammer").
        z_threshold: The minimum z (height) threshold. If hammer z position falls below this, termination is triggered.
            Defaults to 0.4.

    Returns:
        Tensor of shape ``(num_envs,)``: True for environments where hammer has fallen below threshold.
    """
    hammer: RigidObject = env.scene[asset_cfg.name]
    # Get hammer z position in world frame
    hammer_z = hammer.data.root_pos_w[:, 2]
    # Check if hammer z position is below threshold
    fallen = hammer_z < z_threshold
    return fallen
