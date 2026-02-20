# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Residual joint position action: reference_trajectory + residual * scale."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import residual_joint_action_cfg as residual_action_cfgs

logger = logging.getLogger(__name__)


class ResidualJointPositionAction(JointAction):
    """final_action = reference_trajectory + (scale * action + offset) * residual_scale."""

    cfg: residual_action_cfgs.ResidualJointPositionActionCfg

    def __init__(self, cfg: residual_action_cfgs.ResidualJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.command_name = cfg.command_name
        try:
            self._command_term = env.command_manager._terms[self.command_name]
        except KeyError:
            raise ValueError(
                f"Command '{self.command_name}' not found. "
                f"Available: {list(env.command_manager._terms.keys())}"
            )
        self._dim_checked = False
        self._reorder: torch.Tensor | None = None

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        ref = self._env.command_manager.get_command(self.command_name)
        if not self._dim_checked:
            if ref.shape[1] != actions.shape[1]:
                raise ValueError(
                    f"Trajectory joints {ref.shape[1]} != action joints {actions.shape[1]}."
                )
            if hasattr(self._command_term, "joint_names"):
                traj_names = self._command_term.joint_names
                action_names = self._joint_names
                if traj_names != action_names:
                    self._reorder = torch.tensor(
                        [traj_names.index(n) for n in action_names],
                        dtype=torch.long,
                        device=self.device,
                    )
                    logger.warning(
                        "Trajectory joint order differs from action; reordering. "
                        "Traj: %s | Action: %s", traj_names, action_names,
                    )
            self._dim_checked = True
        if self._reorder is not None:
            ref = ref[:, self._reorder]
        residual = actions * self._scale + self._offset
        if self.cfg.clip is not None:
            residual = torch.clamp(residual, self._clip[:, :, 0], self._clip[:, :, 1])
        self._processed_actions = ref + residual * self.cfg.residual_scale

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self.processed_actions, joint_ids=self._joint_ids
        )
