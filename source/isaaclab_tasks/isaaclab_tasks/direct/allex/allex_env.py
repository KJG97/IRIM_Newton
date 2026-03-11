# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal direct env: ALLEX robot only on ground (Newton). For zero_agent / debugging."""

from __future__ import annotations

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

try:
    from isaaclab_newton.assets.articulation import Articulation
except ImportError:
    from isaaclab.assets import Articulation

from isaaclab_tasks.manager_based.manipulation.dexblind_newton.utils.newton_material import (
    set_shape_contact_stiffness,
)

from .allex_env_cfg import ALLEX_MIMIC_SPEC, AllexEnvCfg, AllexEnvNoLeftCfg

# Hard-contact-like: high ke/kd on both table and hammer so neither side is soft (minimal penetration).
_HARD_CONTACT_KE = 50_000_000.0
_HARD_CONTACT_KD = 10_000.0


class AllexEnv(DirectRLEnv):
    """Minimal environment: ALLEX on ground (Newton). NoLeft variant can add hammer/table."""

    cfg: AllexEnvCfg | AllexEnvNoLeftCfg

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self._hard_contact_pending = False
        if getattr(self.cfg, "hammer_cfg", None) is not None:
            self.scene.articulations["hammer"] = Articulation(self.cfg.hammer_cfg)
            self.scene.articulations["table"] = Articulation(self.cfg.table_cfg)
            self._hard_contact_pending = True
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._joint_dof_idx = None
        self._joint_names = None
        self._mimic_overrides = None

    def _ensure_joint_dof_idx(self):
        if self._joint_dof_idx is None:
            # find_joints returns (mask, names, indices)
            _, self._joint_names, self._joint_dof_idx = self.robot.find_joints(".*")
            name_to_idx = {n: i for i, n in enumerate(self._joint_names)}
            mimic_spec = getattr(self.cfg, "mimic_spec", ALLEX_MIMIC_SPEC)
            self._mimic_overrides = []
            for mimic_name, driver_name, coef in mimic_spec:
                if mimic_name in name_to_idx and driver_name in name_to_idx:
                    self._mimic_overrides.append((name_to_idx[mimic_name], name_to_idx[driver_name], coef))

    def _pre_physics_step(self, actions: torch.Tensor):
        if getattr(self, "_hard_contact_pending", False) and "table" in self.scene.articulations:
            ke, kd = _HARD_CONTACT_KE, _HARD_CONTACT_KD
            set_shape_contact_stiffness(self, None, SceneEntityCfg("table"), ke=ke, kd=kd)
            set_shape_contact_stiffness(self, None, SceneEntityCfg("hammer"), ke=ke, kd=kd)
            self._hard_contact_pending = False
        self.actions = actions.clone()

    def _apply_action(self):
        self._ensure_joint_dof_idx()
        current = wp.to_torch(self.robot.data.joint_pos)[:, self._joint_dof_idx]
        target = current + self.actions
        if self._mimic_overrides:
            # Newton equality: set target for driver joints only. Mimic joints are enforced by the solver.
            mimic_positions = {mimic_i for (mimic_i, _, _) in self._mimic_overrides}
            driver_positions = [j for j in range(len(self._joint_dof_idx)) if j not in mimic_positions]
            driver_joint_ids = [self._joint_dof_idx[j] for j in driver_positions]
            target_driver = target[:, driver_positions]
            self.robot.set_joint_position_target(target_driver, joint_ids=driver_joint_ids)
        else:
            self.robot.set_joint_position_target(target, joint_ids=self._joint_dof_idx)

    def _get_observations(self) -> dict:
        self._ensure_joint_dof_idx()
        joint_pos = wp.to_torch(self.robot.data.joint_pos)[:, self._joint_dof_idx]
        return {"policy": joint_pos.clone()}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out
