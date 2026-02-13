# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal direct env: ALLEX robot only on ground (Newton). For zero_agent / debugging."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# Use Newton Articulation when isaaclab_newton is active (dev/newton branch)
try:
    from isaaclab_newton.assets.articulation import Articulation
except ImportError:
    from isaaclab.assets import Articulation

from .allex_env_cfg import ALLEX_MIMIC_SPEC, AllexEnvCfg


class AllexEnv(DirectRLEnv):
    """Minimal environment: spawn ALLEX on a plane, apply actions as joint position offsets."""

    cfg: AllexEnvCfg

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Defer find_joints until after Newton model is built (first step)
        self._joint_dof_idx = None
        self._joint_names = None
        self._mimic_overrides = None  # list of (mimic_idx, driver_idx, polycoef)

    def _ensure_joint_dof_idx(self):
        if self._joint_dof_idx is None:
            # find_joints returns (mask, names, indices)
            _, self._joint_names, self._joint_dof_idx = self.robot.find_joints(".*")
            name_to_idx = {n: i for i, n in enumerate(self._joint_names)}
            self._mimic_overrides = []
            for mimic_name, driver_name, coef in ALLEX_MIMIC_SPEC:
                if mimic_name in name_to_idx and driver_name in name_to_idx:
                    self._mimic_overrides.append((name_to_idx[mimic_name], name_to_idx[driver_name], coef))

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        self._ensure_joint_dof_idx()
        scale = 0.5
        current = wp.to_torch(self.robot.data.joint_pos)[:, self._joint_dof_idx]
        target = current + scale * self.actions
        use_engine_equality = getattr(self.cfg, "use_newton_equality_for_mimic", False)
        if use_engine_equality and self._mimic_overrides:
            # MuJoCo equality only: set target for driver joints only. Mimic joints are enforced
            # by the solver (equality constraint) in each substep; no _poly here.
            mimic_positions = {mimic_i for (mimic_i, _, _) in self._mimic_overrides}
            driver_positions = [j for j in range(len(self._joint_dof_idx)) if j not in mimic_positions]
            driver_joint_ids = [self._joint_dof_idx[j] for j in driver_positions]
            target_driver = target[:, driver_positions]
            self.robot.set_joint_position_target(target_driver, joint_ids=driver_joint_ids)
        else:
            print("use_engine_equality not")


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
