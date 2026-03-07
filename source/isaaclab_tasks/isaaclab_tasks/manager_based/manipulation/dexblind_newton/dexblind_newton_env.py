# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ManagerBasedRLEnv subclass with Newton-compatible reset logic."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim._impl.newton_manager import NewtonManager
from newton.solvers import SolverNotifyFlags


class DexblindNewtonEnv(ManagerBasedRLEnv):
    """Adds explicit articulation state reset that works with the Newton backend.

    Newton's ``scene.reset()`` only zeroes external wrenches; it does **not**
    restore joint positions, velocities, or root poses.  This subclass
    writes them back explicitly, mirroring what the Allegro-Hand DirectRLEnv
    does inside its ``_reset_idx``.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        try:
            from isaaclab.visualizers.debug_panel import set_debug_panel_env
            set_debug_panel_env(self)
        except Exception:
            pass

    def _reset_idx(self, env_ids: Sequence[int]):
        self.curriculum_manager.compute(env_ids=env_ids)
        self.scene.reset(env_ids)

        self._reset_privileged_buffers(env_ids)
        self._reset_articulations(env_ids)

        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        self.extras["log"] = dict()
        for mgr_name in (
            "observation_manager",
            "action_manager",
            "reward_manager",
            "curriculum_manager",
            "command_manager",
            "event_manager",
            "termination_manager",
            "recorder_manager",
        ):
            info = getattr(self, mgr_name).reset(env_ids)
            self.extras["log"].update(info)

        self.episode_length_buf[env_ids] = 0

    def _reset_articulations(self, env_ids: Sequence[int]):
        """Write every articulation back to its ``init_state``."""
        env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long) \
            if not isinstance(env_ids, torch.Tensor) else env_ids

        for articulation in self.scene.articulations.values():
            default_root_state = wp.to_torch(articulation.data.default_root_state)[env_ids_t].clone()
            default_root_state[:, 0:3] += self.scene.env_origins[env_ids_t]
            articulation.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids_t)
            articulation.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids_t)

            if articulation.num_joints > 0:
                default_joint_pos = wp.to_torch(articulation.data.default_joint_pos)[env_ids_t].clone()
                default_joint_vel = wp.to_torch(articulation.data.default_joint_vel)[env_ids_t].clone()
                articulation.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids_t)
                articulation.set_joint_position_target(default_joint_pos, env_ids=env_ids_t)
                articulation.set_joint_velocity_target(default_joint_vel, env_ids=env_ids_t)

        NewtonManager._solver.notify_model_changed(
            SolverNotifyFlags.BODY_PROPERTIES
            | SolverNotifyFlags.JOINT_PROPERTIES
            | SolverNotifyFlags.SHAPE_PROPERTIES
        )

    def _reset_privileged_buffers(self, env_ids: Sequence[int]):
        """Reset stateful buffers used by privileged observation terms."""
        env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long) \
            if not isinstance(env_ids, torch.Tensor) else env_ids

        for attr, default in (
            ("_priv_min_ft_obj_dist", float("inf")),
            ("_priv_grasped_flag", False),
            ("_priv_prev_lifted", False),
            ("_priv_cum_successes", 0.0),
        ):
            if hasattr(self, attr):
                getattr(self, attr)[env_ids_t] = default
