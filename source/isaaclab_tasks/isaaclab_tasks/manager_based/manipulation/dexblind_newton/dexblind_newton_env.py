# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ManagerBasedRLEnv subclass with Newton-compatible reset and hard-contact table."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim._impl.newton_manager import NewtonManager
from newton.solvers import SolverNotifyFlags

from .config.constants import (
    GOAL_POS,
    GOAL_PROXIMITY_THRESHOLD,
    GOAL_POS_RANGE,
    GOAL_ROT,
    GOAL_ROT_RANGE_RAD,
    HARD_CONTACT_KD,
    HARD_CONTACT_KE,
)
from .utils.newton_material import set_shape_contact_stiffness

_NOTIFY_ALL = (
    SolverNotifyFlags.BODY_PROPERTIES
    | SolverNotifyFlags.JOINT_PROPERTIES
    | SolverNotifyFlags.SHAPE_PROPERTIES
)

_PRIVILEGED_BUFFERS: tuple[tuple[str, float | bool], ...] = (
    ("_priv_min_ft_obj_dist", float("inf")),
    ("_priv_prev_lifted", False),
    ("_priv_cum_successes", 0.0),
)

class DexblindNewtonEnv(ManagerBasedRLEnv):
    """Newton-aware env: explicit articulation reset + one-shot table stiffness.

    Newton ``scene.reset()`` only zeroes external wrenches — joint/root state must
    be written back explicitly. On the first ``step()`` call, high contact stiffness
    is applied to the table so fingers do not penetrate.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._hard_contact_pending = True
        try:
            from isaaclab.visualizers.debug_panel import set_debug_panel_env
            set_debug_panel_env(self)
        except Exception:
            pass

    def load_managers(self):
        self._init_dynamic_goal()
        self._init_hammer_initial_relative_pose()
        super().load_managers()
        from .mdp.observations import hammer_relative_pose
        self._hammer_initial_relative_pose.copy_(hammer_relative_pose(self))

    # --------------------------------------------------------------------- #
    # Dynamic goal
    # --------------------------------------------------------------------- #

    def _init_dynamic_goal(self):
        """Create per-env goal buffers (pos/rot)."""
        n, dev = self.num_envs, self.device
        self._dynamic_goal_pos = torch.tensor(GOAL_POS, dtype=torch.float32, device=dev).unsqueeze(0).expand(n, 3).clone()
        self._dynamic_goal_rot = torch.tensor(GOAL_ROT, dtype=torch.float32, device=dev).unsqueeze(0).expand(n, 4).clone()

    def _init_hammer_initial_relative_pose(self):
        """Create per-env buffer for hammer pose at reset (ProprioObs; filled on reset and after load)."""
        n, dev = self.num_envs, self.device
        self._hammer_initial_relative_pose = torch.zeros(n, 7, dtype=torch.float32, device=dev)

    def _reset_goal_to_default(self, env_ids: torch.Tensor):
        """Reset goal to GOAL_POS / GOAL_ROT for given envs."""
        dev = self.device
        self._dynamic_goal_pos[env_ids] = torch.tensor(GOAL_POS, dtype=torch.float32, device=dev)
        self._dynamic_goal_rot[env_ids] = torch.tensor(GOAL_ROT, dtype=torch.float32, device=dev)

    def _resample_goal_near(self, env_ids: torch.Tensor):
        """Resample goal within ±10cm pos / ±30deg rot of the default GOAL_POS / GOAL_ROT."""
        from isaaclab.utils.math import quat_mul

        dev = self.device
        k = env_ids.shape[0]
        base_pos = torch.tensor(GOAL_POS, dtype=torch.float32, device=dev).unsqueeze(0).expand(k, 3)
        pos_delta = (torch.rand(k, 3, device=dev) * 2 - 1) * GOAL_POS_RANGE
        self._dynamic_goal_pos[env_ids] = base_pos + pos_delta

        base_rot = torch.tensor(GOAL_ROT, dtype=torch.float32, device=dev).unsqueeze(0).expand(k, 4)
        axis = torch.randn(k, 3, device=dev)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        angle = (torch.rand(k, device=dev) * 2 - 1) * GOAL_ROT_RANGE_RAD
        half = angle * 0.5
        # quat_xyzw: (x, y, z, w)
        delta_rot = torch.cat([axis * half.unsqueeze(-1).sin(), half.cos().unsqueeze(-1)], dim=-1)
        self._dynamic_goal_rot[env_ids] = quat_mul(delta_rot, base_rot)

    def _maybe_resample_goals(self):
        """Check if hammer is close to current goal; if so, resample."""
        from .mdp.utils import get_body_poses_batched

        n, dev = self.num_envs, self.device
        h_pos, _ = get_body_poses_batched("hammer", n, dev)
        local_pos = h_pos - self.scene.env_origins
        dist = torch.norm(local_pos - self._dynamic_goal_pos, dim=-1)
        close_ids = (dist < GOAL_PROXIMITY_THRESHOLD).nonzero(as_tuple=False).squeeze(-1)
        if close_ids.numel() > 0:
            self._resample_goal_near(close_ids)

    def step(self, action: torch.Tensor):
        if self._hard_contact_pending:
            for asset_name in ("table", "robot"):
                if asset_name in self.scene.articulations:
                    set_shape_contact_stiffness(
                        self, None, SceneEntityCfg(asset_name), ke=HARD_CONTACT_KE, kd=HARD_CONTACT_KD,
                    )
            self._hard_contact_pending = False
        obs, rew, terminated, truncated, info = super().step(action)
        self._maybe_resample_goals()
        return obs, rew, terminated, truncated, info

    # --------------------------------------------------------------------- #
    # Reset
    # --------------------------------------------------------------------- #

    def _reset_idx(self, env_ids: Sequence[int]):
        self.curriculum_manager.compute(env_ids=env_ids)
        self.scene.reset(env_ids)

        env_ids_t = self._to_tensor(env_ids)
        self._reset_privileged_buffers(env_ids_t)
        self._reset_goal_to_default(env_ids_t)
        self._reset_articulations(env_ids_t)

        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        from .mdp.observations import hammer_relative_pose
        self._hammer_initial_relative_pose[env_ids_t] = hammer_relative_pose(self)[env_ids_t]

        self.extras["log"] = {}
        for mgr_name in (
            "observation_manager", "action_manager", "reward_manager",
            "curriculum_manager", "command_manager", "event_manager",
            "termination_manager", "recorder_manager",
        ):
            self.extras["log"].update(getattr(self, mgr_name).reset(env_ids))

        self.episode_length_buf[env_ids] = 0

    def _reset_articulations(self, env_ids_t: torch.Tensor):
        for articulation in self.scene.articulations.values():
            default_root = wp.to_torch(articulation.data.default_root_state)[env_ids_t].clone()
            default_root[:, :3] += self.scene.env_origins[env_ids_t]
            articulation.write_root_pose_to_sim(default_root[:, :7], env_ids=env_ids_t)
            articulation.write_root_velocity_to_sim(default_root[:, 7:], env_ids=env_ids_t)

            if articulation.num_joints > 0:
                jpos = wp.to_torch(articulation.data.default_joint_pos)[env_ids_t].clone()
                jvel = wp.to_torch(articulation.data.default_joint_vel)[env_ids_t].clone()
                articulation.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids_t)
                articulation.set_joint_position_target(jpos, env_ids=env_ids_t)
                articulation.set_joint_velocity_target(jvel, env_ids=env_ids_t)

        NewtonManager._solver.notify_model_changed(_NOTIFY_ALL)

    def _reset_privileged_buffers(self, env_ids_t: torch.Tensor):
        for attr, default in _PRIVILEGED_BUFFERS:
            buf = getattr(self, attr, None)
            if buf is not None:
                buf[env_ids_t] = default

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _to_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids
        return torch.tensor(env_ids, device=self.device, dtype=torch.long)
