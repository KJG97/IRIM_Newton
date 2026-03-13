# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ManagerBasedRLEnv subclass with Newton-compatible reset and hard-contact table."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
import warp as wp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import manager_based_env as _manager_based_env
from isaaclab.managers import SceneEntityCfg
from isaaclab_newton.physics import NewtonManager
from newton.solvers import SolverNotifyFlags

from allex_rl_dexblind.tasks.manager_based.dexblind_newton.config.constants import (
    CON_PER_ENV,
    GOAL_POS,
    GOAL_ROT,
    HAMMER_CONTACT_KD,
    HAMMER_CONTACT_KE,
    LOG_SOLVER_CONVERGENCE_INTERVAL,
    NCON_MARGIN,
    NJMAX_MULTIPLY,
    ROBOT_CONTACT_KD,
    ROBOT_CONTACT_KE,
    TABLE_CONTACT_KD,
    TABLE_CONTACT_KE,
)
from allex_rl_dexblind.tasks.manager_based.dexblind_newton.utils.scene import (
    DexblindInteractiveScene,
)
from allex_rl_dexblind.tasks.manager_based.dexblind_newton.utils.newton_material import (
    set_shape_contact_stiffness,
)
from allex_rl_dexblind.tasks.manager_based.dexblind_newton.utils.dexblind_visualizer import (
    set_visualizer_env,
)
from allex_rl_dexblind.tasks.manager_based.dexblind_newton.utils.solver_conv_viewer import (
    push_solver_conv_line,
    start_solver_conv_viewer,
)

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
        num_envs = cfg.scene.num_envs
        ncon_per_env = CON_PER_ENV + NCON_MARGIN
        nj_per_env = int(ncon_per_env * NJMAX_MULTIPLY)
        cfg.sim.physics.solver_cfg = replace(
            cfg.sim.physics.solver_cfg,
            nconmax=ncon_per_env,
            njmax=nj_per_env,
        )
        sc = cfg.sim.physics.solver_cfg
        total_nconmax = num_envs * sc.nconmax
        total_njmax = num_envs * sc.njmax
        print(
            f"[DexblindNewton] Solver buffers: num_envs={num_envs} | "
            f"per env: nconmax={sc.nconmax}, njmax={sc.njmax} | "
            f"총 nconmax={total_nconmax}, 총 njmax={total_njmax}"
        )
        # Patch NewtonSceneDataProvider.get_contacts so "Show Contacts" in Newton visualizer works
        # (upstream returns None; we supply NewtonManager._contacts so contact normals are drawn)
        from isaaclab.sim.scene_data_providers import newton_scene_data_provider as _nsdp

        if not getattr(_nsdp.NewtonSceneDataProvider, "_dexblind_contacts_patched", False):
            def _get_contacts(self):
                return NewtonManager._contacts

            _nsdp.NewtonSceneDataProvider.get_contacts = _get_contacts
            _nsdp.NewtonSceneDataProvider._dexblind_contacts_patched = True

        # Use DexblindInteractiveScene so SceneCfgNewton.newton_replicate_kwargs
        # (and other non-asset fields) are skipped in _add_entities_from_cfg.
        _orig_scene = _manager_based_env.InteractiveScene
        _manager_based_env.InteractiveScene = DexblindInteractiveScene
        try:
            super().__init__(cfg, render_mode=render_mode, **kwargs)
        finally:
            _manager_based_env.InteractiveScene = _orig_scene
        self._hard_contact_pending = True

    def load_managers(self):
        self._init_dynamic_goal()
        self._init_hammer_initial_relative_pose()
        super().load_managers()
        from allex_rl_dexblind.tasks.manager_based.dexblind_newton.mdp.observations import (
            hammer_relative_pose,
        )
        self._hammer_initial_relative_pose.copy_(hammer_relative_pose(self))
        self._wrap_managers_for_nan_diagnostic()

    def _wrap_managers_for_nan_diagnostic(self):
        """Upstream이 reward/obs NaN을 0으로 덮기 전에 진단이 실행되도록 compute를 래핑."""
        self._nan_diagnostic_done = False
        _reward_compute = self.reward_manager.compute
        def _wrapped_reward_compute(dt):
            out = _reward_compute(dt)
            if not self._nan_diagnostic_done and isinstance(out, torch.Tensor):
                bad = (torch.isnan(out) | torch.isinf(out)).nonzero(as_tuple=False).squeeze(-1)
                if bad.numel() > 0:
                    self._nan_diagnostic_done = True
                    if bad.dim() == 0:
                        bad = bad.unsqueeze(0)
                    from allex_rl_dexblind.tasks.manager_based.dexblind_newton.utils.diagnose_nan import (
                        diagnose_nan_source,
                    )
                    diagnose_nan_source(self, bad[:10], step_dt=self.step_dt)
            return out
        self.reward_manager.compute = _wrapped_reward_compute

        _obs_compute = self.observation_manager.compute
        def _wrapped_obs_compute(update_history=False):
            out = _obs_compute(update_history=update_history)
            if not self._nan_diagnostic_done and isinstance(out, dict):
                for group_name, obs_tensor in out.items():
                    if isinstance(obs_tensor, torch.Tensor) and (torch.isnan(obs_tensor).any() or torch.isinf(obs_tensor).any()):
                        bad = (torch.isnan(obs_tensor) | torch.isinf(obs_tensor)).any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
                        if bad.numel() > 0:
                            self._nan_diagnostic_done = True
                            if bad.dim() == 0:
                                bad = bad.unsqueeze(0)
                            from allex_rl_dexblind.tasks.manager_based.dexblind_newton.utils.diagnose_nan import (
                        diagnose_nan_source,
                    )
                            diagnose_nan_source(self, bad[:10], step_dt=self.step_dt)
                        break
            return out
        self.observation_manager.compute = _wrapped_obs_compute

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

    def step(self, action: torch.Tensor):
        set_visualizer_env(self)
        if self._hard_contact_pending:
            _contact_params = (
                ("table", TABLE_CONTACT_KE, TABLE_CONTACT_KD),
                ("robot", ROBOT_CONTACT_KE, ROBOT_CONTACT_KD),
                ("hammer", HAMMER_CONTACT_KE, HAMMER_CONTACT_KD),
            )
            for asset_name, ke, kd in _contact_params:
                if asset_name in self.scene.articulations or asset_name in self.scene.rigid_objects:
                    set_shape_contact_stiffness(
                        self, None, SceneEntityCfg(asset_name), ke=ke, kd=kd,
                    )
            self._hard_contact_pending = False
        result = super().step(action)
        interval = getattr(
            self.cfg, "log_solver_convergence_interval", LOG_SOLVER_CONVERGENCE_INTERVAL
        )
        if interval > 0:
            env_step = self._sim_step_counter // self.cfg.decimation
            if env_step > 0:
                start_solver_conv_viewer()
                conv = NewtonManager.get_solver_convergence_steps()
                opt = NewtonManager._solver.mjw_model.opt
                max_iter = opt.iterations
                hit = conv["max"] == max_iter
                line = (
                    f"[Solver conv] env_step={env_step} | max={conv['max']} | "
                    f"mean={conv['mean']:.1f} | limit={max_iter}"
                )
                if env_step % interval == 0:
                    push_solver_conv_line(line + (" | hit_limit" if hit else ""), hit_limit=hit)
                elif hit:
                    push_solver_conv_line(line + " | hit_limit", hit_limit=True)
        return result

    # --------------------------------------------------------------------- #
    # Reset
    # --------------------------------------------------------------------- #

    # Articulation names whose root pose/velocity are set by reset events (do not overwrite in _reset_articulations).
    _RESET_ROOT_BY_EVENT = ("robot",)

    def _reset_idx(self, env_ids: Sequence[int]):
        self.curriculum_manager.compute(env_ids=env_ids)
        self.scene.reset(env_ids)

        env_ids_t = self._to_tensor(env_ids)
        self._reset_privileged_buffers(env_ids_t)
        self._reset_goal_to_default(env_ids_t)

        # Articulations first (default state), then events so reset_robot_joints etc. are not overwritten.
        self._reset_articulations(env_ids_t)
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(
                mode="reset", env_ids=env_ids_t, global_env_step_count=env_step_count
            )
        NewtonManager._solver.notify_model_changed(_NOTIFY_ALL)

        self.extras["log"] = {}
        for mgr_name in (
            "observation_manager", "action_manager", "reward_manager",
            "curriculum_manager", "command_manager", "event_manager",
            "termination_manager", "recorder_manager",
        ):
            self.extras["log"].update(getattr(self, mgr_name).reset(env_ids))

        self.episode_length_buf[env_ids] = 0

    def _reset_articulations(self, env_ids_t: torch.Tensor):
        for name, articulation in self.scene.articulations.items():
            skip_root = name in self._RESET_ROOT_BY_EVENT
            if not skip_root:
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
