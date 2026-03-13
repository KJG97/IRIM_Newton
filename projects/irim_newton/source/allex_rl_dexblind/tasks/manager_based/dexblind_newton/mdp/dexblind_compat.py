# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility layer: dexblind MDP symbols migrated from isaaclab_tasks for project self-containment.

Used when upstream isaaclab_tasks no longer contains dexblind (reverted to upstream).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# reference_joint_pos (from dexblind/mdp/observations.py)
# -----------------------------------------------------------------------------


def reference_joint_pos(
    env: ManagerBasedRLEnv,
    command_name: str = "reference_trajectory",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """참조 궤적의 관절 목표 위치 (num_envs, num_joints). joint_names 순서."""
    try:
        ref = env.command_manager.get_command(command_name)
        term = env.command_manager.get_term(command_name)
        names = joint_names if joint_names is not None else getattr(asset_cfg, "joint_names", None)
        if names is not None and hasattr(term, "joint_names") and list(term.joint_names) != list(names):
            traj_names = term.joint_names
            idx = [traj_names.index(n) for n in names]
            ref = ref[:, idx]
    except (KeyError, AttributeError):
        ref = torch.zeros(env.num_envs, len(joint_names) if joint_names else 18, device=env.device)
    return ref


# -----------------------------------------------------------------------------
# step_based_interpolate_fn, _recurse, modify_term_cfg_with_logging (from dexblind/mdp/curriculums.py)
# -----------------------------------------------------------------------------


def _recurse(iv_elem, fv_elem, data_elem, frac):
    if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
        return type(data_elem)(_recurse(iv_e, fv_e, d_e, frac) for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem))
    new_val = frac * (fv_elem - iv_elem) + iv_elem
    if isinstance(data_elem, int):
        return int(new_val)
    return float(new_val)


def step_based_interpolate_fn(
    env: ManagerBasedRLEnv,
    env_id,
    data,
    initial_value,
    final_value,
    start_step: int,
    end_step: int,
    num_steps_per_env: int | None = None,
):
    """학습 스텝 기반 초기값에서 최종값으로의 보간 함수."""
    current_step = env.common_step_counter
    if num_steps_per_env is not None:
        nstep = num_steps_per_env
        start_step_actual = start_step * nstep
        end_step_actual = end_step * nstep
    else:
        start_step_actual = start_step
        end_step_actual = end_step
    if current_step < start_step_actual:
        return mdp.modify_env_param.NO_CHANGE
    if current_step >= end_step_actual:
        frac = 1.0
    else:
        frac = float(current_step - start_step_actual) / float(end_step_actual - start_step_actual)
        frac = max(0.0, min(1.0, frac))
    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)
    return _recurse(initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac)


class modify_term_cfg_with_logging(mdp.modify_term_cfg):
    """로깅을 지원하는 modify_term_cfg 래퍼 (dexblind 호환)."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        address: str,
        modify_fn: callable,
        modify_params: dict | None = None,
    ):
        modify_params = {} if modify_params is None else modify_params
        if not self._get_fn:
            self._get_fn, self._set_fn = self._process_accessors(self._env, self._address)
        data = self._get_fn()
        new_val = modify_fn(self._env, env_ids, data, **modify_params)
        if new_val is not self.NO_CHANGE:
            self._set_fn(new_val)
            if callable(new_val):
                func_name = getattr(new_val, "__name__", "unknown")
                return 1.0 if "zero" in func_name.lower() else 0.0
            if isinstance(new_val, (list, tuple)) and len(new_val) > 0:
                if all(isinstance(x, (int, float)) for x in new_val):
                    return max(abs(x) for x in new_val)
            return new_val
        if callable(data):
            func_name = getattr(data, "__name__", "unknown")
            return 1.0 if "zero" in func_name.lower() else 0.0
        if isinstance(data, (list, tuple)) and len(data) > 0:
            if all(isinstance(x, (int, float)) for x in data):
                return max(abs(x) for x in data)
        return data
