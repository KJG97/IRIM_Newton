# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for dexblind_newton: reference_weight annealing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reference_weight_annealing_fn(
    env: ManagerBasedRLEnv,
    env_ids,
    data: float,
    start_step: int,
    end_step: int,
    initial_weight: float = 1.0,
    final_weight: float = 0.0,
    num_steps_per_env: int | None = None,
) -> float:
    """Linearly anneal ``reference_weight`` from *initial_weight* to *final_weight*.

    When ``num_steps_per_env`` is provided, ``start_step`` / ``end_step`` are
    interpreted as **iteration numbers** (matching RSL-RL logging) and converted
    to env steps internally via ``step = iteration * num_steps_per_env``.

    Returns ``modify_env_param.NO_CHANGE`` before ``start_step`` to avoid
    unnecessary setter overhead.
    """
    current_step = env.common_step_counter

    if num_steps_per_env is not None:
        start_actual = start_step * num_steps_per_env
        end_actual = end_step * num_steps_per_env
    else:
        start_actual = start_step
        end_actual = end_step

    if current_step < start_actual:
        _log_curriculum_value(env, initial_weight, final_weight, initial_weight)
        return mdp.modify_env_param.NO_CHANGE

    if current_step >= end_actual:
        frac = 1.0
    else:
        frac = float(current_step - start_actual) / float(end_actual - start_actual)

    weight = initial_weight + frac * (final_weight - initial_weight)
    _log_curriculum_value(env, initial_weight, final_weight, weight)
    return weight


def randomize_scale_annealing_fn(
    env: ManagerBasedRLEnv,
    env_ids,
    data: float,
    start_step: int,
    end_step: int,
    initial_scale: float = 0.0,
    final_scale: float = 1.0,
    num_steps_per_env: int | None = None,
) -> float:
    """Linearly anneal ``randomize_scale`` from *initial_scale* to *final_scale*.

    Same scheduling logic as :func:`reference_weight_annealing_fn`.
    """
    current_step = env.common_step_counter

    if num_steps_per_env is not None:
        start_actual = start_step * num_steps_per_env
        end_actual = end_step * num_steps_per_env
    else:
        start_actual = start_step
        end_actual = end_step

    if current_step < start_actual:
        scale = initial_scale
    elif current_step >= end_actual:
        scale = final_scale
    else:
        frac = float(current_step - start_actual) / float(end_actual - start_actual)
        scale = initial_scale + frac * (final_scale - initial_scale)

    _log_value(env, "Curriculum/randomize_scale", scale)
    return scale


def reward_weight_annealing_fn(
    env: ManagerBasedRLEnv,
    env_ids,
    data: float,
    start_step: int,
    end_step: int,
    initial_weight: float = 10.0,
    final_weight: float = 1.0,
    num_steps_per_env: int | None = None,
    log_key: str = "Curriculum/reward_weight",
) -> float:
    """Linearly anneal a reward term weight from *initial_weight* to *final_weight*."""
    current_step = env.common_step_counter

    if num_steps_per_env is not None:
        start_actual = start_step * num_steps_per_env
        end_actual = end_step * num_steps_per_env
    else:
        start_actual = start_step
        end_actual = end_step

    if current_step < start_actual:
        w = initial_weight
    elif current_step >= end_actual:
        w = final_weight
    else:
        frac = float(current_step - start_actual) / float(end_actual - start_actual)
        w = initial_weight + frac * (final_weight - initial_weight)

    _log_value(env, log_key, w)
    return w


def _log_value(env: ManagerBasedRLEnv, key: str, value: float):
    """Write a scalar into extras for wandb/tensorboard."""
    if not hasattr(env, "extras") or env.extras is None:
        return
    log = env.extras.get("log", {})
    log[key] = value
    env.extras["log"] = log


def _log_curriculum_value(env: ManagerBasedRLEnv, initial: float, final: float, value: float):
    """Write curriculum progress into extras for wandb/tensorboard."""
    _log_value(env, "Curriculum/reference_weight", value)
