# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Dexblind Newton + ALLEX (No-Left) lift task: same rewards as dexblind, Newton physics."""

import gymnasium as gym

from . import agents

_ENV_CLASS = "isaaclab_tasks.manager_based.manipulation.dexblind_newton.dexblind_newton_env:DexblindNewtonEnv"

gym.register(
    id="Isaac-Dexblind-Newton-Allex-Lift-v0",
    entry_point=_ENV_CLASS,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexblind_newton_allex_env_cfg:DexblindNewtonAllexLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexblindNewtonAllexPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexblind-Newton-Allex-Lift-Play-v0",
    entry_point=_ENV_CLASS,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexblind_newton_allex_env_cfg:DexblindNewtonAllexLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexblindNewtonAllexPPORunnerCfg",
    },
)
