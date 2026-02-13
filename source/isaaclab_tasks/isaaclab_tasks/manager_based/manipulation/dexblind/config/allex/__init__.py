# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dextra Kuka Allegro environments.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Dexblind Lift Environments
gym.register(
    id="Isaac-Dexblind-Allex-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexblind_allex_env_cfg:DexblindAllexLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexblindAllexPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Dexblind-Allex-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexblind_allex_env_cfg:DexblindAllexLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexblindAllexPPORunnerCfg",
    },
)


# ============================================================================
# Chunked Trajectory Action Environments (별도 파일에서 로드)
# PPO가 궤적 인덱스와 청크 길이를 직접 선택하는 새로운 학습 방식
# Action space: 19 (1 index_offset + 18 residual joints)
# Temporal Ensemble (k=10)으로 부드러운 전환
#
# 로그 분리: experiment_name="dexblind_allex_chunked", wandb_project="dexblind_chunked"
# ============================================================================

gym.register(
    id="Isaac-Dexblind-Allex-Chunked-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexblind_allex_chunked_env_cfg:DexblindAllexChunkedLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_chunked_cfg:DexblindAllexChunkedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexblind-Allex-Chunked-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexblind_allex_chunked_env_cfg:DexblindAllexChunkedLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_chunked_cfg:DexblindAllexChunkedPPORunnerCfg",
    },
)
