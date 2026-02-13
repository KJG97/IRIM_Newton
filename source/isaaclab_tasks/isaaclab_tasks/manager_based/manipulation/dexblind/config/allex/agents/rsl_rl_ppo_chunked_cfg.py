# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Chunked Trajectory Action 환경용 PPO 설정.

일반 Residual RL과 분리된 로그/wandb 저장을 위해 별도의 설정 파일 사용.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DexblindAllexChunkedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Chunked Trajectory Action 환경용 PPO Runner 설정.
    
    로그 및 wandb가 일반 환경과 분리되어 저장됨:
    - experiment_name: "dexblind_allex_chunked"
    - wandb_project: "dexblind_chunked"
    """
    num_steps_per_env = 32
    obs_groups = {"policy": ["policy", "proprio"], "critic": ["policy", "proprio"]}
    max_iterations = 15000
    save_interval = 100
    # Chunked 환경용 별도 이름
    experiment_name = "dexblind_allex_chunked"
    logger = "wandb"
    wandb_project = "dexblind_chunked"
    # 안정적인 학습: 탐색 과다 방지
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0, 
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # 0.02→0.005: 탐색 줄이고 exploitation 증가!
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,  # 원래대로
        schedule="adaptive",
        gamma=0.99,  # 원래대로
        lam=0.95,
        desired_kl=0.01,  # 원래대로
        max_grad_norm=1.0,
    )

