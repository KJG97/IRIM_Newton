# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
LSTM = False
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

if LSTM:
    @configclass
    class DexblindNewtonAllexPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = 16
        obs_groups = {
            "policy": ["policy", "proprio"],
            "critic": ["policy", "privileged"],
        }
        max_iterations = 10000
        save_interval = 100
        experiment_name = "dexblind_newton_allex"
        logger = "wandb"
        wandb_project = "dexblind_newton_allex"
        # Actor: LSTM[1024] backbone + MLP [1024, 512, 256, 128]; Critic: MLP [1024, 512, 256, 128] only
        policy = RslRlPpoActorCriticRecurrentCfg(
            class_name="ActorCriticRecurrent",
            init_noise_std=1.0,
            noise_std_type="log",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[1024, 512, 256, 128],
            critic_hidden_dims=[1024, 512, 256, 128],
            activation="elu",
            rnn_type="lstm",
            rnn_hidden_dim=1024,
            rnn_num_layers=1,
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=5e-3,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.016,
            max_grad_norm=1.0,
        )
else:
    @configclass
    class DexblindNewtonAllexPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = 16
        obs_groups = {
            "policy": ["policy", "proprio"],
            "critic": ["policy", "proprio"],
        }
        max_iterations = 10000
        save_interval = 100
        experiment_name = "dexblind_newton_allex"
        logger = "wandb"
        wandb_project = "dexblind_newton_allex"
        policy = RslRlPpoActorCriticCfg(
            class_name="ActorCritic",
            init_noise_std=1.0,
            noise_std_type="log",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.016,
            max_grad_norm=1.0,
        )