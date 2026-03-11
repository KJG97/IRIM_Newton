# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
LSTM = True
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

if LSTM:
    # Register the custom module into rsl_rl's on_policy_runner namespace
    # so that eval(class_name) inside OnPolicyRunner._setup_model can resolve it.
    from isaaclab_tasks.manager_based.manipulation.dexblind_newton.actor_critic_recurrent_actor_only import (
        ActorCriticRecurrentActorOnly,
    )
    import rsl_rl.runners.on_policy_runner as _opr_module

    _opr_module.ActorCriticRecurrentActorOnly = ActorCriticRecurrentActorOnly

    @configclass
    class DexblindNewtonAllexPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = 16
        obs_groups = {
            "policy": ["policy", "proprio"],
            "critic": ["policy_critic", "privileged"],
        }
        max_iterations = 6000
        save_interval = 100
        experiment_name = "dexblind_newton_allex"
        logger = "wandb"
        wandb_project = "dexblind_newton_allex"
        # Actor: LSTM → MLP (dexblind-sized); Critic: smaller MLP (obs ~95 dims).
        # See docs/network_size_review.md for rationale.
        policy = RslRlPpoActorCriticRecurrentCfg(
            class_name="ActorCriticRecurrentActorOnly",
            init_noise_std=1.0,
            noise_std_type="log",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[256, 256, 128],
            activation="elu",
            rnn_type="lstm",
            rnn_hidden_dim=512,
            rnn_num_layers=1,
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
else:
    @configclass
    class DexblindNewtonAllexPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = 16
        obs_groups = {
            "policy": ["policy", "proprio"],
            "critic": ["policy_critic", "privileged"],
        }
        max_iterations = 6000
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
