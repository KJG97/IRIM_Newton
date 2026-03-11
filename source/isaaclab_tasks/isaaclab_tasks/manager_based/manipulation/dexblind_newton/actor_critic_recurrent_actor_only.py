# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCriticRecurrent variant: LSTM backbone on Actor only, MLP-only Critic.

Inherits from ActorCriticRecurrent but replaces the critic memory (RNN) with a
simple pass-through so the critic receives raw observations directly into its MLP.
This is useful for asymmetric actor-critic setups where the critic has access to
privileged information and does not benefit from temporal memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization, Memory
from rsl_rl.utils import unpad_trajectories


class ActorCriticRecurrentActorOnly(nn.Module):
    """Actor uses LSTM + MLP; Critic uses MLP only (no recurrence)."""

    is_recurrent = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrentActorOnly.__init__ got unexpected arguments, "
                "which will be ignored: " + str(list(kwargs.keys())),
            )
        super().__init__()

        self.obs_groups = obs_groups

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, (
                "ActorCriticRecurrentActorOnly only supports 1D observations."
            )
            num_actor_obs += obs[obs_group].shape[-1]

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, (
                "ActorCriticRecurrentActorOnly only supports 1D observations."
            )
            num_critic_obs += obs[obs_group].shape[-1]

        # --- Actor: LSTM backbone → MLP head ---
        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.actor = MLP(rnn_hidden_dim, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()
        print(f"Actor RNN: {self.memory_a}")
        print(f"Actor MLP: {self.actor}")

        # --- Critic: MLP only (no recurrence) ---
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()
        print(f"Critic MLP (no RNN): {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        self.memory_a.reset(dones)

    def forward(self):
        raise NotImplementedError

    def update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, masks=None, hidden_states=None):
        obs_actor = self.get_actor_obs(obs)
        obs_actor = self.actor_obs_normalizer(obs_actor)
        out_mem = self.memory_a(obs_actor, masks, hidden_states).squeeze(0)
        self.update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs_actor = self.get_actor_obs(obs)
        obs_actor = self.actor_obs_normalizer(obs_actor)
        out_mem = self.memory_a(obs_actor).squeeze(0)
        return self.actor(out_mem)

    def evaluate(self, obs, masks=None, hidden_states=None):
        obs_critic = self.get_critic_obs(obs)
        # In batch mode (PPO update), obs is padded trajectories [time, traj, dim].
        # Unpad to [time * envs_in_batch, dim] so the MLP gets 2D input.
        if masks is not None:
            obs_critic = unpad_trajectories(obs_critic, masks)
        obs_critic = self.critic_obs_normalizer(obs_critic)
        return self.critic(obs_critic)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        actor_hs = self.memory_a.hidden_states
        if actor_hs is None:
            return None, None
        if isinstance(actor_hs, tuple):
            critic_hs = tuple(torch.zeros_like(h) for h in actor_hs)
        else:
            critic_hs = torch.zeros_like(actor_hs)
        return actor_hs, critic_hs

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
