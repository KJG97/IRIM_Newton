# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reference trajectory command for dexblind_newton (resolves path to dexblind_newton/data or dexblind/data)."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import reference_trajectory_command_cfg as ref_cmd_cfgs

logger = logging.getLogger(__name__)

EXCLUDED_ROLL_JOINTS = (
    "R_Index_Roll_Joint",
    "R_Middle_Roll_Joint",
    "R_Ring_Roll_Joint",
    "R_Little_Roll_Joint",
)


def _resolve_trajectory_path(trajectory_file: str) -> str:
    """Resolve relative path: dexblind_newton/data first, then dexblind/data."""
    if os.path.isabs(trajectory_file):
        return trajectory_file
    # dexblind_newton/mdp/commands -> dexblind_newton
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(base_dir, "data", trajectory_file)
    if os.path.isfile(path):
        return path
    # Fallback: sibling dexblind/data
    sibling_dexblind = os.path.join(os.path.dirname(base_dir), "dexblind", "data", trajectory_file)
    if os.path.isfile(sibling_dexblind):
        return sibling_dexblind
    return path


def _load_and_filter_trajectory(path: str, device: torch.device):
    data = np.load(path)
    times = torch.tensor(data["times"], dtype=torch.float32, device=device)
    positions = torch.tensor(data["positions"], dtype=torch.float32, device=device)
    names_raw = (
        data["joint_names"].tolist()
        if isinstance(data["joint_names"], np.ndarray)
        else list(data["joint_names"])
    )
    sample_rate = float(data["sample_rate"]) if "sample_rate" in data else 50.0
    keep = [i for i, name in enumerate(names_raw) if name not in EXCLUDED_ROLL_JOINTS]
    positions = positions[:, keep]
    joint_names = [names_raw[i] for i in keep]
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    return times, positions, joint_names, duration, sample_rate


class ReferenceTrajectoryCommand(CommandTerm):
    cfg: ref_cmd_cfgs.ReferenceTrajectoryCommandCfg

    def __init__(self, cfg: ref_cmd_cfgs.ReferenceTrajectoryCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        path = _resolve_trajectory_path(cfg.trajectory_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        self.times, self.positions, self.joint_names, self.duration, self.sample_rate = (
            _load_and_filter_trajectory(path, self.device)
        )
        self.num_frames, self.num_joints = self.positions.shape
        self.loop = cfg.loop
        self.loop_start_time = cfg.loop_start_time
        if self.loop_start_time is not None:
            assert 0.0 <= self.loop_start_time < self.duration
        self.playback_time = torch.zeros(self.num_envs, device=self.device)
        self.command_buffer = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self._last_dt = 1.0 / self.sample_rate
        self._initial_playback_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["trajectory_progress"] = torch.zeros(self.num_envs, device=self.device)
        logger.info(
            "Reference trajectory loaded %s: duration=%.2fs, frames=%s, joints=%s (loop=%s)",
            path, self.duration, self.num_frames, self.num_joints, self.loop,
        )

    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer

    def compute(self, dt: float):
        self._last_dt = dt
        super().compute(dt)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.playback_time[ids] = 0.0
        self._initial_playback_done[ids] = False
        self._sample_trajectory()
        return extras

    def _update_metrics(self):
        if self.duration > 0.0:
            self.metrics["trajectory_progress"] = torch.clamp(
                self.playback_time / self.duration, 0.0, 1.0
            )
        else:
            self.metrics["trajectory_progress"] = torch.zeros_like(self.playback_time)

    def _resample_command(self, env_ids: Sequence[int]):
        self.playback_time[env_ids] = 0.0
        self._initial_playback_done[env_ids] = False

    def _update_command(self):
        self.playback_time += self._last_dt * self.cfg.playback_speed
        if self.loop:
            if self.loop_start_time is not None:
                just_done = ~self._initial_playback_done & (self.playback_time >= self.duration)
                if just_done.any():
                    self._initial_playback_done[just_done] = True
                    self.playback_time[just_done] = self.loop_start_time
                loop_len = self.duration - self.loop_start_time + 1e-6
                wrap = self._initial_playback_done & (self.playback_time >= self.duration)
                if wrap.any():
                    self.playback_time[wrap] = self.loop_start_time + torch.fmod(
                        self.playback_time[wrap] - self.loop_start_time, loop_len
                    )
                still_initial = ~self._initial_playback_done
                if still_initial.any():
                    self.playback_time[still_initial] = torch.clamp(
                        self.playback_time[still_initial], 0.0, self.duration
                    )
            else:
                wrap = self.playback_time >= self.duration
                if wrap.any():
                    self.playback_time[wrap] = torch.fmod(
                        self.playback_time[wrap], self.duration + 1e-6
                    )
        else:
            self.playback_time = torch.clamp(self.playback_time, 0.0, self.duration)
        self._sample_trajectory()

    def _sample_trajectory(self):
        t = torch.clamp(self.playback_time, self.times[0], self.times[-1])
        idx = torch.clamp(
            torch.searchsorted(self.times, t, right=False), 0, self.num_frames - 2
        )
        t0, t1 = self.times[idx], self.times[idx + 1]
        dt = t1 - t0
        blend = torch.where(dt > 1e-6, (t - t0) / dt, torch.zeros_like(t)).unsqueeze(-1)
        self.command_buffer = (1.0 - blend) * self.positions[idx] + blend * self.positions[idx + 1]

    def get_joint_index(self, joint_name: str) -> int:
        if joint_name not in self.joint_names:
            raise ValueError(f"Joint '{joint_name}' not in trajectory: {self.joint_names}")
        return self.joint_names.index(joint_name)

    def get_joint_indices(self, joint_names: list[str]) -> list[int]:
        return [self.get_joint_index(n) for n in joint_names]
