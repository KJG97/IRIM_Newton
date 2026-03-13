# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""NaN 원인 추적: 어떤 observation/reward term 또는 raw state에서 NaN이 발생하는지 진단."""

from __future__ import annotations

import torch
import warp as wp


def diagnose_nan_source(env, bad_env_ids: list[int] | torch.Tensor, step_dt: float = 0.0):
    """NaN이 발생한 env에 대해 1) raw state, 2) observation term별, 3) reward term별로 검사해 범인 출력.

    사용: 학습 중 Bad env ids가 찍힌 뒤, 동일 env로 한 스텝 재현 후 이 함수 호출.
    """
    if isinstance(bad_env_ids, torch.Tensor):
        bad_env_ids = bad_env_ids.cpu().tolist()
    if isinstance(bad_env_ids, (int, float)):
        bad_env_ids = [int(bad_env_ids)]
    if not bad_env_ids:
        return

    dev = env.device
    n = env.num_envs
    bad = torch.tensor(bad_env_ids, device=dev, dtype=torch.long)

    print("\n" + "=" * 60)
    print("[diagnose_nan] Raw physics state (Newton body_q / body_qd)")
    print("=" * 60)

    from allex_rl_dexblind.tasks.manager_based.dexblind_newton.mdp.utils import (
        _get_body_indices,
    )
    from isaaclab_newton.physics import NewtonManager

    state = NewtonManager._state_0
    body_q = wp.to_torch(state.body_q)
    body_qd = wp.to_torch(state.body_qd)

    for suffix in ("hammer", "Origin_Body", "Right_Hand_base"):
        idx = _get_body_indices(suffix, n, dev)
        for eid in bad_env_ids[:5]:
            b = int(idx[eid].item())
            if b < 0:
                print(f"  {suffix} env {eid}: body index = -1 (not found)")
                continue
            pos = body_q[b, :3].cpu().tolist()
            quat = body_q[b, 3:7].cpu().tolist()
            lin = body_qd[b, :3].cpu().tolist()
            ang = body_qd[b, 3:6].cpu().tolist()
            has_nan = any(torch.isnan(body_q[b]).cpu().tolist()) or any(torch.isnan(body_qd[b]).cpu().tolist())
            nan_tag = " [NaN]" if has_nan else ""
            print(f"  {suffix} env {eid} (body_id={b}){nan_tag}: pos={pos} quat={quat}")
            print(f"    lin_vel={lin} ang_vel={ang}")

    print("\n" + "-" * 60)
    print("[diagnose_nan] Asset default state (reset 시 참조하는 값)")
    print("-" * 60)

    for asset_name in ("hammer", "robot", "table"):
        if asset_name not in env.scene.rigid_objects and asset_name not in env.scene.articulations:
            continue
        asset = env.scene[asset_name]
        if hasattr(asset.data, "default_root_pose"):
            pose = wp.to_torch(asset.data.default_root_pose)[bad]
            vel = wp.to_torch(asset.data.default_root_vel)[bad] if hasattr(asset.data, "default_root_vel") else None
            for i, eid in enumerate(bad_env_ids[:5]):
                p = pose[i].cpu().tolist()
                has_nan = torch.isnan(pose[i]).any().item()
                print(f"  {asset_name} env {eid} default_pose: pos={p[:3]} quat={p[3:7]} {'[NaN]' if has_nan else ''}")
                if vel is not None:
                    v = vel[i].cpu().tolist()
                    print(f"    default_vel: lin={v[:3]} ang={v[3:6]}")
        if hasattr(asset.data, "default_joint_pos") and getattr(asset.data.default_joint_pos, "shape", None):
            jpos = wp.to_torch(asset.data.default_joint_pos)[bad]
            for i, eid in enumerate(bad_env_ids[:5]):
                if torch.isnan(jpos[i]).any():
                    print(f"  {asset_name} env {eid} default_joint_pos: [NaN]")

    print("\n" + "-" * 60)
    print("[diagnose_nan] Observation terms (proprio 그룹 term별)")
    print("-" * 60)

    obs_manager = env.observation_manager
    if "proprio" in getattr(obs_manager, "_group_obs_term_names", {}):
        term_names = obs_manager._group_obs_term_names["proprio"]
        term_cfgs = obs_manager._group_obs_term_cfgs["proprio"]
        for term_name, term_cfg in zip(term_names, term_cfgs):
            try:
                obs = term_cfg.func(env, **term_cfg.params)
                if isinstance(obs, torch.Tensor):
                    bad_obs = obs[bad]
                    has_nan = torch.isnan(bad_obs).any().item() or torch.isinf(bad_obs).any().item()
                    if has_nan:
                        print(f"  [NaN/Inf] proprio.{term_name} shape={obs.shape}")
                    else:
                        print(f"  [ok] proprio.{term_name} shape={obs.shape}")
            except Exception as e:
                print(f"  [ERR] proprio.{term_name}: {e}")

    print("\n" + "-" * 60)
    print("[diagnose_nan] Reward terms (term별 raw value)")
    print("-" * 60)

    reward_manager = env.reward_manager
    term_names = getattr(reward_manager, "_term_names", [])
    term_cfgs = getattr(reward_manager, "_term_cfgs", [])
    for name, term_cfg in zip(term_names, term_cfgs):
        if term_cfg.weight == 0.0:
            continue
        try:
            value = term_cfg.func(env, **term_cfg.params)
            if isinstance(value, torch.Tensor):
                bad_val = value[bad]
                has_nan = torch.isnan(bad_val).any().item() or torch.isinf(bad_val).any().item()
                if has_nan:
                    print(f"  [NaN/Inf] reward.{name} weight={term_cfg.weight}")
                else:
                    print(f"  [ok] reward.{name} weight={term_cfg.weight}")
            else:
                print(f"  [?] reward.{name} type={type(value)}")
        except Exception as e:
            print(f"  [ERR] reward.{name}: {e}")

    print("=" * 60 + "\n")
