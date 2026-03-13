# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""InteractiveScene that hides newton_replicate_kwargs from config iteration and uses dexblind_newton_replicate."""

from __future__ import annotations

import functools

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.terrains import TerrainImporterCfg

_ASSET_CFG_TYPES = (
    TerrainImporterCfg,
    ArticulationCfg,
    RigidObjectCfg,
    SensorBaseCfg,
    AssetBaseCfg,
)


class DexblindInteractiveScene(InteractiveScene):
    """Skips non-asset cfg fields (e.g. newton_replicate_kwargs) so parent does not raise."""

    def _add_entities_from_cfg(self):
        cfg = self.cfg
        if getattr(cfg, "newton_replicate_kwargs", None):
            from allex_rl_dexblind.tasks.manager_based.dexblind_newton.utils.cloner import (
                dexblind_newton_replicate,
            )
            self.cloner_cfg.physics_clone_fn = functools.partial(
                dexblind_newton_replicate, **cfg.newton_replicate_kwargs
            )
        saved = {}
        for k in list(cfg.__dict__.keys()):
            if k in InteractiveSceneCfg.__dataclass_fields__:
                continue
            val = cfg.__dict__[k]
            if val is None or isinstance(val, _ASSET_CFG_TYPES):
                continue
            saved[k] = cfg.__dict__.pop(k)
        try:
            super()._add_entities_from_cfg()
        finally:
            cfg.__dict__.update(saved)
