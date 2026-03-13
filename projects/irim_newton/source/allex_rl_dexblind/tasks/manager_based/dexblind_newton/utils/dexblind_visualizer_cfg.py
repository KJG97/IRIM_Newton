# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Dexblind-Newton 전용 Newton visualizer 설정 확장.

원본 isaaclab 코드는 수정하지 않고, GoalMarkerCfg 및 확장 필드를 로컬에서만 정의·관리합니다.
create_visualizer() 오버라이드로 이 cfg가 DexblindNewtonVisualizer에 전달되도록 합니다.
"""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils import configclass
from isaaclab.visualizers import NewtonVisualizerCfg


@configclass
class GoalMarkerCfg:
    """정적 목표 마커 설정 (Newton visualizer, 물리 미참여)."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    color: tuple[float, float, float] = (0.2, 0.85, 0.3)
    usd_path: str | None = None


@configclass
class DexblindNewtonVisualizerCfg(NewtonVisualizerCfg):
    """Newton visualizer 설정의 dexblind_newton 확장. create_visualizer()로 DexblindNewtonVisualizer가 생성됨."""

    goal_markers: list[GoalMarkerCfg] = field(default_factory=list)
    show_collision: bool = False
    show_contacts: bool = False  # 접촉 법선을 녹색 선으로 표시 (collision_filter_parent 등 확인용)
    show_visual: bool = True
    font_scale: float = 2.5
    panel_initial_width: int = 500
    debug_panel_width: int = 800
    debug_panel_height: int = 1000
    camera_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.45)
    background_color: tuple[float, float, float] = (0.53, 0.81, 0.92)
    ground_color: tuple[float, float, float] = (0.18, 0.20, 0.25)

    def create_visualizer(self):
        """이 cfg를 적용하는 DexblindNewtonVisualizer 인스턴스를 반환 (upstream 레지스트리 우회)."""
        from allex_rl_dexblind.tasks.manager_based.dexblind_newton.utils.dexblind_visualizer import (
            DexblindNewtonVisualizer,
        )
        return DexblindNewtonVisualizer(self)
