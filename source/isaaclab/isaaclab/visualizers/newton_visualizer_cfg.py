# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton OpenGL Visualizer."""

from dataclasses import field

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class GoalMarkerCfg:
    """A static visual marker rendered by the Newton visualizer (no physics).

    If ``usd_path`` is set the mesh geometry is loaded from the USD file;
    otherwise a simple box with the given ``scale`` (half-extents) is drawn.
    """

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position (x, y, z) in world frame."""

    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion (w, x, y, z)."""

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Instance scale applied to the mesh (or half-extents when no USD)."""

    color: tuple[float, float, float] = (0.2, 0.85, 0.3)
    """RGB color [0, 1]."""

    usd_path: str | None = None
    """Optional path to a USD file. The first ``UsdGeom.Mesh`` prim found
    in the file is used as the visual geometry."""


@configclass
class NewtonVisualizerCfg(VisualizerCfg):
    """Configuration for Newton OpenGL visualizer.

    Lightweight OpenGL-based visualizer with real-time 3D rendering, interactive
    camera controls, and debug visualization (contacts, joints, springs, COM).

    Use ``font_scale`` to enlarge UI text and ``panel_initial_width`` for the left panel.
    The left panel is resizable at runtime; layout is saved in ``imgui.ini`` (run directory),
    section ``[Window][Newton Viewer v0.2.0]``.

    Requires: pyglet >= 2.1.6, imgui_bundle >= 1.92.0
    """

    visualizer_type: str = "newton"
    """Type identifier for Newton visualizer."""

    window_width: int = 3000
    """Window width in pixels."""

    window_height: int = 2000
    """Window height in pixels."""

    font_scale: float = 2.5
    """ImGui font scale factor (e.g. 1.2 = 20% larger). Applied to all UI text."""

    panel_initial_width: int = 500
    """Initial width of the left control panel in pixels. Applied once at startup each run; panel is resizable at runtime."""

    update_frequency: int = 1
    """Visualizer update frequency (updates every N frames). Lower = more responsive but slower training."""

    show_joints: bool = False
    """Show joint visualization."""

    show_contacts: bool = False
    """Show contact visualization."""

    show_springs: bool = False
    """Show spring visualization."""

    show_com: bool = False
    """Show center of mass visualization."""

    show_collision: bool = False
    """Show collision shapes (e.g. convex hulls) for collision-enabled bodies. Toggle in UI at runtime."""

    show_visual: bool = True
    """Show visual meshes (VISIBLE-flagged shapes). Disable to see only collision geometry. Toggle in UI at runtime."""

    enable_shadows: bool = True
    """Enable shadow rendering."""

    enable_sky: bool = True
    """Enable sky rendering."""

    enable_wireframe: bool = False
    """Enable wireframe rendering."""

    background_color: tuple[float, float, float] = (0.53, 0.81, 0.92)
    """Background/sky color RGB [0,1]."""

    ground_color: tuple[float, float, float] = (0.18, 0.20, 0.25)
    """Ground color RGB [0,1]."""

    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Light color RGB [0,1]."""

    goal_markers: list[GoalMarkerCfg] = field(default_factory=list)
    """Static visual markers rendered as boxes (no physics). Each marker is
    displayed at the given world-frame pose with the specified color.
    Useful for showing goal poses without adding bodies to the physics scene.
    """