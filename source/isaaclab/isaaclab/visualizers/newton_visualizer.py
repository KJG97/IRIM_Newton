# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

from __future__ import annotations

import contextlib
import os
import numpy as np
from typing import Any

import newton
import warp as wp
from newton.viewer import ViewerGL

from .newton_visualizer_cfg import GoalMarkerCfg, NewtonVisualizerCfg
from .visualizer import Visualizer

class NewtonViewerGL(ViewerGL):
    """Wrapper around Newton's ViewerGL with training/rendering pause controls.

    Adds two pause modes:
    - Training pause: Stops physics simulation, continues rendering
    - Rendering pause: Stops rendering updates, continues physics (SPACE key)
    """

    def __init__(self, *args, metadata: dict | None = None, update_frequency: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._paused_training = False
        self._paused_rendering = False
        self._metadata = metadata or {}
        self._fallback_draw_controls = False
        self._update_frequency = update_frequency
        self.show_visual: bool = True
        # Set by NewtonVisualizer from cfg after creation
        self._font_scale: float = 1.0
        self._panel_initial_width: int = 300

        try:
            self.register_ui_callback(self._render_training_controls, position="side")
        except AttributeError:
            self._fallback_draw_controls = True

    def is_training_paused(self) -> bool:
        return self._paused_training

    def is_rendering_paused(self) -> bool:
        return self._paused_rendering

    def _render_training_controls(self, imgui):
        imgui.separator()
        imgui.text("IsaacLab Controls")

        # Pause training/simulation button
        pause_label = "Resume Training" if self._paused_training else "Pause Training"
        if imgui.button(pause_label):
            self._paused_training = not self._paused_training

        # Pause rendering button
        rendering_label = "Resume Rendering" if self._paused_rendering else "Pause Rendering"
        if imgui.button(rendering_label):
            self._paused_rendering = not self._paused_rendering
            self._paused = self._paused_rendering  # Sync with parent class pause state

        # Visualizer update frequency control
        imgui.text("Visualizer Update Frequency")
        current_frequency = self._update_frequency
        changed, new_frequency = imgui.slider_int(
            "##VisualizerUpdateFreq", current_frequency, 1, 20, f"Every {current_frequency} frames"
        )
        if changed:
            self._update_frequency = new_frequency

        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Controls visualizer update frequency\nlower values -> more responsive visualizer but slower"
                " training\nhigher values -> less responsive visualizer but faster training"
            )

    def on_key_press(self, symbol, modifiers):
        if self.ui.is_capturing():
            return

        try:
            import pyglet  # noqa: PLC0415
        except Exception:
            return

        if symbol == pyglet.window.key.SPACE:
            self._paused_rendering = not self._paused_rendering
            self._paused = self._paused_rendering  # Sync with parent class pause state
            return

        super().on_key_press(symbol, modifiers)

    def _should_show_shape(self, flags: int, is_static: bool) -> bool:
        """Toggle visibility of collision-only and visual shapes independently."""
        is_collider = bool(flags & int(newton.ShapeFlags.COLLIDE_SHAPES))
        is_visible = bool(flags & int(newton.ShapeFlags.VISIBLE))
        if is_collider and is_visible:
            return self.show_collision or self.show_visual
        if is_collider and not is_visible:
            return self.show_collision
        if is_visible and not is_collider:
            return self.show_visual
        return super()._should_show_shape(flags, is_static)

    def _render_ui(self):
        if not self._fallback_draw_controls:
            return super()._render_ui()

        # Render base UI first
        super()._render_ui()

        # Then render a small floating window with training controls
        imgui = self.ui.imgui
        # Place near left panel but offset
        from contextlib import suppress

        with suppress(Exception):
            imgui.set_next_window_pos(imgui.ImVec2(320, 10))

        flags = 0
        if imgui.begin("Training Controls", flags=flags):
            self._render_training_controls(imgui)
        imgui.end()
        return None

    def _render_left_panel(self):
        """Override the left panel to remove the base pause checkbox."""
        import newton as nt

        imgui = self.ui.imgui

        # Use theme colors directly
        nav_highlight_color = self.ui.get_theme_color(imgui.Col_.nav_cursor, (1.0, 1.0, 1.0, 1.0))

        # Position and size: apply config once per run so panel_initial_width always takes effect
        io = self.ui.io
        imgui.set_next_window_pos(imgui.ImVec2(10, 10), imgui.Cond_.once.value)
        imgui.set_next_window_size(
            imgui.ImVec2(self._panel_initial_width, io.display_size[1] - 20),
            imgui.Cond_.once.value,
        )

        # Main control panel window - resizable so user can adjust panel size
        flags = 0

        if imgui.begin(f"Newton Viewer v{nt.__version__}", flags=flags):
            imgui.separator()

            header_flags = 0

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("IsaacLab Options"):
                # Render UI callbacks for side panel
                for callback in self._ui_callbacks["side"]:
                    callback(self.ui.imgui)

            # Model Information section
            if self.model is not None:
                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Model Information", flags=header_flags):
                    imgui.separator()
                    num_envs = self._metadata.get("num_envs", 0)
                    imgui.text(f"Environments: {num_envs}")
                    axis_names = ["X", "Y", "Z"]
                    imgui.text(f"Up Axis: {axis_names[self.model.up_axis]}")
                    gravity = wp.to_torch(self.model.gravity)[0]
                    gravity_text = f"Gravity: ({gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f})"
                    imgui.text(gravity_text)

                # Visualization Controls section
                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization", flags=header_flags):
                    imgui.separator()

                    # Joint visualization
                    show_joints = self.show_joints
                    changed, self.show_joints = imgui.checkbox("Show Joints", show_joints)

                    # Contact visualization
                    show_contacts = self.show_contacts
                    changed, self.show_contacts = imgui.checkbox("Show Contacts", show_contacts)

                    # Spring visualization
                    show_springs = self.show_springs
                    changed, self.show_springs = imgui.checkbox("Show Springs", show_springs)

                    # Center of mass visualization
                    show_com = self.show_com
                    changed, self.show_com = imgui.checkbox("Show Center of Mass", show_com)

                    # Collision mesh (convex hull) visualization
                    show_collision = self.show_collision
                    changed, self.show_collision = imgui.checkbox("Show Collision", show_collision)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Show collision shapes (e.g. convex hulls) for collision-enabled bodies")

                    # Visual mesh visualization
                    show_visual = self.show_visual
                    changed, self.show_visual = imgui.checkbox("Show Visual", show_visual)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Show visual meshes (VISIBLE-flagged shapes)")

            # Rendering Options section
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()

                # Sky rendering
                changed, self.renderer.draw_sky = imgui.checkbox("Sky", self.renderer.draw_sky)

                # Shadow rendering
                changed, self.renderer.draw_shadows = imgui.checkbox("Shadows", self.renderer.draw_shadows)

                # Wireframe mode
                changed, self.renderer.draw_wireframe = imgui.checkbox("Wireframe", self.renderer.draw_wireframe)

                # Light color
                changed, self.renderer._light_color = imgui.color_edit3("Light Color", self.renderer._light_color)
                # Sky color
                changed, self.renderer.sky_upper = imgui.color_edit3("Sky Color", self.renderer.sky_upper)
                # Ground color
                changed, self.renderer.sky_lower = imgui.color_edit3("Ground Color", self.renderer.sky_lower)

            # Camera Information section
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Camera"):
                imgui.separator()

                pos = self.camera.pos
                pos_text = f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                imgui.text(pos_text)
                imgui.text(f"FOV: {self.camera.fov:.1f}°")
                imgui.text(f"Yaw: {self.camera.yaw:.1f}°")
                imgui.text(f"Pitch: {self.camera.pitch:.1f}°")

                # Camera controls hint - update to reflect new controls
                imgui.separator()
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*nav_highlight_color))
                imgui.text("Controls:")
                imgui.pop_style_color()
                imgui.text("WASD - Forward/Left/Back/Right")
                imgui.text("QE - Down/Up")
                imgui.text("Left Click - Look around")
                imgui.text("Scroll - Zoom")
                imgui.text("Space - Pause/Resume Rendering")
                imgui.text("H - Toggle UI")
                imgui.text("ESC - Exit")

        imgui.end()
        return


class NewtonVisualizer(Visualizer):
    """Newton OpenGL visualizer for Isaac Lab.

    Lightweight OpenGL-based visualization with training/rendering pause controls.
    """

    def __init__(self, cfg: NewtonVisualizerCfg):
        super().__init__(cfg)
        self.cfg: NewtonVisualizerCfg = cfg
        self._viewer: NewtonViewerGL | None = None
        self._sim_time = 0.0
        self._step_counter = 0
        self._model = None
        self._state = None
        self._update_frequency = cfg.update_frequency
        self._scene_data_provider = None

    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        """Initialize visualizer with scene data."""
        if self._is_initialized:
            return

        # Import NewtonManager for metadata access
        from isaaclab.sim._impl.newton_manager import NewtonManager

        # Store scene data provider for accessing physics state
        if scene_data and "scene_data_provider" in scene_data:
            self._scene_data_provider = scene_data["scene_data_provider"]

        # Get Newton-specific data from scene data provider
        if self._scene_data_provider:
            self._model = self._scene_data_provider.get_model()
            self._state = self._scene_data_provider.get_state()
        else:
            # Fallback: direct access to NewtonManager (for backward compatibility)
            self._model = NewtonManager._model
            self._state = NewtonManager._state_0

        if self._model is None:
            raise RuntimeError("Newton visualizer requires Newton Model. Ensure Newton physics is initialized first.")

        # Build metadata from NewtonManager
        metadata = {
            "physics_backend": "newton",
            "num_envs": NewtonManager._num_envs if NewtonManager._num_envs is not None else 0,
            "gravity_vector": NewtonManager._gravity_vector,
            "clone_physics_only": NewtonManager._clone_physics_only,
        }

        # Create the viewer with metadata
        self._viewer = NewtonViewerGL(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            metadata=metadata,
            update_frequency=self.cfg.update_frequency,
        )

        # Set the model
        self._viewer.set_model(self._model)

        # Disable auto world spacing in Newton Viewer to display envs at actual world positions
        self._viewer.set_world_offsets((0.0, 0.0, 0.0))

        # Configure camera position and orientation (Z-up axis)
        self._viewer.camera.pos = wp.vec3(*self.cfg.camera_position)
        self._viewer.up_axis = 2  # Z-up

        # Calculate pitch and yaw from camera_position and camera_target
        cam_pos = np.array(self.cfg.camera_position, dtype=np.float32)
        cam_target = np.array(self.cfg.camera_target, dtype=np.float32)
        direction = cam_target - cam_pos

        # Calculate yaw and pitch for Z-up coordinate system
        # Yaw: rotation around Z axis (horizontal plane)
        yaw = np.degrees(np.arctan2(direction[1], direction[0]))
        # Pitch: elevation angle
        horizontal_dist = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        pitch = np.degrees(np.arctan2(direction[2], horizontal_dist))

        self._viewer.camera.yaw = float(yaw)
        self._viewer.camera.pitch = float(pitch)

        self._viewer.scaling = 1.0
        self._viewer._paused = False

        # Configure visualization options
        self._viewer.show_joints = self.cfg.show_joints
        self._viewer.show_contacts = self.cfg.show_contacts
        self._viewer.show_springs = self.cfg.show_springs
        self._viewer.show_com = self.cfg.show_com
        self._viewer.show_collision = self.cfg.show_collision
        self._viewer.show_visual = self.cfg.show_visual

        # Configure rendering options
        self._viewer.renderer.draw_shadows = self.cfg.enable_shadows
        self._viewer.renderer.draw_sky = self.cfg.enable_sky
        self._viewer.renderer.draw_wireframe = self.cfg.enable_wireframe

        # Configure colors
        self._viewer.renderer.sky_upper = self.cfg.background_color
        self._viewer.renderer.sky_lower = self.cfg.ground_color
        self._viewer.renderer._light_color = self.cfg.light_color

        # ImGui font scale and panel size (imgui_bundle uses style.font_scale_main)
        self._viewer._font_scale = self.cfg.font_scale
        self._viewer._panel_initial_width = self.cfg.panel_initial_width
        if self._viewer.ui and self._viewer.ui.is_available:
            self._viewer.ui.imgui.get_style().font_scale_main = self.cfg.font_scale

        # Compute env origins for goal markers (same grid as scene cloner)
        num_envs = metadata["num_envs"] or 1
        from isaaclab.cloner.cloner_utils import grid_transforms
        env_origins, _ = grid_transforms(num_envs, spacing=2.0)  # matches SceneCfg.env_spacing

        # Register goal-marker meshes and pre-build per-instance warp arrays
        self._goal_marker_data = self._register_goal_markers(
            self._viewer, self.cfg.goal_markers, env_origins.numpy(),
        )

        # Launch PySide6 debug panel alongside Newton viewer
        from .debug_panel import ensure_debug_panel

        self._debug_panel = ensure_debug_panel()

        self._is_initialized = True

    @staticmethod
    def _load_usd_mesh(usd_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Extract the first UsdGeom.Mesh from a USD file.

        Returns (points_Nx3, indices_flat, normals_Nx3_or_None).
        """
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        stage = Usd.Stage.Open(usd_path)
        mesh_prim = None
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh_prim = prim
                break
        if mesh_prim is None:
            raise RuntimeError(f"No UsdGeom.Mesh found in {usd_path}")

        mesh = UsdGeom.Mesh(mesh_prim)
        pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
        face_vertex_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        face_vertex_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

        # Triangulate (fan from first vertex of each face)
        tri_indices: list[int] = []
        offset = 0
        for nv in face_vertex_counts:
            for k in range(nv - 2):
                tri_indices.extend([
                    face_vertex_indices[offset],
                    face_vertex_indices[offset + k + 1],
                    face_vertex_indices[offset + k + 2],
                ])
            offset += nv
        indices = np.array(tri_indices, dtype=np.uint32)

        normals_attr = mesh.GetNormalsAttr().Get()
        normals = np.array(normals_attr, dtype=np.float32) if normals_attr else None

        return pts, indices, normals

    @staticmethod
    def _register_goal_markers(
        viewer,
        markers: list[GoalMarkerCfg],
        env_origins: np.ndarray,
    ) -> list[tuple[str, str, wp.array, wp.array, wp.array, wp.array]] | None:
        """Register mesh prototypes and build per-instance arrays.

        One instance per env is created for each marker, offset by the
        corresponding env origin so the marker appears at the correct
        relative position in every environment.

        Args:
            viewer: Newton viewer instance.
            markers: Goal marker configurations (local-frame pose).
            env_origins: (num_envs, 3) array of env world offsets.
        """
        if not markers:
            return None

        num_envs = len(env_origins)
        result = []
        for i, m in enumerate(markers):
            mesh_name = f"/goal_mesh_{i}"
            inst_name = f"/goal_marker_{i}"

            if m.usd_path is not None:
                pts, indices, normals = NewtonVisualizer._load_usd_mesh(m.usd_path)
                wp_pts = wp.array(pts, dtype=wp.vec3)
                wp_idx = wp.array(indices, dtype=wp.uint32)
                wp_norms = wp.array(normals, dtype=wp.vec3) if normals is not None else None
                viewer.log_mesh(mesh_name, wp_pts, wp_idx, normals=wp_norms, hidden=True)
            else:
                viewer.log_geo(
                    mesh_name,
                    newton.GeoType.BOX,
                    m.scale,
                    geo_thickness=0.0,
                    geo_is_solid=True,
                )

            xf_list = []
            for e in range(num_envs):
                ox, oy, oz = float(env_origins[e, 0]), float(env_origins[e, 1]), float(env_origins[e, 2])
                pos = (m.pos[0] + ox, m.pos[1] + oy, m.pos[2] + oz)
                xf_list.append(wp.transform(pos, m.rot))

            xforms = wp.array(xf_list, dtype=wp.transform)
            scales = wp.array([wp.vec3(*m.scale)] * num_envs, dtype=wp.vec3)
            colors = wp.array([wp.vec3(*m.color)] * num_envs, dtype=wp.vec3)
            materials = wp.array([wp.vec4(0.0, 0.5, 0.0, 0.0)] * num_envs, dtype=wp.vec4)

            result.append((inst_name, mesh_name, xforms, scales, colors, materials))
        return result

    def step(self, dt: float, state: Any | None = None) -> None:
        """Update visualizer for one step."""
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return

        self._sim_time += dt
        self._step_counter += 1

        # Fetch updated state from scene data provider
        if self._scene_data_provider:
            self._state = self._scene_data_provider.get_state()
        else:
            # Fallback: direct access to NewtonManager
            from isaaclab.sim._impl.newton_manager import NewtonManager

            self._state = NewtonManager._state_0

        # Pump PySide6 event loop and update debug panel data
        from .debug_panel import pump_qt_events

        pump_qt_events()
        if hasattr(self, "_debug_panel") and self._debug_panel is not None:
            self._debug_panel.tick()

        # Only update visualizer at the specified frequency
        update_frequency = self._viewer._update_frequency if self._viewer else self._update_frequency
        if self._step_counter % update_frequency != 0:
            return

        picking_on = (
            hasattr(self, "_debug_panel")
            and self._debug_panel is not None
            and self._debug_panel.picking_enabled
        )

        if picking_on and self._state is not None:
            picking = self._viewer.picking
            ke = self._debug_panel.pick_stiffness
            kd = self._debug_panel.pick_damping
            picking.pick_stiffness = ke
            picking.pick_damping = kd
            import numpy as _np
            ps = picking.pick_state.numpy()
            ps[6] = ke
            ps[7] = kd
            picking.pick_state = wp.array(ps, dtype=float, device=picking.model.device)
            self._viewer.apply_forces(self._state)

        with contextlib.suppress(Exception):
            if not self._viewer.is_paused():
                self._viewer.begin_frame(self._sim_time)
                if self._state is not None:
                    self._viewer.log_state(self._state)
                if self._goal_marker_data:
                    for inst_name, mesh_name, xforms, scales, colors, materials in self._goal_marker_data:
                        self._viewer.log_instances(
                            inst_name, mesh_name, xforms, scales, colors, materials,
                        )
                self._viewer.end_frame()
            else:
                self._viewer._update()

    def close(self) -> None:
        """Close visualizer and clean up resources."""
        if self._is_closed:
            return
        if hasattr(self, "_debug_panel") and self._debug_panel is not None:
            self._debug_panel.close()
            self._debug_panel = None
        if self._viewer is not None:
            self._viewer = None
        self._is_closed = True

    def is_running(self) -> bool:
        """Check if visualizer window is still open."""
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return False
        return self._viewer.is_running()

    def supports_markers(self) -> bool:
        """Newton visualizer does not have this feature yet."""
        return False

    def supports_live_plots(self) -> bool:
        """Newton visualizer does not have this feature yet."""
        return False

    def is_training_paused(self) -> bool:
        """Check if training is paused."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_training_paused()

    def is_rendering_paused(self) -> bool:
        """Check if rendering is paused."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_rendering_paused()
