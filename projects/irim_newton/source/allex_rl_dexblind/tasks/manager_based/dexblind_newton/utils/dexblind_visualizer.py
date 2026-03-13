# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Dexblind-Newton 전용 Visualizer: DexblindNewtonVisualizerCfg를 실제로 적용하는 서브클래스.

Upstream NewtonVisualizer/NewtonViewerGL는 수정하지 않고, 여기서만 확장합니다.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from isaaclab.visualizers.newton_visualizer import NewtonViewerGL, NewtonVisualizer

from allex_rl_dexblind.tasks.manager_based.dexblind_newton.utils.dexblind_visualizer_cfg import (
    DexblindNewtonVisualizerCfg,
    GoalMarkerCfg,
)

logger = logging.getLogger(__name__)

# Env ref for State Observer (Torque 등). set_visualizer_env(env)로 실제 ManagerBasedEnv 전달 시 applied_torque 사용 가능.
_dexblind_visualizer_instance: "DexblindNewtonVisualizer | None" = None


def set_visualizer_env(env) -> None:
    """State Observer(우하단 imgui)가 scene.articulations['robot'].data.applied_torque 등을 쓰도록 env 등록."""
    global _dexblind_visualizer_instance
    if _dexblind_visualizer_instance is not None and getattr(_dexblind_visualizer_instance, "_viewer", None) is not None:
        _dexblind_visualizer_instance._viewer._env_ref = env


if TYPE_CHECKING:
    from isaaclab.sim.scene_data_providers import SceneDataProvider


import math


# ObsState 그룹 표시 이름 (우하단 State Observer용)
OBS_GROUP_DISPLAY: dict[str, str] = {
    "policy": "PolicyCfg",
    "policy_critic": "PolicyCriticCfg",
    "proprio": "ProprioObsCfg",
    "privileged": "PrivilegedObsCfg",
}


def _obs_display_len(shape: tuple[int, ...]) -> int:
    """History 무시: 한 스텝 차원만 반환 (표시/플롯용)."""
    if not shape:
        return 0
    return int(shape[-1]) if len(shape) > 1 else int(np.prod(shape))


_OBS_PLOT_HISTORY_MAX = 400


def _get_joint_list_env0():
    """Return list of (joint_name, q_idx) for env 0, excluding FREE/FIXED. q_idx indexes into state.joint_q."""
    try:
        from isaaclab_newton.physics import NewtonManager
        import newton as nw
    except Exception:
        return []
    model = NewtonManager._model
    if model is None:
        return []
    joint_keys = model.joint_key
    joint_world = model.joint_world.numpy() if hasattr(model.joint_world, "numpy") else model.joint_world
    joint_q_start = model.joint_q_start.numpy()
    joint_type = model.joint_type.numpy()
    excluded = {0, 4, int(getattr(nw.JointType, "FIXED", 4))}
    out = []
    for j in range(model.joint_count):
        if int(joint_world[j]) != 0:
            continue
        if int(joint_type[j]) in excluded:
            continue
        q_idx = int(joint_q_start[j])
        name = joint_keys[j].split("/")[-1] if "/" in joint_keys[j] else joint_keys[j]
        out.append((name, q_idx))
    return out


def _get_joint_list_env0_qd():
    """Return list of (joint_name, qd_start) for env 0, excluding FREE/FIXED.
    qd_start indexes into state.joint_qd and control.joint_f (same DOF indexing as per Newton docs)."""
    try:
        from isaaclab_newton.physics import NewtonManager
        import newton as nw
    except Exception:
        return []
    model = NewtonManager._model
    if model is None:
        return []
    joint_keys = model.joint_key
    joint_world = model.joint_world.numpy() if hasattr(model.joint_world, "numpy") else model.joint_world
    joint_qd_start = model.joint_qd_start.numpy()
    joint_type = model.joint_type.numpy()
    excluded = {0, 4, int(getattr(nw.JointType, "FIXED", 4))}
    out = []
    for j in range(model.joint_count):
        if int(joint_world[j]) != 0:
            continue
        if int(joint_type[j]) in excluded:
            continue
        qd_start = int(joint_qd_start[j])
        name = joint_keys[j].split("/")[-1] if "/" in joint_keys[j] else joint_keys[j]
        out.append((name, qd_start))
    return out


def _get_joint_position_velocity(env_id: int = 0):
    """Return (positions_deg, velocities) for env 0: lists of (name, pos_deg) and (name, vel)."""
    try:
        from isaaclab_newton.physics import NewtonManager
    except Exception:
        return [], []
    state = NewtonManager._state_0
    model = NewtonManager._model
    if state is None or model is None:
        return [], []
    joints = _get_joint_list_env0()
    if not joints:
        return [], []
    joint_q = wp.to_torch(state.joint_q).cpu()
    joint_qd = getattr(state, "joint_qd", None)
    if joint_qd is not None:
        joint_qd = wp.to_torch(joint_qd).cpu()
    pos_rows = []
    vel_rows = []
    for name, q_idx in joints:
        if q_idx < joint_q.shape[0]:
            pos_rows.append((name, math.degrees(float(joint_q[q_idx].item()))))
        else:
            pos_rows.append((name, 0.0))
        if joint_qd is not None and q_idx < joint_qd.shape[0]:
            vel_rows.append((name, float(joint_qd[q_idx].item())))
        else:
            vel_rows.append((name, 0.0))
    return pos_rows, vel_rows


def _get_joint_drive_state_rows(env, env_id: int = 0):
    """Per-joint max torque, Kp, Kd (DriveState) for env 0 from Newton Model.
    Uses model.joint_effort_limit, model.joint_target_ke, model.joint_target_kd indexed by joint_qd_start."""
    try:
        from isaaclab_newton.physics import NewtonManager
    except Exception:
        return []
    model = NewtonManager._model
    if model is None:
        return []
    joints_qd = _get_joint_list_env0_qd()
    if not joints_qd:
        return []
    ke = getattr(model, "joint_target_ke", None)
    kd = getattr(model, "joint_target_kd", None)
    effort_limit = getattr(model, "joint_effort_limit", None)
    try:
        ke_t = wp.to_torch(ke).cpu() if ke is not None else None
        kd_t = wp.to_torch(kd).cpu() if kd is not None else None
        eff_t = wp.to_torch(effort_limit).cpu() if effort_limit is not None else None
    except Exception:
        return [(name, "N/A | N/A | N/A") for name, _ in joints_qd]

    def _at(tensor, idx: int) -> str:
        if tensor is None or tensor.numel() <= idx:
            return "N/A"
        return f"{float(tensor[idx]):.2f}"

    rows = []
    for name, qd_start in joints_qd:
        row = f"{_at(eff_t, qd_start)} | {_at(ke_t, qd_start)} | {_at(kd_t, qd_start)}"
        rows.append((name, row))
    return rows


def _link_mass_rows_from_model(model, body_indices: list[int]) -> list[tuple[str, str]]:
    """Newton model에서 주어진 body 인덱스들의 링크 이름과 질량(kg) 행 반환."""
    if not model or not body_indices:
        return []
    body_keys = getattr(model, "body_key", None)
    body_mass = getattr(model, "body_mass", None)
    if body_mass is None:
        return []
    try:
        mass_arr = body_mass.numpy() if hasattr(body_mass, "numpy") else body_mass
    except Exception:
        return []
    rows = []
    for b in body_indices:
        if b < 0 or b >= len(mass_arr):
            continue
        name = body_keys[b] if body_keys is not None and b < len(body_keys) else f"body_{b}"
        short = str(name).split("/")[-1] if "/" in str(name) else str(name)
        rows.append((short, f"{float(mass_arr[b]):.6f}"))
    return rows


def _link_names_and_mass_rows_from_env(env, env_id: int, short_name: str) -> tuple[list[str], list[tuple[str, str]]] | None:
    """env.scene.articulations에서 short_name과 매칭되는 articulation의 전체 링크 이름 목록과 (이름, 질량) 행 반환.
    Newton body_key가 루트만 매칭될 때 폴백으로 사용. 매칭 실패 시 None."""
    if env is None:
        return None
    arts = getattr(getattr(env, "scene", None), "articulations", None) or {}
    for art in arts.values():
        if art is None or not hasattr(art, "body_names"):
            continue
        names = art.body_names
        if not names:
            continue
        first_short = names[0].split("/")[-1] if "/" in names[0] else names[0]
        if first_short != short_name and short_name not in names[0]:
            continue
        body_mass = getattr(art.data, "body_mass", None)
        if body_mass is None:
            continue
        try:
            mass = wp.to_torch(body_mass).cpu()
        except Exception:
            continue
        n_bodies = mass.shape[-1] if mass.numel() > 0 else 0
        if n_bodies == 0:
            continue
        link_names = []
        rows = []
        for b in range(n_bodies):
            name = names[b] if b < len(names) else f"body_{b}"
            s = name.split("/")[-1] if "/" in name else name
            link_names.append(s)
            if mass.dim() >= 2 and mass.shape[0] > env_id:
                val = f"{float(mass[env_id, b]):.6f}"
            elif mass.dim() == 1:
                val = f"{float(mass[b]):.6f}"
            else:
                val = "N/A"
            rows.append((s, val))
        return (link_names, rows)
    return None


def _get_env_articulations_imgui(selected_env: int):
    """Return list of {key, short, art_idx, root_body, body_indices} for the given env (Newton model)."""
    try:
        from isaaclab_newton.physics import NewtonManager
    except Exception:
        return []
    model = NewtonManager._model
    if model is None:
        return []
    art_keys = model.articulation_key
    art_world = model.articulation_world.numpy() if hasattr(model.articulation_world, "numpy") else model.articulation_world
    body_world = model.body_world.numpy()
    body_keys = model.body_key
    results = []
    for a in range(model.articulation_count):
        if int(art_world[a]) != selected_env:
            continue
        art_key = art_keys[a]
        prefix = art_key.replace("_articulation", "")
        short = prefix.split("/")[-1]
        body_indices = [
            b for b in range(model.body_count)
            if int(body_world[b]) == selected_env
            and (body_keys[b].startswith(prefix) or body_keys[b].startswith(art_key))
        ]
        # 로봇이 Waist_Base 등으로 표시되지만 body_key는 /Robot/... 인 경우: 같은 env 내 /Robot/ 접두사도 포함
        if len(body_indices) <= 1 and short in ("Waist_Base", "Robot"):
            env_prefix = "/World/envs/env_{}".format(selected_env)
            robot_prefix = env_prefix + "/Robot"
            extra = [
                b for b in range(model.body_count)
                if int(body_world[b]) == selected_env and body_keys[b].startswith(robot_prefix)
            ]
            seen = set(body_indices)
            for b in extra:
                if b not in seen:
                    body_indices.append(b)
                    seen.add(b)
        body_indices = sorted(body_indices, key=lambda b: body_keys[b])
        root_body = body_indices[0] if body_indices else -1
        results.append({
            "key": art_key,
            "short": short,
            "art_idx": a,
            "root_body": root_body,
            "body_indices": body_indices,
        })
    return results


def _robot_link_mass_only_rows(env, eid: int, scene_key: str = "robot"):
    """로봇 articulation의 Link Mass (kg) 행만 반환 (EnvState용)."""
    if env is None:
        return []
    arts = getattr(getattr(env, "scene", None), "articulations", None) or {}
    robot = arts.get(scene_key)
    if robot is None or not hasattr(robot, "data"):
        return []
    rows = []
    body_mass = getattr(robot.data, "body_mass", None)
    body_names = getattr(robot, "body_names", None)
    if body_mass is not None and body_names is not None:
        mass = wp.to_torch(body_mass).cpu()
        n_bodies = mass.shape[-1] if mass.numel() > 0 else 0
        if n_bodies > 0:
            rows.append(("Link Mass (kg)", ""))
            for b in range(n_bodies):
                name = body_names[b] if b < len(body_names) else f"body_{b}"
                short = name.split("/")[-1] if "/" in name else name
                if mass.dim() >= 2 and mass.shape[0] > eid:
                    val = f"{float(mass[eid, b]):.6f}"
                elif mass.dim() == 1:
                    val = f"{float(mass[b]):.6f}"
                else:
                    val = "N/A"
                rows.append((short, val))
    return rows


def _robot_link_mass_and_joint_gains_rows(env, eid: int, scene_key: str = "robot"):
    """로봇 articulation이 있으면 (Link Mass, Joint Gains)용 (prop, val) 행 리스트 반환."""
    if env is None:
        return []
    arts = getattr(getattr(env, "scene", None), "articulations", None) or {}
    robot = arts.get(scene_key)
    if robot is None or not hasattr(robot, "data"):
        return []
    rows = []
    body_mass = getattr(robot.data, "body_mass", None)
    body_names = getattr(robot, "body_names", None)
    if body_mass is not None and body_names is not None:
        mass = wp.to_torch(body_mass).cpu()
        n_bodies = mass.shape[-1] if mass.numel() > 0 else 0
        if n_bodies > 0:
            rows.append(("", ""))
            rows.append(("Link Mass (kg)", ""))
            for b in range(n_bodies):
                name = body_names[b] if b < len(body_names) else f"body_{b}"
                short = name.split("/")[-1] if "/" in name else name
                if mass.dim() >= 2 and mass.shape[0] > eid:
                    val = f"{float(mass[eid, b]):.6f}"
                elif mass.dim() == 1:
                    val = f"{float(mass[b]):.6f}"
                else:
                    val = "N/A"
                rows.append((short, val))
    stiff = getattr(robot.data, "joint_stiffness", None)
    damp = getattr(robot.data, "joint_damping", None)
    effort_lim = getattr(robot.data, "joint_effort_limits", None)
    joint_names = getattr(robot, "joint_names", None)
    if joint_names is None:
        return rows
    n_j = len(joint_names)
    if n_j == 0:
        return rows
    stiff_t = wp.to_torch(stiff).cpu() if stiff is not None else None
    damp_t = wp.to_torch(damp).cpu() if damp is not None else None
    eff_t = wp.to_torch(effort_lim).cpu() if effort_lim is not None else None
    if stiff_t is not None and (stiff_t.numel() == 0 or stiff_t.shape[-1] < n_j):
        stiff_t = None
    if damp_t is not None and (damp_t.numel() == 0 or damp_t.shape[-1] < n_j):
        damp_t = None
    if eff_t is not None and (eff_t.numel() == 0 or eff_t.shape[-1] < n_j):
        eff_t = None

    def _at(tensor, env_i: int, j: int) -> str:
        if tensor is None:
            return "N/A"
        if tensor.dim() >= 2 and tensor.shape[0] > env_i and tensor.shape[1] > j:
            return f"{float(tensor[env_i, j]):.2f}"
        if tensor.dim() == 1 and tensor.shape[0] > j:
            return f"{float(tensor[j]):.2f}"
        return "N/A"

    rows.append(("", ""))
    rows.append(("Joint Gains", "Max tau | Kp | Kd"))
    for j in range(n_j):
        name = joint_names[j] if j < len(joint_names) else f"joint_{j}"
        short = name.split("/")[-1] if "/" in name else name
        rows.append((short, f"{_at(eff_t, eid, j)} | {_at(stiff_t, eid, j)} | {_at(damp_t, eid, j)}"))
    return rows


class DexblindNewtonViewerGL(NewtonViewerGL):
    """NewtonViewerGL 확장: show_collision, show_visual, 패널 너비/폰트 적용, 우하단 Env State 디버그 패널."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_collision = False
        self.show_visual = True
        self._dexblind_panel_width = 300
        self._dexblind_font_scale = 1.0
        self._env_ref = None
        self._debug_page = 0  # 0 = JointState, 1 = EnvState, 2 = ObsState
        self._debug_env_objects = []
        self._debug_selected_obj_idx = -1
        self._debug_selected_link_idx_in_list = -1
        self._debug_panel_w = 380
        self._debug_panel_h = 440
        self._show_debug_panel = False
        # ObsState (우하단)
        self._debug_obs_group = ""
        self._debug_obs_term_key = ""  # "group/term"
        self._debug_obs_history: dict[str, deque] = {}  # key -> deque of np.ndarray (display_len,)
        self._debug_obs_step = 0

    def log_contacts(self, contacts, state):
        """접촉 법선을 빨간색·굵은 선으로 그림 (기본 녹색 대신)."""
        if not self.show_contacts:
            self.log_lines("/contacts", None, None, None)
            return
        num_contacts = contacts.rigid_contact_count.numpy()[0]
        max_contacts = contacts.rigid_contact_max
        if self._contact_points0 is None or len(self._contact_points0) < max_contacts:
            self._contact_points0 = wp.array(np.zeros((max_contacts, 3)), dtype=wp.vec3, device=self.device)
            self._contact_points1 = wp.array(np.zeros((max_contacts, 3)), dtype=wp.vec3, device=self.device)
        if max_contacts > 0:
            from newton._src.viewer.kernels import compute_contact_lines
            wp.launch(
                kernel=compute_contact_lines,
                dim=max_contacts,
                inputs=[
                    state.body_q,
                    self.model.shape_body,
                    self.model.shape_world,
                    self.world_offsets,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    0.1,
                ],
                outputs=[self._contact_points0, self._contact_points1],
                device=self.device,
            )
        if num_contacts > 0:
            starts = self._contact_points0[:num_contacts]
            ends = self._contact_points1[:num_contacts]
        else:
            starts = wp.array([], dtype=wp.vec3, device=self.device)
            ends = wp.array([], dtype=wp.vec3, device=self.device)
        self.log_lines("/contacts", starts, ends, (1.0, 0.0, 0.0), width=0.5)

    def _should_show_shape(self, flags: int, is_static: bool) -> bool:
        try:
            import newton
            is_collider = bool(flags & int(newton.ShapeFlags.COLLIDE_SHAPES))
            is_visible = bool(flags & int(newton.ShapeFlags.VISIBLE))
            if is_collider and is_visible:
                return self.show_collision or self.show_visual
            if is_collider:
                return self.show_collision
            if is_visible:
                return self.show_visual
        except Exception:
            pass
        return True

    def _render_left_panel(self):
        import newton as nt
        imgui = self.ui.imgui
        nav = self.ui.get_theme_color(imgui.Col_.nav_cursor, (1.0, 1.0, 1.0, 1.0))
        io = self.ui.io
        imgui.set_next_window_pos(imgui.ImVec2(10, 10), imgui.Cond_.once.value)
        imgui.set_next_window_size(
            imgui.ImVec2(self._dexblind_panel_width, io.display_size[1] - 20),
            imgui.Cond_.once.value,
        )
        flags = 0
        if imgui.begin(f"Newton Viewer v{nt.__version__}", flags=flags):
            imgui.separator()
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("IsaacLab Options"):
                for cb in self._ui_callbacks["side"]:
                    cb(self.ui.imgui)
            if self.model is not None:
                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Model Information"):
                    imgui.separator()
                    imgui.text(f"Environments: {self._metadata.get('num_envs', 0)}")
                    axis_names = ["X", "Y", "Z"]
                    imgui.text(f"Up Axis: {axis_names[self.model.up_axis]}")
                    g = wp.to_torch(self.model.gravity)[0]
                    imgui.text(f"Gravity: ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f})")
                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization"):
                    imgui.separator()
                    _, self.show_joints = imgui.checkbox("Show Joints", self.show_joints)
                    _, self.show_contacts = imgui.checkbox("Show Contacts", self.show_contacts)
                    _, self.show_springs = imgui.checkbox("Show Springs", self.show_springs)
                    _, self.show_com = imgui.checkbox("Show Center of Mass", self.show_com)
                    _, self.show_collision = imgui.checkbox("Show Collision", self.show_collision)
                    _, self.show_visual = imgui.checkbox("Show Visual", self.show_visual)
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Debug"):
                imgui.separator()
                _, self._show_debug_panel = imgui.checkbox("State Observer (right-bottom)", self._show_debug_panel)
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()
                _, self.renderer.draw_sky = imgui.checkbox("Sky", self.renderer.draw_sky)
                _, self.renderer.draw_shadows = imgui.checkbox("Shadows", self.renderer.draw_shadows)
                _, self.renderer.draw_wireframe = imgui.checkbox("Wireframe", self.renderer.draw_wireframe)
                _, self.renderer._light_color = imgui.color_edit3("Light Color", self.renderer._light_color)
                _, self.renderer.sky_upper = imgui.color_edit3("Upper Sky", self.renderer.sky_upper)
                _, self.renderer.sky_lower = imgui.color_edit3("Lower Sky", self.renderer.sky_lower)
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Camera"):
                imgui.separator()
                pos = self.camera.pos
                imgui.text(f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                imgui.text(f"FOV: {self.camera.fov:.1f}°")
                imgui.separator()
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*nav))
                imgui.text("Controls:")
                imgui.pop_style_color()
                imgui.text("WASD/QE - Move | LClick - Look | Scroll - Zoom")
                imgui.text("Space - Pause | H - UI | ESC - Exit")
        imgui.end()
        self._render_debug_panel_right_bottom()
        return

    def _render_debug_panel_right_bottom(self):
        if not self._show_debug_panel:
            return
        imgui = self.ui.imgui
        io = self.ui.io
        w = self._debug_panel_w
        h = self._debug_panel_h
        x = io.display_size[0] - w - 10
        y = io.display_size[1] - h - 10
        imgui.set_next_window_pos(imgui.ImVec2(x, y), imgui.Cond_.once.value)
        imgui.set_next_window_size(imgui.ImVec2(w, h), imgui.Cond_.once.value)
        flags = 0
        if not imgui.begin("State Observer", flags=flags):
            imgui.end()
            return
        env_id = 0
        is_joint = self._debug_page == 0
        if is_joint:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.5, 0.2, 0.8))
        if imgui.button("JointState"):
            self._debug_page = 0
        if is_joint:
            imgui.pop_style_color()
        imgui.same_line()
        is_env = self._debug_page == 1
        if is_env:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.5, 0.2, 0.8))
        if imgui.button("EnvState"):
            self._debug_page = 1
        if is_env:
            imgui.pop_style_color()
        imgui.same_line()
        is_obs = self._debug_page == 2
        if is_obs:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.5, 0.2, 0.8))
        if imgui.button("ObsState"):
            self._debug_page = 2
        if is_obs:
            imgui.pop_style_color()
        imgui.separator()
        if self._debug_page == 0:
            self._render_joint_state_page(imgui, w, env_id)
        elif self._debug_page == 1:
            self._render_env_state_page(imgui, w, env_id)
        else:
            self._render_obs_state_page(imgui, w, env_id)
        imgui.end()

    def _render_joint_state_page(self, imgui, w: float, env_id: int):
        pos_rows, vel_rows = _get_joint_position_velocity(env_id)
        env = self._env_ref
        drive_rows = _get_joint_drive_state_rows(env, env_id)
        col2_x = max(450.0, w * 0.52)

        def draw_two_cols(name_str: str, val_str: str) -> None:
            imgui.text(name_str)
            imgui.same_line(col2_x)
            imgui.text(val_str)

        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("Position"):
            for name, val in pos_rows:
                draw_two_cols(name, f"{val:.3f} deg")
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("Velocity"):
            for name, val in vel_rows:
                draw_two_cols(name, f"{val:.4f}")
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("DriveState (max tau | Kp | Kd)"):
            if not drive_rows:
                imgui.text("(no data)")
            else:
                for name, val in drive_rows:
                    draw_two_cols(name, val)

    def _render_env_state_page(self, imgui, w: float, env_id: int):
        if not self._debug_env_objects:
            if self.model is not None:
                self._debug_env_objects = _get_env_articulations_imgui(env_id)
            if not self._debug_env_objects:
                imgui.text("(no objects)")
                return
        imgui.text("Object (env 0):")
        for i, obj in enumerate(self._debug_env_objects):
            imgui.push_id(f"obj_{i}")
            short = obj.get("short", "")
            selected = i == self._debug_selected_obj_idx
            if selected:
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.5, 0.2, 0.8))
            if imgui.button(short):
                self._debug_selected_obj_idx = i
                self._debug_selected_link_idx_in_list = -1
            if selected:
                imgui.pop_style_color()
            imgui.pop_id()
            if i < len(self._debug_env_objects) - 1:
                imgui.same_line()
        imgui.separator()
        if self._debug_selected_obj_idx < 0 or self._debug_selected_obj_idx >= len(self._debug_env_objects):
            return
        obj = self._debug_env_objects[self._debug_selected_obj_idx]
        short = obj.get("short", "")
        body_indices = obj.get("body_indices", [])
        col2_x = max(450.0, w * 0.52)

        # 1) env.scene.articulations에서 매칭되는 articulation이 있으면 전체 링크 사용 (Newton body_key가 루트만 나올 때 폴백)
        link_names = []
        rows = []
        env = self._env_ref
        env_link_data = _link_names_and_mass_rows_from_env(env, env_id, short) if env else None
        if env_link_data is not None:
            link_names, rows = env_link_data

        # 2) env에서 못 가져왔으면 Newton model 기준
        if not link_names and not rows and self.model is not None and body_indices:
            body_keys = self.model.body_key
            for b in body_indices:
                if b < len(body_keys):
                    name = body_keys[b]
                    link_names.append(name.split("/")[-1] if "/" in name else name)
            rows = _link_mass_rows_from_model(self.model, body_indices)

        if link_names and rows:
            n_links = len(link_names)
            if self._debug_selected_link_idx_in_list < 0 or self._debug_selected_link_idx_in_list >= n_links:
                self._debug_selected_link_idx_in_list = -1
            imgui.text(f"{short} - Links (click to select)")
            imgui.push_id("LinkButtons")
            for idx, name in enumerate(link_names):
                imgui.push_id(idx)
                is_selected = idx == self._debug_selected_link_idx_in_list
                if is_selected:
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.5, 0.2, 0.8))
                if imgui.button(name):
                    self._debug_selected_link_idx_in_list = idx
                if is_selected:
                    imgui.pop_style_color()
                imgui.pop_id()
            imgui.pop_id()
            if self._debug_selected_link_idx_in_list >= 0 and self._debug_selected_link_idx_in_list < len(rows):
                sel_prop, sel_val = rows[self._debug_selected_link_idx_in_list]
                imgui.separator()
                imgui.text(f"Mass (kg): {sel_prop} = {sel_val}")
            return
        imgui.text(f"Selected: {short}")
        if not link_names and not body_indices:
            imgui.text("(no bodies)")

    def _get_obs_term_current_values(self, env_id: int, group: str, term_name: str, shape: tuple[int, ...]) -> np.ndarray | None:
        """env observation_manager에서 해당 term의 마지막 타임스텝 값 반환 (history=1). 예외 시 None (시뮬 유지)."""
        try:
            env = self._env_ref
            if env is None or not hasattr(env, "observation_manager"):
                return None
            obs_mgr = env.observation_manager
            obs_buf = getattr(obs_mgr, "_obs_buffer", None)
            if obs_buf is None or group not in obs_buf:
                return None
            try:
                import torch
            except ImportError:
                return None
            data = obs_buf[group]
            display_len = _obs_display_len(shape)
            if isinstance(data, torch.Tensor):
                term_names = obs_mgr.active_terms.get(group, [])
                term_dims = obs_mgr.group_obs_term_dim.get(group, [])
                idx = 0
                for n, sh in zip(term_names, term_dims):
                    length = int(np.prod(sh))
                    if n == term_name:
                        vals = data[env_id, idx : idx + length].detach().cpu().numpy()
                        return vals[-display_len:] if len(vals) >= display_len else vals
                    idx += length
                return None
            if isinstance(data, dict) and term_name in data:
                t = data[term_name]
                if isinstance(t, torch.Tensor) and t.dim() >= 1:
                    flat = t[env_id].detach().cpu().flatten().numpy()
                    return flat[-display_len:] if len(flat) >= display_len else flat
        except Exception:
            pass
        return None

    def _render_obs_state_page(self, imgui, w: float, env_id: int) -> None:
        env = self._env_ref
        if env is None or not hasattr(env, "observation_manager"):
            imgui.text("(no env / observation manager)")
            return
        obs_mgr = env.observation_manager
        groups = list(obs_mgr.active_terms.keys())
        if not groups:
            imgui.text("(no observation groups)")
            return
        # 그룹 버튼
        imgui.text("Group:")
        for g in groups:
            label = OBS_GROUP_DISPLAY.get(g, g)
            imgui.push_id(f"obs_grp_{g}")
            sel = self._debug_obs_group == g
            if sel:
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.5, 0.2, 0.8))
            if imgui.button(label):
                self._debug_obs_group = g
                self._debug_obs_term_key = ""
            if sel:
                imgui.pop_style_color()
            imgui.pop_id()
            imgui.same_line()
        imgui.new_line()
        imgui.separator()
        if not self._debug_obs_group:
            return
        group = self._debug_obs_group
        term_names = obs_mgr.active_terms.get(group, [])
        term_dims = obs_mgr.group_obs_term_dim.get(group, [])
        if not term_names:
            imgui.text("(no terms)")
            return
        imgui.text("Term (click to plot):")
        for name, shape in zip(term_names, term_dims):
            display_len = _obs_display_len(shape)
            key = f"{group}/{name}"
            imgui.push_id(key)
            sel = self._debug_obs_term_key == key
            if sel:
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.5, 0.2, 0.8))
            if imgui.button(f"{name} [{display_len}]"):
                self._debug_obs_term_key = key
                if key not in self._debug_obs_history:
                    self._debug_obs_history[key] = deque(maxlen=_OBS_PLOT_HISTORY_MAX)
            if sel:
                imgui.pop_style_color()
            imgui.pop_id()
            imgui.same_line()
        imgui.new_line()
        if not self._debug_obs_term_key:
            return
        parts = self._debug_obs_term_key.split("/", 1)
        if len(parts) != 2:
            return
        g_name, term_name = parts
        term_names = obs_mgr.active_terms.get(g_name, [])
        term_dims = obs_mgr.group_obs_term_dim.get(g_name, [])
        shape = None
        for n, sh in zip(term_names, term_dims):
            if n == term_name:
                shape = sh
                break
        if shape is None:
            return
        try:
            vals = self._get_obs_term_current_values(env_id, g_name, term_name, shape)
            if vals is not None and len(vals) > 0:
                if self._debug_obs_term_key not in self._debug_obs_history:
                    self._debug_obs_history[self._debug_obs_term_key] = deque(maxlen=_OBS_PLOT_HISTORY_MAX)
                self._debug_obs_history[self._debug_obs_term_key].append(vals.copy())
                self._debug_obs_step += 1
        except Exception:
            pass
        imgui.separator()
        imgui.text(f"Plot: {term_name} (separate window)")
        self._render_obs_plot_window(imgui, self._debug_obs_term_key, term_name, _obs_display_len(shape))

    def _draw_obs_plot_fallback(
        self, imgui, history: deque, display_dim: int, plot_w: float, plot_h: float
    ) -> None:
        """ImPlot 없을 때 ImGui DrawList로 단순 시계열 선 그래프 그림."""
        try:
            imgui.text("(ImGui plot)")
            pad = 8
            region_w = max(100, plot_w - 2 * pad)
            region_h = max(80, plot_h - 60)
            imgui.dummy(imgui.ImVec2(region_w, region_h))
            rmin = imgui.get_item_rect_min()
            rmax = imgui.get_item_rect_max()
            draw_list = imgui.get_window_draw_list()
            if draw_list is None:
                return
            n = len(history)
            if n < 2:
                return
            px_min = rmin.x + pad
            px_max = rmax.x - pad
            py_min = rmin.y + pad
            py_max = rmax.y - pad
            nd = min(display_dim, 4)
            colors = [
                (1.0, 0.3, 0.3, 1.0),
                (0.3, 0.8, 0.3, 1.0),
                (0.3, 0.4, 1.0, 1.0),
                (1.0, 0.75, 0.2, 1.0),
            ]
            for d in range(nd):
                vals = [float(h[d]) if d < len(h) else 0.0 for h in history]
                v_min, v_max = min(vals), max(vals)
                span = (v_max - v_min) or 1.0
                pts = []
                for i, v in enumerate(vals):
                    tx = (i / (n - 1)) if n > 1 else 0.0
                    ty = (v - v_min) / span
                    x = px_min + tx * (px_max - px_min)
                    y = py_max - ty * (py_max - py_min)
                    pts.append((x, y))
                col = colors[d % len(colors)]
                try:
                    col_u32 = imgui.get_color_u32(imgui.ImVec4(col[0], col[1], col[2], col[3]))
                except Exception:
                    col_u32 = 0xFF00FF00
                for j in range(len(pts) - 1):
                    draw_list.add_line(
                        imgui.ImVec2(pts[j][0], pts[j][1]),
                        imgui.ImVec2(pts[j + 1][0], pts[j + 1][1]),
                        col_u32,
                        1.5,
                    )
        except Exception:
            imgui.text("(plot fallback unavailable)")

    def _render_obs_plot_window(self, imgui, key: str, term_name: str, display_dim: int) -> None:
        """플롯을 별도 ImGui 창에 그림. ImPlot 실패 시 DrawList fallback."""
        plot_window_title = f"Obs Plot: {term_name}"
        plot_w, plot_h = 520, 320
        flags = 0
        if not imgui.begin(plot_window_title, None, flags):
            imgui.end()
            return
        history = self._debug_obs_history.get(key)
        if not history:
            imgui.text("(no data)")
            imgui.end()
            return
        n = len(history)
        last_vals = history[-1] if history else None
        if last_vals is not None:
            n_show = min(12, len(last_vals))
            vals_str = ", ".join(f"{float(last_vals[i]):.4g}" for i in range(n_show))
            if len(last_vals) > n_show:
                vals_str += ", ..."
            imgui.text(f"Last ({len(last_vals)} dims): {vals_str}")
        if n < 2:
            imgui.text("(collecting...)")
            imgui.end()
            return
        # Newton 뷰어는 ImPlot::CreateContext()를 호출하지 않으므로 ImPlot 사용 시 에러 발생.
        # ImGui DrawList fallback만 사용 (ImPlot 호출 제거).
        self._draw_obs_plot_fallback(imgui, history, display_dim, plot_w, plot_h)
        imgui.end()


class DexblindNewtonVisualizer(NewtonVisualizer):
    """DexblindNewtonVisualizerCfg를 적용하는 Newton Visualizer 서브클래스."""

    def __init__(self, cfg: DexblindNewtonVisualizerCfg):
        super().__init__(cfg)
        self.cfg: DexblindNewtonVisualizerCfg = cfg
        self._goal_marker_data: list | None = None

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        if self._is_initialized:
            logger.debug("[DexblindNewtonVisualizer] already initialized.")
            return
        if scene_data_provider is None:
            raise RuntimeError("Newton visualizer requires a scene_data_provider.")
        self._scene_data_provider = scene_data_provider
        metadata = scene_data_provider.get_metadata()
        self._env_ids = self._compute_visualized_env_ids()
        if self._env_ids:
            get_filtered = getattr(scene_data_provider, "get_newton_model_for_env_ids", None)
            self._model = get_filtered(self._env_ids) if callable(get_filtered) else scene_data_provider.get_newton_model()
        else:
            self._model = scene_data_provider.get_newton_model()
        self._state = scene_data_provider.get_newton_state(self._env_ids)

        cfg = self.cfg
        self._viewer = DexblindNewtonViewerGL(
            width=cfg.window_width,
            height=cfg.window_height,
            metadata=metadata,
            update_frequency=cfg.update_frequency,
        )
        self._viewer.set_model(self._model)
        self._viewer.set_world_offsets((0.0, 0.0, 0.0))
        self._viewer.up_axis = 2
        self._viewer.scaling = 1.0
        self._viewer._paused = False

        cam_pos = getattr(cfg, "camera_position", (0.0, 0.0, 0.0))
        cam_tgt = getattr(cfg, "camera_target", (0.0, 0.0, 0.45))
        self._apply_camera_pose((cam_pos, cam_tgt))
        self._last_camera_pose = (cam_pos, cam_tgt)

        self._viewer.show_joints = cfg.show_joints
        self._viewer.show_contacts = cfg.show_contacts
        self._viewer.show_springs = cfg.show_springs
        self._viewer.show_com = cfg.show_com
        self._viewer.show_collision = getattr(cfg, "show_collision", False)
        self._viewer.show_visual = getattr(cfg, "show_visual", True)
        self._viewer._dexblind_panel_width = getattr(cfg, "panel_initial_width", 500)
        self._viewer._dexblind_font_scale = getattr(cfg, "font_scale", 2.5)
        self._viewer._debug_panel_w = getattr(cfg, "debug_panel_width", 380)
        self._viewer._debug_panel_h = getattr(cfg, "debug_panel_height", 440)

        self._viewer.renderer.draw_shadows = cfg.enable_shadows
        self._viewer.renderer.draw_sky = cfg.enable_sky
        self._viewer.renderer.draw_wireframe = cfg.enable_wireframe
        sky_upper = getattr(cfg, "background_color", (0.53, 0.81, 0.92))
        sky_lower = getattr(cfg, "ground_color", (0.18, 0.20, 0.25))
        self._viewer.renderer.sky_upper = sky_upper
        self._viewer.renderer.sky_lower = sky_lower
        self._viewer.renderer._light_color = getattr(cfg, "light_color", (1.0, 1.0, 1.0))
        self._viewer._env_ref = scene_data_provider

        if getattr(self._viewer.ui, "imgui", None) and getattr(self._viewer.ui.imgui.get_style(), "font_scale_main", None) is not None:
            try:
                self._viewer.ui.imgui.get_style().font_scale_main = self._viewer._dexblind_font_scale
            except Exception:
                pass

        num_envs = metadata.get("num_envs", 1) or 1
        try:
            from isaaclab.cloner.cloner_utils import grid_transforms
            env_origins, _ = grid_transforms(num_envs, spacing=2.0)
            env_origins_np = env_origins.numpy()
        except Exception:
            env_origins_np = np.zeros((num_envs, 3), dtype=np.float32)
        self._goal_marker_data = _register_goal_markers(self._viewer, getattr(cfg, "goal_markers", []) or [], env_origins_np)

        global _dexblind_visualizer_instance
        _dexblind_visualizer_instance = self

        logger.info(
            "[DexblindNewtonVisualizer] initialized | camera_pos=%s camera_target=%s",
            self._viewer.camera.pos,
            cam_tgt,
        )
        self._is_initialized = True

    def step(self, dt: float, state: Any | None = None) -> None:
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return

        self._sim_time += dt
        self._step_counter += 1
        if self.cfg.camera_source == "usd_path":
            self._update_camera_from_usd_path()
        self._state = self._scene_data_provider.get_newton_state(self._env_ids)
        contacts = None
        if self._viewer.show_contacts:
            contacts_data = self._scene_data_provider.get_contacts()
            if isinstance(contacts_data, dict):
                contacts = contacts_data.get("contacts", contacts_data)
            else:
                contacts = contacts_data
        update_frequency = self._viewer._update_frequency if self._viewer else self._update_frequency
        if self._step_counter % update_frequency != 0:
            return

        try:
            if not self._viewer.is_paused():
                self._viewer.begin_frame(self._sim_time)
                if self._state is not None:
                    body_q = getattr(self._state, "body_q", None)
                    if hasattr(body_q, "shape") and body_q.shape[0] == 0:
                        self._viewer.end_frame()
                        return
                    self._viewer.log_state(self._state)
                    if contacts is not None and hasattr(self._viewer, "log_contacts"):
                        try:
                            self._viewer.log_contacts(contacts, self._state)
                        except RuntimeError as exc:
                            logger.debug("[DexblindNewtonVisualizer] log_contacts failed: %s", exc)
                    if self._goal_marker_data:
                        for inst_name, mesh_name, xforms, scales, colors, materials in self._goal_marker_data:
                            self._viewer.log_instances(inst_name, mesh_name, xforms, scales, colors, materials)
                self._viewer.end_frame()
            else:
                self._viewer._update()
        except RuntimeError as exc:
            logger.debug("[DexblindNewtonVisualizer] Viewer update failed: %s", exc)

    def close(self) -> None:
        global _dexblind_visualizer_instance
        if self._is_closed:
            return
        if _dexblind_visualizer_instance is self:
            _dexblind_visualizer_instance = None
        super().close()


def _load_usd_mesh(usd_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(usd_path)
    mesh_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = prim
            break
    if mesh_prim is None:
        raise RuntimeError(f"No UsdGeom.Mesh in {usd_path}")
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices_flat = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    tri_list = []
    off = 0
    for nv in counts:
        for k in range(nv - 2):
            tri_list.extend([indices_flat[off], indices_flat[off + k + 1], indices_flat[off + k + 2]])
        off += nv
    indices = np.array(tri_list, dtype=np.uint32)
    normals_attr = mesh.GetNormalsAttr().Get()
    normals = np.array(normals_attr, dtype=np.float32) if normals_attr else None
    return pts, indices, normals


def _register_goal_markers(
    viewer,
    markers: list[GoalMarkerCfg],
    env_origins: np.ndarray,
) -> list[tuple[str, str, wp.array, wp.array, wp.array, wp.array]] | None:
    if not markers:
        return None
    try:
        import newton
    except ImportError:
        return None
    result = []
    num_envs = len(env_origins)
    for i, m in enumerate(markers):
        mesh_name = f"/goal_mesh_{i}"
        inst_name = f"/goal_marker_{i}"
        if getattr(m, "usd_path", None):
            pts, indices, normals = _load_usd_mesh(m.usd_path)
            wp_pts = wp.array(pts, dtype=wp.vec3)
            wp_idx = wp.array(indices, dtype=wp.uint32)
            wp_norms = wp.array(normals, dtype=wp.vec3) if normals is not None else None
            viewer.log_mesh(mesh_name, wp_pts, wp_idx, normals=wp_norms, hidden=True)
        else:
            scale = getattr(m, "scale", (1.0, 1.0, 1.0))
            viewer.log_geo(mesh_name, newton.GeoType.BOX, scale, geo_thickness=0.0, geo_is_solid=True)
        pos = getattr(m, "pos", (0.0, 0.0, 0.0))
        rot = getattr(m, "rot", (1.0, 0.0, 0.0, 0.0))
        color = getattr(m, "color", (0.2, 0.85, 0.3))
        scale = getattr(m, "scale", (1.0, 1.0, 1.0))
        xf_list = []
        for e in range(num_envs):
            ox, oy, oz = float(env_origins[e, 0]), float(env_origins[e, 1]), float(env_origins[e, 2])
            xf_list.append(wp.transform((pos[0] + ox, pos[1] + oy, pos[2] + oz), rot))
        xforms = wp.array(xf_list, dtype=wp.transform)
        scales = wp.array([wp.vec3(*scale)] * num_envs, dtype=wp.vec3)
        colors = wp.array([wp.vec3(*color)] * num_envs, dtype=wp.vec3)
        materials = wp.array([wp.vec4(0.0, 0.5, 0.0, 0.0)] * num_envs, dtype=wp.vec4)
        result.append((inst_name, mesh_name, xforms, scales, colors, materials))
    return result
