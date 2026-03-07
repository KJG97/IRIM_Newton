"""PySide6 debug panel that auto-launches with the Newton visualizer.

Provides a dockable, always-on-top window for inspecting per-env simulation
state at runtime. The panel is created once during ``NewtonVisualizer.initialize()``
and its Qt event loop is pumped every ``NewtonVisualizer.step()``.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

_STYLE = """
QMainWindow, QWidget, QScrollArea {
    background-color: #2b2b2b;
}
QLabel, QTableWidget, QHeaderView, QHeaderView::section, QLineEdit, QPushButton {
    color: #e0e0e0;
    font-size: 13px;
}
QTableWidget {
    background-color: #333333;
    gridline-color: #555555;
    selection-background-color: #505050;
}
QHeaderView::section {
    background-color: #3a3a3a;
    border: 1px solid #555555;
    padding: 2px 4px;
}
QLineEdit {
    background-color: #3a3a3a;
    border: 1px solid #555555;
    padding: 2px 4px;
}
QPushButton {
    background-color: #4a4a4a;
    border: 1px solid #666666;
    padding: 4px 12px;
}
QPushButton:hover {
    background-color: #5a5a5a;
}
QPushButton[active="true"] {
    background-color: #2a6496;
    border: 1px solid #3a8fd6;
}
QPushButton[selected="true"] {
    background-color: #2a7a3a;
    border: 1px solid #3ad64a;
}
"""

_PLOT_HISTORY = 200
_PLOT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#e6beff",
    "#ff6347", "#00ced1", "#ff1493", "#7fff00", "#1e90ff",
    "#ff4500", "#2e8b57", "#daa520", "#8a2be2", "#00fa9a",
]

_PAGE_JOINT = 0
_PAGE_ENV = 1
_PAGE_OBS = 2
_PAGE_EVENT = 3


def _restyle_btn(btn: QPushButton) -> None:
    btn.style().unpolish(btn)
    btn.style().polish(btn)


_PLOT_REDRAW_INTERVAL_MS = 200


class ObsPlotWindow(QMainWindow):
    """Per-term plot window with dim checkboxes and live matplotlib graph.

    Rendering is throttled to at most once per _PLOT_REDRAW_INTERVAL_MS to
    avoid slowing down the simulation loop.
    """

    def __init__(self, term_name: str, dim: int, dim_labels: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Obs Plot — {term_name}")
        self.setStyleSheet(_STYLE)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        self.term_name = term_name
        self.dim = dim
        self._labels = dim_labels if dim_labels and len(dim_labels) == dim else [str(i) for i in range(dim)]
        self._history: list[deque] = [deque(maxlen=_PLOT_HISTORY) for _ in range(dim)]
        self._active_dims: set[int] = set()
        self._step_counter = 0
        self._dirty = False
        self._last_draw_ms = 0

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # matplotlib figure
        self._fig = Figure(figsize=(6, 3.5), dpi=100, facecolor="#2b2b2b")
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#333333")
        self._ax.tick_params(colors="#aaaaaa", labelsize=9)
        for spine in self._ax.spines.values():
            spine.set_color("#555555")
        self._ax.set_xlabel("step", color="#aaaaaa", fontsize=10)
        self._ax.set_ylabel("value", color="#aaaaaa", fontsize=10)
        self._ax.set_title(term_name, color="#e0e0e0", fontsize=12)
        self._canvas = FigureCanvasQTAgg(self._fig)
        root.addWidget(self._canvas, 1)

        # checkbox grid
        cb_container = QWidget()
        self._cb_layout = QGridLayout(cb_container)
        self._cb_layout.setContentsMargins(4, 4, 4, 4)
        self._cb_layout.setSpacing(2)
        self._checkboxes: list[QCheckBox] = []
        cols = max(4, min(6, dim // 4 + 1))
        for i in range(dim):
            label = self._labels[i]
            cb = QCheckBox(label)
            cb.setStyleSheet(f"color: {_PLOT_COLORS[i % len(_PLOT_COLORS)]}; font-size: 11px;")
            cb.stateChanged.connect(lambda state, idx=i: self._on_checkbox(idx, state))
            self._cb_layout.addWidget(cb, i // cols, i % cols)
            self._checkboxes.append(cb)

        # select-all / deselect-all
        btn_row = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_all.setFixedWidth(50)
        btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton("None")
        btn_none.setFixedWidth(50)
        btn_none.clicked.connect(self._deselect_all)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(cb_container)
        scroll.setMaximumHeight(min(160, (dim // cols + 2) * 22))

        root.addLayout(btn_row)
        root.addWidget(scroll)

        self.resize(700, 450)

    def _on_checkbox(self, idx: int, state: int) -> None:
        if state == Qt.CheckState.Checked.value:
            self._active_dims.add(idx)
        else:
            self._active_dims.discard(idx)
        self._redraw()

    def _select_all(self) -> None:
        for cb in self._checkboxes:
            cb.setChecked(True)

    def _deselect_all(self) -> None:
        for cb in self._checkboxes:
            cb.setChecked(False)

    def push_values(self, vals: np.ndarray) -> None:
        """Append one timestep of values (1-D array of length self.dim)."""
        self._step_counter += 1
        for i in range(min(len(vals), self.dim)):
            self._history[i].append(float(vals[i]))
        self._dirty = True

    def _redraw(self) -> None:
        self._last_draw_ms = int(time.monotonic() * 1000)
        self._dirty = False

        ax = self._ax
        ax.cla()
        ax.set_facecolor("#333333")
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        ax.set_xlabel("step", color="#aaaaaa", fontsize=10)
        ax.set_ylabel("value", color="#aaaaaa", fontsize=10)
        ax.set_title(self.term_name, color="#e0e0e0", fontsize=12)
        for spine in ax.spines.values():
            spine.set_color("#555555")

        for i in sorted(self._active_dims):
            if i >= self.dim:
                continue
            data = list(self._history[i])
            if not data:
                continue
            x_end = self._step_counter
            x_start = x_end - len(data)
            xs = list(range(x_start, x_end))
            color = _PLOT_COLORS[i % len(_PLOT_COLORS)]
            ax.plot(xs, data, color=color, linewidth=1.2, label=self._labels[i])

        if self._active_dims:
            leg = ax.legend(
                loc="upper left", fontsize=8, ncol=min(len(self._active_dims), 6),
                facecolor="#3a3a3a", edgecolor="#555555", labelcolor="#e0e0e0",
            )
            leg.set_alpha(0.85)

        self._fig.tight_layout(pad=1.0)
        self._canvas.draw_idle()

    def tick(self) -> None:
        if not self.isVisible() or not self._active_dims or not self._dirty:
            return
        now_ms = int(time.monotonic() * 1000)
        if now_ms - self._last_draw_ms < _PLOT_REDRAW_INTERVAL_MS:
            return
        self._redraw()


class DebugPanel(QMainWindow):
    """Debug panel showing per-env joint positions and environment object state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Isaac Lab — Debug Panel")
        self.setStyleSheet(_STYLE)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- env selector row ---
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Env ID:"))
        self._env_input = QLineEdit("0")
        self._env_input.setFixedWidth(60)
        selector_row.addWidget(self._env_input)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply)
        selector_row.addWidget(self._apply_btn)
        self._picking_cb = QCheckBox("Picking")
        self._picking_cb.setToolTip("Enable right-click drag to apply forces on bodies")
        selector_row.addWidget(self._picking_cb)
        self._status_label = QLabel("")
        selector_row.addWidget(self._status_label, 1)
        root.addLayout(selector_row)

        # --- picking force slider row (log-scale: 0.01 ~ 10000) ---
        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("Pick Force:"))
        self._pick_slider = QSlider(Qt.Orientation.Horizontal)
        self._pick_slider.setRange(0, 600)
        self._pick_slider.setValue(200)
        self._pick_slider.setToolTip("Log-scale picking stiffness (ke). Damping = ke × 0.1")
        self._pick_slider.valueChanged.connect(self._on_pick_force_changed)
        pick_row.addWidget(self._pick_slider, 1)
        self._pick_force_label = QLabel("")
        self._pick_force_label.setFixedWidth(160)
        pick_row.addWidget(self._pick_force_label)
        root.addLayout(pick_row)
        self._on_pick_force_changed(self._pick_slider.value())

        # --- tab buttons ---
        tab_row = QHBoxLayout()
        self._btn_joint = QPushButton("JointState")
        self._btn_env = QPushButton("EnvState")
        self._btn_obs = QPushButton("ObsState")
        self._btn_event = QPushButton("EventState")
        self._btn_joint.clicked.connect(lambda: self._switch_page(_PAGE_JOINT))
        self._btn_env.clicked.connect(lambda: self._switch_page(_PAGE_ENV))
        self._btn_obs.clicked.connect(lambda: self._switch_page(_PAGE_OBS))
        self._btn_event.clicked.connect(lambda: self._switch_page(_PAGE_EVENT))
        tab_row.addWidget(self._btn_joint)
        tab_row.addWidget(self._btn_env)
        tab_row.addWidget(self._btn_obs)
        tab_row.addWidget(self._btn_event)
        tab_row.addStretch(1)
        root.addLayout(tab_row)

        # --- stacked pages ---
        self._pages = QStackedWidget()
        root.addWidget(self._pages, 1)

        # Page 0: JointState
        self._pages.addWidget(self._make_joint_page())

        # Page 1: EnvState
        self._pages.addWidget(self._make_env_page())

        # Page 2: ObsState
        self._pages.addWidget(self._make_obs_page())

        # Page 3: EventState
        self._pages.addWidget(self._make_event_page())

        # state
        self._selected_env: int = 0
        self._live: bool = False
        self._applied: bool = False
        self._env_ref = None
        self._plot_windows: dict[str, ObsPlotWindow] = {}
        self._switch_page(_PAGE_JOINT)

        self.resize(500, 700)
        self._move_to_top_right()

    def _move_to_top_right(self, margin: int = 25) -> None:
        """Position the panel at the top-right corner of the primary screen."""
        from PySide6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + geo.width() - self.width() - margin
        y = geo.y() + margin
        self.move(x, y)

    # ------------------------------------------------------------------ #
    #  Page factories                                                     #
    # ------------------------------------------------------------------ #

    def _make_joint_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Joint", "Position (°)"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        scroll.setWidget(table)
        self._joint_table = table
        return scroll

    def _make_env_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # object selector buttons (populated on Apply)
        self._obj_btn_row = QHBoxLayout()
        self._obj_btn_row.addStretch(1)
        layout.addLayout(self._obj_btn_row)

        # object detail table
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Property", "Value"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        scroll.setWidget(table)
        self._env_table = table
        layout.addWidget(scroll, 1)

        self._obj_buttons: list[QPushButton] = []
        self._env_objects: list[dict] = []
        self._selected_obj_idx: int = -1
        return page

    # ------------------------------------------------------------------ #
    #  Tab switching                                                      #
    # ------------------------------------------------------------------ #

    def _switch_page(self, idx: int) -> None:
        self._pages.setCurrentIndex(idx)
        for btn, page_id in [
            (self._btn_joint, _PAGE_JOINT),
            (self._btn_env, _PAGE_ENV),
            (self._btn_obs, _PAGE_OBS),
            (self._btn_event, _PAGE_EVENT),
        ]:
            btn.setProperty("active", idx == page_id)
            _restyle_btn(btn)

    # ------------------------------------------------------------------ #
    #  Apply                                                              #
    # ------------------------------------------------------------------ #

    def _on_apply(self) -> None:
        text = self._env_input.text().strip()
        try:
            env_id = int(text)
        except ValueError:
            self._status_label.setText("Invalid env ID")
            self._live = False
            self._applied = False
            return

        self._selected_env = env_id
        self._live = True
        self._applied = True
        self._status_label.setText(f"Watching env {env_id}")
        self._refresh_joints()
        self._build_object_buttons()
        self._build_obs_group_buttons()
        self._build_event_term_buttons()

    # ------------------------------------------------------------------ #
    #  JointState                                                         #
    # ------------------------------------------------------------------ #

    def _refresh_joints(self) -> None:
        import warp as wp
        from isaaclab.sim._impl.newton_manager import NewtonManager
        import newton as nw

        model = NewtonManager._model
        state = NewtonManager._state_0
        if model is None or state is None:
            self._status_label.setText("Newton model not ready")
            self._live = False
            return

        joint_keys: list[str] = model.joint_key
        joint_world = model.joint_world.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_q = wp.to_torch(state.joint_q).cpu()
        joint_type = model.joint_type.numpy()

        # Exclude FREE (0), FIXED (4), and FixedJoint by enum for clarity
        excluded_joint_types = {0, 4, int(nw.JointType.FIXED)}

        rows: list[tuple[str, float]] = []
        for j in range(model.joint_count):
            if joint_world[j] != self._selected_env:
                continue
            if int(joint_type[j]) in excluded_joint_types:
                continue
            q_idx = int(joint_q_start[j])
            q_val = float(joint_q[q_idx].item())
            name = joint_keys[j].split("/")[-1]
            rows.append((name, math.degrees(q_val)))

        if not rows:
            self._status_label.setText(f"Env {self._selected_env}: no joints found")
            self._joint_table.setRowCount(0)
            return

        self._joint_table.setRowCount(len(rows))
        for i, (name, deg) in enumerate(rows):
            self._joint_table.setItem(i, 0, QTableWidgetItem(name))
            val_item = QTableWidgetItem(f"{deg:.2f}")
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._joint_table.setItem(i, 1, val_item)

        self._status_label.setText(f"Env {self._selected_env}: {len(rows)} joints")

    # ------------------------------------------------------------------ #
    #  EnvState — object discovery & buttons                              #
    # ------------------------------------------------------------------ #

    def _get_env_articulations(self) -> list[dict]:
        """Return articulation info dicts for the selected env.

        Uses body_key path prefix matching to reliably associate bodies
        with their articulation, avoiding joint-range ambiguity.
        """
        from isaaclab.sim._impl.newton_manager import NewtonManager

        model = NewtonManager._model
        if model is None:
            return []

        art_keys: list[str] = model.articulation_key
        art_world = model.articulation_world.numpy() if hasattr(model.articulation_world, "numpy") else model.articulation_world
        body_world = model.body_world.numpy()
        body_keys: list[str] = model.body_key

        results: list[dict] = []
        for a in range(model.articulation_count):
            if int(art_world[a]) != self._selected_env:
                continue

            art_key = art_keys[a]
            # e.g. "/World/envs/env_1/Robot_articulation" → prefix "/World/envs/env_1/Robot"
            prefix = art_key.replace("_articulation", "")
            short = prefix.split("/")[-1]

            body_indices: list[int] = []
            for b in range(model.body_count):
                if int(body_world[b]) != self._selected_env:
                    continue
                if body_keys[b].startswith(prefix):
                    body_indices.append(b)

            root_body = body_indices[0] if body_indices else -1

            results.append({
                "key": art_key,
                "short": short,
                "art_idx": a,
                "root_body": root_body,
                "body_indices": body_indices,
            })
        return results

    def _build_object_buttons(self) -> None:
        """Discover articulations in env and create a button per object."""
        # Clear old buttons
        for btn in self._obj_buttons:
            self._obj_btn_row.removeWidget(btn)
            btn.deleteLater()
        self._obj_buttons.clear()
        self._env_objects = self._get_env_articulations()
        self._selected_obj_idx = -1
        self._env_table.setRowCount(0)

        for i, obj in enumerate(self._env_objects):
            btn = QPushButton(obj["short"])
            btn.setProperty("selected", False)
            btn.clicked.connect(lambda checked, idx=i: self._select_object(idx))
            self._obj_btn_row.insertWidget(self._obj_btn_row.count() - 1, btn)
            self._obj_buttons.append(btn)

    def _select_object(self, idx: int) -> None:
        self._selected_obj_idx = idx
        for i, btn in enumerate(self._obj_buttons):
            btn.setProperty("selected", i == idx)
            _restyle_btn(btn)
        self._refresh_env_object()

    # ------------------------------------------------------------------ #
    #  EnvState — helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_body_in_env(model, body_keys, body_world, env_id: int, suffix: str) -> int:
        """Find a body index by suffix (e.g. 'Origin_Body') within a given env. Returns -1 if not found."""
        for b in range(model.body_count):
            if int(body_world[b]) != env_id:
                continue
            if body_keys[b].endswith(suffix) or body_keys[b].endswith(f"/{suffix}"):
                return b
        return -1

    @staticmethod
    def _body_q_to_pos_quat(body_q_cpu, body_idx: int):
        """Extract (pos[3], quat_xyzw[4]) from Newton body_q (pos3 + quat_xyzw)."""
        import torch
        pose = body_q_cpu[body_idx]
        pos = torch.tensor([float(pose[0]), float(pose[1]), float(pose[2])])
        qx, qy, qz, qw = float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])
        quat_xyzw = torch.tensor([qx, qy, qz, qw])
        return pos, quat_xyzw

    @staticmethod
    def _relative_pose(origin_pos, origin_quat_xyzw, target_pos, target_quat_xyzw):
        """Compute target pose relative to origin frame. All tensors are 1-D (unbatched).

        The returned quaternion is canonicalized so that w >= 0 to match
        the standard positive-w convention used by most trajectory data.
        """
        from isaaclab.utils.math import quat_inv, quat_mul, quat_apply
        q_inv = quat_inv(origin_quat_xyzw.unsqueeze(0)).squeeze(0)
        rel_pos = quat_apply(q_inv.unsqueeze(0), (target_pos - origin_pos).unsqueeze(0)).squeeze(0)
        rel_quat = quat_mul(q_inv.unsqueeze(0), target_quat_xyzw.unsqueeze(0)).squeeze(0)
        if float(rel_quat[3]) < 0:
            rel_quat = -rel_quat
        return rel_pos, rel_quat

    @staticmethod
    def _fmt_pos(pos) -> list[tuple[str, str]]:
        return [
            ("pos x", f"{float(pos[0]):.4f}"),
            ("pos y", f"{float(pos[1]):.4f}"),
            ("pos z", f"{float(pos[2]):.4f}"),
        ]

    @staticmethod
    def _fmt_quat_xyzw(q) -> tuple[str, str]:
        """Format xyzw quat as display string in (w,x,y,z) order."""
        return ("quat (w,x,y,z)", f"{float(q[3]):.4f}, {float(q[0]):.4f}, {float(q[1]):.4f}, {float(q[2]):.4f}")

    # ------------------------------------------------------------------ #
    #  EnvState — object detail table                                     #
    # ------------------------------------------------------------------ #

    def _refresh_env_object(self) -> None:
        if self._selected_obj_idx < 0 or self._selected_obj_idx >= len(self._env_objects):
            self._env_table.setRowCount(0)
            return

        import warp as wp
        from isaaclab.sim._impl.newton_manager import NewtonManager

        model = NewtonManager._model
        state = NewtonManager._state_0
        if model is None or state is None:
            return

        obj = self._env_objects[self._selected_obj_idx]
        body_idx = obj["root_body"]
        body_indices: list[int] = obj.get("body_indices", [])
        is_robot = any(
            "Origin_Body" in model.body_key[b]
            for b in body_indices
        ) if body_indices else False

        body_q_cpu = wp.to_torch(state.body_q).cpu() if state.body_q is not None else None
        body_keys: list[str] = model.body_key
        body_world = model.body_world.numpy()

        rows: list[tuple[str, str]] = []
        rows.append(("name", obj["key"]))

        origin_idx = self._find_body_in_env(model, body_keys, body_world, self._selected_env, "Origin_Body")
        origin_pos = origin_quat = None
        if origin_idx >= 0 and body_q_cpu is not None:
            origin_pos, origin_quat = self._body_q_to_pos_quat(body_q_cpu, origin_idx)

        if is_robot:
            # --- Robot: show Right_Hand_base task-space pose relative to Origin_Body ---
            hand_idx = self._find_body_in_env(model, body_keys, body_world, self._selected_env, "Right_Hand_base")
            if hand_idx >= 0 and body_q_cpu is not None and origin_pos is not None:
                hand_pos, hand_quat = self._body_q_to_pos_quat(body_q_cpu, hand_idx)
                rel_pos, rel_quat = self._relative_pose(origin_pos, origin_quat, hand_pos, hand_quat)
                rows.append(("", ""))
                rows.append(("--- Right_Hand_base (task-space) ---", "rel. to Origin_Body"))
                rows.extend(self._fmt_pos(rel_pos))
                rows.append(self._fmt_quat_xyzw(rel_quat))
                rows.append(("", ""))
                rows.append(("--- Right_Hand_base (world) ---", ""))
                rows.extend(self._fmt_pos(hand_pos))
                rows.append(self._fmt_quat_xyzw(hand_quat))
            elif hand_idx < 0:
                rows.append(("Right_Hand_base", "NOT FOUND"))

            if origin_pos is not None:
                rows.append(("", ""))
                rows.append(("--- Origin_Body (world) ---", ""))
                rows.extend(self._fmt_pos(origin_pos))
                rows.append(self._fmt_quat_xyzw(origin_quat))
        else:
            if body_idx >= 0 and body_q_cpu is not None:
                obj_pos, obj_quat = self._body_q_to_pos_quat(body_q_cpu, body_idx)
                if origin_pos is not None:
                    rel_pos, rel_quat = self._relative_pose(origin_pos, origin_quat, obj_pos, obj_quat)
                    rows.append(("", ""))
                    rows.append(("--- pose (task-space) ---", "rel. to Origin_Body"))
                    rows.extend(self._fmt_pos(rel_pos))
                    rows.append(self._fmt_quat_xyzw(rel_quat))
                rows.append(("", ""))
                rows.append(("--- pose (world) ---", ""))
                rows.extend(self._fmt_pos(obj_pos))
                rows.append(self._fmt_quat_xyzw(obj_quat))

        # -- velocity --
        if body_idx >= 0 and state.body_qd is not None:
            body_qd = wp.to_torch(state.body_qd).cpu()
            vel = body_qd[body_idx]
            rows.append(("", ""))
            rows.append(("lin vel", f"({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})"))
            rows.append(("ang vel", f"({vel[3]:.3f}, {vel[4]:.3f}, {vel[5]:.3f})"))

        # -- mass --
        if body_idx >= 0 and model.body_mass is not None:
            mass = float(model.body_mass.numpy()[body_idx])
            rows.append(("mass", f"{mass:.6f}"))

        # -- collision & material --
        if body_indices and model.shape_body is not None:
            import newton as nw

            collide_flag = int(nw.ShapeFlags.COLLIDE_SHAPES)
            shape_body = model.shape_body.numpy()
            shape_flags = model.shape_flags.numpy()
            body_set = set(body_indices)
            visible_flag = int(nw.ShapeFlags.VISIBLE)
            all_shape_indices = [
                s for s in range(model.shape_count)
                if int(shape_body[s]) in body_set
                and not (int(shape_flags[s]) & visible_flag and not int(shape_flags[s]) & collide_flag)
            ]
            shape_indices = [
                s for s in all_shape_indices
                if int(shape_flags[s]) & collide_flag
            ]

            n_total = len(all_shape_indices)
            n_collide = len(shape_indices)
            if n_total > 0:
                if n_collide == n_total:
                    col_str = f"ENABLED ({n_collide}/{n_total})"
                elif n_collide == 0:
                    col_str = f"DISABLED (0/{n_total})"
                else:
                    col_str = f"PARTIAL ({n_collide}/{n_total})"
                rows.append(("collision", col_str))

            if shape_indices:
                mu = model.shape_material_mu.numpy()
                torsional = model.shape_material_torsional_friction.numpy()
                rolling = model.shape_material_rolling_friction.numpy()
                restitution = model.shape_material_restitution.numpy()
                ke = model.shape_material_ke.numpy()
                kd = model.shape_material_kd.numpy()

                def _summarize(arr, indices, fmt=".4f"):
                    vals = [float(arr[s]) for s in indices]
                    lo, hi = min(vals), max(vals)
                    if abs(hi - lo) < 1e-8:
                        return f"{lo:{fmt}}"
                    return f"{lo:{fmt}} ~ {hi:{fmt}}"

                rows.append(("", ""))
                rows.append(("--- material ---", f"({len(shape_indices)} collision shapes)"))
                rows.append(("mu (friction)", _summarize(mu, shape_indices)))
                rows.append(("torsional friction", _summarize(torsional, shape_indices)))
                rows.append(("rolling friction", _summarize(rolling, shape_indices)))
                rows.append(("restitution", _summarize(restitution, shape_indices)))
                rows.append(("ke (stiffness)", _summarize(ke, shape_indices, ".2f")))
                rows.append(("kd (damping)", _summarize(kd, shape_indices, ".2f")))

        self._env_table.setRowCount(len(rows))
        for i, (prop, val) in enumerate(rows):
            self._env_table.setItem(i, 0, QTableWidgetItem(prop))
            val_item = QTableWidgetItem(val)
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._env_table.setItem(i, 1, val_item)

    # ------------------------------------------------------------------ #
    #  ObsState — observation groups & terms                              #
    # ------------------------------------------------------------------ #

    def set_env(self, env) -> None:
        """Register the ManagerBasedRLEnv so ObsState can read observations."""
        self._env_ref = env

    def _make_obs_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self._obs_group_btn_row = QHBoxLayout()
        self._obs_group_btn_row.addStretch(1)
        layout.addLayout(self._obs_group_btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Group / Term", "Dim", "Values"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        scroll.setWidget(table)
        self._obs_table = table
        layout.addWidget(scroll, 1)

        self._obs_group_buttons: list[QPushButton] = []
        self._selected_obs_group: str = ""
        self._obs_term_info: list[tuple[str, int, list[str] | None]] = []
        table.cellClicked.connect(self._on_obs_cell_clicked)
        return page

    def _build_obs_group_buttons(self) -> None:
        for btn in self._obs_group_buttons:
            self._obs_group_btn_row.removeWidget(btn)
            btn.deleteLater()
        self._obs_group_buttons.clear()
        self._selected_obs_group = ""
        self._obs_table.setRowCount(0)

        env = self._env_ref
        if env is None or not hasattr(env, "observation_manager"):
            return

        obs_mgr = env.observation_manager
        for group_name in obs_mgr.active_terms:
            btn = QPushButton(group_name)
            btn.setProperty("selected", False)
            btn.clicked.connect(lambda checked, g=group_name: self._select_obs_group(g))
            self._obs_group_btn_row.insertWidget(
                self._obs_group_btn_row.count() - 1, btn
            )
            self._obs_group_buttons.append(btn)

    def _select_obs_group(self, group_name: str) -> None:
        self._selected_obs_group = group_name
        for btn in self._obs_group_buttons:
            btn.setProperty("selected", btn.text() == group_name)
            _restyle_btn(btn)
        self._refresh_obs()

    @staticmethod
    def _extract_dim_labels(obs_mgr, group: str, term_name: str) -> list[str] | None:
        """Try to extract meaningful dimension labels from the term's config."""
        cfgs = getattr(obs_mgr, "_group_obs_term_cfgs", {}).get(group, [])
        names = obs_mgr.active_terms.get(group, [])
        for cfg, name in zip(cfgs, names):
            if name != term_name:
                continue
            params = getattr(cfg, "params", {})
            asset_cfg = params.get("asset_cfg")
            if asset_cfg is not None:
                jn = getattr(asset_cfg, "joint_names", None)
                if jn and isinstance(jn, (list, tuple)):
                    return list(jn)
            jn = params.get("joint_names")
            if jn and isinstance(jn, (list, tuple)):
                return list(jn)
        return None

    def _refresh_obs(self) -> None:
        import torch

        env = self._env_ref
        if env is None or not hasattr(env, "observation_manager"):
            self._obs_table.setRowCount(0)
            return

        obs_mgr = env.observation_manager
        group = self._selected_obs_group
        if group not in obs_mgr.active_terms:
            self._obs_table.setRowCount(0)
            return

        term_names = obs_mgr.active_terms[group]
        term_dims = obs_mgr.group_obs_term_dim[group]

        obs_buf = obs_mgr._obs_buffer
        if obs_buf is None or group not in obs_buf:
            self._obs_table.setRowCount(0)
            return

        data = obs_buf[group]
        eid = self._selected_env

        rows: list[tuple[str, int, str, str, list[str] | None]] = []

        if isinstance(data, torch.Tensor):
            idx = 0
            for name, shape in zip(term_names, term_dims):
                length = int(np.prod(shape))
                vals = data[eid, idx : idx + length].detach().cpu()
                dim_str = "×".join(str(d) for d in shape)
                val_str = self._format_obs_values(vals)
                labels = self._extract_dim_labels(obs_mgr, group, name)
                rows.append((name, length, dim_str, val_str, labels))
                self._push_to_plot(group, name, vals.numpy())
                idx += length
        elif isinstance(data, dict):
            for name in term_names:
                if name not in data:
                    continue
                t = data[name]
                if isinstance(t, torch.Tensor) and t.dim() >= 1:
                    vals = t[eid].detach().cpu().flatten()
                    dim_str = "×".join(str(d) for d in t.shape[1:])
                    val_str = self._format_obs_values(vals)
                    labels = self._extract_dim_labels(obs_mgr, group, name)
                    rows.append((name, vals.numel(), dim_str, val_str, labels))
                    self._push_to_plot(group, name, vals.numpy())

        self._obs_term_info = [(name, dim, labels) for name, dim, _, _, labels in rows]

        self._obs_table.setRowCount(len(rows))
        for i, (name, _dim, dim_str, val_str, _labels) in enumerate(rows):
            name_item = QTableWidgetItem(f"📈 {name}")
            name_item.setToolTip("Click to open plot window")
            name_item.setForeground(QColor("#6cb0ff"))
            self._obs_table.setItem(i, 0, name_item)
            dim_item = QTableWidgetItem(dim_str)
            dim_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._obs_table.setItem(i, 1, dim_item)
            val_item = QTableWidgetItem(val_str)
            val_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._obs_table.setItem(i, 2, val_item)

        self._status_label.setText(
            f"Env {eid} | {group}: {len(rows)} terms"
        )

    def _push_to_plot(self, group: str, term_name: str, vals: np.ndarray) -> None:
        key = f"{group}/{term_name}"
        win = self._plot_windows.get(key)
        if win is not None and win.isVisible():
            win.push_values(vals)

    def _on_obs_cell_clicked(self, row: int, col: int) -> None:
        if col != 0 or row < 0 or row >= len(self._obs_term_info):
            return
        term_name, dim, labels = self._obs_term_info[row]
        self._open_plot_window(term_name, dim, labels)

    def _open_plot_window(self, term_name: str, dim: int, dim_labels: list[str] | None = None) -> None:
        key = f"{self._selected_obs_group}/{term_name}"
        if key in self._plot_windows and self._plot_windows[key].isVisible():
            self._plot_windows[key].raise_()
            self._plot_windows[key].activateWindow()
            return
        win = ObsPlotWindow(key, dim, dim_labels=dim_labels)
        win.show()
        self._plot_windows[key] = win

    @staticmethod
    def _format_obs_values(vals) -> str:
        """Format a 1-D tensor into a compact string for the table cell."""
        n = vals.numel()
        if n <= 8:
            return " ".join(f"{v:.4f}" for v in vals.tolist())
        head = " ".join(f"{v:.4f}" for v in vals[:4].tolist())
        tail = " ".join(f"{v:.4f}" for v in vals[-4:].tolist())
        return f"{head}  …  {tail}  ({n})"

    # ------------------------------------------------------------------ #
    #  EventState — event term inspection                                 #
    # ------------------------------------------------------------------ #

    def _make_event_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self._event_btn_row = QHBoxLayout()
        self._event_btn_row.addStretch(1)
        layout.addLayout(self._event_btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Property", "Value"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        scroll.setWidget(table)
        self._event_table = table
        layout.addWidget(scroll, 1)

        self._event_buttons: list[QPushButton] = []
        self._selected_event_term: str = ""
        self._event_term_list: list[str] = []
        return page

    def _build_event_term_buttons(self) -> None:
        for btn in self._event_buttons:
            self._event_btn_row.removeWidget(btn)
            btn.deleteLater()
        self._event_buttons.clear()
        self._selected_event_term = ""
        self._event_term_list.clear()
        self._event_table.setRowCount(0)

        env = self._env_ref
        if env is None or not hasattr(env, "event_manager"):
            return

        ev_mgr = env.event_manager
        for mode, names in ev_mgr._mode_term_names.items():
            for name in names:
                label = f"{name} [{mode}]"
                self._event_term_list.append(name)
                btn = QPushButton(label)
                btn.setProperty("selected", False)
                btn.clicked.connect(
                    lambda checked, n=name: self._select_event_term(n)
                )
                self._event_btn_row.insertWidget(
                    self._event_btn_row.count() - 1, btn
                )
                self._event_buttons.append(btn)

    def _select_event_term(self, term_name: str) -> None:
        self._selected_event_term = term_name
        for btn in self._event_buttons:
            btn.setProperty("selected", term_name in btn.text())
            _restyle_btn(btn)
        self._refresh_event_detail()

    def _refresh_event_detail(self) -> None:
        """Read the current simulation values affected by the selected event term."""
        import warp as wp

        env = self._env_ref
        if env is None or not hasattr(env, "event_manager"):
            self._event_table.setRowCount(0)
            return

        eid = self._selected_env
        term = self._selected_event_term
        rows: list[tuple[str, str]] = []
        rows.append(("term", term))
        rows.append(("env_id", str(eid)))

        ev_mgr = env.event_manager
        term_cfg = None
        for mode, names in ev_mgr._mode_term_names.items():
            for i, n in enumerate(names):
                if n == term:
                    term_cfg = ev_mgr._mode_term_cfgs[mode][i]
                    rows.append(("mode", mode))
                    break
            if term_cfg is not None:
                break

        if term_cfg is None:
            rows.append(("error", "term config not found"))
            self._fill_event_table(rows)
            return

        params = term_cfg.params
        rows.append(("", ""))

        # Determine which asset and what data to show
        asset_cfg = params.get("asset_cfg")
        asset_name = asset_cfg.name if asset_cfg is not None else None

        if asset_name is not None and asset_name in env.scene.keys():
            asset = env.scene[asset_name]
            rows.append(("asset", asset_name))

            func_name = ""
            if hasattr(term_cfg.func, "__name__"):
                func_name = term_cfg.func.__name__
            elif hasattr(term_cfg.func, "__class__"):
                func_name = term_cfg.func.__class__.__name__
            rows.append(("func", func_name))
            rows.append(("", ""))

            if "stiffness" in term or "actuator_gains" in func_name:
                rows.extend(self._read_joint_property(asset, eid, "joint_stiffness", "stiffness"))
                rows.append(("", ""))
                rows.extend(self._read_joint_property(asset, eid, "joint_damping", "damping"))
            elif "friction" in term and "joint" in term:
                rows.extend(self._read_joint_property(asset, eid, "joint_friction_coeff", "joint_friction"))
            elif "mass" in term:
                rows.extend(self._read_body_property(asset, eid, "body_mass", "mass"))
            elif "friction" in term:
                rows.extend(self._read_shape_friction(asset_name, eid))
            elif "reset" in term and "hammer" in term.lower():
                rows.extend(self._read_root_pose(asset, eid))
            elif "reset" in term and "joint" in term:
                rows.extend(self._read_joint_property(asset, eid, "joint_pos", "joint_pos"))
            elif "force" in term:
                rows.extend(self._read_external_wrench(asset, eid))
            else:
                rows.extend(self._read_root_pose(asset, eid))
        else:
            # Multi-asset terms (e.g. reset_table_hammer_height)
            for key, val in params.items():
                if hasattr(val, "name") and val.name in env.scene.keys():
                    asset = env.scene[val.name]
                    rows.append(("", ""))
                    rows.append((f"--- {val.name} ---", ""))
                    rows.extend(self._read_root_pose(asset, eid))

        self._fill_event_table(rows)

    def _fill_event_table(self, rows: list[tuple[str, str]]) -> None:
        self._event_table.setRowCount(len(rows))
        for i, (prop, val) in enumerate(rows):
            self._event_table.setItem(i, 0, QTableWidgetItem(prop))
            val_item = QTableWidgetItem(val)
            val_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._event_table.setItem(i, 1, val_item)

    @staticmethod
    def _read_joint_property(
        asset, eid: int, data_attr: str, label: str
    ) -> list[tuple[str, str]]:
        """Read a per-joint property from the asset data for a single env."""
        import warp as wp

        rows: list[tuple[str, str]] = []
        data_obj = asset.data
        prop = getattr(data_obj, data_attr, None)
        if prop is None:
            rows.append((label, "N/A"))
            return rows

        vals = wp.to_torch(prop)
        if vals.dim() < 2:
            rows.append((label, "unexpected shape"))
            return rows

        env_vals = vals[eid].detach().cpu()
        joint_names = asset.joint_names if hasattr(asset, "joint_names") else None

        rows.append((f"--- {label} ({env_vals.numel()} joints) ---", ""))
        for j in range(env_vals.numel()):
            jname = joint_names[j] if joint_names and j < len(joint_names) else str(j)
            short = jname.split("/")[-1] if "/" in jname else jname
            rows.append((short, f"{float(env_vals[j]):.6f}"))
        return rows

    @staticmethod
    def _read_body_property(
        asset, eid: int, data_attr: str, label: str
    ) -> list[tuple[str, str]]:
        """Read a per-body property from the asset data for a single env."""
        import warp as wp

        rows: list[tuple[str, str]] = []
        prop = getattr(asset.data, data_attr, None)
        if prop is None:
            rows.append((label, "N/A"))
            return rows

        vals = wp.to_torch(prop)
        if vals.dim() < 2:
            rows.append((label, "unexpected shape"))
            return rows

        env_vals = vals[eid].detach().cpu()
        body_names = asset.body_names if hasattr(asset, "body_names") else None

        rows.append((f"--- {label} ({env_vals.numel()} bodies) ---", ""))
        for b in range(env_vals.numel()):
            bname = body_names[b] if body_names and b < len(body_names) else str(b)
            short = bname.split("/")[-1] if "/" in bname else bname
            rows.append((short, f"{float(env_vals[b]):.6f}"))
        return rows

    @staticmethod
    def _read_shape_friction(asset_name: str, eid: int) -> list[tuple[str, str]]:
        """Read shape_material_mu for all collision shapes of an asset in a given env."""
        import warp as wp
        from isaaclab.sim._impl.newton_manager import NewtonManager

        rows: list[tuple[str, str]] = []
        model = NewtonManager._model
        if model is None:
            rows.append(("shape_friction", "model not ready"))
            return rows

        import newton as nw

        collide_flag = int(nw.ShapeFlags.COLLIDE_SHAPES)
        body_keys = model.body_key
        body_world = model.body_world.numpy()
        shape_body = model.shape_body.numpy()
        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_key
        mu = model.shape_material_mu.numpy()

        target_bodies: set[int] = set()
        for b in range(model.body_count):
            if int(body_world[b]) != eid:
                continue
            bk = body_keys[b]
            if bk.endswith(f"/{asset_name}") or f"/{asset_name}/" in bk:
                target_bodies.add(b)

        shapes: list[tuple[int, str, float]] = []
        for s in range(model.shape_count):
            sb = int(shape_body[s])
            if sb not in target_bodies:
                continue
            if not (int(shape_flags[s]) & collide_flag):
                continue
            sname = shape_keys[s].split("/")[-1] if "/" in shape_keys[s] else shape_keys[s]
            shapes.append((s, sname, float(mu[s])))

        rows.append((f"--- shape friction ({len(shapes)} shapes) ---", ""))
        for s_idx, sname, mu_val in shapes:
            rows.append((sname, f"mu={mu_val:.4f}"))
        return rows

    @staticmethod
    def _read_root_pose(asset, eid: int) -> list[tuple[str, str]]:
        """Read root pose (pos + quat) for a single env."""
        import warp as wp

        rows: list[tuple[str, str]] = []
        root_pos = getattr(asset.data, "root_pos_w", None)
        root_quat = getattr(asset.data, "root_quat_w", None)
        if root_pos is None:
            rows.append(("root_pose", "N/A"))
            return rows

        pos = wp.to_torch(root_pos)[eid].detach().cpu()
        rows.append(("pos x", f"{float(pos[0]):.4f}"))
        rows.append(("pos y", f"{float(pos[1]):.4f}"))
        rows.append(("pos z", f"{float(pos[2]):.4f}"))

        if root_quat is not None:
            q = wp.to_torch(root_quat)[eid].detach().cpu()
            rows.append(("quat (w,x,y,z)", f"{float(q[3]):.4f}, {float(q[0]):.4f}, {float(q[1]):.4f}, {float(q[2]):.4f}"))
        return rows

    @staticmethod
    def _read_external_wrench(asset, eid: int) -> list[tuple[str, str]]:
        """Read the current external wrench buffer for a single env."""
        import warp as wp

        rows: list[tuple[str, str]] = []
        wrench_attr = getattr(asset.data, "_sim_bind_body_external_wrench", None)
        if wrench_attr is None:
            rows.append(("external_wrench", "N/A"))
            return rows

        wrench = wp.to_torch(wrench_attr)
        if wrench.dim() < 2:
            rows.append(("external_wrench", "unexpected shape"))
            return rows

        env_wrench = wrench[eid].detach().cpu()
        body_names = asset.body_names if hasattr(asset, "body_names") else None
        n_bodies = env_wrench.shape[0]

        has_nonzero = False
        for b in range(n_bodies):
            w = env_wrench[b]
            if w.abs().max().item() > 1e-8:
                has_nonzero = True
                bname = body_names[b] if body_names and b < len(body_names) else str(b)
                short = bname.split("/")[-1] if "/" in bname else bname
                force_str = f"({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f})"
                torque_str = f"({w[3]:.3f}, {w[4]:.3f}, {w[5]:.3f})"
                rows.append((f"{short} force", force_str))
                rows.append((f"{short} torque", torque_str))

        if not has_nonzero:
            rows.append(("external_wrench", "all zero"))
        return rows

    # ------------------------------------------------------------------ #
    #  Tick (called every sim step)                                       #
    # ------------------------------------------------------------------ #

    @property
    def picking_enabled(self) -> bool:
        return self._picking_cb.isChecked()

    def _slider_to_ke(self, value: int) -> float:
        """Map slider 0..600 → ke 0.01..10000 on a log10 scale."""
        return 10.0 ** (value / 100.0 - 2.0)

    @property
    def pick_stiffness(self) -> float:
        return self._slider_to_ke(self._pick_slider.value())

    @property
    def pick_damping(self) -> float:
        return self.pick_stiffness * 0.1

    def _on_pick_force_changed(self, value: int) -> None:
        ke = self._slider_to_ke(value)
        kd = ke * 0.1
        if ke >= 100:
            self._pick_force_label.setText(f"ke={ke:.0f}  kd={kd:.0f}")
        elif ke >= 1:
            self._pick_force_label.setText(f"ke={ke:.1f}  kd={kd:.2f}")
        else:
            self._pick_force_label.setText(f"ke={ke:.3f}  kd={kd:.4f}")

    def tick(self) -> None:
        if not self._live or not self._applied:
            return

        has_open_plots = any(w.isVisible() for w in self._plot_windows.values())

        if self.isVisible():
            page = self._pages.currentIndex()
            if page == _PAGE_JOINT:
                self._refresh_joints()
            elif page == _PAGE_ENV and self._selected_obj_idx >= 0:
                self._refresh_env_object()
            elif page == _PAGE_OBS and self._selected_obs_group:
                self._refresh_obs()
                for win in self._plot_windows.values():
                    if win.isVisible():
                        win.tick()
                return
            elif page == _PAGE_EVENT and self._selected_event_term:
                self._refresh_event_detail()

        if has_open_plots and self._selected_obs_group:
            self._collect_and_push_obs()
            for win in self._plot_windows.values():
                if win.isVisible():
                    win.tick()

    def _collect_and_push_obs(self) -> None:
        """Push obs data to plot windows even when ObsState tab is not active."""
        import torch

        env = self._env_ref
        if env is None or not hasattr(env, "observation_manager"):
            return
        obs_mgr = env.observation_manager
        group = self._selected_obs_group
        if group not in obs_mgr.active_terms:
            return
        obs_buf = obs_mgr._obs_buffer
        if obs_buf is None or group not in obs_buf:
            return

        data = obs_buf[group]
        eid = self._selected_env
        term_names = obs_mgr.active_terms[group]
        term_dims = obs_mgr.group_obs_term_dim[group]

        if isinstance(data, torch.Tensor):
            idx = 0
            for name, shape in zip(term_names, term_dims):
                length = int(np.prod(shape))
                vals = data[eid, idx : idx + length].detach().cpu().numpy()
                self._push_to_plot(group, name, vals)
                idx += length
        elif isinstance(data, dict):
            for name in term_names:
                if name not in data:
                    continue
                t = data[name]
                if isinstance(t, torch.Tensor) and t.dim() >= 1:
                    vals = t[eid].detach().cpu().flatten().numpy()
                    self._push_to_plot(group, name, vals)


# ------------------------------------------------------------------ #
#  Module-level singleton                                             #
# ------------------------------------------------------------------ #

_qt_app: QApplication | None = None
_debug_panel: DebugPanel | None = None


def ensure_debug_panel() -> DebugPanel:
    global _qt_app, _debug_panel

    if _debug_panel is not None and _debug_panel.isVisible():
        return _debug_panel

    if _qt_app is None:
        _qt_app = QApplication.instance() or QApplication([])

    _debug_panel = DebugPanel()
    _debug_panel.show()
    return _debug_panel


def set_debug_panel_env(env) -> None:
    """Register a ManagerBasedRLEnv with the debug panel for ObsState display."""
    if _debug_panel is not None:
        _debug_panel.set_env(env)


def pump_qt_events() -> None:
    if _qt_app is not None:
        _qt_app.processEvents()
