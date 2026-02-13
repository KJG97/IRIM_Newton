# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Joint slider agent: same-process PySide GUI to control joint targets.

Run: ./isaaclab.sh -p scripts/environments/joint_slider_agent.py --task Isaac-Allex-Direct-NoLeft-v0 --visualizer newton

No TCP; GUI runs in the same process. Only active (driver) joints; mimic/passive excluded. Sliders in degrees.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Joint slider agent: PySide GUI for joint control.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import warp as wp

from isaaclab.utils import close_simulation, is_simulation_running
from isaaclab.utils.timer import Timer

Timer.enable = False
Timer.enable_display_output = False

import isaaclab_tasks_experimental  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.direct.allex.allex_env_cfg import ALLEX_MIMIC_SPEC

# PySide6 (same process, no TCP)
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QScrollArea,
        QSlider,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QScrollArea,
        QSlider,
        QWidget,
    )

# Action scale used in AllexEnv._apply_action (target = current + scale * actions)
ALLEX_ACTION_SCALE = 0.5

# Slider integer range; value mapped to [lower_deg, upper_deg]
SLIDER_RESOLUTION = 10000


def _rad2deg(r: float) -> float:
    return math.degrees(r)


def _deg2rad(d: float) -> float:
    return math.radians(d)


def _poly_scalar(q_rad: float, c0: float, c1: float, c2: float, c3: float, c4: float) -> float:
    """q_mimic = c0 + c1*q + c2*q^2 + c3*q^3 + c4*q^4 (scalar, for passive target display)."""
    q2 = q_rad * q_rad
    q3 = q2 * q_rad
    q4 = q3 * q_rad
    return c0 + c1 * q_rad + c2 * q2 + c3 * q3 + c4 * q4


# GUI colors: dark gray bg, bold; coupling-free=green, active(coupling)=red, passive=light red
STYLE_DARK = """
    QMainWindow, QWidget, QScrollArea { background-color: #3d3d3d; }
    QLabel, QSlider { font-weight: bold; color: #e0e0e0; }
"""
STYLE_DRIVER_NO_COUPLING = "font-weight: bold; color: #90EE90;"   # 연두
STYLE_DRIVER_COUPLING = "font-weight: bold; color: #FF0000;"     # 빨강 (Active)
STYLE_PASSIVE = "font-weight: bold; color: #FFB0B0;"             # 연한 빨강


class JointSliderWindow(QMainWindow):
    """Slider GUI: drivers (green/red by coupling), passive rows under active with current/target from _poly."""

    def __init__(
        self,
        driver_names: list[str],
        driver_lower_rad: list[float],
        driver_upper_rad: list[float],
        driver_full_indices: list[int],
        num_full: int,
        driver_initial_rad: list[float] | None = None,
        is_coupling_driver: list[bool] | None = None,
        driver_mimics_info: list[list[tuple[str, int, tuple[float, ...]]]] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Joint Slider — target (°)")
        self.setStyleSheet(STYLE_DARK)
        self._driver_names = driver_names
        self._lower_rad = driver_lower_rad
        self._upper_rad = driver_upper_rad
        self._driver_full_indices = driver_full_indices
        self._num_full = num_full
        self._n = len(driver_names)
        self._sliders: list[QSlider] = []
        self._current_labels: list[QLabel] = []
        self._target_labels: list[QLabel] = []
        if is_coupling_driver is None or len(is_coupling_driver) != self._n:
            is_coupling_driver = [False] * self._n
        self._is_coupling_driver = is_coupling_driver
        if driver_mimics_info is None or len(driver_mimics_info) != self._n:
            driver_mimics_info = [[] for _ in range(self._n)]
        self._driver_mimics_info = driver_mimics_info
        # passive rows: (driver_k, polycoef, current_lbl, target_lbl, mimic_full_idx)
        self._passive_display: list[tuple[int, tuple[float, ...], QLabel, QLabel, int]] = []
        if driver_initial_rad is None or len(driver_initial_rad) != self._n:
            driver_initial_rad = [0.0] * self._n
        self._driver_initial_rad = driver_initial_rad

        central = QWidget()
        self.setCentralWidget(central)
        layout = QFormLayout(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #3d3d3d;")
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: #3d3d3d;")
        scroll_layout = QFormLayout(scroll_content)
        scroll.setWidget(scroll_content)

        for i in range(self._n):
            name = driver_names[i]
            driver_style = STYLE_DRIVER_COUPLING if self._is_coupling_driver[i] else STYLE_DRIVER_NO_COUPLING
            lo_rad, hi_rad = driver_lower_rad[i], driver_upper_rad[i]
            current_lbl = QLabel("0.0°")
            current_lbl.setMinimumWidth(56)
            current_lbl.setStyleSheet(driver_style)
            self._current_labels.append(current_lbl)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, SLIDER_RESOLUTION)
            initial_deg = _rad2deg(self._driver_initial_rad[i])
            slider.setValue(self._deg_to_slider(i, initial_deg))
            slider.valueChanged.connect(lambda v, idx=i: self._on_slider(idx, v))
            self._sliders.append(slider)
            target_lbl = QLabel("0.0°")
            target_lbl.setMinimumWidth(56)
            target_lbl.setStyleSheet(driver_style)
            self._target_labels.append(target_lbl)
            row_widget = QWidget()
            row_widget.setStyleSheet("background-color: #3d3d3d;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(driver_style)
            row_layout.addWidget(name_lbl)
            row_layout.addWidget(QLabel("current:"))
            row_layout.addWidget(current_lbl)
            row_layout.addWidget(QLabel("target:"))
            row_layout.addWidget(target_lbl)
            row_layout.addWidget(slider, 1)
            scroll_layout.addRow(row_widget)
            self._update_target_label(i)

            # Passive (mimic) rows under this driver: current = sim value, target = _poly(driver)
            for mimic_name, mimic_full_idx, polycoef in self._driver_mimics_info[i]:
                p_cur = QLabel("0.0°")
                p_cur.setMinimumWidth(56)
                p_cur.setStyleSheet(STYLE_PASSIVE)
                p_tgt = QLabel("0.0°")
                p_tgt.setMinimumWidth(56)
                p_tgt.setStyleSheet(STYLE_PASSIVE)
                self._passive_display.append((i, polycoef, p_cur, p_tgt, mimic_full_idx))
                pw = QWidget()
                pw.setStyleSheet("background-color: #3d3d3d;")
                pl = QHBoxLayout(pw)
                pl.setContentsMargins(0, 0, 0, 0)
                pname = QLabel("  └ " + mimic_name)
                pname.setStyleSheet(STYLE_PASSIVE)
                pl.addWidget(pname)
                pl.addWidget(QLabel("current:"))
                pl.addWidget(p_cur)
                pl.addWidget(QLabel("target:"))
                pl.addWidget(p_tgt)
                scroll_layout.addRow(pw)

        layout.addRow(scroll)
        self.resize(500, 800)

    def _slider_to_deg(self, slider_idx: int, slider_val: int) -> float:
        lo, hi = self._lower_rad[slider_idx], self._upper_rad[slider_idx]
        lo_deg, hi_deg = _rad2deg(lo), _rad2deg(hi)
        return lo_deg + (hi_deg - lo_deg) * (slider_val / SLIDER_RESOLUTION)

    def _deg_to_slider(self, slider_idx: int, deg: float) -> int:
        lo, hi = self._lower_rad[slider_idx], self._upper_rad[slider_idx]
        lo_deg, hi_deg = _rad2deg(lo), _rad2deg(hi)
        if hi_deg <= lo_deg:
            return 0
        t = (deg - lo_deg) / (hi_deg - lo_deg)
        val = int(0.5 + t * SLIDER_RESOLUTION)
        return max(0, min(SLIDER_RESOLUTION, val))

    def _on_slider(self, idx: int, value: int) -> None:
        self._update_target_label(idx)
        self._update_passive_targets_for_driver(idx)

    def _update_target_label(self, idx: int) -> None:
        deg = self._slider_to_deg(idx, self._sliders[idx].value())
        self._target_labels[idx].setText(f"{deg:.1f}°")

    def _update_passive_targets_for_driver(self, driver_k: int) -> None:
        """Update target labels of passive rows that follow driver_k (target = _poly(driver))."""
        q_driver_rad = _deg2rad(self._slider_to_deg(driver_k, self._sliders[driver_k].value()))
        for dk, polycoef, _cur_lbl, tgt_lbl, _mimic_idx in self._passive_display:
            if dk != driver_k:
                continue
            c0, c1, c2, c3, c4 = (polycoef + (0.0,) * 5)[:5]
            q_mimic_rad = _poly_scalar(q_driver_rad, c0, c1, c2, c3, c4)
            tgt_lbl.setText(f"{_rad2deg(q_mimic_rad):.1f}°")

    def get_target_positions(self, current_full: list[float]) -> list[float]:
        """Build full (num_full) target in rad: driver = slider value (rad), mimic = current."""
        target = list(current_full)
        for k in range(self._n):
            j = self._driver_full_indices[k]
            deg = self._slider_to_deg(k, self._sliders[k].value())
            target[j] = _deg2rad(deg)
        return target

    def update_current_labels_rad(self, full_positions_rad: list[float]) -> None:
        """Update current (°) for drivers and passive rows; passive target = _poly(driver)."""
        if len(full_positions_rad) != self._num_full:
            return
        for k in range(self._n):
            j = self._driver_full_indices[k]
            deg = _rad2deg(full_positions_rad[j])
            self._current_labels[k].setText(f"{deg:.1f}°")
        for driver_k, polycoef, cur_lbl, tgt_lbl, mimic_idx in self._passive_display:
            cur_lbl.setText(f"{_rad2deg(full_positions_rad[mimic_idx]):.1f}°")
            q_driver_rad = _deg2rad(self._slider_to_deg(driver_k, self._sliders[driver_k].value()))
            c0, c1, c2, c3, c4 = (polycoef + (0.0,) * 5)[:5]
            q_mimic_rad = _poly_scalar(q_driver_rad, c0, c1, c2, c3, c4)
            tgt_lbl.setText(f"{_rad2deg(q_mimic_rad):.1f}°")


def main():
    """Joint slider agent: env loop + same-process PySide window."""
    qt_app = QApplication.instance() or QApplication([])

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    env.reset()

    unwrapped = env.unwrapped
    device = unwrapped.device
    num_envs = unwrapped.num_envs
    action_scale = ALLEX_ACTION_SCALE
    first_step_done = False
    robot = None
    joint_dof_idx = None
    slider_window: JointSliderWindow | None = None

    while is_simulation_running(simulation_app, unwrapped.sim):
        with torch.inference_mode():
            if not first_step_done:
                actions = torch.zeros(env.action_space.shape, device=device)
                env.step(actions)
                unwrapped._ensure_joint_dof_idx()
                robot = unwrapped.robot
                joint_names = unwrapped._joint_names
                joint_dof_idx = unwrapped._joint_dof_idx
                lower_w = wp.to_torch(robot.data.joint_pos_limits_lower)[0]
                upper_w = wp.to_torch(robot.data.joint_pos_limits_upper)[0]
                lower = [float(lower_w[i].item()) for i in joint_dof_idx]
                upper = [float(upper_w[i].item()) for i in joint_dof_idx]
                # Only active (driver) joints; exclude mimic/passive from ALLEX_MIMIC_SPEC
                mimic_names = {m[0] for m in ALLEX_MIMIC_SPEC}
                driver_full_indices = [i for i, n in enumerate(joint_names) if n not in mimic_names]
                driver_names = [joint_names[i] for i in driver_full_indices]
                driver_lower = [lower[i] for i in driver_full_indices]
                driver_upper = [upper[i] for i in driver_full_indices]
                num_full = len(joint_names)
                name_to_full_idx = {joint_names[i]: i for i in range(num_full)}
                driver_name_to_k = {driver_names[k]: k for k in range(len(driver_names))}
                driver_mimics_info = [[] for _ in range(len(driver_names))]
                for mimic_name, driver_name, polycoef in ALLEX_MIMIC_SPEC:
                    if driver_name not in driver_name_to_k or mimic_name not in name_to_full_idx:
                        continue
                    k = driver_name_to_k[driver_name]
                    mimic_full_idx = name_to_full_idx[mimic_name]
                    driver_mimics_info[k].append((mimic_name, mimic_full_idx, polycoef))
                is_coupling_driver = [len(driver_mimics_info[k]) > 0 for k in range(len(driver_names))]
                current_rad = wp.to_torch(robot.data.joint_pos)[0, joint_dof_idx].cpu().tolist()
                driver_initial_rad = [current_rad[i] for i in driver_full_indices]
                slider_window = JointSliderWindow(
                    driver_names,
                    driver_lower,
                    driver_upper,
                    driver_full_indices,
                    num_full,
                    driver_initial_rad=driver_initial_rad,
                    is_coupling_driver=is_coupling_driver,
                    driver_mimics_info=driver_mimics_info,
                )
                slider_window.show()
                print(f"[INFO]: Joint slider ready: {len(driver_names)} active joints (degree), {num_full - len(driver_names)} mimic excluded")
                first_step_done = True
                continue

            # Process Qt events so window stays responsive
            qt_app.processEvents()
            if slider_window is not None and not slider_window.isVisible():
                break

            # Current positions (env 0, action order) — full 31-dim
            joint_pos = wp.to_torch(robot.data.joint_pos)[0, joint_dof_idx]
            current_list = joint_pos.cpu().tolist()
            slider_window.update_current_labels_rad(current_list)

            # Target from GUI (full 31-dim: driver from sliders, mimic = current)
            target = slider_window.get_target_positions(current_list)
            target_t = torch.tensor(target, device=device, dtype=torch.float32)
            current_t = joint_pos
            actions = (target_t - current_t) / action_scale
            actions = torch.clamp(actions, -1.0, 1.0)

            env.step(actions)

    if slider_window is not None:
        slider_window.close()
    env.close()


if __name__ == "__main__":
    main()
    close_simulation(simulation_app)
