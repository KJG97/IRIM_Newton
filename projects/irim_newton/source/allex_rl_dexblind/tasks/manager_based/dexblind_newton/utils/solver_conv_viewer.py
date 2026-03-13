# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Solver conv log: main process appends to a temp file; viewer is a separate process that tails it."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading

_LOG_FILE = None
_STARTED = False
_LOCK = threading.Lock()


def start_solver_conv_viewer() -> None:
    """Create log file, open it (line-buffered), start viewer subprocess. Idempotent."""
    global _LOG_FILE, _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
        path = os.path.join(tempfile.gettempdir(), f"solver_conv_{os.getpid()}.log")
        _LOG_FILE = open(path, "a", encoding="utf-8", buffering=1)
    subprocess.Popen(
        [sys.executable, __file__, path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


_HIT_PREFIX = "\x01HIT\x01"


def push_solver_conv_line(line: str, hit_limit: bool = False) -> None:
    """Append one line to the log file. If hit_limit, viewer will show it in red."""
    if _LOG_FILE is None:
        return
    try:
        if hit_limit:
            line = _HIT_PREFIX + line
        _LOG_FILE.write(line + "\n")
    except OSError:
        pass


def _run_viewer_ui(log_path: str) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.title("Solver conv")
    root.geometry("2400x2400")
    root.resizable(True, True)
    frame = tk.Frame(root, padx=4, pady=4)
    frame.pack(fill=tk.BOTH, expand=True)
    scrollbar = tk.Scrollbar(frame, width=24)
    text = tk.Text(
        frame, font=("Consolas", 30), wrap=tk.WORD, height=1, state=tk.NORMAL,
        yscrollcommand=scrollbar.set,
    )
    text.tag_configure("hit_limit", foreground="red")
    scrollbar.config(command=text.yview)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    _HIT = "\x01HIT\x01"
    _buf: list[str] = [""]

    def process_line(raw: str) -> None:
        if raw.startswith(_HIT):
            text.insert(tk.END, raw[len(_HIT) :] + "\n", "hit_limit")
        else:
            text.insert(tk.END, raw + "\n")
        text.see(tk.END)

    try:
        with open(log_path, "r", encoding="utf-8") as fd:
            def poll() -> None:
                try:
                    new = fd.read()
                    if new:
                        _buf[0] += new
                        lines = _buf[0].split("\n")
                        _buf[0] = lines[-1]
                        for ln in lines[:-1]:
                            if ln:
                                process_line(ln)
                except OSError:
                    pass
                if root.winfo_exists():
                    root.after(100, poll)

            root.after(100, poll)
            root.mainloop()
    except OSError:
        root.destroy()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m ...solver_conv_viewer <log_file_path>")
    _run_viewer_ui(sys.argv[1])
