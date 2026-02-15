#!/usr/bin/env python3
"""ALLEX_newton.usd를 no_left와 일치시키기: 루트 prim /ALLEX -> /allex_contact_sensor, DefaultPrim 설정."""
from __future__ import annotations

import os
import sys

from pxr import Sdf, Usd


def main() -> int:
    base = os.path.join(os.path.dirname(__file__), "..", "source", "isaaclab_assets", "allex_usd")
    if not os.path.isdir(base):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "isaaclab_assets", "allex_usd"))
    path = os.path.join(base, "ALLEX_newton.usd")
    if not os.path.isfile(path):
        path = os.path.abspath("source/isaaclab_assets/allex_usd/ALLEX_newton.usd")
    if not os.path.isfile(path):
        print("ALLEX_newton.usd not found", file=sys.stderr)
        return 1

    stage = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    if not stage:
        print("Failed to open stage", file=sys.stderr)
        return 1

    prim_ALLEX = stage.GetPrimAtPath("/ALLEX")
    if not prim_ALLEX.IsValid():
        print("/ALLEX prim not found", file=sys.stderr)
        return 1

    # no_left와 동일하게: 루트 로봇 prim 이름을 allex_contact_sensor 로 변경
    editor = Usd.NamespaceEditor(stage)
    if not editor.RenamePrim(prim_ALLEX, "allex_contact_sensor"):
        print("RenamePrim failed", file=sys.stderr)
        return 1
    if not editor.ApplyEdits():
        print("ApplyEdits failed", file=sys.stderr)
        return 1

    # DefaultPrim 을 allex_contact_sensor 로 설정
    stage.SetDefaultPrim(stage.GetPrimAtPath("/allex_contact_sensor"))
    stage.GetRootLayer().Save()
    print("ALLEX_newton.usd updated: /ALLEX -> /allex_contact_sensor, DefaultPrim = /allex_contact_sensor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
