#!/usr/bin/env python3
"""Set all mesh-related scale (0.001,0.001,0.001) to (1,1,1) in ALLEX_newton.usd to match no_left."""
from __future__ import annotations

import os
import sys

from pxr import Usd, UsdGeom


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "..", "source", "isaaclab_assets", "allex_usd")
    if not os.path.isdir(base):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "isaaclab_assets", "allex_usd"))
    path = os.path.join(base, "ALLEX_newton.usd")
    if not os.path.isfile(path):
        path = os.path.abspath("source/isaaclab_assets/allex_usd/ALLEX_newton.usd")
    if not os.path.isfile(path):
        print("ALLEX_newton.usd not found", file=sys.stderr)
        sys.exit(1)

    stage = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    if not stage:
        print("Failed to open stage", file=sys.stderr)
        sys.exit(1)

    target_scale = (0.001, 0.001, 0.001)
    new_scale = (1.0, 1.0, 1.0)
    count = 0
    for prim in stage.Traverse():
        xform = UsdGeom.Xformable(prim)
        if not xform:
            continue
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() != UsdGeom.XformOp.TypeScale:
                continue
            val = op.Get()
            if val is None:
                continue
            try:
                v = (float(val[0]), float(val[1]), float(val[2]))
            except (TypeError, IndexError):
                continue
            if abs(v[0] - target_scale[0]) < 1e-6 and abs(v[1] - target_scale[1]) < 1e-6 and abs(v[2] - target_scale[2]) < 1e-6:
                op.Set(new_scale)
                count += 1
                print(f"  {prim.GetPath()} scale {v} -> (1,1,1)")

    if count == 0:
        print("No scale (0.001,0.001,0.001) found.")
        return
    stage.GetRootLayer().Save()
    print(f"Updated {count} scale(s). Saved.")


if __name__ == "__main__":
    main()
