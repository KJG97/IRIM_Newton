#!/usr/bin/env python3
"""Print all Xform scale ops and their values in allex_test.usd."""
from __future__ import annotations

import os
import sys

from pxr import Usd, UsdGeom


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "..", "..", "source", "isaaclab_assets", "allex_usd")
    path = os.path.join(base, "allex_test.usd")
    if not os.path.isfile(path):
        path = os.path.abspath("source/isaaclab_assets/allex_usd/allex_test.usd")
    if not os.path.isfile(path):
        print("allex_test.usd not found", file=sys.stderr)
        sys.exit(1)

    stage = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    if not stage:
        print("Failed to open stage", file=sys.stderr)
        sys.exit(1)

    for prim in stage.Traverse():
        xform = UsdGeom.Xformable(prim)
        if not xform:
            continue
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() != UsdGeom.XformOp.TypeScale:
                continue
            val = op.Get()
            if val is not None:
                print(f"{prim.GetPath()}  scale = {val}  (type {type(val)})")


if __name__ == "__main__":
    main()
