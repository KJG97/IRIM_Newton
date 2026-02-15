#!/usr/bin/env python3
"""Set all mesh-related scale to (1,1,1) in allex_test.usd to match no_left.

- Scale 0.001 lives under instanceable prims (e.g. .../visuals/..., .../collisions/...).
  USD does not allow authoring on instance **prototypes**, so we must set
  instanceable=False on those prims first, then fix scale on their descendants.
- Replaces (0.001, 0.001, 0.001) with (1, 1, 1).
- If --force is given, sets every xformOp:scale to (1, 1, 1) regardless of current value.
"""
from __future__ import annotations

import argparse
import os
import sys

from pxr import Usd, UsdGeom


def _fix_prim_scale(prim, new_scale, target_scale, force_all_to_one, tol) -> int:
    """Fix scale on a single prim. Returns number of attributes changed (0 or 1)."""
    xform = UsdGeom.Xformable(prim)
    if not xform:
        return 0
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
        if force_all_to_one:
            if abs(v[0] - 1.0) > tol or abs(v[1] - 1.0) > tol or abs(v[2] - 1.0) > tol:
                op.Set(new_scale)
                return 1
        else:
            if (
                abs(v[0] - target_scale[0]) < tol
                and abs(v[1] - target_scale[1]) < tol
                and abs(v[2] - target_scale[2]) < tol
            ):
                op.Set(new_scale)
                return 1
    return 0


def fix_scales_in_stage(
    stage: Usd.Stage,
    *,
    target_scale=(0.001, 0.001, 0.001),
    new_scale=(1.0, 1.0, 1.0),
    force_all_to_one: bool = False,
    tol: float = 1e-6,
) -> int:
    """Uninstance visuals/collisions, then set scale to new_scale. Returns count changed."""
    count = 0

    # 1) Set instanceable=False on all instanceable prims so we can author on their children.
    #    (USD does not allow editing prototype content; we must "break" the instance.)
    uninstanced = []
    for prim in stage.Traverse():
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
            uninstanced.append(str(prim.GetPath()))
    if uninstanced:
        print(f"Uninstanced {len(uninstanced)} prim(s) (e.g. .../visuals, .../collisions).")

    # 2) Regular traversal: fold unitsResolve and fix scale on every prim (including ex-instance children)
    for prim in stage.Traverse():
        xform = UsdGeom.Xformable(prim)
        if not xform:
            continue
        units_attr = prim.GetAttribute("xformOp:scale:unitsResolve")
        if units_attr and units_attr.HasAuthoredValue():
            scale_attr = prim.GetAttribute("xformOp:scale")
            if scale_attr and scale_attr.HasAuthoredValue():
                cur = scale_attr.Get()
                u = units_attr.Get()
                if cur is not None and u is not None:
                    folded = (float(cur[0]) * float(u[0]), float(cur[1]) * float(u[1]), float(cur[2]) * float(u[2]))
                    scale_attr.Set(folded)
                    prim.RemoveProperty("xformOp:scale:unitsResolve")
                    count += 1
                    print(f"  {prim.GetPath()} folded unitsResolve -> scale {folded}")
        n = _fix_prim_scale(prim, new_scale, target_scale, force_all_to_one, tol)
        if n:
            count += n
            print(f"  {prim.GetPath()} scale -> {new_scale}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Set mesh scale to 1.0 in allex_test.usd (like no_left).")
    parser.add_argument("--force", action="store_true", help="Set every xformOp:scale to (1,1,1) regardless of value")
    parser.add_argument("path", nargs="?", default=None, help="USD file path (default: allex_usd/allex_test.usd)")
    args = parser.parse_args()

    if args.path and os.path.isfile(args.path):
        path = os.path.abspath(args.path)
    else:
        base = os.path.join(os.path.dirname(__file__), "..", "..", "source", "isaaclab_assets", "allex_usd")
        if not os.path.isdir(base):
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "isaaclab_assets", "allex_usd"))
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

    new_scale = (1.0, 1.0, 1.0)
    count = fix_scales_in_stage(
        stage,
        target_scale=(0.001, 0.001, 0.001),
        new_scale=new_scale,
        force_all_to_one=args.force,
    )

    if count == 0:
        print("No scale changes needed (0.001 not found; use --force to set all scales to 1.0).")
        return
    stage.GetRootLayer().Save()
    print(f"Updated {count} scale(s). Saved.")


if __name__ == "__main__":
    main()
