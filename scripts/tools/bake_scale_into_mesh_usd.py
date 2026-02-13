#!/usr/bin/env python3
"""Bake xform scale (e.g. 0.001 for mm→m) into mesh vertices in a USD file.

Newton's add_usd() does not apply USD xform scale when loading mesh geometry,
so assets with a parent scale (e.g. 0.001) appear 1000x too large in the Newton
visualizer. This script multiplies mesh points by the scale and sets the
xform scale to (1,1,1), so both Omniverse and Newton show correct size.

Usage:
  conda run -n isaaclab python scripts/tools/bake_scale_into_mesh_usd.py [input.usd] [output.usd]
  If output is omitted, writes to input.usd (overwrite).
  Default input: source/isaaclab_assets/allex_usd/ALLEX_newton_no_left.usd
"""
from pathlib import Path
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom


def get_local_scale(prim: Usd.Prim) -> tuple[float, float, float] | None:
    """Get xformOp:scale value if present."""
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            val = op.Get()
            return (float(val[0]), float(val[1]), float(val[2]))
    return None


def set_local_scale(prim: Usd.Prim, scale: tuple[float, float, float]) -> None:
    """Set or add xformOp:scale and ensure it's in xformOpOrder."""
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    scale_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if scale_op is None:
        scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    scale_op.Set(Gf.Vec3d(*scale))


def get_descendant_meshes(stage: Usd.Stage, root_path: Sdf.Path) -> list:
    """Return all Mesh prims under root_path (excluding root itself)."""
    meshes = []
    for prim in stage.Traverse():
        if prim.GetPath() == root_path:
            continue
        if not str(prim.GetPath()).startswith(root_path.pathString + "/"):
            continue
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
    return meshes


def collect_scale_prims_and_meshes(stage: Usd.Stage, target_scale: tuple[float, float, float]):
    """Find prims with the given scale and their descendant Mesh prims.
    Returns list of (scale_prim, [mesh_prim, ...])."""
    scale_prim_to_meshes = []
    for prim in stage.Traverse():
        s = get_local_scale(prim)
        if s is None or abs(s[0] - target_scale[0]) > 1e-9:
            continue
        meshes = get_descendant_meshes(stage, prim.GetPath())
        if meshes:
            scale_prim_to_meshes.append((prim, meshes))
    return scale_prim_to_meshes


def bake_scale_into_meshes(
    stage: Usd.Stage,
    target_scale: tuple[float, float, float] = (0.001, 0.001, 0.001),
) -> int:
    """Multiply mesh points under prims with target_scale by that scale, then set scale to (1,1,1).
    Returns number of meshes modified."""
    pairs = collect_scale_prims_and_meshes(stage, target_scale)
    count = 0
    for scale_prim, mesh_prims in pairs:
        sx, sy, sz = target_scale
        for mesh_prim in mesh_prims:
            mesh = UsdGeom.Mesh(mesh_prim)
            pts_attr = mesh.GetPointsAttr()
            if not pts_attr:
                continue
            pts = np.asarray(pts_attr.Get(), dtype=np.float64)
            pts = pts * np.array([sx, sy, sz], dtype=np.float64)
            pts_attr.Set(pts)
            count += 1
        set_local_scale(scale_prim, (1.0, 1.0, 1.0))
    return count


def main():
    repo = Path(__file__).resolve().parents[2]
    default_input = repo / "source/isaaclab_assets/allex_usd/ALLEX_newton_no_left.usd"

    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    else:
        input_path = default_input

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path

    if not input_path.is_absolute():
        input_path = repo / input_path
    if not output_path.is_absolute():
        output_path = repo / output_path

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    stage = Usd.Stage.Open(str(input_path))
    n = bake_scale_into_meshes(stage, target_scale=(0.001, 0.001, 0.001))
    print(f"Baked scale 0.001 into {n} mesh(es).")

    if output_path == input_path:
        # Export in place: need to get root layer and export
        stage.GetRootLayer().Export(str(output_path))
        print(f"Overwrote: {output_path}")
    else:
        stage.GetRootLayer().Export(str(output_path))
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
