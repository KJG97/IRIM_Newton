#!/usr/bin/env python3
"""Compare mesh scale between ALLEX_Right_Arm.usd and ALLEX_newton_no_left.usd."""
from pathlib import Path

try:
    from pxr import Usd, UsdGeom, Gf
except ImportError:
    print("pxr not available in current env. Try: conda activate isaaclab")
    raise


def get_xform_scale(prim) -> Gf.Vec3d | None:
    """Get scale from prim or its xformable schema."""
    if not prim:
        return None
    xform = UsdGeom.Xformable(prim)
    if not xform:
        return None
    ops = xform.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            return op.Get()
    return None


def get_world_transform_scale(prim) -> Gf.Vec3d | None:
    """Get scale component from full world transform (includes parent hierarchy)."""
    if not prim:
        return None
    xform = UsdGeom.Xformable(prim)
    if not xform:
        return None
    world = xform.ComputeLocalToWorldTransform(0)
    # Extract scale from matrix (length of column vectors for scale)
    import math
    sx = math.sqrt(world[0][0]**2 + world[1][0]**2 + world[2][0]**2)
    sy = math.sqrt(world[0][1]**2 + world[1][1]**2 + world[2][1]**2)
    sz = math.sqrt(world[0][2]**2 + world[1][2]**2 + world[2][2]**2)
    return Gf.Vec3d(sx, sy, sz)


def collect_mesh_scales(stage, prefix=""):
    """Collect path -> local scale for all Mesh prims (and their parent scales)."""
    scales = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh):
            # Mesh: check parent chain for scale
            p = prim
            scale = None
            while p:
                s = get_xform_scale(p)
                if s is not None:
                    scale = s
                    break
                p = p.GetParent()
            if scale is not None:
                scales[path] = scale
            else:
                scales[path] = "(no local scale)"
        elif prim.IsA(UsdGeom.Xformable):
            s = get_xform_scale(prim)
            if s is not None:
                scales[path] = s
    return scales


def collect_all_xform_scales(stage):
    """Collect all prims that have scale op and their scale value."""
    out = []
    for prim in stage.Traverse():
        s = get_xform_scale(prim)
        if s is not None:
            out.append((str(prim.GetPath()), s))
    return out


def main():
    base = Path(__file__).resolve().parents[2] / "source/isaaclab_assets/allex_usd"
    right_usd = base / "ALLEX_Right_Arm.usd"
    no_left_usd = base / "ALLEX_newton_no_left.usd"

    for label, path in [("ALLEX_Right_Arm.usd", right_usd), ("ALLEX_newton_no_left.usd", no_left_usd)]:
        if not path.exists():
            print(f"Not found: {path}")
            continue
        stage = Usd.Stage.Open(str(path))
        print(f"\n=== {label} ===")
        scales = collect_all_xform_scales(stage)
        if not scales:
            print("  (no xform scale ops found)")
        for prim_path, s in scales:
            print(f"  {prim_path}: scale = ({s[0]:.6f}, {s[1]:.6f}, {s[2]:.6f})")

    # Summary: compare first few scales if both have meshes
    print("\n--- Summary ---")
    s1 = Usd.Stage.Open(str(right_usd))
    s2 = Usd.Stage.Open(str(no_left_usd))
    scales_right = collect_all_xform_scales(s1)
    scales_no_left = collect_all_xform_scales(s2)
    if scales_right and scales_no_left:
        # Compare by path or first N
        for (p1, v1) in scales_right[:5]:
            print(f"  Right: {p1} -> {v1}")
        for (p2, v2) in scales_no_left[:5]:
            print(f"  NoLeft: {p2} -> {v2}")


if __name__ == "__main__":
    main()
