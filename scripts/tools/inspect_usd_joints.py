#!/usr/bin/env python3
"""Print joint names and DOF order from a USD articulation. Usage: python inspect_usd_joints.py <path/to/file.usd> [prim_path]"""

import sys
from pathlib import Path

# Add isaaclab source for USD/pxr
_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

def main():
    from pxr import Usd, UsdPhysics

    if len(sys.argv) < 2:
        print("Usage: python inspect_usd_joints.py <path/to/file.usd> [root_prim_path]", file=sys.stderr)
        sys.exit(1)
    usd_path = Path(sys.argv[1]).resolve()
    root_path = sys.argv[2] if len(sys.argv) > 2 else None

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open: {usd_path}", file=sys.stderr)
        sys.exit(2)

    def is_joint(prim):
        return (
            prim.IsA(UsdPhysics.RevoluteJoint)
            or prim.IsA(UsdPhysics.PrismaticJoint)
            or prim.IsA(UsdPhysics.SphericalJoint)
            or prim.IsA(UsdPhysics.FixedJoint)
            or prim.IsA(UsdPhysics.DistanceJoint)
            or (hasattr(UsdPhysics, "D6Joint") and prim.IsA(UsdPhysics.D6Joint))
            or (hasattr(UsdPhysics, "Joint") and prim.IsA(UsdPhysics.Joint))
        )

    def _traverse(prim):
        yield prim
        for c in prim.GetAllChildren():
            for p in _traverse(c):
                yield p

    def _joint_type(prim):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            return "Revolute"
        if prim.IsA(UsdPhysics.PrismaticJoint):
            return "Prismatic"
        if prim.IsA(UsdPhysics.FixedJoint):
            return "Fixed"
        if prim.IsA(UsdPhysics.SphericalJoint):
            return "Spherical"
        if prim.IsA(UsdPhysics.DistanceJoint):
            return "Distance"
        if hasattr(UsdPhysics, "D6Joint") and prim.IsA(UsdPhysics.D6Joint):
            return "D6"
        return "Joint"

    joints = []
    if root_path:
        root = stage.GetPrimAtPath(root_path)
        if not root:
            print(f"Prim not found: {root_path}", file=sys.stderr)
            sys.exit(3)
        for prim in root.GetAllChildren():
            for p in _traverse(prim):
                if is_joint(p):
                    joints.append((p.GetPath().pathString, p.GetName(), _joint_type(p)))
    else:
        for prim in stage.Traverse():
            if is_joint(prim):
                joints.append((prim.GetPath().pathString, prim.GetName(), _joint_type(prim)))

    # Sort by path for consistent order (often matches DOF order)
    joints.sort(key=lambda x: x[0])
    print(f"# USD: {usd_path}")
    print(f"# Total joint prims: {len(joints)}")
    print("# path | name | type")
    print("-" * 80)
    for path_str, name, jtype in joints:
        print(f"  {path_str!r} | {name!r} | {jtype}")
    # Names only, one per line (easy to copy into config)
    print()
    print("# Joint names (order as in USD):")
    names = [j[1] for j in joints]
    for n in names:
        print(n)
    print()
    print(f"# DOF count (excluding Fixed): {sum(1 for j in joints if j[2] != 'Fixed')}")

if __name__ == "__main__":
    main()
