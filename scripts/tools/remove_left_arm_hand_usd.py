#!/usr/bin/env python3
"""
Remove left arm and left hand (links + joints) from an ALLEX USD file.
Optionally convert the two neck joints to Fixed joints.

Target for removal: all prims whose name starts with "L_" or "Left_" (ALLEX 왼팔/왼손).

Neck joints: prims named like Neck_Pitch_Joint, Neck_Yaw_Joint (or containing "Neck" and
"Pitch"/"Yaw") are converted from Revolute (or other) to FixedJoint, preserving body0/body1
and local pos/rot.

Requires: pxr (USD). Run with Isaac Sim Python or isaaclab conda env.
  ./isaaclab.sh -c "python scripts/tools/remove_left_arm_hand_usd.py <input.usd> [output.usd]"
  If output is omitted, overwrites input.
  Example:
  python remove_left_arm_hand_usd.py source/isaaclab_assets/allex_usd/ALLEX_XML_test.usd source/isaaclab_assets/allex_usd/ALLEX_XML_no_left.usd
"""

import sys
from pathlib import Path


def _is_joint(prim, UsdPhysics):
    return (
        prim.IsA(UsdPhysics.RevoluteJoint)
        or prim.IsA(UsdPhysics.PrismaticJoint)
        or prim.IsA(UsdPhysics.SphericalJoint)
        or prim.IsA(UsdPhysics.FixedJoint)
        or (hasattr(UsdPhysics, "D6Joint") and prim.IsA(UsdPhysics.D6Joint))
    )


def _is_neck_joint(prim, UsdPhysics):
    """목 2개 조인트: Neck_Pitch_Joint, Neck_Yaw_Joint (또는 이름에 Neck + Pitch/Yaw 포함)."""
    if not _is_joint(prim, UsdPhysics) or prim.IsA(UsdPhysics.FixedJoint):
        return False
    name = prim.GetName()
    return "Neck" in name and ("Pitch" in name or "Yaw" in name)


def _convert_neck_joints_to_fixed(stage, UsdPhysics):
    """Convert all neck joints (Revolute etc.) to FixedJoint; return number converted."""
    to_convert = []
    for prim in stage.Traverse():
        if _is_neck_joint(prim, UsdPhysics):
            to_convert.append(prim)

    for prim in to_convert:
        path = prim.GetPath()
        body0_rel = prim.GetRelationship("body0")
        body1_rel = prim.GetRelationship("body1")
        targets0 = body0_rel.GetTargets() if body0_rel else []
        targets1 = body1_rel.GetTargets() if body1_rel else []
        if len(targets0) != 1 or len(targets1) != 1:
            print(f"  [WARN] Skipping {path}: body0/body1 targets not single.", file=sys.stderr)
            continue

        # Copy local frame for consistency
        local_pos0 = prim.GetAttribute("localPos0").Get() if prim.HasAttribute("localPos0") else None
        local_rot0 = prim.GetAttribute("localRot0").Get() if prim.HasAttribute("localRot0") else None
        local_pos1 = prim.GetAttribute("localPos1").Get() if prim.HasAttribute("localPos1") else None
        local_rot1 = prim.GetAttribute("localRot1").Get() if prim.HasAttribute("localRot1") else None

        stage.RemovePrim(path)
        fixed = UsdPhysics.FixedJoint.Define(stage, path)
        fixed.CreateBody0Rel().SetTargets(targets0)
        fixed.CreateBody1Rel().SetTargets(targets1)
        if local_pos0 is not None:
            fixed.CreateLocalPos0Attr().Set(local_pos0)
        if local_rot0 is not None:
            fixed.CreateLocalRot0Attr().Set(local_rot0)
        if local_pos1 is not None:
            fixed.CreateLocalPos1Attr().Set(local_pos1)
        if local_rot1 is not None:
            fixed.CreateLocalRot1Attr().Set(local_rot1)
        print(f"  Converted to Fixed: {path}")

    return len(to_convert)


def main():
    from pxr import Usd, UsdPhysics

    if len(sys.argv) < 2:
        print("Usage: python remove_left_arm_hand_usd.py <input.usd> [output.usd]", file=sys.stderr)
        sys.exit(1)
    input_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else input_path

    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    stage = Usd.Stage.Open(str(input_path))
    if not stage:
        print(f"Failed to open: {input_path}", file=sys.stderr)
        sys.exit(3)

    # 1) 목 2개 조인트를 Fixed로 변경 (L_/Left_ 제거 전에 수행)
    n_fixed = _convert_neck_joints_to_fixed(stage, UsdPhysics)
    if n_fixed:
        print(f"Neck joints converted to Fixed: {n_fixed}")

    # 2) L_, Left_ 접두사 링크·조인트 제거 (ALLEX 왼팔/왼손)
    to_remove = set()
    for prim in stage.Traverse():
        name = prim.GetName()
        if name.startswith("L_") or name.startswith("Left_"):
            to_remove.add(prim.GetPath())

    root_paths = [p for p in to_remove if p.GetParentPath() not in to_remove]
    root_paths.sort(key=lambda p: len(p.pathString), reverse=True)

    removed = []
    for path in root_paths:
        prim = stage.GetPrimAtPath(path)
        if prim:
            removed.append(path.pathString)
            stage.RemovePrim(path)

    print(f"Removed {len(removed)} root prim(s) (L_/Left_* and their descendants):")
    for r in removed:
        print(f"  {r}")
    print(f"Total L_*/Left_* prims removed: {len(to_remove)}")

    stage.GetRootLayer().Export(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
