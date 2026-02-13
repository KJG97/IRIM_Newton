#!/usr/bin/env python3
"""
Align ALLEX_newton_no_left.usd stage tree structure to match ALLEX_Right_Arm.usd.

- Does NOT remove any links or joints; all content is preserved.
- Renames the articulation root to /URDF_ALLEX_RightArm (same as Right_Arm).
- Reorders direct children under root: Looks, joints, then all link/joint prims
  (same pattern as Right_Arm: Looks, joints, root_joint, base_link, ...).

Requires: pxr (USD). Run with: conda activate isaaclab && python this_script.py [input.usd] [output.usd]
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from pxr import Sdf, Usd, UsdPhysics

    if len(sys.argv) < 2:
        print(
            "Usage: python align_no_left_structure_to_right_arm.py <no_left.usd> [output.usd]",
            file=sys.stderr,
        )
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

    root_path = "/ALLEX"
    new_root_path = "/URDF_ALLEX_RightArm"

    root = stage.GetPrimAtPath(root_path)
    if not root:
        print(f"Articulation root not found: {root_path}", file=sys.stderr)
        sys.exit(4)

    # 1) Reorder direct children of /ALLEX to match Right_Arm pattern:
    #    Looks, joints, then root_joint (if any), then all link Xforms.
    child_names = [c.GetName() for c in root.GetChildren()]
    right_arm_style_order = ["Looks", "joints"]
    # Right_Arm has root_joint; no_left may not - include if present
    if "root_joint" in child_names:
        right_arm_style_order.append("root_joint")
    # Then all other prims in stable order (links, etc.)
    rest = [n for n in child_names if n not in right_arm_style_order]
    desired_order = right_arm_style_order + rest

    if desired_order != child_names:
        layer = stage.GetRootLayer()
        prim_spec = layer.GetPrimAtPath(root_path)
        if prim_spec:
            prim_spec.ApplyNameChildrenOrder(desired_order)
            print(f"Reordered /ALLEX children: Looks, joints, then {len(rest)} prims.")
        else:
            print("Warning: Could not get PrimSpec for reorder; skipping.", file=sys.stderr)
    else:
        print("Child order already matches (Looks, joints, ...).")

    # 2) Create new root prim at /URDF_ALLEX_RightArm (same as Right_Arm)
    new_root = stage.DefinePrim(new_root_path, "Xform")
    if not new_root:
        print(f"Failed to create prim: {new_root_path}", file=sys.stderr)
        sys.exit(5)

    # Apply ArticulationRootAPI so the new root is the articulation root when loaded
    api = UsdPhysics.ArticulationRootAPI.Apply(new_root)
    if not api:
        print("Warning: Could not apply ArticulationRootAPI to new root.", file=sys.stderr)

    # 3) Internal reference: URDF_ALLEX_RightArm references /ALLEX (all content preserved)
    refs = new_root.GetReferences()
    refs.AddInternalReference(Sdf.Path(root_path))
    print(f"Added internal reference: {new_root_path} -> {root_path}")

    # 4) Set default prim so loaders open the same root path as Right_Arm
    stage.SetDefaultPrim(new_root)
    print(f"Set default prim to: {new_root_path}")

    stage.GetRootLayer().Export(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
