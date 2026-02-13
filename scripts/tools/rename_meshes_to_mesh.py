#!/usr/bin/env python3
"""
Rename all Mesh prims in a USD to 'mesh' (with mesh_1, mesh_2, ... when multiple
meshes share the same parent so sibling names stay unique).

Usage: python rename_meshes_to_mesh.py <input.usd> [output.usd]
If output is omitted, overwrites input.
Requires: pxr (USD). Run with: conda run -n isaaclab python this_script.py ...
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from pxr import Sdf, Usd, UsdGeom

    if len(sys.argv) < 2:
        print(
            "Usage: python rename_meshes_to_mesh.py <input.usd> [output.usd]",
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

    # Collect all Mesh prim paths
    mesh_paths: list[Sdf.Path] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_paths.append(prim.GetPath())

    if not mesh_paths:
        print("No Mesh prims found.")
        stage.GetRootLayer().Export(str(output_path))
        return

    # Group by parent path; assign mesh, mesh_1, mesh_2 per parent
    from collections import defaultdict

    by_parent: dict[Sdf.Path, list[Sdf.Path]] = defaultdict(list)
    for p in mesh_paths:
        by_parent[p.GetParentPath()].append(p)

    renames: list[tuple[Sdf.Path, Sdf.Path]] = []
    for parent_path, paths in sorted(by_parent.items(), key=lambda x: str(x[0])):
        for i, old_path in enumerate(sorted(paths, key=lambda x: str(x))):
            new_name = "mesh" if i == 0 else f"mesh_{i}"
            new_path = parent_path.AppendChild(new_name)
            if old_path != new_path:
                renames.append((old_path, new_path))

    if not renames:
        print("All Mesh prims already named 'mesh' (or mesh_N).")
        stage.GetRootLayer().Export(str(output_path))
        return

    # Apply renames: deepest paths first to avoid mid-rename path clashes
    renames.sort(key=lambda r: (-r[0].pathElementCount, str(r[0])))

    editor = Usd.NamespaceEditor(stage)
    for old_path, new_path in renames:
        if not editor.MovePrimAtPath(old_path, new_path):
            print(f"Warning: could not move {old_path} -> {new_path}", file=sys.stderr)
    if not editor.ApplyEdits():
        print("NamespaceEditor.ApplyEdits() failed.", file=sys.stderr)
        sys.exit(4)

    print(f"Renamed {len(renames)} Mesh prim(s) to mesh / mesh_N.")
    stage.GetRootLayer().Export(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
