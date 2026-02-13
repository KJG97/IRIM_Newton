#!/usr/bin/env python3
"""Inspect USD structure: hierarchy, Mesh prims, references, payloads. Usage: python inspect_usd_structure.py <file.usd> [root_prim]"""

import sys
from pathlib import Path

def main():
    from pxr import Usd, UsdGeom

    if len(sys.argv) < 2:
        print("Usage: python inspect_usd_structure.py <file.usd> [root_prim]", file=sys.stderr)
        sys.exit(1)
    usd_path = Path(sys.argv[1]).resolve()
    root_path = sys.argv[2] if len(sys.argv) > 2 else "/"

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open: {usd_path}", file=sys.stderr)
        sys.exit(2)

    root = stage.GetPrimAtPath(root_path) if root_path != "/" else stage.GetPseudoRoot()
    if not root:
        root = stage.GetPseudoRoot()

    def traverse(prim, depth=0):
        yield depth, prim
        for c in prim.GetAllChildren():
            for d, p in traverse(c, depth + 1):
                yield d, p

    # Collect: type, has ref, has payload, is mesh
    lines = []
    mesh_paths = []
    ref_paths = []
    payload_paths = []

    start = root.GetPath().pathString
    for depth, prim in traverse(root):
        path = prim.GetPath().pathString
        if path == "":
            continue
        ty = prim.GetTypeName()
        refs = prim.GetMetadata("references")
        payload = prim.GetMetadata("payload")
        is_mesh = prim.IsA(UsdGeom.Mesh)
        has_ref = prim.HasAuthoredReferences()
        has_payload = prim.HasAuthoredPayloads()

        indent = "  " * depth
        extra = []
        if is_mesh:
            mesh_paths.append(path)
            extra.append("Mesh")
        if has_ref:
            ref_paths.append(path)
            extra.append("ref")
        if has_payload:
            payload_paths.append(path)
            extra.append("payload")
        if extra:
            lines.append(f"{indent}{path}  [{ty}]  {', '.join(extra)}")
        else:
            lines.append(f"{indent}{path}  [{ty}]")

    print(f"# USD: {usd_path}")
    print(f"# Root: {root_path}")
    print(f"# Total prims (from root): {len(lines)}")
    print()
    for line in lines[:300]:  # cap output
        print(line)
    if len(lines) > 300:
        print(f"... and {len(lines) - 300} more prims")
    print()
    print(f"# Mesh prims: {len(mesh_paths)}")
    for p in mesh_paths[:50]:
        print(f"  {p}")
    if len(mesh_paths) > 50:
        print(f"  ... and {len(mesh_paths) - 50} more")
    print()
    print(f"# Prims with references: {len(ref_paths)}")
    for p in ref_paths[:30]:
        prim = stage.GetPrimAtPath(p)
        refs = prim.GetMetadata("references") if prim else None
        print(f"  {p}  -> {refs}")
    if len(ref_paths) > 30:
        print(f"  ... and {len(ref_paths) - 30} more")
    print()
    print(f"# Prims with payloads: {len(payload_paths)}")
    for p in payload_paths[:20]:
        prim = stage.GetPrimAtPath(p)
        pl = prim.GetPayloads() if prim else []
        print(f"  {p}  -> {list(pl)}")

if __name__ == "__main__":
    main()
