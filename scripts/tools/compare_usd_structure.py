#!/usr/bin/env python3
"""
Compare two USD files for articulation/mesh structure differences.
Usage: python compare_usd_structure.py <file1.usd> <file2.usd>
"""

import sys
from pathlib import Path
from collections import defaultdict

def analyze(stage, name):
    from pxr import Usd, UsdGeom, UsdPhysics

    info = {"name": name, "path": str(stage.GetRootLayer().identifier)}

    # Default prim
    dp = stage.GetDefaultPrim()
    info["default_prim"] = dp.GetPath().pathString if dp else None
    root_path = info["default_prim"] or "/"
    root = stage.GetPrimAtPath(root_path)

    # Root specifier & type
    info["root_type"] = root.GetTypeName() if root else None
    info["root_specifier"] = str(root.GetSpecifier()) if root else None

    # APIs on root
    apis = []
    if root:
        for api in root.GetAppliedSchemas():
            apis.append(api)
    info["root_apis"] = apis

    # ArticulationRootAPI: which prim has it (Isaac/Omniverse may use different schema)
    art_root_path = None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_root_path = prim.GetPath().pathString
            break
    info["articulation_root_prim"] = art_root_path

    # RigidBodyAPI: count and paths (first 20)
    rb_paths = []
    for prim in stage.Traverse():
        if UsdPhysics.RigidBodyAPI(prim):
            rb_paths.append(prim.GetPath().pathString)
    info["rigid_body_count"] = len(rb_paths)
    info["rigid_body_paths_sample"] = rb_paths[:20]

    # Direct children of default prim (the "link" level)
    link_level = []
    if root:
        for c in root.GetChildren():
            link_level.append((c.GetName(), c.GetTypeName()))
    info["root_direct_children"] = link_level

    # Mesh prims: under root only, and total
    meshes_under_root = []
    total_meshes = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            total_meshes += 1
            path = prim.GetPath().pathString
            if path.startswith(root_path + "/") or path == root_path:
                meshes_under_root.append(path)
    info["mesh_count_total"] = total_meshes
    info["mesh_count_under_root"] = len(meshes_under_root)
    info["mesh_paths_under_root_sample"] = meshes_under_root[:25]

    # Subtree depth: max depth under root
    def max_depth(prim, d=0):
        if not prim:
            return d
        m = d
        for c in prim.GetAllChildren():
            m = max(m, max_depth(c, d + 1))
        return m
    info["max_depth_under_root"] = max_depth(root) if root else 0

    # References / Payloads anywhere (could affect loading)
    refs = []
    for prim in stage.Traverse():
        if prim.HasAuthoredReferences():
            refs.append(prim.GetPath().pathString)
        if prim.HasAuthoredPayloads():
            refs.append(prim.GetPath().pathString + " (payload)")
    info["refs_payloads"] = refs[:15]

    # Check: do link prims (direct children of root) have RigidBodyAPI?
    link_rb = []
    if root:
        for c in root.GetChildren():
            if UsdPhysics.RigidBodyAPI(c):
                link_rb.append(c.GetName())
    info["root_children_with_rb"] = link_rb

    return info


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_usd_structure.py <file1.usd> <file2.usd>", file=sys.stderr)
        sys.exit(1)
    from pxr import Usd

    a_path = Path(sys.argv[1]).resolve()
    b_path = Path(sys.argv[2]).resolve()
    sa = Usd.Stage.Open(str(a_path))
    sb = Usd.Stage.Open(str(b_path))
    if not sa or not sb:
        print("Failed to open one or both files", file=sys.stderr)
        sys.exit(2)

    name_a = a_path.name
    name_b = b_path.name
    info_a = analyze(sa, name_a)
    info_b = analyze(sb, name_b)

    def p(key, label=None):
        lbl = label or key
        va = info_a.get(key)
        vb = info_b.get(key)
        same = va == vb
        print(f"\n--- {lbl} ---")
        print(f"  {name_a}: {va}")
        print(f"  {name_b}: {vb}")
        if not same:
            print("  >> DIFFERENT")

    print("=" * 60)
    print("USD STRUCTURE COMPARISON")
    print("=" * 60)
    p("default_prim", "Default prim (load target)")
    p("root_type", "Root prim type")
    p("root_apis", "Applied APIs on root")
    p("articulation_root_prim", "Prim with ArticulationRootAPI")
    p("rigid_body_count", "RigidBodyAPI count (whole stage)")
    p("mesh_count_total", "Total Mesh prims (whole stage)")
    p("mesh_count_under_root", "Mesh prims under default prim")
    p("max_depth_under_root", "Max depth under root")

    print("\n--- Root direct children (link level) ---")
    print(f"  {name_a} ({len(info_a['root_direct_children'])}):")
    for nm, ty in info_a["root_direct_children"][:25]:
        print(f"    {nm} [{ty}]")
    if len(info_a["root_direct_children"]) > 25:
        print(f"    ... +{len(info_a['root_direct_children']) - 25} more")
    print(f"  {name_b} ({len(info_b['root_direct_children'])}):")
    for nm, ty in info_b["root_direct_children"][:25]:
        print(f"    {nm} [{ty}]")
    if len(info_b["root_direct_children"]) > 25:
        print(f"    ... +{len(info_b['root_direct_children']) - 25} more")

    print("\n--- Root children WITH RigidBodyAPI ---")
    print(f"  {name_a}: {info_a['root_children_with_rb'][:30]}")
    print(f"  {name_b}: {info_b['root_children_with_rb'][:30]}")
    if set(info_a["root_children_with_rb"]) != set(info_b["root_children_with_rb"]):
        print("  >> DIFFERENT (Newton may use this to decide which links to create)")

    print("\n--- Mesh paths under root (sample) ---")
    print(f"  {name_a}:")
    for m in info_a["mesh_paths_under_root_sample"]:
        print(f"    {m}")
    print(f"  {name_b}:")
    for m in info_b["mesh_paths_under_root_sample"]:
        print(f"    {m}")

    print("\n--- RigidBody paths (sample) ---")
    print(f"  {name_a}: {info_a['rigid_body_paths_sample']}")
    print(f"  {name_b}: {info_b['rigid_body_paths_sample']}")

    print("\n--- References/Payloads ---")
    print(f"  {name_a}: {info_a['refs_payloads']}")
    print(f"  {name_b}: {info_b['refs_payloads']}")


if __name__ == "__main__":
    main()
