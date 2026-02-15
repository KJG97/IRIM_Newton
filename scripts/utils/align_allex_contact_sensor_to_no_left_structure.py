#!/usr/bin/env python3
"""
allex_contact_sensor.usd를 ALLEX_newton_no_left.usd와 동일한 트리 구조로 맞춤.
- 왼팔/왼손은 유지 (제거하지 않음).
- Waist_Base 아래 중첩된 링크들을 루트(allex_contact_sensor) 직하위로 평탄화.
- 모든 'site' prim 및 그 자손 제거.
- no_left처럼 joints, Looks, worldBody를 루트 직하위에 두고, 링크들 순서 정렬.

사용: conda activate isaaclab && python align_allex_contact_sensor_to_no_left_structure.py [input.usd] [output.usd]
"""

from __future__ import annotations

import sys
from pathlib import Path


def _all_prim_paths_under(layer, parent_path: str) -> list[str]:
    """List all prim paths under parent_path in the layer (breadth-first)."""
    from pxr import Sdf

    result = []
    spec = layer.GetPrimAtPath(parent_path)
    if not spec:
        return result
    stack = [parent_path]
    while stack:
        path_str = stack.pop()
        result.append(path_str)
        spec = layer.GetPrimAtPath(path_str)
        if spec:
            for name in spec.nameChildren:
                child_name = name.name if hasattr(name, "name") else str(name)
                stack.append(path_str + "/" + child_name)
    return result


def _remove_sites_from_layer(layer, root_path: str = "/allex_contact_sensor") -> None:
    """Remove every prim whose path contains a 'sites' path component."""
    from pxr import Sdf

    # Collect all prim paths that are 'sites' or under 'sites'
    to_remove = []
    for path_str in _all_prim_paths_under(layer, root_path):
        parts = path_str.split("/")
        if "sites" in parts:
            to_remove.append(path_str)
    # Remove deepest first
    to_remove.sort(key=lambda p: p.count("/"), reverse=True)
    edit = Sdf.BatchNamespaceEdit()
    for path_str in to_remove:
        edit.Add(Sdf.Path(path_str), Sdf.Path.emptyPath)
    layer.Apply(edit)


def main() -> None:
    from pxr import Sdf, Usd, UsdGeom

    if len(sys.argv) < 2:
        print(
            "Usage: python align_allex_contact_sensor_to_no_left_structure.py <allex_contact_sensor.usd> [output.usd]",
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

    src_layer = stage.GetRootLayer()
    # Use /allex_contact_sensor if present, else use default prim (e.g. allex_contact_sensor_03)
    root = stage.GetPrimAtPath("/allex_contact_sensor") or stage.GetDefaultPrim()
    if not root:
        print("No root prim (allex_contact_sensor or defaultPrim) found", file=sys.stderr)
        sys.exit(4)
    root_path = str(root.GetPath())
    waist_base_path = root_path + "/Waist_Base"
    waist_base = stage.GetPrimAtPath(waist_base_path)
    if not waist_base:
        print(f"Waist_Base not found under {root_path}", file=sys.stderr)
        sys.exit(4)

    # Copy entire source layer so instancing (Flattened_Prototype_*, etc.) is preserved
    dst_layer = Sdf.Layer.CreateAnonymous(".usdc")
    for root_prim in src_layer.rootPrims.values():
        src_root_path = root_prim.path
        Sdf.CopySpec(src_layer, src_root_path, dst_layer, src_root_path)

    # 1) Flatten: move each child of Waist_Base to root (same layer)
    waist_base_spec = dst_layer.GetPrimAtPath(waist_base_path)
    if waist_base_spec:
        child_names = [str(c.name) if hasattr(c, "name") else str(c) for c in waist_base_spec.nameChildren]
        for name in child_names:
            src_child_path = waist_base_path + "/" + name
            dst_child_path = root_path + "/" + name
            if dst_layer.GetPrimAtPath(src_child_path):
                Sdf.CopySpec(dst_layer, Sdf.Path(src_child_path), dst_layer, Sdf.Path(dst_child_path))
        # After copy, root/Waist_Base is the link (overwrote wrapper). Remove only the
        # old nested links still under root/Waist_Base (siblings of the link content).
        # I.e. remove root/Waist_Base/X for each X in child_names that is not "Waist_Base".
        to_remove = []
        for name in child_names:
            if name == "Waist_Base":
                continue
            p = waist_base_path + "/" + name
            if dst_layer.GetPrimAtPath(p):
                to_remove.append(p)
        if to_remove:
            edit = Sdf.BatchNamespaceEdit()
            for path_str in to_remove:
                edit.Add(Sdf.Path(path_str), Sdf.Path.emptyPath)
            dst_layer.Apply(edit)

    # 2) Remove all 'sites' prims and their descendants
    _remove_sites_from_layer(dst_layer, root_path)

    # 3) Ensure Looks, worldBody exist (empty, like no_left)
    for name, type_name in [("Looks", "Scope"), ("worldBody", "Xform")]:
        path = root_path + "/" + name
        if not dst_layer.GetPrimAtPath(path):
            Sdf.CreatePrimInLayer(dst_layer, Sdf.Path(path))
            dst_layer.GetPrimAtPath(path).typeName = type_name

    # 4) Reorder root children: joints, Looks, worldBody first (no_left style)
    no_left_first = ["joints", "Looks", "worldBody"]
    dst_root_spec = dst_layer.GetPrimAtPath(root_path)
    current_children = [str(c.name) if hasattr(c, "name") else str(c) for c in dst_root_spec.nameChildren]
    rest = [n for n in current_children if n not in no_left_first]
    desired_order = no_left_first + rest
    if set(desired_order) == set(current_children):
        dst_root_spec.ApplyNameChildrenOrder(desired_order)

    # 5) DefaultPrim (keep same root name, e.g. allex_contact_sensor or allex_contact_sensor_03)
    dst_layer.defaultPrim = root_path.split("/")[-1]

    # 6) Save
    dst_layer.Export(str(output_path))
    print(f"Saved: {output_path}")
    print(f"  - Flattened Waist_Base children under {root_path}")
    print("  - Removed all 'sites' prims")
    print("  - Root order: joints, Looks, worldBody, then links (left arm/hand kept)")


if __name__ == "__main__":
    main()
