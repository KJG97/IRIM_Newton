#!/usr/bin/env python3
"""
USD 파일에서 존재하지 않는 Flattened_Prototype_* 참조를 제거하여
뷰어/Isaac Sim에서 파일이 열리도록 함.
참조만 제거하고 메시 데이터는 복구하지 않음(계층은 유지됨).

사용: conda activate isaaclab && python strip_broken_usd_references.py <file.usd> [output.usd]
"""

from __future__ import annotations

import sys
from pathlib import Path


def _list_has_bad_ref(list_op, bad_prefix: str) -> bool:
    if list_op is None:
        return False
    for op_name in ("prependedItems", "appendedItems", "explicitItems"):
        items = getattr(list_op, op_name, None) or []
        for item in items:
            for attr in ("assetPath", "path", "primPath"):
                v = getattr(item, attr, None)
                if v is not None and bad_prefix in str(v):
                    return True
    return False


def _strip_bad_refs_from_spec(spec, bad_prefix: str = "Flattened_Prototype") -> bool:
    """Clear referenceList/payloadList if they contain refs to bad_prefix. Returns True if cleared."""
    changed = False
    for list_attr in ("referenceList", "payloadList"):
        list_op = getattr(spec, list_attr, None)
        if list_op is not None and _list_has_bad_ref(list_op, bad_prefix):
            try:
                list_op.Clear()
                changed = True
            except Exception:
                pass
    return changed


def _traverse_and_strip(layer, prim_path_str: str, bad_prefix: str) -> bool:
    from pxr import Sdf

    spec = layer.GetPrimAtPath(prim_path_str)
    if not spec:
        return False
    changed = _strip_bad_refs_from_spec(spec, bad_prefix)
    for c in spec.nameChildren:
        child_name = c.name if hasattr(c, "name") else str(c)
        if _traverse_and_strip(layer, prim_path_str + "/" + child_name, bad_prefix):
            changed = True
    return changed


def main() -> None:
    from pxr import Sdf, Usd

    if len(sys.argv) < 2:
        print("Usage: python strip_broken_usd_references.py <input.usd> [output.usd]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else input_path

    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    # Open as layer (avoid stage resolution which triggers the errors)
    layer = Sdf.Layer.FindOrOpen(str(input_path))
    if not layer:
        print(f"Failed to open layer: {input_path}", file=sys.stderr)
        sys.exit(3)

    # Traverse and strip bad refs from all prims
    changed = False
    for root_prim in layer.rootPrims.values():
        path_str = str(root_prim.path)
        if _traverse_and_strip(layer, path_str, "Flattened_Prototype"):
            changed = True

    if changed:
        layer.Export(str(output_path))
        print(f"Saved (broken refs stripped): {output_path}")
    else:
        if input_path != output_path:
            layer.Export(str(output_path))
            print(f"Saved (no bad refs found): {output_path}")
        else:
            print("No broken Flattened_Prototype refs found; file unchanged.")


if __name__ == "__main__":
    main()
