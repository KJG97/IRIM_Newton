#!/usr/bin/env python3
"""Inspect allex_test.usd: sublayers, refs, payloads, and scales in all layers."""
from __future__ import annotations

import os
import sys

from pxr import Usd, UsdGeom, Sdf


def get_prim_stack(prim):
    """Return list of (layer, path) that contribute to this prim."""
    stack = prim.GetPrimStack()
    return [(str(s.layer.identifier), str(s.path)) for s in stack]


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "..", "..", "source", "isaaclab_assets", "allex_usd")
    path = os.path.join(base, "allex_test.usd")
    if not os.path.isfile(path):
        path = os.path.abspath("source/isaaclab_assets/allex_usd/allex_test.usd")
    if not os.path.isfile(path):
        print("allex_test.usd not found", file=sys.stderr)
        sys.exit(1)

    stage = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    if not stage:
        print("Failed to open stage", file=sys.stderr)
        sys.exit(1)

    root = stage.GetRootLayer()
    print("Root layer:", root.identifier)
    print("Sublayers:", root.subLayerPaths if hasattr(root, "subLayerPaths") else "N/A")
    for i, sub in enumerate(root.subLayerPaths):
        print(f"  [{i}] {sub}")

    # Find refs and payloads
    refs_found = []
    for prim in stage.Traverse():
        refs = prim.GetMetadata("references")
        if refs:
            for r in refs.prependedItems + refs.appendedItems:
                refs_found.append((str(prim.GetPath()), str(r.assetPath)))
        payload = prim.GetPayloads()
        if payload:
            for p in payload:
                refs_found.append((str(prim.GetPath()), f"payload:{p}"))
    print("\nReferences/Payloads (first 20):")
    for path, ref in refs_found[:20]:
        print(f"  {path} -> {ref}")

    # Check layer stack for any layer that might have scale
    def scales_in_layer(layer: Sdf.Layer, prefix_path: str = ""):
        for path, spec in layer.iteritems("/"):
            if spec.specType == Sdf.SpecTypeAttribute:
                aname = path.split("/")[-1]
                if "scale" in aname.lower() or "xformOp" in aname:
                    attr = layer.GetAttributeAtPath(path)
                    if attr:
                        val = attr.default
                        if val is not None and "scale" in aname.lower():
                            print(f"    {path} = {val}")

    print("\nScales in root layer (iter):")
    for path, spec in root.iteritems("/"):
        if spec.specType == Sdf.SpecTypeAttribute:
            aname = path.split("/")[-1]
            if "scale" in aname.lower():
                attr = root.GetAttributeAtPath(path)
                if attr:
                    print(f"  {path} = {attr.default}")

    # Also traverse and print any scale that is NOT (1,1,1)
    print("\nPrims with scale != (1,1,1):")
    for prim in stage.Traverse():
        xform = UsdGeom.Xformable(prim)
        if not xform:
            continue
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() != UsdGeom.XformOp.TypeScale:
                continue
            val = op.Get()
            if val is None:
                continue
            v = (float(val[0]), float(val[1]), float(val[2]))
            if abs(v[0] - 1.0) > 1e-6 or abs(v[1] - 1.0) > 1e-6 or abs(v[2] - 1.0) > 1e-6:
                print(f"  {prim.GetPath()} scale = {v}")


if __name__ == "__main__":
    main()
