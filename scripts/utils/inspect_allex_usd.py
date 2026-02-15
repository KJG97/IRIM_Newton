#!/usr/bin/env python3
"""Inspect ALLEX_newton.usd and ALLEX_newton_no_left.usd: mesh scales, references, prims."""
from __future__ import annotations

import os
import sys

from pxr import Kind, Usd, UsdGeom, UsdShade, Sdf


def get_asset_paths(stage: Usd.Stage) -> list[tuple[str, str, dict]]:
    """Collect prim path, asset path (reference/payload), and scale if any."""
    result = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        info = {}
        # scale
        xform = UsdGeom.Xformable(prim)
        if xform:
            order = xform.GetOrderedXformOps()
            for op in order:
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    info["scale"] = op.Get()
                    break
        # mesh scale on Mesh
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            # no direct mesh scale in UsdGeom.Mesh; scale usually on parent Xform
            pass
        # references and payloads
        refs = prim.GetMetadata("references")
        if refs:
            info["references"] = [str(r.assetPath) for r in refs.prependedItems] + [
                str(r.assetPath) for r in refs.appendedItems
            ]
        payload = prim.GetPayloads()
        if payload:
            info["payload"] = [str(p.assetPath) for p in payload]
        if info:
            result.append((path, path, info))
        # external reference via attribute
        for attr in prim.GetAttributes():
            if "asset" in str(attr.GetTypeName()).lower():
                val = attr.Get()
                if val:
                    result.append((path, str(val.path), {"attr": attr.GetName(), "asset": str(val)}))
    return result


def get_all_xform_scales(stage: Usd.Stage) -> dict[str, tuple[float, float, float]]:
    """Get scale xform op for every prim that has one."""
    scales = {}
    for prim in stage.Traverse():
        xform = UsdGeom.Xformable(prim)
        if not xform:
            continue
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scales[str(prim.GetPath())] = op.Get()
                break
    return scales


def get_references_and_payloads(stage: Usd.Stage) -> list[tuple[str, str, str]]:
    """(prim_path, type 'references'|'payload', asset_path)."""
    out = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        refs = prim.GetMetadata("references")
        if refs:
            for r in refs.prependedItems:
                out.append((path, "references", str(r.assetPath)))
            for r in refs.appendedItems:
                out.append((path, "references", str(r.assetPath)))
        try:
            payload_list = prim.GetPayloads()
            if hasattr(payload_list, "prependedItems"):
                for p in payload_list.prependedItems:
                    out.append((path, "payload", str(p.assetPath)))
                for p in payload_list.appendedItems:
                    out.append((path, "payload", str(p.assetPath)))
            elif hasattr(payload_list, "GetAddedOrExplicitItems"):
                for p in payload_list.GetAddedOrExplicitItems():
                    out.append((path, "payload", str(p.assetPath)))
        except (TypeError, AttributeError):
            pass
    return out


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "source", "isaaclab_assets", "allex_usd")
    if not os.path.isdir(base):
        base = os.path.join(os.path.dirname(__file__), "..", "..", "source", "isaaclab_assets", "allex_usd")
    no_left_path = os.path.join(base, "ALLEX_newton_no_left.usd")
    newton_path = os.path.join(base, "ALLEX_newton.usd")
    if not os.path.isfile(no_left_path):
        no_left_path = os.path.abspath("source/isaaclab_assets/allex_usd/ALLEX_newton_no_left.usd")
        newton_path = os.path.abspath("source/isaaclab_assets/allex_usd/ALLEX_newton.usd")
    if not os.path.isfile(no_left_path):
        print("ALLEX_newton_no_left.usd not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(newton_path):
        print("ALLEX_newton.usd not found", file=sys.stderr)
        sys.exit(1)

    stage_nl = Usd.Stage.Open(no_left_path)
    stage_n = Usd.Stage.Open(newton_path)

    print("=== ALLEX_newton_no_left.usd: scales (path -> scale) ===")
    scales_nl = get_all_xform_scales(stage_nl)
    for p, s in sorted(scales_nl.items()):
        print(f"  {p} -> {s}")

    print("\n=== ALLEX_newton.usd: scales (path -> scale) ===")
    scales_n = get_all_xform_scales(stage_n)
    for p, s in sorted(scales_n.items()):
        print(f"  {p} -> {s}")

    print("\n=== Scale comparison (in no_left but different or missing in newton) ===")
    for path, scale_nl in sorted(scales_nl.items()):
        scale_n = scales_n.get(path)
        if scale_n is None:
            print(f"  MISS in newton: {path} (no_left has {scale_nl})")
        elif scale_n != scale_nl:
            print(f"  DIFF: {path}  no_left={scale_nl}  newton={scale_n}")

    print("\n=== References/Payloads: ALLEX_newton_no_left ===")
    for path, typ, asset in get_references_and_payloads(stage_nl):
        print(f"  {path}  {typ}: {asset}")

    print("\n=== References/Payloads: ALLEX_newton ===")
    refs_n = get_references_and_payloads(stage_n)
    for path, typ, asset in refs_n:
        print(f"  {path}  {typ}: {asset}")

    # Collect all asset paths from prim attributes (e.g. mesh source)
    def get_asset_attrs(stage: Usd.Stage) -> list[tuple[str, str]]:
        out = []
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                if "asset" in str(attr.GetTypeName()).lower():
                    val = attr.Get()
                    if val and str(val.path):
                        out.append((str(prim.GetPath()), str(val)))
        return out

    print("\n=== ALLEX_newton mesh asset path validation (existence under allex_usd) ===")
    usd_dir = os.path.dirname(newton_path)
    validated = set()
    for path, typ, asset in refs_n:
        if not asset or asset in validated:
            continue
        validated.add(asset)
        if asset.startswith("./") or not os.path.isabs(asset):
            full = os.path.normpath(os.path.join(usd_dir, asset))
        else:
            full = asset
        exists = os.path.isfile(full)
        print(f"  {'OK' if exists else 'MISS'}: {asset}  -> {full}")
    asset_attrs = get_asset_attrs(stage_n)
    for prim_path, asset in asset_attrs:
        if not asset or asset in validated:
            continue
        validated.add(asset)
        if asset.startswith("./") or not os.path.isabs(asset):
            full = os.path.normpath(os.path.join(usd_dir, asset))
        else:
            full = asset
        exists = os.path.isfile(full)
        print(f"  {'OK' if exists else 'MISS'} (attr): {asset}  -> {full}")


if __name__ == "__main__":
    main()
