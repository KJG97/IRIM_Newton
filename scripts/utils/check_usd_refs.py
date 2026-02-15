#!/usr/bin/env python3
"""Check reference/payload/sublayer paths in ALLEX_newton_no_left.usd and ALLEX_newton.usd."""
from __future__ import annotations

import os
import re
import sys

from pxr import Sdf, Usd


def collect_all_asset_paths(stage: Usd.Stage, base_path: str) -> list[tuple[str, str, str]]:
    """(source: 'sublayer'|'ref'|'payload'|'attr', prim_or_layer_path, asset_path)."""
    out = []
    root = stage.GetRootLayer()
    for i, sub in enumerate(root.subLayerPaths):
        out.append(("sublayer", root.identifier, sub))
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        refs = prim.GetMetadata("references")
        if refs:
            for r in refs.prependedItems:
                out.append(("references", path, str(r.assetPath)))
            for r in refs.appendedItems:
                out.append(("references", path, str(r.assetPath)))
        try:
            pl = prim.GetPayloads()
            if hasattr(pl, "prependedItems"):
                for p in pl.prependedItems:
                    out.append(("payload", path, str(p.assetPath)))
                for p in pl.appendedItems:
                    out.append(("payload", path, str(p.assetPath)))
        except Exception:
            pass
        for attr in prim.GetAttributes():
            val = attr.Get()
            if val is None:
                continue
            s = str(val).strip()
            if not s:
                continue
            if "asset" in str(attr.GetTypeName()).lower() or "path" in attr.GetName().lower():
                out.append(("attr", path, s))
            if "\\\\" in s or (len(s) > 2 and s[1] == ":" and (s[2] == "\\" or s[2] == "/")):
                out.append(("attr_winpath", path, s))
    return out


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "source", "isaaclab_assets", "allex_usd")
    if not os.path.isdir(base):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "isaaclab_assets", "allex_usd"))
    no_left_path = os.path.join(base, "ALLEX_newton_no_left.usd")
    newton_path = os.path.join(base, "ALLEX_newton.usd")
    if not os.path.isfile(no_left_path):
        no_left_path = os.path.abspath("source/isaaclab_assets/allex_usd/ALLEX_newton_no_left.usd")
        newton_path = os.path.abspath("source/isaaclab_assets/allex_usd/ALLEX_newton.usd")
    if not os.path.isfile(no_left_path) or not os.path.isfile(newton_path):
        print("USD files not found", file=sys.stderr)
        sys.exit(1)

    print("=== ALLEX_newton_no_left.usd ===\nRoot layer:", no_left_path)
    stage_nl = Usd.Stage.Open(no_left_path)
    root_nl = stage_nl.GetRootLayer()
    print("Sublayers:", root_nl.subLayerPaths)
    paths_nl = collect_all_asset_paths(stage_nl, base)
    for src, where, asset in paths_nl:
        print(f"  [{src}] {where} -> {asset}")
    win_nl = [p for p in paths_nl if "\\\\" in p[2] or (len(p[2]) > 2 and p[2][1] == ":" and p[2][2] in "\\/")]
    if win_nl:
        print("  >>> Windows-style paths:", win_nl)
    allex_nl = [p for p in paths_nl if "allex_contact_sensor" in p[2].lower()]
    print("  Contains 'allex_contact_sensor':", allex_nl if allex_nl else "none")

    print("\n=== ALLEX_newton.usd ===\nRoot layer:", newton_path)
    stage_n = Usd.Stage.Open(newton_path)
    root_n = stage_n.GetRootLayer()
    print("Sublayers:", root_n.subLayerPaths)
    paths_n = collect_all_asset_paths(stage_n, base)
    for src, where, asset in paths_n:
        print(f"  [{src}] {where} -> {asset}")
    win_n = [p for p in paths_n if "\\\\" in p[2] or (len(p[2]) > 2 and p[2][1] == ":" and p[2][2] in "\\/")]
    if win_n:
        print("  >>> Windows-style paths:", win_n)
    allex_n = [p for p in paths_n if "allex_contact_sensor" in p[2].lower()]
    print("  Contains 'allex_contact_sensor':", allex_n if allex_n else "none")


if __name__ == "__main__":
    main()
