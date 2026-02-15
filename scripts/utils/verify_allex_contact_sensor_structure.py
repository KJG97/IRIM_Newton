#!/usr/bin/env python3
"""allex_contact_sensor.usd가 no_left와 동일한 구조인지 검증 (sites 제거, 평탄화, 왼팔 유지)."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from pxr import Sdf

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("source/isaaclab_assets/allex_usd/allex_contact_sensor.usd")
    path = path.resolve()
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    layer = Sdf.Layer.FindOrOpen(str(path))
    if not layer:
        print(f"Failed to open: {path}", file=sys.stderr)
        sys.exit(2)

    root_path = "/allex_contact_sensor"
    root = layer.GetPrimAtPath(root_path)
    if not root:
        print("Root /allex_contact_sensor not found", file=sys.stderr)
        sys.exit(3)

    children = [str(c.name) if hasattr(c, "name") else str(c) for c in root.nameChildren]

    # 1) Flat: no nested Waist_Base/Waist_Yaw_link
    nested = layer.GetPrimAtPath(root_path + "/Waist_Base/Waist_Yaw_link")
    flat = nested is None

    # 2) No sites
    def count_sites(spec, path_str):
        n = 1 if "sites" in path_str.split("/") else 0
        for c in spec.nameChildren:
            cn = c.name if hasattr(c, "name") else str(c)
            child_spec = layer.GetPrimAtPath(path_str + "/" + cn)
            if child_spec:
                n += count_sites(child_spec, path_str + "/" + cn)
        return n

    n_sites = count_sites(root, root_path)
    no_sites = n_sites == 0

    # 3) Left arm/hand present
    left = [n for n in children if n.startswith("L_") or n.startswith("Left_")]
    left_kept = len(left) > 0

    # 4) Order: joints, Looks, worldBody first
    order_ok = children[:3] == ["joints", "Looks", "worldBody"]

    # 5) Waist_Base link present (as direct child)
    has_waist_base_link = "Waist_Base" in children

    # 6) defaultPrim
    default_ok = layer.defaultPrim == "allex_contact_sensor"

    print("=== allex_contact_sensor.usd structure verification ===\n")
    print(f"  Flat (no Waist_Base wrapper):     {flat}")
    print(f"  Sites removed:                    {no_sites} (count={n_sites})")
    print(f"  Left arm/hand kept:               {left_kept} ({len(left)} prims)")
    print(f"  Root order (joints,Looks,worldBody first): {order_ok}")
    print(f"  Waist_Base link present:          {has_waist_base_link}")
    print(f"  defaultPrim = allex_contact_sensor: {default_ok}")
    print(f"  Total root children:              {len(children)}")
    print()
    if flat and no_sites and left_kept and order_ok and default_ok:
        print("OK: Structure matches no_left (with left arm/hand).")
        if not has_waist_base_link:
            print("Note: Waist_Base link missing (run align script on original USD to preserve it).")
    else:
        print("Mismatch: some checks failed.")
        sys.exit(4)


if __name__ == "__main__":
    main()
