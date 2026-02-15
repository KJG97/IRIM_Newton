#!/usr/bin/env python3
"""allex_test.usd 전체 메시 버텍스를 mm → m 로 베이크 (×0.001). no_left와 동일한 크기로 로드되도록.

Newton은 메시 지오메트리에 xform scale을 적용하지 않으므로, 버텍스를 미터 단위로 넣어야 함.
스케일은 이미 fix_allex_test_usd_scales.py 로 (1,1,1) 처리된 상태를 가정.

Usage:
  python scripts/utils/bake_allex_test_mesh_to_meters.py [allex_test.usd]
  출력 경로 생략 시 입력 파일 덮어씀.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

# mm → m (no_left와 동일 단위로)
SCALE = (0.001, 0.001, 0.001)


def bake_all_meshes_to_scale(stage: Usd.Stage, scale_xyz: tuple[float, float, float]) -> int:
    """스테이지 내 모든 Mesh prim의 points에 scale_xyz를 곱하고, extent를 갱신. 반환: 수정된 메시 수."""
    sx, sy, sz = scale_xyz
    scale_arr = np.array([sx, sy, sz], dtype=np.float64)
    count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr()
        if not pts_attr or not pts_attr.HasAuthoredValue():
            continue
        pts = np.asarray(pts_attr.Get(), dtype=np.float64)
        pts = pts * scale_arr
        pts_attr.Set(pts.tolist())
        # extent 갱신 (min/max)
        if pts.size > 0:
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            extent_attr = mesh.GetExtentAttr()
            if extent_attr:
                extent_attr.Set([Gf.Vec3f(mn[0], mn[1], mn[2]), Gf.Vec3f(mx[0], mx[1], mx[2])])
        count += 1
    return count


def main():
    repo = Path(__file__).resolve().parents[2]
    default_path = repo / "source/isaaclab_assets/allex_usd/allex_test.usd"
    path = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else default_path
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if not stage:
        print("Failed to open stage", file=sys.stderr)
        sys.exit(1)

    n = bake_all_meshes_to_scale(stage, SCALE)
    print(f"Baked scale {SCALE} into {n} mesh(es).")
    stage.GetRootLayer().Save()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
