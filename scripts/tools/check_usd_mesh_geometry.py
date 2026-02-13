#!/usr/bin/env python3
"""
USD 파일 내 모든 Mesh prim의 geometry(vertices/indices) 채움 여부를 검사합니다.
- points(vertices) 개수, faceVertexIndices(또는 faceVertexCounts) 유무/개수 출력
- 빈 메시(0 vertices 또는 0 faces) 목록과 채워진 메시 개수 요약

Usage:
  python scripts/tools/check_usd_mesh_geometry.py <file.usd> [root_prim]
  python scripts/tools/check_usd_mesh_geometry.py ALLEX_newton_no_left.usd
  python scripts/tools/check_usd_mesh_geometry.py ALLEX_Right_Arm.usd
"""

import sys
from pathlib import Path


def get_mesh_geometry_info(stage, root_path: str):
    """root_path 아래(또는 전체) Mesh prim을 순회하며 vertices/indices 개수 수집."""
    from pxr import UsdGeom

    results = []  # (path, num_points, num_face_vertices, is_empty)
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if root_path and root_path != "/" and not path.startswith(root_path + "/") and path != root_path:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        num_points = 0
        num_face_vertices = 0
        points_attr = mesh.GetPointsAttr()
        if points_attr:
            points = points_attr.Get()
            num_points = len(points) if points is not None else 0
        # faceVertexIndices (또는 faceVertexCounts로 face 수 추정)
        face_indices_attr = mesh.GetFaceVertexIndicesAttr()
        if face_indices_attr:
            indices = face_indices_attr.Get()
            num_face_vertices = len(indices) if indices is not None else 0
        else:
            face_counts_attr = mesh.GetFaceVertexCountsAttr()
            if face_counts_attr:
                counts = face_counts_attr.Get()
                if counts is not None:
                    num_face_vertices = sum(int(c) for c in counts)
        is_empty = num_points == 0 or num_face_vertices == 0
        results.append((path, num_points, num_face_vertices, is_empty))
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_usd_mesh_geometry.py <file.usd> [root_prim]", file=sys.stderr)
        sys.exit(1)

    from pxr import Usd

    usd_path = Path(sys.argv[1]).resolve()
    root_path = sys.argv[2].strip() if len(sys.argv) > 2 else None

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open: {usd_path}", file=sys.stderr)
        sys.exit(2)

    if not root_path:
        default_prim = stage.GetDefaultPrim()
        root_path = default_prim.GetPath().pathString if default_prim else "/"
    print(f"# USD: {usd_path.name}")
    print(f"# Root: {root_path}")
    print()

    rows = get_mesh_geometry_info(stage, root_path)
    empty = [r for r in rows if r[3]]
    filled = [r for r in rows if not r[3]]

    print(f"총 Mesh 개수: {len(rows)}")
    print(f"  - geometry 있음 (vertices>0, faceVertices>0): {len(filled)}")
    print(f"  - 빈 메시 (vertices=0 또는 faceVertices=0): {len(empty)}")
    print()

    print("--- Mesh별 상세 (path, num_points, num_face_vertices, empty) ---")
    for path, n_pts, n_fv, is_empty in rows:
        flag = " [EMPTY]" if is_empty else ""
        print(f"  {path}")
        print(f"    points={n_pts}, faceVertexIndices/vertices={n_fv}{flag}")
    print()

    if empty:
        print("--- 빈 메시 목록 ---")
        for path, n_pts, n_fv, _ in empty:
            print(f"  {path}  (points={n_pts}, faceVerts={n_fv})")
    else:
        print("--- 빈 메시 없음 (모든 Mesh에 geometry 있음) ---")

    if filled and len(filled) <= 30:
        print()
        print("--- 채워진 메시 목록 ---")
        for path, n_pts, n_fv, _ in filled:
            print(f"  {path}  (points={n_pts}, faceVerts={n_fv})")
    elif filled:
        print()
        print(f"--- 채워진 메시 샘플 (처음 20개, 총 {len(filled)}개) ---")
        for path, n_pts, n_fv, _ in filled[:20]:
            print(f"  {path}  (points={n_pts}, faceVerts={n_fv})")

    return 0 if not empty else 1


if __name__ == "__main__":
    sys.exit(main())
