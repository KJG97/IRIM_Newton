#!/usr/bin/env python3
"""Compare mesh load scale between allex_test.usd and ALLEX_newton_no_left.usd.

no_left = 정상 크기. allex_test와 스케일 차이 원인 검토.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from pxr import Usd, UsdGeom


def get_scale_from_xform(prim) -> tuple[float, float, float] | None:
    """Prim의 xformOp:scale 값. 없으면 None."""
    xform = UsdGeom.Xformable(prim)
    if not xform:
        return None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            v = op.Get()
            if v is not None:
                return (float(v[0]), float(v[1]), float(v[2]))
    return None


def get_world_scale(prim) -> tuple[float, float, float]:
    """Prim의 local-to-world 변환에서 스케일 성분만 추출 (메시 로드 시 적용되는 스케일)."""
    xform = UsdGeom.Xformable(prim)
    if not xform:
        return (1.0, 1.0, 1.0)
    world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    m = np.array(world)
    # Scale from 3x3 rotation/scale block
    scale = np.sqrt(np.sum(m[:3, :3] ** 2, axis=0))
    return (float(scale[0]), float(scale[1]), float(scale[2]))


def collect_mesh_scale_info(stage: Usd.Stage, root_path: str) -> list[tuple[str, tuple, tuple]]:
    """(path, local_scale, world_scale) for prims that have scale or are mesh parents."""
    out = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root_path):
            continue
        local = get_scale_from_xform(prim)
        world = get_world_scale(prim)
        # 메시를 가진 prim 또는 scale이 있는 prim만 (요약)
        if local is not None or prim.IsA(UsdGeom.Mesh) or "visuals" in path or "collisions" in path:
            out.append((path, local or (1, 1, 1), world))
    return out


def normalize_path_for_compare(path: str, root_a: str, root_b: str) -> str:
    """비교용: root 제거 후 상대 경로처럼 (같은 링크/이름이면 매칭)."""
    for r in (root_a, root_b, "/ALLEX/", "/World/envs/env_0/Robot/"):
        if path.startswith(r):
            return path[len(r) :].rstrip("/") or "/"
    return path


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "..", "source", "isaaclab_assets", "allex_usd")
    path_test = os.path.join(base, "allex_test.usd")
    path_noleft = os.path.join(base, "ALLEX_newton_no_left.usd")
    if not os.path.isfile(path_test):
        path_test = os.path.abspath("source/isaaclab_assets/allex_usd/allex_test.usd")
    if not os.path.isfile(path_noleft):
        path_noleft = os.path.abspath("source/isaaclab_assets/allex_usd/ALLEX_newton_no_left.usd")
    if not os.path.isfile(path_test) or not os.path.isfile(path_noleft):
        print("USD files not found", file=sys.stderr)
        sys.exit(1)

    stage_test = Usd.Stage.Open(path_test, Usd.Stage.LoadAll)
    stage_nl = Usd.Stage.Open(path_noleft, Usd.Stage.LoadAll)

    # no_left: 루트가 /World/envs/env_0/Robot 인지 /allex_... 인지 확인
    def find_root(stage):
        for p in stage.Traverse():
            path = str(p.GetPath())
            if "ALLEX" in path or "Robot" in path or "allex" in path.lower():
                parts = path.split("/")
                return "/" + "/".join(parts[1:2])  # /ALLEX or /World
        return "/"

    root_test = "/ALLEX"
    root_nl = "/World"
    for p in stage_nl.Traverse():
        path = str(p.GetPath())
        if "envs" in path or "Robot" in path:
            root_nl = "/World/envs/env_0/Robot"
            break

    info_test = collect_mesh_scale_info(stage_test, root_test)
    info_nl = collect_mesh_scale_info(stage_nl, root_nl)

    # no_left 경로: Robot 아래가 링크명과 비슷할 것
    def rel_test(p):
        if p.startswith("/ALLEX/"):
            return p.replace("/ALLEX/", "", 1)
        return p

    def rel_nl(p):
        if "/Robot/" in p:
            return p.split("/Robot/")[-1]
        if p.startswith("/World/"):
            return p.replace("/World/envs/env_0/Robot/", "", 1) if "Robot" in p else p
        return p

    # 매칭: 같은 상대 경로끼리 비교
    by_rel_test = {rel_test(p): (local, world) for p, local, world in info_test}
    by_rel_nl = {rel_nl(p): (local, world) for p, local, world in info_nl}

    print("=== 1) allex_test.usd 루트 및 스케일 요약 ===")
    print(f"  Root: {root_test}")
    # 샘플: 첫 링크 하나의 전체 체인
    sample = [x for x in info_test if "Waist_Base" in x[0] and ("visuals" in x[0] or "collisions" in x[0])][:6]
    for path, local, world in sample:
        print(f"  {path}")
        print(f"    local_scale={local}  world_scale={world}")

    print("\n=== 2) no_left.usd 루트 및 스케일 요약 ===")
    print(f"  Root: {root_nl}")
    sample_nl = [x for x in info_nl if "Waist_Base" in x[0] and ("visuals" in x[0] or "collisions" in x[0])][:6]
    for path, local, world in sample_nl:
        print(f"  {path}")
        print(f"    local_scale={local}  world_scale={world}")

    print("\n=== 3) 동일 상대경로 비교 (world_scale 다른 것만) ===")
    common = set(by_rel_test.keys()) & set(by_rel_nl.keys())
    for rel in sorted(common):
        (local_t, world_t) = by_rel_test[rel]
        (local_nl, world_nl) = by_rel_nl[rel]
        if abs(world_t[0] - world_nl[0]) > 1e-6 or abs(world_t[1] - world_nl[1]) > 1e-6 or abs(world_t[2] - world_nl[2]) > 1e-6:
            print(f"  {rel}")
            print(f"    allex_test local={local_t} world={world_t}")
            print(f"    no_left    local={local_nl} world={world_nl}")

    print("\n=== 4) no_left에만 있는 경로 (allex_test에 없음) ===")
    only_nl = set(by_rel_nl.keys()) - set(by_rel_test.keys())
    for rel in sorted(only_nl)[:20]:
        print(f"  {rel}")
    if len(only_nl) > 20:
        print(f"  ... and {len(only_nl) - 20} more")

    print("\n=== 5) allex_test에만 있는 경로 (no_left에 없음) ===")
    only_test = set(by_rel_test.keys()) - set(by_rel_nl.keys())
    for rel in sorted(only_test)[:20]:
        print(f"  {rel}")
    if len(only_test) > 20:
        print(f"  ... and {len(only_test) - 20} more")

    # 6) 스케일이 1이 아닌 prim (allex_test)
    print("\n=== 6) allex_test에서 world_scale != (1,1,1) 인 prim ===")
    non_one = [(rel, w) for rel, (loc, w) in by_rel_test.items() if abs(w[0] - 1) > 1e-6 or abs(w[1] - 1) > 1e-6 or abs(w[2] - 1) > 1e-6]
    for rel, w in sorted(non_one)[:30]:
        print(f"  {rel}  world_scale={w}")
    if len(non_one) > 30:
        print(f"  ... and {len(non_one) - 30} more")

    print("\n=== 7) no_left에서 world_scale != (1,1,1) 인 prim ===")
    non_one_nl = [(rel, w) for rel, (loc, w) in by_rel_nl.items() if abs(w[0] - 1) > 1e-6 or abs(w[1] - 1) > 1e-6 or abs(w[2] - 1) > 1e-6]
    for rel, w in sorted(non_one_nl)[:30]:
        print(f"  {rel}  world_scale={w}")
    if len(non_one_nl) > 30:
        print(f"  ... and {len(non_one_nl) - 30} more")


def report_summary():
    """요약: 두 파일이 메시 로드 시 동일 스케일이 되려면 필요한 조건."""
    base = os.path.join(os.path.dirname(__file__), "..", "..", "source", "isaaclab_assets", "allex_usd")
    path_test = os.path.join(base, "allex_test.usd")
    path_noleft = os.path.join(base, "ALLEX_newton_no_left.usd")
    if not os.path.isfile(path_test):
        path_test = os.path.abspath("source/isaaclab_assets/allex_usd/allex_test.usd")
    if not os.path.isfile(path_noleft):
        path_noleft = os.path.abspath("source/isaaclab_assets/allex_usd/ALLEX_newton_no_left.usd")

    st = Usd.Stage.Open(path_test, Usd.Stage.LoadAll)
    sn = Usd.Stage.Open(path_noleft, Usd.Stage.LoadAll)

    mpu_t = UsdGeom.GetStageMetersPerUnit(st)
    mpu_n = UsdGeom.GetStageMetersPerUnit(sn)
    print("=== 스테이지 단위 ===")
    print(f"  allex_test  metersPerUnit = {mpu_t}")
    print(f"  no_left     metersPerUnit = {mpu_n}")
    if mpu_t != mpu_n:
        print("  -> 차이 있음. 동일하게 맞추려면 UsdGeom.SetStageMetersPerUnit(stage, value) 사용.")

    print("\n=== 메시 extent (동일하면 같은 지오메트리) ===")
    for stage, label in [(st, "allex_test"), (sn, "no_left")]:
        root = "/ALLEX" if "allex_test" in label else "/allex_contact_sensor"
        for p in stage.Traverse():
            if p.IsA(UsdGeom.Mesh) and "Waist_Base" in str(p.GetPath()):
                attr = p.GetAttribute("extent")
                if attr:
                    ext = attr.Get()
                    print(f"  {label} {p.GetPath()} extent[0]={ext[0]}")
                    break
        else:
            print(f"  {label} Waist_Base mesh not found")

    print("\n=== 비(1,1,1) 로컬 스케일 개수 ===")
    for stage, label in [(st, "allex_test"), (sn, "no_left")]:
        n = 0
        for p in stage.Traverse():
            loc = get_scale_from_xform(p)
            if loc and (abs(loc[0] - 1) > 1e-6 or abs(loc[1] - 1) > 1e-6 or abs(loc[2] - 1) > 1e-6):
                n += 1
        print(f"  {label}: {n} prims with local scale != (1,1,1)")
    print("\n(no_left=0, allex_test=0 이면 둘 다 스케일 1로 통일된 상태. 차이나면 fix_allex_test_usd_scales.py 실행)")


if __name__ == "__main__":
    main()
    print("\n" + "=" * 60)
    report_summary()
