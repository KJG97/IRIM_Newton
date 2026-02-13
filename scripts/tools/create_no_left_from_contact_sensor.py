#!/usr/bin/env python3
"""
allex_contact_sensor.usd(XML에서 최근 변환)를 열어
왼팔/왼손(L_*, Left_*) prim을 제거한 뒤 ALLEX_newton_no_left.usd 로 저장합니다.

1. contact_sensor.usd 열기 (payload 포함)
2. Export로 payload 해석된 단일 파일 생성
3. 해당 레이어에서 왼팔/왼손 경로를 Sdf.BatchNamespaceEdit으로 제거
4. ALLEX_newton_no_left.usd 에 덮어쓰기

사용:
  python create_no_left_from_contact_sensor.py
  python create_no_left_from_contact_sensor.py --dry-run  # 제거 대상만 출력
"""

import argparse
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create no-left USD from contact_sensor.usd")
    parser.add_argument("--dry-run", action="store_true", help="Only print prims to remove, do not save")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    src_usd = repo / "source/isaaclab_assets/allex_usd/allex_model_mjcf_250903/allex_contact_sensor/allex_contact_sensor.usd"
    dst_usd = repo / "source/isaaclab_assets/allex_usd/ALLEX_newton_no_left.usd"

    if not src_usd.is_file():
        print(f"소스 파일 없음: {src_usd}", file=sys.stderr)
        return 1

    from pxr import Usd, Sdf

    print("열기:", src_usd)
    stage = Usd.Stage.Open(str(src_usd))
    if not stage:
        print("Stage 열기 실패", file=sys.stderr)
        return 2

    with tempfile.NamedTemporaryFile(suffix=".usd", delete=False) as f:
        tmp_path = f.name
    try:
        print("Export (payload 해석)...")
        stage.Export(tmp_path)

        stage2 = Usd.Stage.Open(tmp_path)
        layer = stage2.GetRootLayer()
        paths = [p.GetPath() for p in stage2.Traverse()]
        # 1) 왼팔/왼손 링크(바디) 제거
        to_remove_links = [
            p for p in paths
            if "/Waist_Base/L_" in p.pathString or "/Waist_Base/Left_" in p.pathString
        ]
        # 2) 왼팔/왼손 조인트 제거 — 남기면 Newton이 "Multiple joints lead to body" 오류 발생
        to_remove_joints = [
            p for p in paths
            if "/allex_contact_sensor/joints/" in p.pathString
            and (p.name.startswith("L_") or p.name.startswith("Left_"))
        ]
        to_remove = sorted(
            to_remove_links + to_remove_joints,
            key=lambda x: x.pathElementCount,
            reverse=True,
        )

        print(f"제거할 prim 수: 링크 {len(to_remove_links)}, 조인트 {len(to_remove_joints)}, 합계 {len(to_remove)}")
        if args.dry_run:
            for p in to_remove[:25]:
                print(" ", p.pathString)
            if len(to_remove) > 25:
                print(" ... 외", len(to_remove) - 25, "개")
            return 0

        edit = Sdf.BatchNamespaceEdit()
        for p in to_remove:
            edit.Add(p, Sdf.Path())
        if not layer.Apply(edit):
            print("BatchNamespaceEdit Apply 실패", file=sys.stderr)
            return 3

        if not layer.Export(str(dst_usd)):
            print(f"저장 실패: {dst_usd}", file=sys.stderr)
            return 4

        print(f"저장 완료: {dst_usd}")
        verify = Usd.Stage.Open(str(dst_usd))
        print(f"확인: prim 수 = {sum(1 for _ in verify.Traverse())}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
