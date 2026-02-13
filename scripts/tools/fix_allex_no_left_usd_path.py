#!/usr/bin/env python3
"""
ALLEX_newton_no_left.usd 내부의 Windows 절대 경로를
allex_usd 기준 상대 경로(allex_model_mjcf_250903/mesh)로 치환합니다.
바이너리 USD에서 동일 길이(43자)로만 치환하므로 파일 구조는 유지됩니다.

사용: python fix_allex_no_left_usd_path.py [path_to_usd]
기본: source/isaaclab_assets/allex_usd/ALLEX_newton_no_left.usd
"""

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) > 1:
        usd_path = Path(sys.argv[1]).resolve()
    else:
        repo = Path(__file__).resolve().parents[2]
        usd_path = repo / "source/isaaclab_assets/allex_usd/ALLEX_newton_no_left.usd"

    if not usd_path.is_file():
        print(f"파일 없음: {usd_path}", file=sys.stderr)
        sys.exit(1)

    old_path = b"C:\\isaac-sim\\assets\\allex_model_mjcf_250903"
    # 상대 경로 (USD가 allex_usd/ 에 있으므로 allex_model_mjcf_250903/mesh 가 같은 폴더 내 mesh)
    new_path_raw = "allex_model_mjcf_250903/mesh"
    if len(new_path_raw) > len(old_path):
        print("새 경로가 43자 초과라 동일 길이 치환 불가.", file=sys.stderr)
        sys.exit(2)
    new_path = new_path_raw.encode("utf-8").ljust(len(old_path), b" ")

    data = usd_path.read_bytes()
    if old_path not in data:
        print("대상 경로가 파일에 없습니다. 이미 수정되었거나 다른 파일입니다.")
        return

    data = data.replace(old_path, new_path, 1)
    usd_path.write_bytes(data)
    print(f"치환 완료: {usd_path}")
    print(f"  이전: {old_path.decode()!r}")
    print(f"  이후: {new_path.decode().rstrip()!r}")


if __name__ == "__main__":
    main()
