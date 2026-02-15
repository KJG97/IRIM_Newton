#!/usr/bin/env python3
"""
allex_contact_sensor.usd가 뷰어/앱에서 열리도록 함.
- Stage를 열고(경고 무시) 모든 reference/payload를 제거해
  Flattened_Prototype 미해결 참조를 없앰. 계층은 유지되고 메시는 빈 상태가 됨.
- 또는 ASCII로 내보내서 에디터에서 내용 확인 가능: --usda 옵션.

사용:
  conda activate isaaclab
  python fix_allex_contact_sensor_open.py source/isaaclab_assets/allex_usd/allex_contact_sensor.usd
  python fix_allex_contact_sensor_open.py source/isaaclab_assets/allex_usd/allex_contact_sensor.usd --usda  # ASCII로 저장
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from pxr import Sdf, Usd

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ascii_out = "--usda" in sys.argv or "--ascii" in sys.argv

    if not args:
        print("Usage: python fix_allex_contact_sensor_open.py <input.usd> [output.usd] [--usda]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args[0]).resolve()
    output_path = Path(args[1]).resolve() if len(args) > 1 else input_path

    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    # Open stage (will emit Unresolved reference warnings)
    stage = Usd.Stage.Open(str(input_path))
    if not stage:
        print("Failed to open stage", file=sys.stderr)
        sys.exit(3)

    # Edit root layer and remove all references and payloads from every prim
    layer = stage.GetRootLayer()
    stage.SetEditTarget(stage.GetRootLayer())
    for prim in stage.Traverse():
        refs = prim.GetReferences()
        payloads = prim.GetPayloads()
        if refs:
            refs.ClearReferences()
        if payloads:
            payloads.ClearPayloads()

    if ascii_out:
        out = output_path.with_suffix(".usda") if output_path.suffix.lower() == ".usd" else Path(str(output_path) + ".usda")
        stage.Export(str(out))
        print(f"Exported ASCII: {out}")
    else:
        stage.Export(str(output_path))
        print(f"Saved (all refs/payloads cleared): {output_path}")
    print("File should open in viewers now (geometry will be empty).")


if __name__ == "__main__":
    main()
