#!/usr/bin/env python3
"""
스폰을 시뮬레이션: no_left USD를 /World/envs/env_0/Robot 에 reference로 붙인 뒤
해당 경로 아래 구조(링크/메시 개수)를 덤프합니다.
Reference가 풀렸을 때 전체가 보이는지 확인용.

Usage: conda run -n isaaclab python scripts/tools/check_stage_after_reference.py <path/to/ALLEX_newton_no_left.usd>
"""

import sys
from pathlib import Path

def main():
    from pxr import Usd, UsdGeom, UsdPhysics

    if len(sys.argv) < 2:
        print("Usage: python check_stage_after_reference.py <ALLEX_newton_no_left.usd>", file=sys.stderr)
        sys.exit(1)
    usd_path = Path(sys.argv[1]).resolve()
    if not usd_path.exists():
        print(f"Not found: {usd_path}", file=sys.stderr)
        sys.exit(2)

    # 새 스테이지에 reference 붙이기 (스폰과 유사하게)
    stage = Usd.Stage.CreateInMemory()
    stage.SetDefaultPrim(stage.DefinePrim("/World"))
    env = stage.DefinePrim("/World/envs/env_0")
    robot_prim = env.GetStage().DefinePrim("/World/envs/env_0/Robot")
    robot_prim.GetReferences().AddReference(str(usd_path))

    # Reference 붙인 후 composed subtree 확인
    robot = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    if not robot:
        print("Robot prim not found")
        sys.exit(3)

    def count(prim):
        meshes, bodies, xforms = 0, 0, 0
        for p in prim.GetAllChildren():
            if p.IsA(UsdGeom.Mesh):
                meshes += 1
            if UsdPhysics.RigidBodyAPI(p):
                bodies += 1
            if p.GetTypeName() == "Xform":
                xforms += 1
            m, b, x = count(p)
            meshes, bodies, xforms = meshes + m, bodies + b, xforms + x
        return meshes, bodies, xforms

    meshes, bodies, xforms = count(robot)
    direct = list(robot.GetChildren())
    direct_names = [c.GetName() for c in direct]
    link_like = [n for n in direct_names if "link" in n or "Base" in n or "Hand" in n or "Palm" in n]

    print(f"=== Reference at /World/envs/env_0/Robot ({usd_path.name}) ===")
    print(f"Direct children count: {len(direct)}")
    print(f"Link-like direct children (sample): {link_like[:15]} ...")
    print(f"Under Robot - Meshes: {meshes}, RigidBodyAPI: {bodies}, Xforms: {xforms}")
    if meshes >= 100 and bodies >= 30:
        print("=> OK: Reference composition shows full structure (all links + meshes).")
        print("   If Newton still shows only base, the cause is likely in newton add_usd/load_visual_shapes.")
    else:
        print("=> WARNING: Fewer meshes/bodies than expected. Reference may not be fully composed here.")

if __name__ == "__main__":
    main()
