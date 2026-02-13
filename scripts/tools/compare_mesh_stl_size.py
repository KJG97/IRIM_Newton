#!/usr/bin/env python3
"""Compare actual mesh geometry size (bounding box) of STL files between
   allex_model_mjcf_250903/mesh and URDF_ALLEX_RightArm/meshes.
"""
from pathlib import Path
import struct


def read_stl_vertices(path: Path):
    """Read all vertices from binary or ASCII STL. Returns list of (x,y,z)."""
    data = path.read_bytes()
    # Binary: first 5 bytes are not "solid" (ASCII)
    if data[:5].lower() == b"solid" and b"endsolid" in data:
        return _read_ascii_stl(data)
    return _read_binary_stl(data)


def _read_binary_stl(data: bytes):
    if len(data) < 84:
        return []
    n_tri_file = struct.unpack_from("<I", data, 80)[0]
    # 50 bytes per tri; we read 48 bytes at off+12, so need off+60 <= len(data)
    max_tri = (len(data) - 84 - 60 + 50) // 50 if len(data) >= 144 else 0
    n_tri = min(n_tri_file, max(0, max_tri))
    verts = []
    for i in range(n_tri):
        off = 84 + i * 50
        if off + 60 > len(data):
            break
        vs = struct.unpack_from("<12f", data, off + 12)  # skip normal, read v0,v1,v2
        verts.extend([(vs[0], vs[1], vs[2]), (vs[3], vs[4], vs[5]), (vs[6], vs[7], vs[8])])
    return verts


def _read_ascii_stl(data: bytes):
    text = data.decode("utf-8", errors="ignore")
    verts = []
    lines = text.replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("vertex "):
            parts = line.split()
            if len(parts) >= 4:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        i += 1
    return verts


def bbox(verts):
    if not verts:
        return None, None
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def extent(min_pt, max_pt):
    return (
        max_pt[0] - min_pt[0],
        max_pt[1] - min_pt[1],
        max_pt[2] - min_pt[2],
    )


def main():
    base = Path(__file__).resolve().parents[2] / "source/isaaclab_assets/allex_usd"
    mjcf_mesh_dir = base / "allex_model_mjcf_250903/mesh"
    urdf_meshes_dir = base / "URDF_ALLEX_RightArm/meshes"

    # Map URDF mesh name -> possible MJCF equivalents (right arm / forearm / elbow etc.)
    # URDF: base_link, elbow, forearm, shoulder_pitch, shoulder_roll, shoulder_yaw, tcp, wrist_pitch, wrist_roll
    name_map = [
        ("elbow.STL", "ALLEX_Right_Elbow_Frame.stl"),
        ("forearm.STL", "ALLEX_Right_Forearm_Base_Cover.stl"),
        ("shoulder_pitch.STL", "ALLEX_Right_Shoulder_Pitch_Frame.stl"),  # MJCF has different names
        ("shoulder_roll.STL", "ALLEX_Right_Shoulder_Yaw_Frame.stl"),
        ("shoulder_yaw.STL", "ALLEX_Right_Shoulder_Yaw_Frame.stl"),
        ("wrist_pitch.STL", "ALLEX_Right_Wrist_Pitch_Frame.stl"),
        ("wrist_roll.STL", "ALLEX_Right_Wrist_Roll_Frame.stl"),
    ]

    def report_dir(title, d: Path):
        print(f"\n=== {title} ({d}) ===")
        stls = sorted(d.glob("*.stl")) or sorted(d.glob("*.STL"))
        for stl in stls:
            verts = read_stl_vertices(stl)
            if not verts:
                print(f"  {stl.name}: (read failed or empty)")
                continue
            mn, mx = bbox(verts)
            ext = extent(mn, mx)
            diag = (ext[0] ** 2 + ext[1] ** 2 + ext[2] ** 2) ** 0.5
            print(f"  {stl.name}")
            print(f"    extent (dx,dy,dz) = ({ext[0]:.6f}, {ext[1]:.6f}, {ext[2]:.6f}) [m or mm]")
            print(f"    diagonal ≈ {diag:.6f}  (bbox min={mn}, max={mx})")

    report_dir("URDF_ALLEX_RightArm/meshes", urdf_meshes_dir)
    report_dir("allex_model_mjcf_250903/mesh (first 15 Right-arm related)", mjcf_mesh_dir)

    # Direct comparison for mappable pairs
    print("\n--- Direct comparison (same body part) ---")
    mjcf_stls = {p.name: p for p in (mjcf_mesh_dir.glob("*.stl") or mjcf_mesh_dir.glob("*.STL"))}
    for urdf_name, mjcf_name in name_map:
        urdf_path = urdf_meshes_dir / urdf_name
        mjcf_path = mjcf_stls.get(mjcf_name)
        if not urdf_path.exists():
            continue
        if not mjcf_path:
            print(f"  {urdf_name} vs {mjcf_name}: MJCF file not found")
            continue
        v_urdf = read_stl_vertices(urdf_path)
        v_mjcf = read_stl_vertices(mjcf_path)
        if not v_urdf or not v_mjcf:
            continue
        mn_u, mx_u = bbox(v_urdf)
        mn_m, mx_m = bbox(v_mjcf)
        ext_u = extent(mn_u, mx_u)
        ext_m = extent(mn_m, mx_m)
        ratio_x = ext_m[0] / ext_u[0] if ext_u[0] else 0
        ratio_y = ext_m[1] / ext_u[1] if ext_u[1] else 0
        ratio_z = ext_m[2] / ext_u[2] if ext_u[2] else 0
        print(f"  {urdf_name}")
        print(f"    URDF extent: ({ext_u[0]:.6f}, {ext_u[1]:.6f}, {ext_u[2]:.6f})")
        print(f"    MJCF extent: ({ext_m[0]:.6f}, {ext_m[1]:.6f}, {ext_m[2]:.6f})")
        print(f"    ratio MJCF/URDF: ({ratio_x:.4f}, {ratio_y:.4f}, {ratio_z:.4f})  <- if ~1000 then MJCF is in mm, URDF in m")
        print()

    # Summary: typical scale difference
    print("\n--- Summary ---")
    urdf_elbow = urdf_meshes_dir / "elbow.STL"
    mjcf_elbow = mjcf_mesh_dir / "ALLEX_Right_Elbow_Frame.stl"
    if urdf_elbow.exists() and mjcf_elbow.exists():
        eu = extent(*bbox(read_stl_vertices(urdf_elbow)))
        em = extent(*bbox(read_stl_vertices(mjcf_elbow)))
        r = [em[i] / eu[i] for i in range(3) if eu[i] > 1e-9]
        if r:
            avg_ratio = sum(r) / len(r)
            print(f"  Elbow: URDF extent ~ {eu}, MJCF extent ~ {em}")
            print(f"  Average extent ratio (MJCF/URDF) ≈ {avg_ratio:.2f}")
            if avg_ratio > 100:
                print("  => MJCF meshes are in LARGER units (e.g. mm) than URDF (m).")
            elif avg_ratio < 0.001:
                print("  => MJCF meshes are in SMALLER units than URDF.")
            else:
                print("  => Similar scale; ratio suggests unit conversion factor.")


if __name__ == "__main__":
    main()
