"""Generate occupancy.ply for clutter scenes from their self-contained USD meshes.

The real ESDF (occupancy.ply) lives in the gated `mp3d_*.tar.gz` dataset, so we
reconstruct a point cloud of the scene's occupied geometry from the USD mesh
surface vertices (world-space). This replaces the placeholder boundary-box
occupancy used for the smoke test.
"""
import os
import sys

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import open3d as o3d
from pxr import Gf, Usd, UsdGeom


def extract_mesh_points(usd_path: str, floor_z: float = 0.18) -> np.ndarray | None:
    """Collect world-space mesh vertices above the floor (obstacles), dropping the ground plane."""
    stage = Usd.Stage.Open(usd_path)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    all_pts = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts is None or len(pts) == 0:
            continue
        xf = xform_cache.GetLocalToWorldTransform(prim)
        for p in pts:
            wp = xf.Transform(p)
            # Keep only points above the ground plane (walls/furniture), so the
            # floor is traversable in the 2D occupancy grid.
            if wp[2] > floor_z:
                all_pts.append((wp[0], wp[1], wp[2]))
    if not all_pts:
        return None
    return np.asarray(all_pts, dtype=np.float64)


def main():
    scenes_dir = os.environ.get("SCENE_DIR", "/mnt/data3/cqz/nav/scenes")
    count = 0
    for clutter_type in ("cluttered_easy", "cluttered_hard"):
        cdir = os.path.join(scenes_dir, clutter_type)
        if not os.path.isdir(cdir):
            continue
        for scene in sorted(os.listdir(cdir)):
            scene_dir = os.path.join(cdir, scene)
            if not os.path.isdir(scene_dir):
                continue
            usd_files = sorted(
                f for f in os.listdir(scene_dir)
                if f.endswith(".usd") and "noMDL" not in f and "scale" not in f
            )
            if not usd_files:
                continue
            usd_path = os.path.join(scene_dir, usd_files[0])
            pts = extract_mesh_points(usd_path)
            if pts is None or len(pts) == 0:
                print(f"[skip] {clutter_type}/{scene}: no mesh found in {usd_files[0]}")
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            out_path = os.path.join(scene_dir, "occupancy.ply")
            o3d.io.write_point_cloud(out_path, pcd)
            count += 1
            print(
                f"[ok] {clutter_type}/{scene}: {len(pts)} pts "
                f"x[{pts[:, 0].min():.1f},{pts[:, 0].max():.1f}] "
                f"y[{pts[:, 1].min():.1f},{pts[:, 1].max():.1f}]"
            )
    print(f"generated {count} occupancy.ply files")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
