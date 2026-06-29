#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_fishmap.py — Step 1: 从标定数据生成鱼眼查找表

核心思想:
  鱼眼图每个像素 → 世界方向射线 → 记录所有可见摄像头 → 原图 (u, v)

投影方式:
  圆形等距鱼眼 (equidistant circular fisheye)
  - 圆心 = 天顶
  - 圆周 = 地平线
  - 径向: r ∝ zenith_angle (天顶角)

输出 fishmap.bin (二进制):
  头: magic(4B) + version(4=I) + width(I) + height(I) + max_cam(I) = 20B
  数据: width×height × (1 + max_cam×3)×f32 = {count, cam0,u0,v0, cam1,u1,v1, ...}
    count=可见相机数, cam_i=-1 表示空槽位

Usage:
  python generate_fishmap.py --calib calibration_result.json \\
      [--size 3000] [--image_w 2560] [--image_h 1440] \\
      [--max_zenith 90] --out fishmap.bin
"""

from __future__ import annotations

import argparse
import json
import math
import struct

import numpy as np

from fit_sky_camera import rot_yaw_pitch_roll

MAX_CAM_PER_PIXEL = 4  # 每个像素最多存 4 个相机


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate fisheye lookup map")
    ap.add_argument("--calib", default="calibration_result_lite.json",
                    help="Calibration JSON from fit_sky_camera.py")
    ap.add_argument("--out", default="fishmap_lite.bin",
                    help="Output binary lookup table")
    ap.add_argument("--size", type=int, default=3000,
                    help="Output image size (square, pixels)")
    ap.add_argument("--image_w", type=int, default=2560,
                    help="Source image width in pixels")
    ap.add_argument("--image_h", type=int, default=1440,
                    help="Source image height in pixels")
    ap.add_argument("--max_zenith", type=float, default=80.0,
                    help="Max zenith angle (deg) at circle edge; 90=horizon, <90=crop sky")
    ap.add_argument("--tile_size", type=int, default=512,
                    help="Tile size (px) for memory-efficient processing (default 512)")
    args = ap.parse_args()

    # ── 1. 加载标定数据 ────────────────────────────────────────────
    with open(args.calib, "r", encoding="utf-8") as f:
        calib = json.load(f)

    cx = calib["principal_point"]["cx"]
    cy = calib["principal_point"]["cy"]
    b1 = calib["radial_model"]["b1"]
    b3 = calib["radial_model"]["b3"]
    b5 = calib["radial_model"]["b5"]

    cam_ids = sorted(calib["cameras"].keys(), key=int)
    n_cam = len(cam_ids)
    print(f"Loaded {n_cam} cameras: {cam_ids}")
    print(f"Principal point: ({cx:.2f}, {cy:.2f})")
    print(f"Radial model: b1={b1:.4f}, b3={b3:.6e}, b5={b5:.6e}")

    # 预计算每台相机的旋转矩阵和主点偏移
    cam_data = []
    for cid in cam_ids:
        c = calib["cameras"][cid]
        R = rot_yaw_pitch_roll(c["Az_rad"], c["Alt_rad"], c["Roll_rad"])
        cam_data.append({
            "R": R,
            "dcx": c["principal_point_offset_px"]["dcx"],
            "dcy": c["principal_point_offset_px"]["dcy"],
        })

    # ── 2. 分块构建鱼眼查找表 (tiling, 降低峰值内存) ──────────
    SZ = args.size
    M = MAX_CAM_PER_PIXEL
    ch = 1 + M * 3
    T = args.tile_size

    print(f"\nBuilding fishmap: {SZ}×{SZ}, tile={T}×{T}, max_cam={M} ...")

    # memmap 输出文件 (数据从字节 20 开始, 避开文件头)
    tmp_path = str(args.out) + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(b"\0" * 20)  # placeholder header
        f.truncate(20 + SZ * SZ * ch * 4)  # 预分配完整文件大小
    out_data = np.memmap(tmp_path, dtype=np.float32, mode="r+",
                         shape=(SZ, SZ, ch), offset=20)

    cx_px = (SZ - 1) * 0.5
    cy_px = (SZ - 1) * 0.5
    R_px = SZ * 0.5
    max_zenith_rad = math.radians(args.max_zenith)
    iw, ih = float(args.image_w), float(args.image_h)
    u_min, u_max = 0.0, iw - 0.5
    v_min, v_max = 0.0, ih - 0.5

    total_covered = 0
    overlap_2 = overlap_3 = overlap_4 = 0

    n_tiles_h = (SZ + T - 1) // T
    n_tiles_w = (SZ + T - 1) // T
    tile_idx = 0

    for ty in range(0, SZ, T):
        tile_h = min(T, SZ - ty)
        for tx in range(0, SZ, T):
            tile_idx += 1
            tile_w = min(T, SZ - tx)

            # ── Tile 内的世界射线 ──
            yy, xx = np.mgrid[ty:ty + tile_h, tx:tx + tile_w].astype(np.float64)
            dx = xx - cx_px
            dy = yy - cy_px
            r_px_t = np.hypot(dx, dy)

            in_circle = r_px_t <= R_px

            zenith_rad = np.where(in_circle, (r_px_t / R_px) * max_zenith_rad, 0.0)
            alt_rad = np.where(in_circle, math.pi / 2.0 - zenith_rad, 0.0)
            az_rad = np.where(in_circle, np.arctan2(dx, -dy), 0.0)

            cos_alt = np.cos(alt_rad)
            sin_alt = np.sin(alt_rad)
            sin_az = np.sin(az_rad)
            cos_az = np.cos(az_rad)

            world_rays = np.stack([
                cos_alt * sin_az,
                cos_alt * cos_az,
                sin_alt,
            ], axis=-1)

            circle_mask = in_circle & (zenith_rad <= max_zenith_rad)

            # ── 本 tile 的相机存储 ──
            pixel_count = np.zeros((tile_h, tile_w), dtype=np.int32)
            cam_store = np.full((tile_h, tile_w, M), -1, dtype=np.int32)
            u_store = np.zeros((tile_h, tile_w, M), dtype=np.float32)
            v_store = np.zeros((tile_h, tile_w, M), dtype=np.float32)

            for cam_idx, cd in enumerate(cam_data):
                R = cd["R"]
                dcx, dcy = cd["dcx"], cd["dcy"]

                v_cam = world_rays @ R

                behind = v_cam[:, :, 2] <= 1e-9

                norm = np.linalg.norm(v_cam, axis=-1)
                cos_theta = np.clip(v_cam[:, :, 2] / norm, -1.0, 1.0)
                theta = np.arccos(cos_theta)

                r = np.zeros_like(theta)
                vt = ~behind
                t = theta[vt]
                t2 = t ** 2
                r[vt] = b1 * t + b3 * t * t2 + b5 * t * t2 ** 2

                x = v_cam[:, :, 0]
                y = v_cam[:, :, 1]
                norm_xy = np.hypot(x, y)
                zn = norm_xy < 1e-12
                u = cx + dcx + np.where(zn, 0.0, r * x / np.where(zn, 1.0, norm_xy))
                v = cy + dcy + np.where(zn, 0.0, r * y / np.where(zn, 1.0, norm_xy))

                cam_valid = (
                    circle_mask
                    & ~behind
                    & (u >= u_min) & (u <= u_max)
                    & (v >= v_min) & (v <= v_max)
                )

                has_space = cam_valid & (pixel_count < M)
                n_stored = np.count_nonzero(has_space)

                if n_stored > 0:
                    slot = pixel_count[has_space]
                    cam_store[has_space, slot] = cam_idx
                    u_store[has_space, slot] = u[has_space].astype(np.float32)
                    v_store[has_space, slot] = v[has_space].astype(np.float32)
                    pixel_count[has_space] += 1

            # ── 写入 memmap ──
            out_data[ty:ty + tile_h, tx:tx + tile_w, 0] = pixel_count.astype(np.float32)
            for s in range(M):
                out_data[ty:ty + tile_h, tx:tx + tile_w, 1 + s * 3 + 0] = cam_store[:, :, s].astype(np.float32)
                out_data[ty:ty + tile_h, tx:tx + tile_w, 1 + s * 3 + 1] = u_store[:, :, s]
                out_data[ty:ty + tile_h, tx:tx + tile_w, 1 + s * 3 + 2] = v_store[:, :, s]

            total_covered += np.count_nonzero(pixel_count >= 1)
            overlap_2 += np.count_nonzero(pixel_count >= 2)
            overlap_3 += np.count_nonzero(pixel_count >= 3)
            overlap_4 += np.count_nonzero(pixel_count >= 4)

            print(f"  Tile {tile_idx:>3d}/{n_tiles_h * n_tiles_w}  "
                  f"({tx:>4d},{ty:>4d})  "
                  f"covered={np.count_nonzero(pixel_count >= 1)}")

        out_data.flush()

    total_px = int(math.pi * R_px * R_px)
    coverage = total_covered / total_px * 100.0 if total_px else 0.0
    print(f"\nTotal covered: {total_covered}/{total_px} ({coverage:.1f}%)")
    print(f"Overlap pixels: 2+:{overlap_2}  3+:{overlap_3}  4+:{overlap_4}")

    # ── 写入 header 并重命名 ──────────────────────────────────
    out_data.flush()
    del out_data
    with open(tmp_path, "r+b") as f:
        f.seek(0)
        f.write(struct.pack("<4sIIII", b"FISH", 4, SZ, SZ, M))
    import os
    os.replace(tmp_path, args.out)

    file_size_mb = 20 + ch * SZ * SZ * 4
    print(f"Saved: {args.out}  ({file_size_mb / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
