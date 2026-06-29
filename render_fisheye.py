#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_fisheye.py — Step 2: 用查找表 + N 张原图渲染鱼眼合成图

- 第一步: 统一所有原图的亮度和白平衡
- 第二步: 距离羽化 (distance feathering) — 每台相机在自身覆盖区边缘
  权重平滑降至 0, 重叠区自动归一化, 单相机区不受影响

Usage:
  python render_fisheye.py --map fishmap.bin \\
      --images cam1.jpg cam2.jpg ... cam6.jpg \\
      --out fisheye.png
"""

from __future__ import annotations

import argparse
import os
import re
import struct

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


# ── 图像归一化 ──────────────────────────────────────────────────

def compute_image_stats(img: np.ndarray):
    """计算图像有效区域 (非纯黑/纯饱和) 的 per-channel 均值."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 5) & (gray < 250)  # 排除死黑和过曝
    if mask.sum() < 100:
        mask = np.ones(img.shape[:2], dtype=bool)
    means = img[mask].mean(axis=0)  # B, G, R
    return means.astype(np.float64)


def normalize_images(images: list[np.ndarray]) -> list[np.ndarray]:
    """
    将所有图像归一化到相同的全局均值.
    返回归一化后的图像列表.
    """
    n = len(images)
    if n <= 1:
        return images

    stats = [compute_image_stats(im) for im in images]
    global_mean = np.mean(stats, axis=0)  # (3,)

    print("\nWhite-balance + brightness normalization:")
    print(f"  Target mean (B,G,R): ({global_mean[0]:.1f}, {global_mean[1]:.1f}, {global_mean[2]:.1f})")

    result = []
    for i, (img, sm) in enumerate(zip(images, stats)):
        scale = np.where(sm > 1, global_mean / sm, 1.0).astype(np.float64)
        corrected = (img.astype(np.float64) * scale).clip(0, 255).astype(np.uint8)
        print(f"  cam{i+1}: was ({sm[0]:.1f},{sm[1]:.1f},{sm[2]:.1f}) → "
              f"scale=({scale[0]:.3f},{scale[1]:.3f},{scale[2]:.3f})")
        result.append(corrected)

    return result


# ── 查找表读取 ──────────────────────────────────────────────────

def read_fishmap(path: str):
    """
    读取 v4 fishmap.bin.
    返回: (counts, cam_store, u_store, v_store, W, H, M)
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"FISH":
            raise ValueError(f"Not a valid fishmap file (magic={magic!r})")
        version, W, H = struct.unpack("<III", f.read(12))

        if version == 4:
            M = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"Need v4 fishmap, got version={version}")

        ch = 1 + M * 3
        raw = np.frombuffer(f.read(), dtype=np.float32).reshape(H, W, ch)

    counts = raw[:, :, 0].astype(np.int32)
    cam_store = np.full((H, W, M), -1, dtype=np.int32)
    u_store = np.zeros((H, W, M), dtype=np.float32)
    v_store = np.zeros((H, W, M), dtype=np.float32)

    for s in range(M):
        cam_store[:, :, s] = raw[:, :, 1 + s*3 + 0].astype(np.int32)
        u_store[:, :, s]   = raw[:, :, 1 + s*3 + 1]
        v_store[:, :, s]   = raw[:, :, 1 + s*3 + 2]

    print(f"Loaded fishmap: {W}×{H}, version={version}, max_cam={M}")
    return counts, cam_store, u_store, v_store, W, H, M


# ── 渲染 ────────────────────────────────────────────────────────

def _build_camera_mask(cam_store: np.ndarray, u_store: np.ndarray,
                       v_store: np.ndarray, cam_i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提取 cam_i 在 fishmap 中的覆盖区和采样坐标.
    返回 (pix_mask, map_x, map_y):
      pix_mask: (H,W) bool, 该相机的有效像素
      map_x/y:  (H,W) float32, remap 坐标 (无效处填 -1)
    """
    H, W, M = cam_store.shape
    mask_per_slot = cam_store == cam_i  # (H,W,M)
    map_x = np.full((H, W), -1.0, dtype=np.float32)
    map_y = np.full((H, W), -1.0, dtype=np.float32)
    pix_mask = np.zeros((H, W), dtype=bool)
    for s in range(M):
        slot_mask = mask_per_slot[:, :, s]
        if not np.any(slot_mask):
            continue
        map_x[slot_mask] = u_store[slot_mask, s]
        map_y[slot_mask] = v_store[slot_mask, s]
        pix_mask[slot_mask] = True
    return pix_mask, map_x, map_y


def render_fisheye(
    counts: np.ndarray,
    cam_store: np.ndarray,
    u_store: np.ndarray,
    v_store: np.ndarray,
    src_images: list[np.ndarray],
    feather_px: float = 50.0,
) -> np.ndarray:
    """
    距离羽化多相机融合.

    - 对每台相机, 从其覆盖区边缘向内做距离变换 → per-pixel 权重
    - 权重在边缘 = 0, feather_px 像素后 = 1 (线性 ramp)
    - 最终: out = Σ(warped_i × w_i) / Σ w_i (权重自动归一化)

    Parameters
    ----------
    counts     : (H,W) int32
    cam_store  : (H,W,M) int32
    u_store    : (H,W,M) float32
    v_store    : (H,W,M) float32
    src_images : 原图列表 (已归一化)
    feather_px : 羽化宽度 (像素), 0 = 硬边界

    Returns: (H,W,3) uint8
    """
    H, W, M = cam_store.shape
    n_cam = len(src_images)

    acc        = np.zeros((H, W, 3), dtype=np.float32)
    weight_sum = np.zeros((H, W),    dtype=np.float32)

    for cam_i in range(n_cam):
        pix_mask, map_x, map_y = _build_camera_mask(cam_store, u_store, v_store, cam_i)
        n_px = np.count_nonzero(pix_mask)
        if n_px == 0:
            print(f"  Camera {cam_i+1}:  no coverage — skip")
            continue

        # ── remap ──────────────────────────────────────────
        warped = cv2.remap(
            src_images[cam_i], map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        ).astype(np.float32)

        # ── 距离羽化权重 ────────────────────────────────────
        if feather_px > 0.1:
            dist = distance_transform_edt(pix_mask.astype(np.uint8))
            w = np.clip(q, 0.0, 1.0).astype(np.float32)
        else:
            w = pix_mask.astype(np.float32)

        acc        += warped * w[:, :, np.newaxis]
        weight_sum += w
        print(f"  Camera {cam_i+1}: {n_px:>8d} pixels, "
              f"feather weight min={w[pix_mask].min():.3f} max={w[pix_mask].max():.3f}")

    # 归一化 (防除零)
    ws = np.where(weight_sum > 1e-9, weight_sum, 1.0)
    out = (acc / ws[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

    n_multi = np.count_nonzero(counts >= 2)
    print(f"  Feather width: {feather_px:.0f} px  |  blended (≥2 cam): {n_multi}")
    return out


# ── 径向平衡 ────────────────────────────────────────────────────

def radial_balance(img: np.ndarray, max_zenith_deg: float = 90.0,
                   poly_order: int = 3, n_rings: int = 40) -> np.ndarray:
    """
    拟合平滑径向亮度曲线, 消除相机间系统亮度偏移.

    思路:
      1. 计算每个天顶角环的 per-channel 均值
      2. 用低阶多项式拟合 → 平滑曲线
      3. 逐像素校正: out = img × smooth_profile / raw_profile

    单相机区自然过渡, 不会出现突兀的亮度跳变.
    """
    H, W = img.shape[:2]
    cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
    R = min(W, H) * 0.5

    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot(xx - cx, yy - cy)
    in_circle = dist <= R

    # 天顶角 (度)
    zenith = np.where(in_circle, (dist / R) * max_zenith_deg, -1.0)

    # 每环 per-channel 均值
    ring_centers = []
    ring_means = []  # list of (B,G,R)
    for i in range(n_rings):
        z_lo = (i / n_rings) * max_zenith_deg
        z_hi = ((i + 1) / n_rings) * max_zenith_deg
        mask = in_circle & (zenith >= z_lo) & (zenith < z_hi)
        n = mask.sum()
        if n < 50:
            continue
        ring_centers.append((z_lo + z_hi) / 2)
        ring_means.append(img[mask].mean(axis=0).astype(np.float64))

    ring_centers = np.array(ring_centers)
    ring_means = np.array(ring_means)  # (n_valid, 3)

    # 多项式拟合 → 平滑曲线
    fit_coeffs = []
    smooth_vals = np.zeros_like(ring_means)
    for ch in range(3):
        coeff = np.polyfit(ring_centers, ring_means[:, ch], poly_order)
        fit_coeffs.append(coeff)
        smooth_vals[:, ch] = np.polyval(coeff, ring_centers)

    # 校正因子: smooth / raw (每个环)
    ratio_per_ring = np.where(ring_means > 1, smooth_vals / ring_means, 1.0)

    print(f"\nRadial balance (poly order={poly_order}):")
    max_dev = np.abs(1.0 - ratio_per_ring).max(axis=0)
    print(f"  Max per-ring correction: B={max_dev[0]:.3f}  G={max_dev[1]:.3f}  R={max_dev[2]:.3f}")

    # 逐像素插值校正因子
    correction_map = np.ones((H, W, 3), dtype=np.float64)
    for ch in range(3):
        # 对每个有效像素, 用其天顶角查插值后的校正因子
        valid = in_circle
        z_valid = zenith[valid]
        # 线性插值: 环中心 → 像素天顶角
        corr_interp = np.interp(z_valid, ring_centers, ratio_per_ring[:, ch])
        cmap_ch = np.ones((H, W), dtype=np.float64)
        cmap_ch[valid] = corr_interp
        correction_map[:, :, ch] = cmap_ch

    out = (img.astype(np.float64) * correction_map).clip(0, 255).astype(np.uint8)
    return out


# ── 自动发现图片 ──────────────────────────────────────────────

def auto_discover_camera_images(n_cam_expected: int = 6,
                                search_dir: str = ".") -> list[str]:
    """
    自动扫描 search_dir 下所有文件，
    从文件名中提取 "通道N" 的通道号，按 1..n_cam_expected 排序返回完整路径。
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    pattern = re.compile(r"通道(\d+)")
    found: dict[int, str] = {}

    for fname in os.listdir(search_dir):
        m = pattern.search(fname)
        if not m:
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in img_exts:
            continue
        ch = int(m.group(1))
        found[ch] = os.path.join(search_dir, fname)

    result: list[str] = []
    for ch in range(1, n_cam_expected + 1):
        if ch not in found:
            raise FileNotFoundError(
                f"未找到通道 {ch} 的图片（当前目录下应包含文件名带"
                f"“通道{ch}”的图片）"
            )
        result.append(found[ch])
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Render fisheye from lookup map + images")
    ap.add_argument("--map", default="fishmap.bin",
                    help="Lookup table from generate_fishmap.py")
    ap.add_argument("--images", nargs="*", default=None,
                    help="Source images, ordered by camera index (1..N)")
    ap.add_argument("--image_dir", default=None,
                    help="Directory containing source images")
    ap.add_argument("--image_ext", default=".jpg",
                    help="Image extension (used with --image_dir)")
    ap.add_argument("--n_cam", type=int, default=6,
                    help="Number of cameras (used with --image_dir)")
    ap.add_argument("--auto_detect", type=int, default=None,
                    help="1=auto-discover images by channel in filename "
                         "(default: auto when --images and --image_dir not set)")
    ap.add_argument("--out", default="fisheye.png",
                    help="Output fisheye image path")
    ap.add_argument("--feather_px", type=float, default=300.0,
                    help="Distance-feather width in pixels (0=hard edge)")
    ap.add_argument("--radial_balance", type=int, default=1,
                    help="1=auto-correct inter-camera brightness offsets (default), 0=off")
    args = ap.parse_args()

    # ── 1. 自动检测或手动指定图片 ────────────────────────────
    if args.images:
        img_paths = args.images
    elif args.image_dir:
        img_paths = [
            os.path.join(args.image_dir, f"cam{i}{args.image_ext}")
            for i in range(1, args.n_cam + 1)
        ]
    elif args.auto_detect is not False:
        img_paths = auto_discover_camera_images(n_cam_expected=args.n_cam)
        print(f"  Auto-detected {len(img_paths)} camera images:")
        for i, p in enumerate(img_paths, 1):
            print(f"    ch{i}: {os.path.basename(p)}")
    else:
        img_paths = [f"cam{i}.jpg" for i in range(1, args.n_cam + 1)]

    src_images = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  WARNING: cannot open '{p}' — skipping")
            src_images.append(np.zeros((10, 10, 3), dtype=np.uint8))
        else:
            print(f"  Loaded: {p}  ({img.shape[1]}×{img.shape[0]})")
            src_images.append(img)

    n_cam_loaded = len([img for img in src_images if img.size > 100])
    print(f"Loaded {n_cam_loaded}/{len(img_paths)} images")

    # ── 2. 白平衡 + 亮度归一化 ─────────────────────────────────
    src_images = normalize_images(src_images)

    # ── 3. 加载查找表 ──────────────────────────────────────────
    counts, cam_store, u_store, v_store, W, H, M = read_fishmap(args.map)

    # ── 4. 渲染 ────────────────────────────────────────────────
    print(f"\nRendering (feather={args.feather_px:.0f}px) ...")
    out = render_fisheye(counts, cam_store, u_store, v_store,
                         src_images, feather_px=args.feather_px)

    # ── 5. 径向平衡 ────────────────────────────────────────────
    if args.radial_balance:
        out = radial_balance(out)

    # ── 6. 左右翻转并保存 ──────────────────────────────────────
    out = cv2.flip(out, 1)
    cv2.imwrite(args.out, out)
    print(f"\nSaved: {args.out}  ({W}×{H})")


if __name__ == "__main__":
    main()
