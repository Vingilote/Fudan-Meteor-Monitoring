"""
pixel_to_radec.py — 像素坐标 → 赤道坐标 (RA/Dec) 反向推算

输入: 像素坐标 (u, v) + 相机编号
输出: 赤经 RA (deg)、赤纬 Dec (deg)

推算链路:
  像素 (u,v)
    → 径向距离 r_px (相对主点)
    → 牛顿法反演 θ (r = b1·θ + b3·θ³ + b5·θ⁵)
    → 相机系方向 v_cam = [sinθ·cosφ, sinθ·sinφ, cosθ]
    → 世界系方向 v_world = R @ v_cam
    → 地平系 Alt/Az
    → 赤道系 RA/Dec (astropy)

Usage:
  python pixel_to_radec.py --calib calibration_result.json \
      --obs_time "2026-06-09T16:00:00" --lat 31.3 --lon 121.5 --height 25 \
      --cam 1 --u 1280 --v 720

  # 或从 CSV 批量转换
  python pixel_to_radec.py --calib calibration_result.json \
      --csv pixels.csv --obs_time "2026-06-09T16:00:00"
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Tuple

import numpy as np

from astropy.coordinates import AltAz, SkyCoord, EarthLocation
import astropy.units as u

from fit_sky_camera import azalt_to_world_vec, project_world_to_pixel, rot_yaw_pitch_roll, star_altaz, theta_to_radius


def _ra_hms(ra_deg: float) -> str:
    """赤经 (度) → 'Xh Xm Xs' 字符串"""
    total_h = ra_deg / 15.0
    h = int(total_h)
    remainder_m = (total_h - h) * 60.0
    m = int(remainder_m)
    s = (remainder_m - m) * 60.0
    return f"{h}h {m:02d}m {s:.1f}s"

def invert_radial(r_px: float, b1: float, b3: float, b5: float,
                  max_iter: int = 30, tol: float = 1e-10) -> float:
    """
    牛顿法反演 r(θ) → θ。
    f(θ) = b1·θ + b3·θ³ + b5·θ⁵ - r_px
    f'(θ) = b1 + 3·b3·θ² + 5·b5·θ⁴
    """
    # 初始猜测
    if b1 > 1e-9:
        theta = r_px / b1
    else:
        theta = 0.5
    theta = min(max(theta, 0.0), math.radians(89.9))

    for _ in range(max_iter):
        t2 = theta * theta
        t3 = t2 * theta
        t4 = t2 * t2

        f = b1 * theta + b3 * t3 + b5 * t4 * theta - r_px
        df = b1 + 3.0 * b3 * t2 + 5.0 * b5 * t4

        if abs(df) < 1e-15:
            break

        step = f / df
        theta -= step
        theta = min(max(theta, 0.0), math.radians(89.9))

        if abs(step) < tol:
            break

    return theta


def pixel_to_altaz(
    u: float, v: float,
    cam_id: str,
    calib: dict,
) -> Tuple[float, float]:
    """
    像素坐标 (u, v) → 地平坐标 (az_deg, alt_deg)

    Parameters
    ----------
    u, v   : 像素坐标 (x=列, y=行), 左上角为原点
    cam_id : 相机编号 (str, 如 "1", "2"...)
    calib  : calibration_result.json

    Returns
    -------
    az_deg  : 方位角 (度), East of North
    alt_deg : 高度角 (度)
    """
    # 读取全局模型
    cx = calib["principal_point"]["cx"]
    cy = calib["principal_point"]["cy"]
    b1 = calib["radial_model"]["b1"]
    b3 = calib["radial_model"]["b3"]
    b5 = calib["radial_model"]["b5"]

    # 读取该相机的参数
    if cam_id not in calib["cameras"]:
        raise KeyError(f"相机 '{cam_id}' 不在标定文件中 (可用: {list(calib['cameras'].keys())})")

    cam = calib["cameras"][cam_id]
    dcx = cam["principal_point_offset_px"]["dcx"]
    dcy = cam["principal_point_offset_px"]["dcy"]
    Az = cam["Az_rad"]
    Alt = cam["Alt_rad"]
    Roll = cam["Roll_rad"]

    # ── 1. 像素 → 径向距离 ──────────────────────────────────
    dx = u - (cx + dcx)
    dy = v - (cy + dcy)
    r_px = math.hypot(dx, dy)

    # ── 2. 反演 θ ──────────────────────────────────────────
    theta = invert_radial(r_px, b1, b3, b5)

    # ── 3. 相机系方向 v_cam ────────────────────────────────
    # 相机系: z=光轴向前, x=右, y=下
    # 像素 (u,v) 中: dx = x方向, dy = y方向(下)
    # 对应 v_cam = [sinθ·cosφ, sinθ·sinφ, cosθ], φ = atan2(dy, dx)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    if r_px < 1e-12:
        # 光学中心: 方向 = 光轴 = +z
        v_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        phi = math.atan2(dy, dx)  # y↓ 在图像中向下
        v_cam = np.array([
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta,
        ], dtype=np.float64)

    # ── 4. 世界系方向 v_world = R @ v_cam ──────────────────
    R = rot_yaw_pitch_roll(Az, Alt, Roll)
    v_world = R @ v_cam

    # ── 5. 世界系 → 地平 Az/Alt ────────────────────────────
    # 世界系: x=East, y=North, z=Up
    az_rad = math.atan2(v_world[0], v_world[1])   # East of North
    alt_rad = math.asin(np.clip(v_world[2], -1.0, 1.0))

    az_deg = math.degrees(az_rad)
    alt_deg = math.degrees(alt_rad)

    return az_deg % 360.0, alt_deg


def altaz_to_radec(
    az_deg: float, alt_deg: float,
    obs_time_utc: str,
    lat: float, lon: float, height_m: float,
) -> Tuple[float, float]:
    """
    地平坐标 (Az, Alt) → 赤道坐标 (RA_deg, Dec_deg)
    """
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height_m * u.m)
    altaz_frame = AltAz(obstime=obs_time_utc, location=loc, pressure=1013 * u.hPa, temperature=25 * u.deg_C)
    sky = SkyCoord(az=az_deg * u.deg, alt=alt_deg * u.deg, frame=altaz_frame)
    icrs = sky.transform_to("icrs")
    assert icrs is not None

    return float(icrs.ra.deg), float(icrs.dec.deg)


def radec_to_altaz_geometric(
    ra_deg: float, dec_deg: float,
    obs_time_utc: str,
    lat: float, lon: float, height_m: float,
) -> Tuple[float, float]:
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height_m * u.m)
    # pressure=0 禁用大气折射
    altaz_frame = AltAz(obstime=obs_time_utc, location=loc, pressure=0 * u.hPa)
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    altaz = sky.transform_to(altaz_frame)
    assert altaz is not None

    return float(altaz.az.deg), float(altaz.alt.deg)


def pixel_to_radec(
    u: float, v: float,
    cam_id: str,
    calib: dict,
    obs_time_utc: str,
    lat: float, lon: float, height_m: float,
) -> dict[str, float]:
    """
    完整管线: 像素 → RA/Dec + 无折射几何 AltAz

    Returns
    -------
    dict with keys:
      ra_deg, dec_deg        — ICRS 赤道坐标
      az_apparent, alt_apparent — 视位置 (含折射) 地平坐标
      az_geometric, alt_geometric — 几何 (无折射) 地平坐标
    """
    az_app, alt_app = pixel_to_altaz(u, v, cam_id, calib)
    ra_deg, dec_deg = altaz_to_radec(az_app, alt_app, obs_time_utc, lat, lon, height_m)
    az_geo, alt_geo = radec_to_altaz_geometric(ra_deg, dec_deg, obs_time_utc, lat, lon, height_m)

    return {
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "az_apparent": az_app,
        "alt_apparent": alt_app,
        "az_geometric": az_geo,
        "alt_geometric": alt_geo,
    }


def batch_convert_csv(
    csv_path: str,
    calib: dict,
    obs_time_utc: str,
    lat: float, lon: float, height_m: float,
    out_path: str | None = None,
) -> None:
    """从 CSV (camera, x, y) 批量转换为 (ra_deg, dec_deg)。"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"camera", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {sorted(required)}")

    rows_out: list[dict] = []
    for _, row in df.iterrows():
        cam = row["camera"]
        u = float(row["x"])
        v = float(row["y"])
        try:
            astro = pixel_to_radec(u, v, cam, calib, obs_time_utc, lat, lon, height_m)
            entry = {"camera": cam, "x": u, "y": v, **astro}
        except Exception as e:
            print(f"  跳过 camera={cam} (x={u}, y={v}): {e}")
            entry = {"camera": cam, "x": u, "y": v,
                     "ra_deg": float("nan"), "dec_deg": float("nan"),
                     "az_apparent": float("nan"), "alt_apparent": float("nan"),
                     "az_geometric": float("nan"), "alt_geometric": float("nan")}
        rows_out.append(entry)

    out_df = pd.DataFrame(rows_out)
    cols = ["camera", "x", "y",
            "ra_deg", "dec_deg",
            "az_apparent", "alt_apparent",
            "az_geometric", "alt_geometric"]
    out_df = out_df[cols]

    out = out_path or csv_path.replace(".csv", "_radec.csv")
    out_df.to_csv(out, index=False)
    print(f"\n批量转换完成: {len(out_df)} 行 → {out}")


def self_check(args, calib: dict, cam_id: str, u: float, v: float) -> None:
    """正向投影再反算, 验证往返误差。"""
    res = pixel_to_radec(u, v, cam_id, calib, args.obs_time,
                         args.lat, args.lon, args.height_m)

    # 用反算的 RA/Dec 正向投影
    catalog = {"_test_": {"ra_deg": res["ra_deg"], "dec_deg": res["dec_deg"]}}
    az2, alt2 = star_altaz("_test_", args.obs_time, args.lat, args.lon,
                            args.height_m, catalog)
    v_world = azalt_to_world_vec(az2, alt2)

    cx = calib["principal_point"]["cx"]
    cy = calib["principal_point"]["cy"]
    b1 = calib["radial_model"]["b1"]
    b3 = calib["radial_model"]["b3"]
    b5 = calib["radial_model"]["b5"]
    cam = calib["cameras"][cam_id]
    dcx = cam["principal_point_offset_px"]["dcx"]
    dcy = cam["principal_point_offset_px"]["dcy"]
    Az = cam["Az_rad"]
    Alt = cam["Alt_rad"]
    Roll = cam["Roll_rad"]

    params = np.array([cx, cy, b1, b3, b5, dcx, dcy, Az, Alt, Roll], dtype=np.float64)
    u2, v2 = project_world_to_pixel(v_world, params, 0)

    dr = math.hypot(u2 - u, v2 - v)
    print(f"\n  往返自检: 输入=({u:.1f}, {v:.1f}) → 输出=({u2:.2f}, {v2:.2f}) → 误差={dr:.4f} px")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calibration_result_6-1.json",
                    help="标定 JSON 文件路径")
    ap.add_argument("--obs_time", default="2026-05-31T16:00:00",
                    help="观测时间 (UTC ISO 格式)")
    ap.add_argument("--lat", type=float, default=31.3,
                    help="观测点纬度 (deg)")
    ap.add_argument("--lon", type=float, default=121.5,
                    help="观测点经度 (deg)")
    ap.add_argument("--height_m", type=float, default=33.0,
                    help="观测点海拔 (m)")

    # 单像素模式
    ap.add_argument("--cam", type=str, default=None,
                    help="相机编号 (单像素模式)")
    ap.add_argument("--u", type=float, default=None,
                    help="像素 x 坐标")
    ap.add_argument("--v", type=float, default=None,
                    help="像素 y 坐标")

    # 批量模式
    ap.add_argument("--csv", default=None,
                    help="输入 CSV (批量模式), 需含 camera,x,y 列")
    ap.add_argument("--out", default=None,
                    help="输出 CSV 路径 (默认: 输入文件名_radec.csv)")

    # 自检
    ap.add_argument("--check", action="store_true",
                    help="运行往返自检 (需 fit_sky_camera.py 可导入)")

    args = ap.parse_args()

    # 加载标定
    with open(args.calib, "r", encoding="utf-8") as f:
        calib = json.load(f)

    print(f"标定文件: {args.calib}")
    print(f"观测时间: {args.obs_time}")
    print(f"观测点:   lat={args.lat}°  lon={args.lon}°  height={args.height_m}m")

    # ── 批量模式 ──────────────────────────────────────────
    if args.csv:
        batch_convert_csv(args.csv, calib, args.obs_time,
                          args.lat, args.lon, args.height_m, args.out)
        return

    # ── 单像素模式 ────────────────────────────────────────
    if args.cam is None or args.u is None or args.v is None:
        ap.error("单像素模式需要 --cam --u --v (或指定 --csv 批量模式)")

    res = pixel_to_radec(args.u, args.v, args.cam, calib,
                         args.obs_time, args.lat, args.lon, args.height_m)

    print(f"\n相机 {args.cam}  像素 ({args.u:.2f}, {args.v:.2f}):")
    print(f"  RA  = {res['ra_deg']:.6f}°   ({_ra_hms(res['ra_deg'])})")
    print(f"  Dec = {res['dec_deg']:.6f}°")
    print(f"  ──────────────────────────────────")
    print(f"  视位置 (含折射):")
    print(f"    Az  = {res['az_apparent']:.4f}°")
    print(f"    Alt = {res['alt_apparent']:.4f}°")
    print(f"  几何位置 (无折射):")
    print(f"    Az  = {res['az_geometric']:.4f}°")
    print(f"    Alt = {res['alt_geometric']:.4f}°")
    refr = res['alt_apparent'] - res['alt_geometric']
    print(f"  ──────────────────────────────────")
    print(f"  折射量 = {refr:.4f}°  ({refr*60:.2f} arcmin)")

    # 自检
    if args.check:
        self_check(args, calib, args.cam, args.u, args.v)


if __name__ == "__main__":
    main()
