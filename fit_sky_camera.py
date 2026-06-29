"""
Fit sky-camera lens model + per-camera pose from star correspondences.

Input:
  1) observations.csv
     Columns:
       image,camera,star,x,y
     Example:
       cam1_2026-06-01_23-00.png,cam1,Sirius,1523.4,611.2
       cam1_2026-06-01_23-00.png,cam1,Betelgeuse,1140.7,802.9

  2) images (used only for optional visualization; not required for optimization)

  3) A site/time config:
       --lat
       --lon
       --height_m
       --obs_time  (UTC, ISO format, e.g. 2026-06-01T15:00:00)

How it works:
  - Uses astropy to compute Alt/Az of each star from RA/Dec at the observation time and site.
  - Fits a radial camera model:
        r(theta) = b1*theta + b3*theta^3 + b5*theta^5
    where theta is angular distance from the camera optical axis.
  - Fits one pose per camera:
        Az, Alt, Roll
  - Minimizes reprojection residuals with scipy.optimize.least_squares.

Notes:
  - This script assumes your manual labels are reasonably correct.
  - It is designed for your identical-camera multi-view sky setup.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from astropy.coordinates import AltAz, SkyCoord, EarthLocation
import astropy.units as u


# ----------------------------
# Star catalog: loaded from JSON file at runtime.
# RA/Dec are ICRS/J2000.
# ----------------------------
def load_star_catalog(path: str) -> Dict[str, Dict[str, float]]:
    """Load star catalog from a JSON file.

    Expected format:
        {"star_name": {"ra_deg": float, "dec_deg": float}, ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    return catalog


@dataclass
class ObsRow:
    camera: str
    star: str
    x: float
    y: float


def load_obs_csv(path: str) -> List[ObsRow]:
    df = pd.read_csv(path)
    required = {"camera", "star", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    rows = []
    for _, r in df.iterrows():
        rows.append(ObsRow(
            camera=str(r["camera"]),
            star=str(r["star"]),
            x=float(r["x"]),
            y=float(r["y"]),
        ))
    return rows


def star_altaz(star_name: str, obs_time_utc: str, lat: float, lon: float, height_m: float,
               star_catalog: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    if star_name not in star_catalog:
        raise KeyError(
            f"Star '{star_name}' not in catalog. "
            f"Add it to your star_catalog.json or use a more complete catalog."
        )
    star = star_catalog[star_name]
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height_m * u.m)
    sc = SkyCoord(ra=star["ra_deg"] * u.deg, dec=star["dec_deg"] * u.deg, frame="icrs")
    altaz = sc.transform_to(AltAz(obstime=obs_time_utc, location=loc, pressure=1013 * u.hPa, temperature=25 * u.deg_C))
    az_deg = float(altaz.az.deg)   # East of North
    alt_deg = float(altaz.alt.deg)
    return az_deg, alt_deg


def rot_yaw_pitch_roll(Az: float, Alt: float, Roll: float) -> np.ndarray:
    """
    World-to-camera rotation from yaw (Z), pitch (X), roll (Z)-like convention.
    This convention is arbitrary; keep it consistent in optimization and output.
    """
    cy, sy = math.cos(Az),  -math.sin(Az)
    cp, sp = math.sin(Alt), -math.cos(Alt)
    cr, sr = math.cos(Roll), math.sin(Roll)

    R1z = np.array([
        [ cy, -sy, 0.0],
        [ sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    R2x = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  cp, -sp],
        [0.0,  sp,  cp],
    ], dtype=np.float64)

    R3z = np.array([
        [ cr, -sr, 0.0],
        [ sr,  cr, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return R1z @ R2x @ R3z


def azalt_to_world_vec(az_deg: float, alt_deg: float) -> np.ndarray:
    """
    Local horizon frame:
      x = East
      y = North
      z = Up
    Astropy AltAz azimuth is East of North.  :contentReference[oaicite:2]{index=2}
    """
    az = math.radians(az_deg)
    alt = math.radians(alt_deg)
    x = math.cos(alt) * math.sin(az)
    y = math.cos(alt) * math.cos(az)
    z = math.sin(alt)
    v = np.array([x, y, z], dtype=np.float64)
    return v / np.linalg.norm(v)


def world_vec_to_theta(v_cam: np.ndarray) -> float:
    v_cam = v_cam / np.linalg.norm(v_cam)
    z = np.clip(v_cam[2], -1.0, 1.0)
    return math.acos(z)


def theta_to_radius(theta: float, b1: float, b3: float, b5: float) -> float:
    return b1 * theta + b3 * theta**3 + b5 * theta**5


def project_world_to_pixel(
    star_world: np.ndarray,
    params: np.ndarray,
    cam_index: int,
) -> Tuple[float, float]:
    """
    params layout:
      [cx, cy, b1, b3, b5,
       dcx_0, dcy_0, Az_0, Alt_0, Roll_0,
       dcx_1, dcy_1, Az_1, Alt_1, Roll_1,
       ...]
    """
    cx, cy, b1, b3, b5 = params[:5]
    base = 5 + cam_index * 5
    dcx, dcy, Az, Alt, Roll = params[base:base+5]

    R = rot_yaw_pitch_roll(Az, Alt, Roll)
    v_cam = np.transpose(R) @ star_world
    if v_cam[2] <= 1e-9:
        # Star behind the camera. Return a large penalty.
        return float("nan"), float("nan")

    theta = world_vec_to_theta(v_cam)
    r = theta_to_radius(theta, b1, b3, b5)

    # Camera image plane axes:
    # x axis to the right, y axis down.
    # azimuth in camera frame:
    x = v_cam[0]
    y = v_cam[1]
    norm_xy = math.hypot(x, y)
    if norm_xy < 1e-12:
        u = cx + dcx
        v = cy + dcy
    else:
        u = cx + dcx + r * (x / norm_xy)
        v = cy + dcy + r * (y / norm_xy)
    return u, v


def pack_initial_params(image_w: int, image_h: int) -> np.ndarray:
    # Rough guess from your 83° HFOV and 4mm lens.
    # Use weakly informative starting values.
    cx0 = image_w / 2.0
    cy0 = image_h / 2.0

    # For 2560x1440 and 83° HFOV, a rough focal length in pixels is around:
    # fx ≈ (W/2)/tan(HFOV/2)
    fx0 = (image_w / 2.0) / math.tan(math.radians(83.0 / 2.0))

    # Use a slightly larger value for radial scale than fx0; the polynomial will adapt.
    b1_0 = fx0
    b3_0 = 0.0
    b5_0 = 0.0

    p = [cx0, cy0, b1_0, b3_0, b5_0]

    # One pose per camera. Initial guess can be zero; you can improve this later manually.
    # Per camera: [dcx, dcy, Az, Alt, Roll]
    p.extend(
        [0.0, 0.0, math.radians(0),   math.radians(30), 0.0,
         0.0, 0.0, math.radians(90),  math.radians(30), 0.0,
         0.0, 0.0, math.radians(-90), math.radians(30), 0.0,
         0.0, 0.0, math.radians(80),  math.radians(75), 0.0,
         0.0, 0.0, math.radians(-90), math.radians(70), 0.0,
         0.0, 0.0, math.radians(180), math.radians(30), 0.0]
    )

    return np.array(p, dtype=np.float64)


def build_residuals(
    params: np.ndarray,
    obs_rows: List[ObsRow],
    star_dirs: Dict[str, np.ndarray],
    camera_to_index: Dict[str, int],
    robust_pixel_sigma: float = 3.0,
) -> np.ndarray:
    residuals = []
    for row in obs_rows:
        if row.star not in star_dirs:
            continue
        if row.camera not in camera_to_index:
            continue

        cam_idx = camera_to_index[row.camera]
        star_world = star_dirs[row.star]
        u_pred, v_pred = project_world_to_pixel(star_world, params, cam_idx)
        if not np.isfinite(u_pred) or not np.isfinite(v_pred):
            # Penalize stars behind camera strongly.
            residuals.extend([1000.0, 1000.0])
            continue

        du = (u_pred - row.x) / robust_pixel_sigma
        dv = (v_pred - row.y) / robust_pixel_sigma
        residuals.extend([du, dv])
    return np.array(residuals, dtype=np.float64)


def fit_model(
    obs_rows: List[ObsRow],
    obs_time_utc: str,
    lat: float,
    lon: float,
    height_m: float,
    image_w: int,
    image_h: int,
    catalog_path: str,
    max_nfev: int = 500,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, np.ndarray]]:
    star_catalog = load_star_catalog(catalog_path)
    cameras = sorted(set(r.camera for r in obs_rows))
    camera_to_index = {cam: i for i, cam in enumerate(cameras)}

    stars = sorted(set(r.star for r in obs_rows))
    star_dirs: Dict[str, np.ndarray] = {}
    for s in stars:
        az_deg, alt_deg = star_altaz(s, obs_time_utc, lat, lon, height_m, star_catalog)
        star_dirs[s] = azalt_to_world_vec(az_deg, alt_deg)

    x0 = pack_initial_params(image_w, image_h)

    # Loose bounds:
    # cx, cy near image center but allowed to move.
    # b1 positive, polynomial coefficients modest.
    lower = [ 0.45 * image_w, 0.45 * image_h, 10.0, -1e6, -1e9]
    upper = [ 0.55 * image_w, 0.55 * image_h,  1e5,  1e6,  1e9]

    # Add pose bounds for each camera:
    # dcx, dcy, Az, Alt, Roll in radians.
    for _ in cameras:
        lower.extend([-50.0, -50.0, -math.pi,          0, -math.pi/6])
        upper.extend([ 50.0,  50.0,  math.pi,  math.pi/2,  math.pi/6])

    lower = np.array(lower, dtype=np.float64)
    upper = np.array(upper, dtype=np.float64)

    def fun(p: np.ndarray) -> np.ndarray:
        return build_residuals(p, obs_rows, star_dirs, camera_to_index)

    result = least_squares(
        fun,
        x0=x0,
        bounds=(lower, upper),
        loss="linear",   # robust to bad star labels / detection errors
        f_scale=1.0,
        max_nfev=max_nfev,
        verbose=2,
    )

    if not result.success:
        print("Optimization finished with status:", result.status)
        print("Message:", result.message)

    return result.x, camera_to_index, star_dirs


def write_calibration_json(
    path: str,
    params: np.ndarray,
    camera_to_index: Dict[str, int],
) -> None:
    cx, cy, b1, b3, b5 = params[:5]
    cameras = sorted(camera_to_index, key=lambda k: camera_to_index[k])

    data = {
        "model": "theta_to_radius_polynomial",
        "principal_point": {"cx": float(cx), "cy": float(cy)},
        "radial_model": {"b1": float(b1), "b3": float(b3), "b5": float(b5)},
        "cameras": {}
    }

    for cam in cameras:
        idx = camera_to_index[cam]
        base = 5 + idx * 5
        dcx, dcy, Az, Alt, Roll = params[base:base+5]
        data["cameras"][cam] = {
            "principal_point_offset_px": {"dcx": float(dcx), "dcy": float(dcy)},
            "Az_rad": float(Az),
            "Alt_rad": float(Alt),
            "Roll_rad": float(Roll),
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# def undistort_points_from_model(
#     pts_xy: np.ndarray,
#     params: np.ndarray,
# ) -> np.ndarray:
#     """
#     Convert pixel points to normalized ray directions in camera coordinates.
#     Returns Nx3 rays.
#     """
#     cx, cy, b1, b3, b5 = params[:5]

#     rays = []
#     for (u, v) in pts_xy:
#         dx = float(u) - cx
#         dy = float(v) - cy
#         r = math.hypot(dx, dy)
#         if r < 1e-12:
#             rays.append(np.array([0.0, 0.0, 1.0], dtype=np.float64))
#             continue

#         # Invert r(theta) numerically using a small Newton loop.
#         theta = min(max(r / max(b1, 1e-9), 0.0), math.radians(89.0))
#         for _ in range(20):
#             f = theta_to_radius(theta, b1, b3, b5) - r
#             df = b1 + 3.0 * b3 * theta**2 + 5.0 * b5 * theta**4
#             if abs(df) < 1e-12:
#                 break
#             step = f / df
#             theta -= step
#             theta = min(max(theta, 0.0), math.radians(89.9))
#             if abs(step) < 1e-10:
#                 break

#         az = math.atan2(dy, dx)
#         # local camera ray: z forward
#         x = math.sin(theta) * math.cos(az)
#         y = math.sin(theta) * math.sin(az)
#         z = math.cos(theta)
#         rays.append(np.array([x, y, z], dtype=np.float64))
#     return np.vstack(rays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="observations.csv")
    ap.add_argument("--catalog", default="star_catalog.json")
    ap.add_argument("--obs_time", default="2026-06-09T16:00:00")
    ap.add_argument("--lat", type=float, default=31.3, help="observer latitude in degrees")
    ap.add_argument("--lon", type=float, default=121.5, help="observer longitude in degrees")
    ap.add_argument("--height_m", type=float, default=33.0, help="observer height in meters")
    ap.add_argument("--image_w", type=int, default=2560)
    ap.add_argument("--image_h", type=int, default=1440)
    ap.add_argument("--max_nfev", type=int, default=500)
    ap.add_argument("--out", default="calibration_result.json")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    obs_rows = load_obs_csv(args.obs)
    if not obs_rows:
        raise RuntimeError("No observations found.")

    params, camera_to_index, star_dirs = fit_model(
        obs_rows=obs_rows,
        obs_time_utc=args.obs_time,
        lat=args.lat,
        lon=args.lon,
        height_m=args.height_m,
        image_w=args.image_w,
        image_h=args.image_h,
        catalog_path=args.catalog,
        max_nfev=args.max_nfev,
    )

    write_calibration_json(args.out, params, camera_to_index)
    print(f"\nSaved calibration to: {args.out}")

    # Report fitted parameters.
    cx, cy, b1, b3, b5 = params[:5]
    print("\nGlobal model:")
    print(f"  cx = {cx:.3f}")
    print(f"  cy = {cy:.3f}")
    print(f"  b1 = {b1:.6f}")
    print(f"  b3 = {b3:.6e}")
    print(f"  b5 = {b5:.6e}")

    print("\nPer-camera poses (radians):")
    cameras = sorted(camera_to_index, key=lambda k: camera_to_index[k])
    for cam in cameras:
        idx = camera_to_index[cam]
        base = 5 + idx * 5
        dcx, dcy, Az, Alt, Roll = params[base:base+5]
        print(f"  {cam}: dcx={dcx:.3f}, dcy={dcy:.3f}, Az={Az:.6f}, Alt={Alt:.6f}, Roll={Roll:.6f}")

    # Compute RMS reprojection error in pixels.
    residuals = build_residuals(params, obs_rows, star_dirs, camera_to_index, robust_pixel_sigma=1.0)
    if residuals.size:
        # residuals are in pixels because sigma=1.0 here
        rms = math.sqrt(np.mean(residuals**2))
        print(f"\nRMS reprojection error: {rms:.3f} px")

    if args.debug:
        # Show per-row residuals for debugging.
        print("\nPer-observation residuals:")
        for row in obs_rows:
            cam_idx = camera_to_index[row.camera]
            star_world = star_dirs[row.star]
            u_pred, v_pred = project_world_to_pixel(star_world, params, cam_idx)
            du = u_pred - row.x
            dv = v_pred - row.y
            err = math.hypot(du, dv)
            print(f"{row.camera:5s} {row.star:12s} "
                  f"obs=({row.x:8.2f},{row.y:8.2f}) pred=({u_pred:8.2f},{v_pred:8.2f}) "
                  f"err={err:7.3f}px")


if __name__ == "__main__":
    main()