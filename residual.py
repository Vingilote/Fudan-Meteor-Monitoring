import fit_sky_camera as fsc
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_residual_vectors(
    params,
    obs_rows,
    star_dirs,
    camera_to_index,
    image_w=2560,
    image_h=1440,
):
    xs = []
    ys = []

    dus = []
    dvs = []

    for row in obs_rows:

        star_world = star_dirs[row.star]

        cam_idx = camera_to_index[row.camera]

        u_pred, v_pred = fsc.project_world_to_pixel(
            star_world,
            params,
            cam_idx,
        )

        du = u_pred - row.x
        dv = v_pred - row.y

        xs.append(row.x)
        ys.append(row.y)

        dus.append(du)
        dvs.append(dv)

    plt.figure(figsize=(12,7))

    plt.quiver(
        xs,
        ys,
        dus,
        dvs,
        angles='xy',
        scale_units='xy',
        scale=1,
    )

    plt.xlim(0,image_w)
    plt.ylim(image_h,0)

    plt.title("Residual vectors")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid(True)

    plt.show()

def plot_residual_magnitude(
    params,
    obs_rows,
    star_dirs,
    camera_to_index,
):
    xs=[]
    ys=[]
    errs=[]

    for row in obs_rows:

        star_world = star_dirs[row.star]

        cam_idx = camera_to_index[row.camera]

        u_pred,v_pred = fsc.project_world_to_pixel(
            star_world,
            params,
            cam_idx
        )

        err=np.hypot(
            u_pred-row.x,
            v_pred-row.y
        )

        xs.append(row.x)
        ys.append(row.y)
        errs.append(err)

    plt.figure(figsize=(12,7))

    sc=plt.scatter(
        xs,
        ys,
        c=errs,
        s=40
    )

    plt.colorbar(sc,label="error(px)")

    plt.xlim(0,2560)
    plt.ylim(1440,0)

    plt.title("Residual magnitude")

    plt.show()

def plot_theta_error(
    params,
    obs_rows,
    star_dirs,
    camera_to_index,
):
    thetas=[]
    errs=[]

    for row in obs_rows:

        cam_idx=camera_to_index[row.camera]

        base=5+cam_idx*3

        yaw,pitch,roll=params[
            base:base+3
        ]

        R=fsc.rot_yaw_pitch_roll(
            yaw,
            pitch,
            roll
        )

        star_world=star_dirs[row.star]

        v_cam=R @ star_world

        theta=np.arccos(
            np.clip(v_cam[2],-1,1)
        )

        u_pred,v_pred=fsc.project_world_to_pixel(
            star_world,
            params,
            cam_idx
        )

        err=np.hypot(
            u_pred-row.x,
            v_pred-row.y
        )

        thetas.append(
            np.degrees(theta)
        )

        errs.append(err)

    plt.figure(figsize=(8,5))

    plt.scatter(
        thetas,
        errs,
        s=15
    )

    plt.xlabel("theta (deg)")
    plt.ylabel("error (px)")

    plt.grid(True)

    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="observations.csv", help="observations.csv")
    ap.add_argument("--catalog", default="star_catalog.json")
    ap.add_argument("--obs_time", default="2026-06-09T16:00:00", help="UTC ISO time, e.g. 2026-06-01T15:00:00")
    ap.add_argument("--lat", type=float, default=31.3, help="observer latitude in degrees")
    ap.add_argument("--lon", type=float, default=121.5, help="observer longitude in degrees")
    ap.add_argument("--height_m", type=float, default=25.0, help="observer height in meters")
    ap.add_argument("--image_w", type=int, default=2560)
    ap.add_argument("--image_h", type=int, default=1440)
    ap.add_argument("--max_nfev", type=int, default=2000)
    ap.add_argument("--out", default="calibration_result.json")
    ap.add_argument("--show_error", action="store_true")
    args = ap.parse_args()
    obs_rows = fsc.load_obs_csv(args.obs)
    params, camera_to_index, star_dirs = fsc.fit_model(        
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

    plot_residual_vectors(
        params,
        obs_rows,
        star_dirs,
        camera_to_index,
    )

    plot_residual_magnitude(
        params,
        obs_rows,
        star_dirs,
        camera_to_index,
    )

    plot_theta_error(
        params,
        obs_rows,
        star_dirs,
        camera_to_index,
    )

main()