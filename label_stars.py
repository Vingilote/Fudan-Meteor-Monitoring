import cv2
import numpy as np
import pandas as pd
import os
import re
import argparse

WINDOW = "Star Label"

points = []
img = None
display = None

ZOOM = 2


def refine_star_center(gray, x, y, win=15):
    h, w = gray.shape

    x0 = max(0, x - win)
    y0 = max(0, y - win)
    x1 = min(w, x + win + 1)
    y1 = min(h, y + win + 1)

    roi = gray[y0:y1, x0:x1]

    if roi.size == 0:
        return float(x), float(y)

    _, thresh = cv2.threshold(
        roi,
        np.percentile(roi, 99),
        255,
        cv2.THRESH_BINARY
    )

    ys, xs = np.where(thresh > 0)

    if len(xs) < 3:
        return float(x), float(y)

    weights = roi[ys, xs].astype(np.float64)

    cx = np.sum(xs * weights) / np.sum(weights)
    cy = np.sum(ys * weights) / np.sum(weights)

    return (
        x0 + cx,
        y0 + cy
    )


def redraw():
    global display

    display = img.copy()

    for p in points:
        x, y, star = p

        cv2.circle(
            display,
            (int(round(x)), int(round(y))),
            8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            star,
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


def mouse(event, x, y, flags, param):

    global points

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    x = int(x / ZOOM)
    y = int(y / ZOOM)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cx, cy = refine_star_center(gray, x, y)

    print()
    star = input("Star name: ").strip()

    if not star:
        return

    points.append((cx, cy, star))

    print(
        f"Added {star}: "
        f"x={cx:.2f} y={cy:.2f}"
    )

    redraw()


def save_csv(camera_name, csv_file):

    rows = []

    for x, y, star in points:
        rows.append({
            "camera": camera_name,
            "star": star,
            "x": x,
            "y": y
        })

    df = pd.DataFrame(rows)

    if os.path.exists(csv_file):
        df.to_csv(
            csv_file,
            mode="a",
            header=False,
            index=False
        )
    else:
        df.to_csv(
            csv_file,
            index=False
        )

    print()
    print(f"Saved {len(df)} points")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img",
        default=None,
        help="Path to image (omit for auto-discovery by channel)"
    )

    parser.add_argument(
        "--cam",
        type=int,
        required=True,
        help="Camera channel number (1-6)"
    )

    parser.add_argument(
        "--csv",
        default="observations_lite.csv"
    )

    args = parser.parse_args()

    ch = args.cam
    if ch < 1 or ch > 6:
        raise RuntimeError("--cam must be between 1 and 6")

    camera_name = ch

    # ── 确定图片路径 ────────────────────────────────────────
    if args.img:
        img_path = args.img
        print(f"\nUsing specified image: {img_path}")
    else:
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        pattern = re.compile(r"通道" + str(ch) + r"(?:[^\d]|$)")
        found = None
        for fname in os.listdir("."):
            m = pattern.search(fname)
            if not m:
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in img_exts:
                continue
            found = fname
            break

        if found is None:
            raise RuntimeError(
                f"未在当前目录找到通道 {ch} 的图片 "
                f"(文件名应包含“通道{ch}”)"
            )
        img_path = found
        print(f"\nAuto-detected: {img_path}  →  {camera_name}")

    global img

    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError(
            f"Cannot open {img_path}"
        )

    redraw()

    cv2.namedWindow(
        WINDOW,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(WINDOW, 1600, 900)

    cv2.setMouseCallback(
        WINDOW,
        mouse
    )

    print()
    print("Left click : add star")
    print("u          : undo")
    print("s          : save")
    print("q          : quit")
    print()

    while True:

        show = cv2.resize(
            display,
            None,
            fx=ZOOM,
            fy=ZOOM
        )

        cv2.imshow(
            WINDOW,
            show
        )

        key = cv2.waitKey(30)

        if key == ord('u'):

            if points:
                removed = points.pop()
                print(
                    "Removed:",
                    removed[2]
                )
                redraw()

        elif key == ord('s'):

            save_csv(
                camera_name,
                args.csv
            )

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()