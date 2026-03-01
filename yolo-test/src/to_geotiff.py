"""
Convert rectified plan-view images to GeoTIFF and Cloud Optimized GeoTIFF (COG).

The spatial reference is reconstructed from:
  - the camera position in UTM MGA94 (read from the reference-keypoints JSON)
  - the xlim / ylim / dx values used during rectification in align.py

The local coordinate frame is centred at the camera position, so:
    UTM_Easting  = local_X + camera_pos_utm[0]
    UTM_Northing = local_Y + camera_pos_utm[1]

Usage
-----
    python to_geotiff.py \\
        --input-dir  ../rectified \\
        --reference-keypoints ../reference/reference.json \\
        --xlim -200 300 \\
        --ylim -150 250 \\
        --dx 0.5 \\
        --epsg 28355 \\
        --output-dir ../geotiffs
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import rasterio.shutil


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_camera_pos_utm(keypoints_path: str) -> np.ndarray:
    """Read camera_pos from the reference keypoints JSON.

    Returns
    -------
    (3,) float64 array: [Easting, Northing, Elevation] in UTM/MGA94 metres.
    """
    with open(keypoints_path) as f:
        data = json.load(f)
    if "camera_pos" not in data:
        raise ValueError(
            f"'camera_pos' key not found in {keypoints_path}.\n"
            "Add the camera UTM position as:  \"camera_pos\": [X, Y, Z]"
        )
    return np.array(data["camera_pos"], dtype=np.float64)


def build_transform(
    xlim: tuple,
    ylim: tuple,
    dx: float,
    camera_pos_utm: np.ndarray,
):
    """Build a rasterio Affine transform for a plan-view image.

    xlim / ylim are in local (camera-centred) coordinates.
    camera_pos_utm is added back to recover absolute UTM coordinates.

    The plan-view image is north-up:
      - column 0  → UTM Easting  = xlim[0] + camera_pos_utm[0]
      - row 0     → UTM Northing = ylim[1] + camera_pos_utm[1]  (northernmost)
    """
    utm_west  = xlim[0] + camera_pos_utm[0]
    utm_north = ylim[1] + camera_pos_utm[1]
    return from_origin(utm_west, utm_north, dx, dx)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_geotiff(
    img_bgr: np.ndarray,
    output_path: str,
    transform,
    epsg: int,
) -> None:
    """Write a BGR image as a georeferenced GeoTIFF (RGB band order)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width, bands = img_rgb.shape

    profile = {
        "driver":    "GTiff",
        "dtype":     "uint8",
        "width":     width,
        "height":    height,
        "count":     bands,
        "crs":       f"EPSG:{epsg}",
        "transform": transform,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for i in range(bands):
            dst.write(img_rgb[:, :, i], i + 1)


def write_cog(
    img_bgr: np.ndarray,
    output_path: str,
    transform,
    epsg: int,
    overview_levels: list = None,
    blocksize: int = 512,
) -> None:
    """Write a BGR image as a Cloud Optimized GeoTIFF.

    Strategy
    --------
    1. Write a regular GeoTIFF to a temp file.
    2. Build overview pyramids (for efficient zoom-out rendering).
    3. Copy to the final path with tiled layout and deflate compression —
       the combination that satisfies the COG spec.
    """
    if overview_levels is None:
        overview_levels = [2, 4, 8, 16, 32]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width, bands = img_rgb.shape

    profile = {
        "driver":    "GTiff",
        "dtype":     "uint8",
        "width":     width,
        "height":    height,
        "count":     bands,
        "crs":       f"EPSG:{epsg}",
        "transform": transform,
    }

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(tmp_fd)

    try:
        # Step 1: write to temp
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for i in range(bands):
                dst.write(img_rgb[:, :, i], i + 1)

        # Step 2: build overviews in-place
        with rasterio.open(tmp_path, "r+") as dst:
            dst.build_overviews(overview_levels, Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

        # Step 3: copy as tiled COG
        cog_profile = profile.copy()
        cog_profile.update({
            "tiled":              True,
            "blockxsize":         blocksize,
            "blockysize":         blocksize,
            "compress":           "deflate",
            "copy_src_overviews": True,
        })
        with rasterio.open(tmp_path) as src:
            rasterio.shutil.copy(src, output_path, **cog_profile)

    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert rectified plan-view images to GeoTIFF and COG"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing rectified plan-view images.",
    )
    parser.add_argument(
        "--reference-keypoints", required=True,
        help="JSON file used in align.py (must contain 'camera_pos').",
    )
    parser.add_argument(
        "--xlim", type=float, nargs=2, required=True,
        metavar=("E_MIN", "E_MAX"),
        help="Easting extent in local (camera-centred) metres — "
             "the same value passed to --xlim in align.py.",
    )
    parser.add_argument(
        "--ylim", type=float, nargs=2, required=True,
        metavar=("N_MIN", "N_MAX"),
        help="Northing extent in local metres — "
             "the same value passed to --ylim in align.py.",
    )
    parser.add_argument(
        "--dx", type=float, required=True,
        help="Ground-sampling distance in metres/pixel — "
             "the same value passed to --dx in align.py.",
    )
    parser.add_argument(
        "--epsg", type=int, required=True,
        help="EPSG code for the MGA94 zone "
             "(e.g. 28354 zone 54, 28355 zone 55, 28356 zone 56).",
    )
    parser.add_argument(
        "--output-dir", default="./geotiffs",
        help="Output directory (default: ./geotiffs).",
    )
    parser.add_argument(
        "--no-cog", action="store_true",
        help="Write GeoTIFF only; skip COG output.",
    )
    parser.add_argument(
        "--blocksize", type=int, default=512,
        help="Tile block size in pixels for COG (default: 512).",
    )
    args = parser.parse_args()

    # ── Camera position ───────────────────────────────────────────────────
    camera_pos_utm = load_camera_pos_utm(args.reference_keypoints)
    print(
        f"Camera UTM     : E={camera_pos_utm[0]:.2f}  "
        f"N={camera_pos_utm[1]:.2f}  Z={camera_pos_utm[2]:.2f}  "
        f"EPSG:{args.epsg}"
    )

    utm_west  = args.xlim[0] + camera_pos_utm[0]
    utm_east  = args.xlim[1] + camera_pos_utm[0]
    utm_south = args.ylim[0] + camera_pos_utm[1]
    utm_north = args.ylim[1] + camera_pos_utm[1]
    print(
        f"UTM extent     : E=[{utm_west:.2f}, {utm_east:.2f}]  "
        f"N=[{utm_south:.2f}, {utm_north:.2f}]  dx={args.dx} m/px"
    )

    transform = build_transform(
        tuple(args.xlim), tuple(args.ylim), args.dx, camera_pos_utm
    )

    # ── Collect input images ──────────────────────────────────────────────
    input_dir = Path(args.input_dir)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.tif")
    image_paths = []
    for ext in exts:
        image_paths.extend(input_dir.glob(ext))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting {len(image_paths)} image(s) → {out_dir}\n")

    n_ok = 0
    for img_path in image_paths:
        stem = img_path.stem

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] {img_path.name}: cannot read image")
            continue

        # GeoTIFF
        geotiff_path = str(out_dir / f"{stem}.tif")
        write_geotiff(img, geotiff_path, transform, args.epsg)
        print(f"  {img_path.name} → {Path(geotiff_path).name}")

        # COG
        if not args.no_cog:
            cog_path = str(out_dir / f"{stem}_cog.tif")
            write_cog(img, cog_path, transform, args.epsg,
                      blocksize=args.blocksize)
            print(f"  {img_path.name} → {Path(cog_path).name}  [COG]")

        n_ok += 1

    print(f"\nDone.  {n_ok}/{len(image_paths)} images converted.")


if __name__ == "__main__":
    main()
