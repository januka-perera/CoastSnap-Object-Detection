"""Build annotations.json from .mat GCP files and matched images.

For each .mat file in data/gcp/, extracts the UVpicked ground control points,
scales them from annotation resolution (NU x NV) to actual image resolution,
and writes the result to data/annotations.json.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import scipy.io

GCP_DIR = Path("data/gcp")
IMAGES_DIR = Path("data/images")
OUTPUT_FILE = Path("data/annotations.json")


def mat_to_image_filename(mat_name: str) -> str:
    """Convert a .mat filename to the corresponding .jpg image filename.

    e.g. '...manly.plan.CherylWhite.mat' -> '...manly.snap.CherylWhite.jpg'
    """
    return mat_name.replace(".plan.", ".snap.").replace(".mat", ".jpg")


def main():
    mat_files = sorted(GCP_DIR.glob("*.mat"))
    print(f"Found {len(mat_files)} .mat files in {GCP_DIR}")

    images = []
    skipped_no_image = 0
    skipped_error = 0
    num_landmarks_set = set()

    for mat_path in mat_files:
        img_filename = mat_to_image_filename(mat_path.name)
        img_path = IMAGES_DIR / img_filename

        if not img_path.exists():
            skipped_no_image += 1
            continue

        try:
            # Load .mat file
            data = scipy.io.loadmat(str(mat_path))
            uv = data["metadata"]["gcps"][0, 0]["UVpicked"][0, 0]
            lcp = data["metadata"]["geom"][0, 0]["lcp"][0, 0]
            nu = lcp["NU"][0, 0].item()
            nv = lcp["NV"][0, 0].item()

            # Get actual image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                skipped_error += 1
                print(f"  WARNING: Could not read image {img_path}")
                continue
            img_h, img_w = img.shape[:2]

            # Scale UV from annotation resolution to image resolution
            scale_u = img_w / nu
            scale_v = img_h / nv

            num_points = uv.shape[0]
            num_landmarks_set.add(num_points)

            landmarks = []
            for i in range(num_points):
                landmarks.append({
                    "id": i,
                    "u": round(float(uv[i, 0] * scale_u), 2),
                    "v": round(float(uv[i, 1] * scale_v), 2),
                })

            images.append({
                "filename": img_filename,
                "width": img_w,
                "height": img_h,
                "landmarks": landmarks,
            })

        except Exception as e:
            skipped_error += 1
            print(f"  ERROR processing {mat_path.name}: {e}")

    # Sort by filename for reproducibility
    images.sort(key=lambda x: x["filename"])

    annotations = {"images": images}

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nDone!")
    print(f"  Matched image+mat pairs: {len(images)}")
    print(f"  Skipped (no matching image): {skipped_no_image}")
    print(f"  Skipped (errors): {skipped_error}")
    print(f"  Unique landmark counts: {num_landmarks_set}")
    print(f"  Written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
