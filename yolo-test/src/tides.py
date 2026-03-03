"""
Tide height prediction for CoastSnap images.

Given a CoastSnap image filename (first field is a Unix timestamp) and the
site reference keypoints JSON (containing "camera_pos" in UTM/MGA94 metres),
returns the modelled tide height in metres at the image capture time.

Usage (module)
--------------
    from tides import get_tide_height
    height = get_tide_height("1577820840.Wed.Jan.01....jpg",
                             keypoints_path="../reference/reference.json",
                             utm_zone=56)

Usage (CLI)
-----------
    python tides.py 1577820840.Wed.Jan.01....jpg \\
        --keypoints ../reference/reference.json \\
        --utm-zone 56
"""

import json
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import utm as utm_lib
from eo_tides.model import model_tides
from eo_tides.utils import clip_models


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIDE_MODELS_DIR         = Path(r"C:\Users\z3551540\Code\Github\CoastSnap-Object-Detection\yolo-test\tide_models")
TIDE_MODELS_CLIPPED_DIR = Path(r"C:\Users\z3551540\Code\Github\CoastSnap-Object-Detection\yolo-test\tide_models_clipped")
CLIP_BBOX  = (144.7, -38.9, 152.7, -30.3)   # (lon_min, lat_min, lon_max, lat_max)
TIDE_MODEL = "FES2022_load"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_clipped_models() -> None:
    """Clip the global tide model to the regional bbox once, then cache."""
    if not TIDE_MODELS_CLIPPED_DIR.exists():
        print("Clipping tide models to regional bbox — this runs once only...")
        clip_models(
            input_directory=TIDE_MODELS_DIR,
            output_directory=TIDE_MODELS_CLIPPED_DIR,
            bbox=CLIP_BBOX,
        )


def _load_camera_pos_utm(keypoints_path: str) -> np.ndarray:
    """Return (3,) float64 [Easting, Northing, Elevation] from the JSON."""
    with open(keypoints_path) as f:
        data = json.load(f)
    if "camera_pos" not in data:
        raise ValueError(
            f"'camera_pos' key not found in {keypoints_path}.\n"
            "Add the camera UTM position as:  \"camera_pos\": [X, Y, Z]"
        )
    return np.array(data["camera_pos"], dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tide_height(
    image_name: str,
    keypoints_path: str,
    utm_zone: int,
    northern: bool = False,
    model: str = TIDE_MODEL,
) -> float:
    """Return the modelled tide height in metres for a CoastSnap image.

    Parameters
    ----------
    image_name     : CoastSnap filename, e.g.
                     "1577820840.Wed.Jan.01_05_34_00.AEST.2020.manly.snap.jpg"
                     The first dot-separated field must be a Unix timestamp.
    keypoints_path : Path to the reference keypoints JSON containing
                     "camera_pos" in UTM/MGA94 metres.
    utm_zone       : MGA94/UTM zone number (e.g. 56 for Sydney).
    northern       : True for northern hemisphere, False (default) for southern.
    model          : eo_tides model name (default: "FES2022_load").

    Returns
    -------
    float : tide height in metres, rounded to 2 decimal places.
    """
    # ── Parse timestamp from filename ────────────────────────────────────
    timestamp    = int(Path(image_name).name.split(".")[0])
    capture_time = datetime.fromtimestamp(timestamp, UTC)

    # ── Convert UTM camera position to lat/lon ───────────────────────────
    camera_pos = _load_camera_pos_utm(keypoints_path)
    easting, northing, _ = camera_pos
    lat, lon = utm_lib.to_latlon(easting, northing, utm_zone, northern=northern)

    # ── Query tide model ─────────────────────────────────────────────────
    _ensure_clipped_models()
    result = model_tides(
        x=lon, y=lat,
        time=capture_time,
        model=model,
        directory=TIDE_MODELS_CLIPPED_DIR,
    )
    return float(result["tide_height"].values[0].round(2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Get modelled tide height for a CoastSnap image filename"
    )
    parser.add_argument("image_name",
                        help="CoastSnap image filename (Unix timestamp as first field)")
    parser.add_argument("--keypoints", required=True,
                        help="Path to reference keypoints JSON with 'camera_pos'")
    parser.add_argument("--utm-zone", type=int, required=True,
                        help="MGA94/UTM zone number (e.g. 56 for Sydney)")
    parser.add_argument("--northern", action="store_true",
                        help="Set if site is in the northern hemisphere")
    parser.add_argument("--model", default=TIDE_MODEL,
                        help=f"eo_tides model name (default: {TIDE_MODEL})")
    args = parser.parse_args()

    height = get_tide_height(
        args.image_name,
        args.keypoints,
        args.utm_zone,
        northern=args.northern,
        model=args.model,
    )
    print(f"Tide height: {height} m")
