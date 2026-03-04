"""
Daily tide plot for a CoastSnap image.

Given a CoastSnap image filename (first field is a Unix timestamp) and the
site reference keypoints JSON (containing "camera_pos" in UTM/MGA94 metres),
generates a plot of modelled tide height for the entire day the image was taken,
with a marker at the image capture time.

Optionally, a MATLAB tide .mat file (struct with 'time' in MATLAB datenum and
'level' in metres, stored in local AEST time) can be supplied to overlay the
recorded gauge data on the same axes.

Usage (module)
--------------
    from plot_tides import plot_daily_tides
    plot_daily_tides("1577820840.Wed.Jan.01....jpg",
                     keypoints_path="../reference/reference.json",
                     utm_zone=56)

    # With recorded tide gauge data
    plot_daily_tides("1577820840.Wed.Jan.01....jpg",
                     keypoints_path="../reference/reference.json",
                     utm_zone=56,
                     mat_path="manly_tides.mat")

Usage (CLI)
-----------
    python plot_tides.py 1577820840.Wed.Jan.01....jpg \\
        --keypoints ../reference/reference.json \\
        --utm-zone 56

    # With recorded tide gauge data
    python plot_tides.py 1577820840.Wed.Jan.01....jpg \\
        --keypoints ../reference/reference.json \\
        --utm-zone 56 \\
        --mat manly_tides.mat
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import scipy.io
import utm as utm_lib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from eo_tides.model import model_tides

from tides import (
    TIDE_MODELS_CLIPPED_DIR,
    TIDE_MODEL,
    _ensure_clipped_models,
    _load_camera_pos_utm,
)


# AEST is UTC+10 (no daylight saving adjustment)
_AEST = timezone(timedelta(hours=10))

# MATLAB datenum epoch offset relative to Python's datetime.fromordinal
# MATLAB day 1 = Jan 1, year 0000 (proleptic); Python ordinal 1 = Jan 1, 0001
# Difference is 366 days.
_MATLAB_EPOCH_OFFSET = 366


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matlab_datenum_to_utc(datenums: np.ndarray) -> list[datetime]:
    """Convert an array of MATLAB datenums (AEST local) to UTC datetimes.

    MATLAB datenum stores days since Jan 0, year 0000 (i.e. Dec 30, year -1).
    The tide file timestamps are in AEST (UTC+10), so we subtract 10 h to get
    UTC.
    """
    result = []
    for dn in datenums:
        # Integer part → calendar day (Python ordinal = MATLAB datenum - 366)
        # Fractional part → time of day in days
        ordinal  = int(dn) - _MATLAB_EPOCH_OFFSET
        frac_day = dn - int(dn)
        # fromordinal + fractional day gives the naive local AEST datetime directly.
        # Tagging with _AEST and converting to UTC subtracts the 10 h offset.
        dt_aest = (
            datetime.fromordinal(ordinal) + timedelta(days=frac_day)
        ).replace(tzinfo=_AEST)
        result.append(dt_aest.astimezone(timezone.utc))
    return result


def _load_mat_tides(mat_path: str) -> tuple[list[datetime], np.ndarray]:
    """Load tide gauge data from a MATLAB .mat file.

    Expects the file to contain a struct variable named 'tide' with fields:
      - time  : MATLAB datenum array, timestamps in AEST local time
      - level : float array, tide height in metres

    Returns
    -------
    times  : list of timezone-aware UTC datetimes
    levels : numpy float array of tide heights (m)
    """
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    if "tide" not in mat:
        raise ValueError(
            f"'tide' variable not found in {mat_path}. "
            "Expected a MATLAB struct named 'tide'."
        )

    tide_struct = mat["tide"]
    time_raw  = np.atleast_1d(tide_struct.time).astype(float).ravel()
    level_raw = np.atleast_1d(tide_struct.level).astype(float).ravel()

    # Drop NaN rows
    valid = ~(np.isnan(time_raw) | np.isnan(level_raw))
    times  = _matlab_datenum_to_utc(time_raw[valid])
    levels = level_raw[valid]

    return times, levels


def _filter_to_day(
    times: list[datetime],
    levels: np.ndarray,
    day_start: datetime,
    day_end: datetime,
) -> tuple[list[datetime], np.ndarray]:
    """Return only the (time, level) pairs that fall within [day_start, day_end)."""
    mask = np.array([day_start <= t < day_end for t in times])
    return [t for t, m in zip(times, mask) if m], levels[mask]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_daily_tides(
    image_name: str,
    keypoints_path: str,
    utm_zone: int,
    northern: bool = False,
    model: str = TIDE_MODEL,
    interval_minutes: int = 10,
    mat_path: str | None = None,
    output_path: str | None = None,
) -> None:
    """Generate and show (or save) a daily tide height plot for a CoastSnap image.

    Parameters
    ----------
    image_name       : CoastSnap filename whose first field is a Unix timestamp.
    keypoints_path   : Path to reference keypoints JSON with "camera_pos".
    utm_zone         : MGA94/UTM zone number (e.g. 56 for Sydney).
    northern         : True for northern hemisphere.
    model            : eo_tides model name.
    interval_minutes : Time resolution of the modelled tide curve in minutes.
    mat_path         : Optional path to a MATLAB .mat tide file (struct with
                       'time' in MATLAB datenum AEST and 'level' in metres).
                       When supplied, the recorded gauge data is overlaid on the
                       same axes as the modelled curve.
    output_path      : If given, save the figure to this path instead of showing.
    """
    # ── Parse timestamp ───────────────────────────────────────────────────
    timestamp    = int(Path(image_name).name.split(".")[0])
    capture_time = datetime.fromtimestamp(timestamp, timezone.utc)

    # ── Full-day window (UTC) ─────────────────────────────────────────────
    day_start = capture_time.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end   = day_start + timedelta(days=1)

    # ── Modelled tide curve ───────────────────────────────────────────────
    n_steps = int(24 * 60 / 60) + 1
    times   = pd.date_range(start=day_start, end=day_end, periods=n_steps)

    camera_pos = _load_camera_pos_utm(keypoints_path)
    easting, northing, _ = camera_pos
    lat, lon = utm_lib.to_latlon(easting, northing, utm_zone, northern=northern)
    # lat, lon = -33.796980335609085, 152
    _ensure_clipped_models()
    result = model_tides(
        x=lon, y=lat,
        time=times,
        model=model,
        directory=TIDE_MODELS_CLIPPED_DIR,
    )

    print(result)
    heights = result["tide_height"].values

    capture_result = model_tides(
        x=lon, y=lat,
        time=capture_time,
        model=model,
        directory=TIDE_MODELS_CLIPPED_DIR,
    )
    capture_height = (capture_result["tide_height"].values[0].round(2))

    # ── Recorded gauge data (optional) ────────────────────────────────────
    gauge_times = gauge_levels = None
    if mat_path is not None:
        all_times, all_levels = _load_mat_tides(mat_path)
        gauge_times, gauge_levels = _filter_to_day(
            all_times, all_levels, day_start, day_end
        )
        if len(gauge_times) == 0:
            print(
                f"Warning: no gauge records in {mat_path} fall on "
                f"{day_start.strftime('%Y-%m-%d')} (UTC). "
                "Recorded curve will not be shown."
            )
            gauge_times = gauge_levels = None

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    print(heights)
    # Modelled curve
    ax.plot(times.to_pydatetime(), heights,
            color="steelblue", linewidth=2, label=f"Model ({model})")
    ax.fill_between(times.to_pydatetime(), heights, alpha=0.12, color="steelblue")

    # Recorded gauge curve
    if gauge_times is not None:
        ax.plot(gauge_times, gauge_levels,
                color="darkorange", linewidth=2, linestyle="-",
                label="Recorded gauge (.mat)")

    # Vertical line + dot at capture time
    ax.axvline(capture_time, color="crimson", linewidth=1.5, linestyle="--",
               label=f"Image time ({capture_time.strftime('%H:%M')} UTC)")
    ax.scatter([capture_time], [capture_height], color="crimson", zorder=5, s=60)
    

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    fig.autofmt_xdate()

    date_str = capture_time.strftime("%Y-%m-%d")
    ax.set_title(f"Tide height — {date_str} (UTC)\n{Path(image_name).name}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Tide height (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot daily tide height for a CoastSnap image"
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
    parser.add_argument("--interval", type=int, default=10,
                        help="Modelled tide curve resolution in minutes (default: 10)")
    parser.add_argument("--mat", default=None, dest="mat_path",
                        help="MATLAB .mat tide file to overlay recorded gauge data")
    parser.add_argument("--output", default=None,
                        help="Save figure to this path instead of displaying it")
    args = parser.parse_args()

    plot_daily_tides(
        args.image_name,
        args.keypoints,
        args.utm_zone,
        northern=args.northern,
        model=args.model,
        interval_minutes=args.interval,
        mat_path=args.mat_path,
        output_path=args.output,
    )
