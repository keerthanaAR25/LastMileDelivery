"""
Coordinate validation — Section 19/22.

Flags invalid lat/lng without silently deleting anything (Section 18:
"Do not silently delete. Create flags.").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def flag_invalid_coordinates(
    df: pd.DataFrame,
    lat_cols: list[str],
    lng_cols: list[str],
) -> pd.Series:
    """Returns a boolean Series: True if ANY of the given lat/lng column pairs
    for that row are outside valid ranges or null.

    Real LaDe-D has THREE coordinate pairs per row, not one:
      (lat, lng)                       — order-level location
      (accept_gps_lat, accept_gps_lng)
      (delivery_gps_lat, delivery_gps_lng)
    Nulls are real and observed (JL: 955 rows, YT: 2,422 rows in the
    accept_gps_* columns; 0 elsewhere) — flag, don't silently drop.
    """
    invalid = pd.Series(False, index=df.index)
    for lat_col, lng_col in zip(lat_cols, lng_cols):
        lat = df[lat_col]
        lng = df[lng_col]
        bad = (
            lat.isna() | lng.isna()
            | (lat < -90) | (lat > 90)
            | (lng < -180) | (lng > 180)
            | ((lat == 0) & (lng == 0))  # classic null-island sentinel
        )
        invalid = invalid | bad
    return invalid


def verify_crs_assumption(df: pd.DataFrame, lat_col: str, lng_col: str) -> dict:
    """Cheap sanity check that values look like WGS84 degrees, not e.g.
    a projected CRS in meters (Section 8/91: 'Do not assume CRS.').
    This is a heuristic pre-check, not a substitute for real CRS metadata.
    """
    lat, lng = df[lat_col].dropna(), df[lng_col].dropna()
    looks_like_degrees = lat.between(-90, 90).mean() > 0.99 and lng.between(-180, 180).mean() > 0.99
    return {
        "column_pair": (lat_col, lng_col),
        "looks_like_wgs84_degrees": bool(looks_like_degrees),
        "lat_min": float(lat.min()) if len(lat) else None,
        "lat_max": float(lat.max()) if len(lat) else None,
        "lng_min": float(lng.min()) if len(lng) else None,
        "lng_max": float(lng.max()) if len(lng) else None,
        "note": "Heuristic only — confirm against source metadata before trusting SRID 4326.",
    }
