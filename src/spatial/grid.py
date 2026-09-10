"""
Module 4 — Spatial-Temporal Mining: grid construction + zone aggregation
(Sections 28, 31).

REAL DATA CONSTRAINT: no trajectory or road-network files have been
provided yet, so average_speed and congestion_index (Section 28's full
column list) are NOT computable here. They are left as NULL/NaN and
explicitly excluded from spatial_risk_score's weighted formula (with
weights renormalized over only the available components) rather than
silently defaulting to 0, which would misleadingly suggest "no congestion"
instead of "unknown / not yet measured".

Also real: 37 of Shanghai's 1,483,864 rows (0.002%) have lng/lat far
outside any plausible Shanghai bounding box (min lng 102.08, max lat
39.91 — nowhere near Shanghai's real ~121.x/31.x). These are flagged
via is_geo_outlier and excluded from grid/zone aggregation, not deleted
from the underlying table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402

log = get_module_logger("MODULE_04_SPATIAL")

# Real, city-specific plausible bounding boxes (generous, not tight), used only
# to catch gross GPS errors — NOT to trim genuine delivery spread.
CITY_BBOX = {
    "Shanghai": (120.8, 122.2, 30.6, 31.9),
    "Chongqing": (105.0, 110.5, 28.0, 32.5),
    "Hangzhou": (118.5, 120.9, 29.0, 30.9),
    "Jilin": (125.5, 127.2, 43.0, 44.2),
    "Yantai": (120.0, 122.0, 36.7, 38.2),
}


def flag_geo_outliers(df: pd.DataFrame, city_name: str) -> pd.Series:
    bbox = CITY_BBOX.get(city_name)
    if bbox is None:
        return pd.Series(False, index=df.index)
    lng_min, lng_max, lat_min, lat_max = bbox
    return ~(df["lng"].between(lng_min, lng_max) & df["lat"].between(lat_min, lat_max))


def assign_grid_cell(df: pd.DataFrame, cell_size_km: float = 1.0) -> pd.Series:
    """Simple equirectangular-projected grid. Degrees-per-km varies with
    latitude for longitude; uses the data's own mean latitude for the
    correction factor (documented approximation, not a full projection)."""
    mean_lat_rad = np.radians(df["lat"].mean())
    km_per_deg_lat = 111.0
    km_per_deg_lng = 111.0 * np.cos(mean_lat_rad)

    lat_cell = np.floor(df["lat"] * km_per_deg_lat / cell_size_km).astype("int64")
    lng_cell = np.floor(df["lng"] * km_per_deg_lng / cell_size_km).astype("int64")
    return "grid_" + lat_cell.astype(str) + "_" + lng_cell.astype(str)


def aggregate_zone(df: pd.DataFrame, zone_col: str) -> pd.DataFrame:
    """Real Section-28 zone stats. average_speed/congestion_index omitted
    (see module docstring — not computable without trajectory data)."""
    g = df.groupby(zone_col)
    out = g.agg(
        delivery_count=("order_id", "count"),
        unique_couriers=("courier_id", "nunique"),
        average_duration=("delivery_duration_min", "mean"),
        duration_p90=("delivery_duration_min", lambda x: x.quantile(0.9)),
        risk_rate=("risk_label", "mean"),
        mean_lng=("lng", "mean"),
        mean_lat=("lat", "mean"),
    ).reset_index()
    out["average_speed"] = np.nan       # NOT AVAILABLE — needs trajectory data
    out["congestion_index"] = np.nan    # NOT AVAILABLE — needs trajectory data
    return out


def main(processed_path: str, city_code: str, city_name: str, grid_cell_km: float = 1.0):
    with ModuleRun(log, module="MODULE 04 - SPATIAL ZONES") as run:
        df = pd.read_parquet(processed_path)
        input_rows = len(df)

        df["is_geo_outlier"] = flag_geo_outliers(df, city_name)
        clean = df[~df["is_geo_outlier"]].copy()

        clean["grid_id"] = assign_grid_cell(clean, grid_cell_km)

        grid_stats = aggregate_zone(clean, "grid_id")
        aoi_stats = aggregate_zone(clean, "aoi_id")
        region_stats = aggregate_zone(clean, "region_id")

        out_dir = Path("artifacts/spatial")
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_stats.to_parquet(out_dir / f"grid_stats_{city_code}.parquet", index=False)
        aoi_stats.to_parquet(out_dir / f"aoi_stats_{city_code}.parquet", index=False)
        region_stats.to_parquet(out_dir / f"region_stats_{city_code}.parquet", index=False)

        run.record(
            city=city_code,
            input_rows=input_rows,
            geo_outliers_excluded=int(df["is_geo_outlier"].sum()),
            n_grid_cells=len(grid_stats),
            n_aois=len(aoi_stats),
            n_regions=len(region_stats),
            artifact_location=str(out_dir),
        )
        return {"grid": grid_stats, "aoi": aoi_stats, "region": region_stats, "clean_df": clean}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--city-name", required=True)
    args = parser.parse_args()
    main(args.input, args.city, args.city_name)
