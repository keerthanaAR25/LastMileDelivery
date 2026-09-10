"""
Module 4 — Hotspot analysis (Section 29).

Congestion hotspots are NOT computed here — no trajectory data available
yet (documented limitation, consistent with grid.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402

log = get_module_logger("MODULE_04_HOTSPOTS")


def density_hotspots_grid(grid_stats: pd.DataFrame, top_pct: float = 0.90) -> pd.DataFrame:
    """Grid cells in the top (1-top_pct) of delivery_count — Section 29A."""
    threshold = grid_stats["delivery_count"].quantile(top_pct)
    hot = grid_stats[grid_stats["delivery_count"] >= threshold].copy()
    hot["hotspot_type"] = "DENSITY"
    hot["method"] = "grid"
    hot["score"] = hot["delivery_count"]
    return hot


def risk_hotspots_grid(
    grid_stats: pd.DataFrame, min_deliveries: int = 30, min_risk_rate: float | None = None
) -> pd.DataFrame:
    """Grid cells with elevated risk_rate AND enough volume to be
    statistically meaningful (Section 29B — a delivery-risk hotspot is
    NOT the same thing as a delivery-density hotspot, so this filters and
    scores independently from density_hotspots_grid)."""
    reliable = grid_stats[grid_stats["delivery_count"] >= min_deliveries].copy()
    if min_risk_rate is None:
        min_risk_rate = reliable["risk_rate"].quantile(0.90)
    hot = reliable[reliable["risk_rate"] >= min_risk_rate].copy()
    hot["hotspot_type"] = "RISK"
    hot["method"] = "grid"
    hot["score"] = hot["risk_rate"]
    return hot


def density_hotspots_dbscan(
    df: pd.DataFrame, eps_km: float = 0.5, min_samples: int = 15, sample_n: int | None = 200_000
):
    """DBSCAN clustering directly on lng/lat (Section 29, config
    dbscan_eps_km/dbscan_min_samples). For a city-scale dataset (SH has
    1.48M points), DBSCAN's O(n log n) with a KD-tree is usually tractable,
    but this sandbox has 1 CPU/3.9GB RAM — a documented deterministic
    sample is used by default (`sample_n`) rather than assuming full-scale
    DBSCAN completes in reasonable time. Set sample_n=None to force full data.
    """
    work = df if sample_n is None or len(df) <= sample_n else df.sample(n=sample_n, random_state=42)
    mean_lat_rad = np.radians(work["lat"].mean())
    km_per_deg_lat = 111.0
    km_per_deg_lng = 111.0 * np.cos(mean_lat_rad)

    coords = np.column_stack([
        work["lat"].to_numpy() * km_per_deg_lat,
        work["lng"].to_numpy() * km_per_deg_lng,
    ])
    labels = DBSCAN(eps=eps_km, min_samples=min_samples).fit_predict(coords)

    work = work.copy()
    work["cluster"] = labels
    clusters = work[work["cluster"] >= 0].groupby("cluster").agg(
        delivery_count=("order_id", "count"),
        mean_lng=("lng", "mean"),
        mean_lat=("lat", "mean"),
        risk_rate=("risk_label", "mean"),
    ).reset_index()
    clusters["hotspot_type"] = "DENSITY"
    clusters["method"] = "dbscan"
    clusters["score"] = clusters["delivery_count"]
    return clusters, int((labels == -1).sum()), len(work)


def main(processed_path: str, city_code: str, city_name: str):
    with ModuleRun(log, module="MODULE 04 - HOTSPOTS") as run:
        from src.spatial.grid import main as grid_main
        zones = grid_main(processed_path, city_code, city_name)
        grid_stats, clean = zones["grid"], zones["clean_df"]

        density_hot = density_hotspots_grid(grid_stats)
        risk_hot = risk_hotspots_grid(grid_stats)
        dbscan_hot, n_noise, n_scanned = density_hotspots_dbscan(clean)

        out_dir = Path("artifacts/spatial")
        density_hot.to_parquet(out_dir / f"hotspots_density_grid_{city_code}.parquet", index=False)
        risk_hot.to_parquet(out_dir / f"hotspots_risk_grid_{city_code}.parquet", index=False)
        dbscan_hot.to_parquet(out_dir / f"hotspots_density_dbscan_{city_code}.parquet", index=False)

        run.record(
            city=city_code,
            n_density_hotspots_grid=len(density_hot),
            n_risk_hotspots_grid=len(risk_hot),
            n_density_hotspots_dbscan=len(dbscan_hot),
            dbscan_noise_points=n_noise,
            dbscan_points_scanned=n_scanned,
        )
        return {"density_grid": density_hot, "risk_grid": risk_hot, "density_dbscan": dbscan_hot}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--city-name", required=True)
    args = parser.parse_args()
    main(args.input, args.city, args.city_name)
