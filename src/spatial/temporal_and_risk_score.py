"""
Module 4 — Temporal analysis + spatial risk score (Sections 30-31).

Spatial risk score weights (Section 31, config `spatial_risk_weights`):
    historical_risk_rate: 0.35
    congestion:           0.25   <- NOT AVAILABLE (no trajectory data)
    delivery_density:     0.20
    pattern_risk:         0.20   <- NOT AVAILABLE (no per-zone pattern
                                    attribution built yet; would need
                                    Module 3 patterns joined by AOI, not
                                    just aoi_type — future work)

Per Section 91 ("if a feature cannot be obtained: remove it or derive it
transparently"), unavailable components are dropped and the remaining
weights are renormalized to sum to 1 — NOT silently zero-filled, which
would understate the score without saying so. This is recorded in
`weights_json` for every row, per analytics.spatial_risk's schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402

log = get_module_logger("MODULE_04_TEMPORAL_RISK_SCORE")


def hour_weekday_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["weekday", "accept_hour"]).agg(
        delivery_count=("order_id", "count"),
        risk_rate=("risk_label", "mean"),
        avg_duration=("delivery_duration_min", "mean"),
    ).reset_index()


def daily_risk_trend(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("ds").agg(
        delivery_count=("order_id", "count"),
        risk_rate=("risk_label", "mean"),
        avg_duration=("delivery_duration_min", "mean"),
    ).reset_index().sort_values("ds")


def compute_spatial_risk_score(
    grid_stats: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = weights or {"historical_risk_rate": 0.35, "delivery_density": 0.20}
    total_w = sum(weights.values())
    norm_weights = {k: v / total_w for k, v in weights.items()}

    out = grid_stats.copy()
    dens_min, dens_max = out["delivery_count"].min(), out["delivery_count"].max()
    out["density_component"] = (
        (out["delivery_count"] - dens_min) / (dens_max - dens_min) if dens_max > dens_min else 0.0
    )
    out["historical_risk_component"] = out["risk_rate"]  # already a 0-1 rate
    out["congestion_component"] = None   # NOT AVAILABLE
    out["pattern_risk_component"] = None  # NOT AVAILABLE

    out["spatial_risk_score"] = (
        norm_weights.get("historical_risk_rate", 0) * out["historical_risk_component"]
        + norm_weights.get("delivery_density", 0) * out["density_component"]
    )
    out["weights_used"] = str(norm_weights)
    return out


def main(processed_path: str, city_code: str, city_name: str):
    with ModuleRun(log, module="MODULE 04 - TEMPORAL+RISKSCORE") as run:
        from src.spatial.grid import main as grid_main
        zones = grid_main(processed_path, city_code, city_name)
        clean = zones["clean_df"]

        heatmap = hour_weekday_heatmap(clean)
        trend = daily_risk_trend(clean)
        scored_grid = compute_spatial_risk_score(zones["grid"])

        out_dir = Path("artifacts/spatial")
        heatmap.to_parquet(out_dir / f"hour_weekday_heatmap_{city_code}.parquet", index=False)
        trend.to_parquet(out_dir / f"daily_risk_trend_{city_code}.parquet", index=False)
        scored_grid.to_parquet(out_dir / f"spatial_risk_score_{city_code}.parquet", index=False)

        run.record(
            city=city_code,
            heatmap_cells=len(heatmap),
            trend_days=len(trend),
            top_spatial_risk_score=float(scored_grid["spatial_risk_score"].max()),
            congestion_available=False,
            pattern_risk_available=False,
        )
        return {"heatmap": heatmap, "trend": trend, "scored_grid": scored_grid}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--city-name", required=True)
    args = parser.parse_args()
    main(args.input, args.city, args.city_name)
