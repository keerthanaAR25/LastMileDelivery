"""
Module 2 — Data Audit & Preprocessing (Sections 18-19).

Implements:
  - duplicate detection (exact-row and duplicate package_id)
  - outlier flagging (learned threshold, not hard-coded — Section 20/35)
  - categorical / city normalization
  - orchestrated audit that never silently deletes, only flags
  - data_quality_report.csv / .html generation (Section 18)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.coordinate_validation import flag_invalid_coordinates, verify_crs_assumption  # noqa: E402
from src.preprocessing.time_processing import flag_invalid_timestamps, flag_zero_duration, add_temporal_features  # noqa: E402

# Real LaDe-D observed aoi_type codes (0-15) across all 5 cities. No codebook
# has been published/confirmed — treat as an opaque categorical ID, not a
# semantic label, until documented (Section 91: don't invent meanings).
KNOWN_AOI_TYPE_CODES = set(range(16))

# Known city-name variants -> canonical code. Extend once real data is audited;
# do NOT silently guess codes that haven't been confirmed against the source.
CITY_CANON = {
    "sh": "sh", "shanghai": "sh",
    "hz": "hz", "hangzhou": "hz",
    "cq": "cq", "chongqing": "cq",
    "jl": "jl", "jilin": "jl",
    "yt": "yt", "yantai": "yt",
}


def normalize_city(df: pd.DataFrame, city_col: str = "city") -> pd.DataFrame:
    out = df.copy()
    lowered = out[city_col].astype(str).str.strip().str.lower()
    out["city_normalized"] = lowered.map(CITY_CANON)
    out["city_unrecognized"] = out["city_normalized"].isna()
    return out


def flag_duplicates(df: pd.DataFrame, id_col: str = "order_id") -> pd.DataFrame:
    """Real LaDe-D uses `order_id` as its natural key (confirmed unique
    within every one of the 5 real city files — 0 duplicates found)."""
    out = df.copy()
    out["is_duplicate_row"] = out.duplicated(keep=False)
    out["is_duplicate_id"] = out.duplicated(subset=[id_col], keep=False)
    return out


def flag_duration_outliers(
    df: pd.DataFrame, duration_col: str = "delivery_duration_min",
    lower_pct: float = 1.0, upper_pct: float = 99.0,
) -> pd.Series:
    """Outlier flag learned from the data's own percentile range — not a
    hard-coded cutoff (Section 20/35: thresholds must be learned)."""
    valid = df[duration_col].dropna()
    if valid.empty:
        return pd.Series(False, index=df.index)
    lo, hi = np.percentile(valid, [lower_pct, upper_pct])
    return (df[duration_col] < lo) | (df[duration_col] > hi) | df[duration_col].isna()


def compute_risk_threshold(
    df_train: pd.DataFrame, duration_col: str = "delivery_duration_min", percentile: float = 90.0
) -> float:
    """Section 20: threshold learned ONLY from the training split."""
    return float(np.percentile(df_train[duration_col].dropna(), percentile))


def flag_unknown_aoi_type(df: pd.DataFrame, col: str = "aoi_type") -> pd.Series:
    return ~df[col].isin(KNOWN_AOI_TYPE_CODES)


def run_audit(
    df: pd.DataFrame,
    assumed_year: int = 2022,
    lat_cols=("lat", "accept_gps_lat", "delivery_gps_lat"),
    lng_cols=("lng", "accept_gps_lng", "delivery_gps_lng"),
) -> dict:
    """Runs the full Module 2 audit against the REAL LaDe-D schema and returns:
        {"flagged_df": DataFrame with all flag columns,
         "summary": dict of counts for the quality report}

    Real schema note: there are THREE coordinate pairs (order-level lng/lat,
    accept GPS ping, delivery GPS ping), not the single accept/delivery pair
    assumed in an earlier draft of this module.
    """
    out = df.copy()
    out = normalize_city(out)
    out = flag_duplicates(out, id_col="order_id")
    out = add_temporal_features(out, "accept_time_raw", "delivery_time_raw", "ds", assumed_year)

    out["is_invalid_timestamp"] = flag_invalid_timestamps(out["accept_time"], out["delivery_time"])
    out["is_zero_duration"] = flag_zero_duration(out["accept_time"], out["delivery_time"])
    out["is_invalid_coordinate"] = flag_invalid_coordinates(out, list(lat_cols), list(lng_cols))
    out["is_outlier"] = flag_duration_outliers(out)
    out["is_unknown_aoi_type"] = flag_unknown_aoi_type(out)

    crs_checks = [verify_crs_assumption(out, lat_cols[i], lng_cols[i]) for i in range(len(lat_cols))]

    n = len(out)
    summary = {
        "total_rows": n,
        "assumed_year": assumed_year,
        "duplicate_rows": int(out["is_duplicate_row"].sum()),
        "duplicate_order_ids": int(out["is_duplicate_id"].sum()),
        "invalid_timestamps": int(out["is_invalid_timestamp"].sum()),
        "zero_duration_rows": int(out["is_zero_duration"].sum()),
        "invalid_coordinates": int(out["is_invalid_coordinate"].sum()),
        "outliers": int(out["is_outlier"].sum()),
        "unrecognized_city": int(out["city_unrecognized"].sum()),
        "malformed_ds_vs_accept_time": int(out["is_malformed_ds"].sum()),
        "unknown_aoi_type_code": int(out["is_unknown_aoi_type"].sum()),
        "pct_duplicate_rows": round(100 * out["is_duplicate_row"].sum() / n, 4) if n else 0,
        "pct_invalid_timestamps": round(100 * out["is_invalid_timestamp"].sum() / n, 4) if n else 0,
        "pct_invalid_coordinates": round(100 * out["is_invalid_coordinate"].sum() / n, 4) if n else 0,
        "pct_outliers": round(100 * out["is_outlier"].sum() / n, 4) if n else 0,
        "pct_malformed_ds": round(100 * out["is_malformed_ds"].sum() / n, 4) if n else 0,
        "crs_checks": crs_checks,
    }
    return {"flagged_df": out, "summary": summary}


def write_quality_report(summary: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {k: v for k, v in summary.items() if k != "crs_checks"}
    csv_path = out_dir / "data_quality_report.csv"
    pd.DataFrame([rows]).to_csv(csv_path, index=False)

    html_path = out_dir / "data_quality_report.html"
    crs_html = "".join(
        f"<li>{c['column_pair']}: looks_like_wgs84_degrees={c['looks_like_wgs84_degrees']}, "
        f"lat=[{c['lat_min']:.4f}, {c['lat_max']:.4f}], lng=[{c['lng_min']:.4f}, {c['lng_max']:.4f}]"
        f"<br><em>{c['note']}</em></li>"
        for c in summary["crs_checks"]
    )
    html = f"""
    <html><head><title>NexusFlow Data Quality Report</title></head>
    <body>
    <h1>Data Quality Report</h1>
    <table border="1" cellpadding="4">
    {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows.items())}
    </table>
    <h2>CRS heuristic checks</h2>
    <ul>{crs_html}</ul>
    </body></html>
    """
    html_path.write_text(html)
    return csv_path, html_path
