"""
NexusFlow — Trajectory-Derived Spatial Congestion Index
==========================================================

RUN THIS ON YOUR OWN MACHINE (~15.69GB RAM), NOT IN A CONSTRAINED SANDBOX.

WHAT THIS DOES AND WHY (read before running)
----------------------------------------------
The real, confirmed trajectory file has 51,122,257 rows and exactly these
columns: ds, postman_id, gps_time, lat, lng.

The lat/lng values in this file are PRIVACY-PRESERVING TRANSFORMED
COORDINATES, not real-world GPS. Because of this:
  - We do NOT map-match against roads.csv (there is no real road geometry
    to match a transformed coordinate against).
  - We do NOT convert via EPSG:3857 -> EPSG:4326 (that transform assumes
    the input is a real projected CRS, which this is not).
  - We do NOT compute Haversine distance (Haversine assumes real
    lat/lng degrees on a sphere; a transformed coordinate is not that).
  - The result is NEVER called "road-level congestion" — it is a
    TRAJECTORY-DERIVED SPATIAL CONGESTION INDEX, computed directly from
    courier movement in the transformed coordinate space, gridded into
    cells of a configurable size.

congestion_index := 1 - (avg_speed / reference_speed), clipped to [0, 1],
where reference_speed is each cell's own 90th-percentile observed speed
(a real, data-derived "free-flow" reference — never a live traffic feed,
never fabricated).

REAL RISK CHECKED, NOT ASSUMED: earlier in this project, the delivery
table's accept_time/delivery_time strings turned out to have NO YEAR
embedded, which silently made pandas default to year 1900. gps_time is a
different field and was not specified to have this issue, so this script
does NOT silently "fix" it — it only WARNS if the parsed year looks like
pandas' silent 1900 default, so you can decide whether --assumed-year is
actually needed for your real file. Nothing is forced without you seeing
the warning first.

WHERE TO SAVE THIS FILE
-------------------------
Save as: process_trajectory_locally.py, at the root of your NexusFlow
project directory (next to run_pipeline.py).

EXACT COMMANDS TO RUN
-----------------------
1. Test on a 1% sample first (fast, catches column/parsing issues early):

   python process_trajectory_locally.py `
     --trajectory "data/raw/trajectory/courier_detailed_trajectory_20s.pkl.xz" `
     --output "data/processed/congestion_by_cell_hour.parquet" `
     --sample-frac 0.01

2. Once the sample run's printed summary looks sane (real row counts,
   a plausible speed distribution, a reasonable number of grid cells —
   not all in one cell, not one cell per point), run the full file:

   python process_trajectory_locally.py `
     --trajectory "data/raw/trajectory/courier_detailed_trajectory_20s.pkl.xz" `
     --output "data/processed/congestion_by_cell_hour.parquet"

Both commands also write a small CSV preview to
data/processed/congestion_preview.csv so you can eyeball results without
opening the full Parquet file.
"""
from __future__ import annotations

import argparse
import gc
import lzma
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["ds", "postman_id", "gps_time", "lat", "lng"]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_trajectory(path: str, sample_frac: float | None, seed: int = 42) -> pd.DataFrame:
    log(f"Loading {path} ...")
    p = Path(path)
    if p.suffix == ".xz":
        with lzma.open(p, "rb") as f:
            df = pickle.load(f)
    else:
        df = pd.read_pickle(p)
    log(f"Loaded {len(df):,} rows. Actual columns found: {list(df.columns)}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {REQUIRED_COLUMNS} but these are missing: {missing}. "
            f"Actual columns are {list(df.columns)} — do not guess, inspect and adapt."
        )

    df = df[REQUIRED_COLUMNS].copy()

    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
        log(f"Sampled down to {len(df):,} rows (--sample-frac {sample_frac}, seed={seed})")

    return df


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce", downcast="float")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce", downcast="float")
    df["postman_id"] = pd.to_numeric(df["postman_id"], errors="coerce", downcast="integer")
    df["ds"] = pd.to_numeric(df["ds"], errors="coerce", downcast="integer")
    return df


def parse_gps_time(df: pd.DataFrame, assumed_year: int | None) -> pd.Series:
    """Parses gps_time robustly against the REAL confirmed format.

    CONFIRMED (2026-09, from actual run output against the real 51.1M-row
    file): gps_time values look like '03-28 15:55:03' — "MM-DD HH:MM:SS",
    NO YEAR. This is the exact same year-less pattern already found and
    fixed for the delivery table's accept_time/delivery_time earlier in
    this project (see src/preprocessing/time_processing.py).

    Depending on pandas version, a naive pd.to_datetime() on this format
    either silently defaults to year 1900 (observed in the build sandbox)
    OR fails to parse entirely, 0% valid, with a "could not infer format"
    warning (observed on the user's real machine/pandas version — do not
    assume one behavior over the other). This function detects the
    year-less pattern directly via regex BEFORE attempting any parse, so
    it is robust to both failure modes and does not depend on how a given
    pandas version happens to fail.

    Since this exact format is now directly confirmed (not guessed) to
    match the rest of the LaDe dataset family's known year-less
    convention, the year is applied automatically (default 2022, the same
    documented external assumption used for the delivery data throughout
    this project — see data/raw/README.md) rather than requiring an
    opt-in flag. --assumed-year still overrides the default.
    """
    raw_str = df["gps_time"].astype(str)
    year_missing = raw_str.str.match(r"^\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
    frac_year_missing = year_missing.mean()

    if frac_year_missing > 0.5:
        year_to_use = assumed_year or 2022
        log(
            f"gps_time confirmed year-less for {frac_year_missing:.1%} of rows "
            f"(format 'MM-DD HH:MM:SS', matching the delivery data's known "
            f"issue). Applying assumed_year={year_to_use} explicitly, using "
            f"a fixed format string (faster and more robust than letting "
            f"pandas guess)."
        )
        with_year = np.where(
            year_missing, str(year_to_use) + "-" + raw_str, raw_str
        )
        parsed = pd.to_datetime(with_year, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        n_failed = parsed.isna().sum()
        if n_failed > 0:
            log(f"WARNING: {n_failed:,} of {len(parsed):,} rows still failed "
                f"to parse after applying assumed_year — inspect "
                f"df.loc[parsed.isna(), 'gps_time'].head() for malformed rows.")
    else:
        log(f"Only {frac_year_missing:.1%} of gps_time values look year-less — "
            f"trying a direct parse instead (values may already include a real year).")
        parsed = pd.to_datetime(df["gps_time"], errors="coerce")
        valid_years = parsed.dropna().dt.year
        if len(valid_years) > 0:
            log(f"Direct parse: year range {int(valid_years.min())}-{int(valid_years.max())}, "
                f"{parsed.notna().mean():.1%} valid.")

    n_valid = parsed.notna().sum()
    log(f"gps_time parsing: {n_valid:,}/{len(parsed):,} valid ({n_valid/len(parsed):.2%})")
    if n_valid == 0:
        sample = df["gps_time"].dropna().astype(str).head(10).tolist()
        raise ValueError(
            f"gps_time parsing produced 0 valid timestamps even after the "
            f"year-less fix. Sample raw gps_time values: {sample}. "
            f"The actual timestamp format differs from what was confirmed "
            f"previously — inspect these samples and adjust parse_gps_time "
            f"accordingly rather than guessing further."
        )

    return parsed


def compute_movement(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["postman_id", "gps_time"]).reset_index(drop=True)

    grp = df.groupby("postman_id", sort=False)
    df["prev_lat"] = grp["lat"].shift(1)
    df["prev_lng"] = grp["lng"].shift(1)
    df["prev_time"] = grp["gps_time"].shift(1)

    dx = (df["lng"] - df["prev_lng"]).to_numpy()
    dy = (df["lat"] - df["prev_lat"]).to_numpy()
    df["displacement"] = np.sqrt(dx * dx + dy * dy)

    time_diff_hours = (df["gps_time"] - df["prev_time"]).dt.total_seconds() / 3600.0
    df["time_diff_hours"] = time_diff_hours

    with np.errstate(divide="ignore", invalid="ignore"):
        df["speed"] = df["displacement"] / df["time_diff_hours"]

    del grp, dx, dy, time_diff_hours
    gc.collect()

    n_before = len(df)

    valid = (
        df["lat"].notna() & df["lng"].notna()
        & df["gps_time"].notna() & df["prev_time"].notna()
        & (df["time_diff_hours"] > 0)
        & df["speed"].notna() & np.isfinite(df["speed"])
    )
    df = df[valid].copy()
    n_after_basic = len(df)
    log(f"Basic validity filter: {n_before:,} -> {n_after_basic:,} rows "
        f"({n_before - n_after_basic:,} dropped: missing coords/timestamps, "
        f"non-positive time diffs, or non-finite speed)")

    if len(df) > 0:
        hi_cutoff = df["speed"].quantile(0.995)
        df = df[df["speed"] <= hi_cutoff].copy()
    n_after_outliers = len(df)
    log(f"Percentile-based outlier removal (99.5th pct speed cutoff): "
        f"{n_after_basic:,} -> {n_after_outliers:,} rows")

    df.drop(columns=["prev_lat", "prev_lng", "prev_time"], inplace=True)
    gc.collect()
    return df


def assign_grid_cells(df: pd.DataFrame, cell_size: float) -> pd.DataFrame:
    cell_x = np.floor(df["lng"].to_numpy() / cell_size).astype(np.int64)
    cell_y = np.floor(df["lat"].to_numpy() / cell_size).astype(np.int64)
    df["cell_x"] = cell_x
    df["cell_y"] = cell_y
    df["cell_id"] = [f"cell_{x}_{y}" for x, y in zip(cell_x, cell_y)]
    df["centroid_x"] = (cell_x + 0.5) * cell_size
    df["centroid_y"] = (cell_y + 0.5) * cell_size
    df["hour"] = df["gps_time"].dt.hour
    return df


def compute_congestion(df: pd.DataFrame) -> pd.DataFrame:
    reference = df.groupby("cell_id")["speed"].quantile(0.9).rename("reference_speed")

    agg = df.groupby(["cell_id", "hour"]).agg(
        avg_speed=("speed", "mean"),
        n_observations=("speed", "count"),
        cell_x=("cell_x", "first"),
        cell_y=("cell_y", "first"),
        centroid_x=("centroid_x", "first"),
        centroid_y=("centroid_y", "first"),
    ).reset_index()

    agg = agg.join(reference, on="cell_id")
    agg["congestion_index"] = (1 - agg["avg_speed"] / agg["reference_speed"]).clip(0, 1)
    return agg


def main():
    parser = argparse.ArgumentParser(description="Trajectory-derived spatial congestion index")
    parser.add_argument("--trajectory", required=True, help="Path to courier_detailed_trajectory_20s.pkl.xz")
    parser.add_argument("--output", required=True, help="Output Parquet path, e.g. data/processed/congestion_by_cell_hour.parquet")
    parser.add_argument("--sample-frac", type=float, default=None, help="e.g. 0.01 for a 1%% test run first")
    parser.add_argument("--cell-size", type=float, default=0.01,
                         help="Grid cell size in the SAME raw units as lat/lng in this file "
                              "(an anonymized transformed space, not degrees/meters). "
                              "Inspect the printed coordinate range and adjust so you get a "
                              "reasonable number of cells, not one giant cell or one per point.")
    parser.add_argument("--assumed-year", type=int, default=None,
                         help="Only used if gps_time is found to be year-less (see warning). "
                              "Not applied unless you pass this explicitly.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output_path.parent / "congestion_preview.csv"

    df = load_trajectory(args.trajectory, args.sample_frac, args.seed)
    rows_loaded = len(df)

    df = downcast_dtypes(df)
    df["gps_time"] = parse_gps_time(df, args.assumed_year)

    log(f"lat range: [{df['lat'].min():.6g}, {df['lat'].max():.6g}]  "
        f"lng range: [{df['lng'].min():.6g}, {df['lng'].max():.6g}]  "
        f"(use this to sanity-check --cell-size={args.cell_size})")

    df = compute_movement(df)
    valid_segments = len(df)
    gc.collect()

    df = assign_grid_cells(df, args.cell_size)
    n_cells = df["cell_id"].nunique()
    log(f"Assigned {valid_segments:,} valid segments to {n_cells:,} spatial cells "
        f"(cell_size={args.cell_size})")

    congestion = compute_congestion(df)
    del df
    gc.collect()

    congestion.to_parquet(output_path, index=False)
    congestion.head(1000).to_csv(preview_path, index=False)

    log("")
    log("=" * 70)
    log("PROCESSING SUMMARY")
    log("=" * 70)
    log(f"Rows loaded from trajectory file: {rows_loaded:,}")
    log(f"Valid movement segments after filtering: {valid_segments:,}")
    log(f"Number of spatial cells: {n_cells:,}")
    log(f"Number of (cell, hour) congestion records: {len(congestion):,}")
    log(f"congestion_index stats:\n{congestion['congestion_index'].describe()}")
    log(f"Output Parquet: {output_path.resolve()}")
    log(f"Output preview CSV: {preview_path.resolve()}")
    log("=" * 70)


if __name__ == "__main__":
    main()