"""
Database loading — Section 50.

load_deliveries() takes the Module 2 processed parquet (per city) and loads
it into raw.deliveries via psycopg2 execute_values (fast bulk insert).
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG  # noqa: E402
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402

log = get_module_logger("MODULE_09_DATABASE_LOAD")

DB_COLUMNS = [
    "order_id", "region_id", "city", "courier_id", "lng", "lat",
    "aoi_id", "aoi_type", "accept_time_raw", "accept_gps_time_raw",
    "accept_gps_lng", "accept_gps_lat", "delivery_time_raw", "delivery_gps_time_raw",
    "delivery_gps_lng", "delivery_gps_lat", "ds", "assumed_year",
    "accept_time", "delivery_time", "delivery_duration_min",
    "is_malformed_ds", "risk_label", "source_file",
]


def _to_csv_buffer(df: pd.DataFrame) -> io.StringIO:
    """Builds an in-memory CSV for COPY, in chunks, without materializing a
    Python list-of-tuples for the whole (potentially 1.8M-row) frame — that
    approach OOM'd on this sandbox's 3.9GB RAM / 1 CPU."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    sub = df[DB_COLUMNS]
    for row in sub.itertuples(index=False, name=None):
        writer.writerow(["" if (v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA) else v for v in row])
    buf.seek(0)
    return buf


def load_deliveries(parquet_path: str, conn_str: str | None = None, chunk_size: int = 200_000) -> int:
    conn_str = conn_str or CONFIG.database_url
    df = pd.read_parquet(parquet_path)

    missing = set(DB_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Processed parquet is missing expected columns: {missing}")

    # Stage into a temp table first (no unique constraint), then upsert into
    # raw.deliveries with ON CONFLICT DO NOTHING — COPY itself can't express
    # ON CONFLICT.
    cols_sql = ", ".join(DB_COLUMNS)
    conn = psycopg2.connect(conn_str)
    total = 0
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE TEMP TABLE stage_deliveries (LIKE raw.deliveries INCLUDING DEFAULTS) ON COMMIT DROP;")
            cur.execute(f"ALTER TABLE stage_deliveries DROP COLUMN IF EXISTS delivery_id;")
            for start in range(0, len(df), chunk_size):
                chunk = df.iloc[start:start + chunk_size]
                buf = _to_csv_buffer(chunk)
                cur.copy_expert(f"COPY stage_deliveries ({cols_sql}) FROM STDIN WITH CSV NULL ''", buf)
                total += len(chunk)
                del buf
            cur.execute(
                f"INSERT INTO raw.deliveries ({cols_sql}) "
                f"SELECT {cols_sql} FROM stage_deliveries "
                f"ON CONFLICT (order_id, city) DO NOTHING;"
            )
        conn.commit()
    finally:
        conn.close()

    return total


def main(city_codes: list[str]) -> None:
    with ModuleRun(log, module="MODULE 09 - DB LOAD") as run:
        total = 0
        for city in city_codes:
            path = CONFIG.path("data", "processed", f"deliveries_processed_{city}.parquet")
            if not path.exists():
                log.warning(f"No processed file for {city} at {path}, skipping")
                continue
            n = load_deliveries(str(path))
            total += n
            log.info(f"Loaded {n} rows for city={city}")
        run.record(cities=city_codes, total_rows_loaded=total)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=["cq", "hz", "jl", "sh", "yt"])
    args = parser.parse_args()
    main(args.cities)
