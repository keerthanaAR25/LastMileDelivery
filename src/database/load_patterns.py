"""
Loads mined sequential patterns (Section 25) into analytics.sequential_patterns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG  # noqa: E402
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402

log = get_module_logger("MODULE_09_LOAD_PATTERNS")

COLS = [
    "pattern_id", "pattern", "pattern_length", "support_count", "support_rate",
    "occurrence_count", "risk_count", "risk_rate", "average_duration",
    "duration_p90", "pattern_risk_score", "algorithm",
]


def load_patterns(csv_path: str, conn_str: str | None = None) -> int:
    conn_str = conn_str or CONFIG.database_url
    df = pd.read_csv(csv_path)
    df["spatial_coverage"] = None  # not computable until Module 4 (spatial) runs

    conn = psycopg2.connect(conn_str)
    try:
        with conn.cursor() as cur:
            rows = [tuple(r[c] if pd.notna(r[c]) else None for c in COLS) for _, r in df.iterrows()]
            cols_sql = ", ".join(COLS)
            sql = f"INSERT INTO analytics.sequential_patterns ({cols_sql}) VALUES %s ON CONFLICT (pattern_id) DO NOTHING"
            psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
    finally:
        conn.close()
    return len(df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    with ModuleRun(log, module="MODULE 09 - LOAD PATTERNS") as run:
        n = load_patterns(args.csv)
        run.record(patterns_loaded=n, source=args.csv)
