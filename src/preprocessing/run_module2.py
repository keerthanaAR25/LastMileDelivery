"""
Runs Module 2 (Data Audit + Preprocessing) end-to-end on REAL LaDe-D data
and writes real artifacts. Synthetic sample mode has been removed now that
real data is available (per explicit instruction: "Do not use or generate
synthetic data. Use only these real LaDe-D files for Module 2.").

Usage:
    python -m src.preprocessing.run_module2 --input /path/to/delivery_cq.parquet --city cq
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG  # noqa: E402
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402
from src.preprocessing.cleaning import run_audit, write_quality_report, compute_risk_threshold  # noqa: E402

log = get_module_logger("MODULE_02_AUDIT_PREPROCESSING")

ASSUMED_YEAR = 2022  # external assumption from HF dataset card - see data/raw/README.md


def main(input_path: str, city_code: str) -> pd.DataFrame:
    with ModuleRun(log, module="MODULE 02") as run:
        df = pd.read_parquet(input_path)
        input_rows = len(df)
        df["source_file"] = Path(input_path).name

        # Real source columns are accept_time/delivery_time/accept_gps_time/
        # delivery_gps_time (plain strings, no year). Rename to *_raw so the
        # derived, year-attached timestamps (built below) can occupy the
        # clean accept_time/delivery_time names without collision.
        df = df.rename(columns={
            "accept_time": "accept_time_raw",
            "delivery_time": "delivery_time_raw",
            "accept_gps_time": "accept_gps_time_raw",
            "delivery_gps_time": "delivery_gps_time_raw",
        })

        result = run_audit(df, assumed_year=ASSUMED_YEAR)
        flagged_df = result["flagged_df"]
        summary = result["summary"]

        report_dir = CONFIG.path("reports", "data_report")
        csv_path, html_path = write_quality_report(summary, report_dir)
        pd.DataFrame([{k: v for k, v in summary.items() if k != "crs_checks"}]).to_csv(
            report_dir / f"data_quality_report_{city_code}.csv", index=False
        )

        flagged_df_sorted = flagged_df.sort_values("accept_time")
        n = len(flagged_df_sorted)
        train_end = int(n * CONFIG.get("splits", "train_frac", default=0.70))
        train_df = flagged_df_sorted.iloc[:train_end]

        risk_threshold = compute_risk_threshold(
            train_df, percentile=CONFIG.get("risk_label", "percentile", default=90)
        )
        flagged_df["risk_label"] = (flagged_df["delivery_duration_min"] > risk_threshold).astype("Int64")
        flagged_df.loc[flagged_df["is_invalid_timestamp"], "risk_label"] = pd.NA

        clean_mask = ~(
            flagged_df["is_invalid_timestamp"]
            | flagged_df["is_invalid_coordinate"]
            | flagged_df["is_duplicate_row"]
        )
        processed_dir = CONFIG.path(CONFIG.get("dataset", "processed_dir", default="data/processed"))
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / f"deliveries_processed_{city_code}.parquet"
        flagged_df.to_parquet(out_path, index=False)

        run.record(
            city=city_code,
            input_rows=input_rows,
            output_rows=len(flagged_df),
            clean_rows=int(clean_mask.sum()),
            flagged_rows=int((~clean_mask).sum()),
            malformed_ds_rows=summary["malformed_ds_vs_accept_time"],
            risk_threshold_min=round(risk_threshold, 2),
            risk_positive_rate=round(float(flagged_df["risk_label"].dropna().astype(float).mean()), 4),
            artifact_location=str(out_path),
        )
        log.info(f"Data quality report: {csv_path}, {html_path}")
        log.info(f"Summary: { {k: v for k, v in summary.items() if k != 'crs_checks'} }")

        return flagged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    main(args.input, args.city)
