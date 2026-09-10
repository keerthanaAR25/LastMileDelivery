"""
Module 3 — Sequence construction (Sections 24-25).

SEQUENCE DEFINITION (documented per Section 25's requirement):
  Real LaDe-D has exactly ONE aoi_id/aoi_type per order — there is no
  separate accept-AOI vs delivery-AOI. A sequence therefore cannot be built
  WITHIN a single order; it is built ACROSS the multiple orders one courier
  handles. This is also the more operationally meaningful unit: it captures
  which AOI-type route patterns a courier's day follows, and whether those
  patterns precede excessive-duration risk.

  Sequence unit = (city, courier_id, ds)  — one courier, one calendar day.
  Event        = DELIVERY@aoi_type_<code>, ordered by delivery_time.
  Sequence-level risk label = 1 if ANY delivery in that courier-day has
    risk_label == 1 (excessive-duration risk), else 0.

This is a real design decision made because the confirmed schema does not
support a richer per-order event chain (no MOVE/STOP/HIGH_CONGESTION events
are possible without trajectory data, which has not yet been provided).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402

log = get_module_logger("MODULE_03_SEQUENCE_BUILDER")


def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (city, courier_id, ds) with:
        sequence: list[str] of "aoi_type_<code>" tokens, ordered by delivery_time
        length: int
        risk_flag: 0/1 (any risky delivery in this courier-day)
        n_risky: count of risky deliveries in this courier-day
        avg_duration / duration_p90: stats over deliveries in this sequence
    Excludes rows with null risk_label (malformed timestamps) and null
    delivery_time (can't be ordered).
    """
    clean = df.dropna(subset=["delivery_time", "aoi_type"]).copy()
    clean = clean.sort_values(["courier_id", "ds", "delivery_time"])

    def _agg(g: pd.DataFrame) -> pd.Series:
        seq = [f"aoi_type_{int(t)}" for t in g["aoi_type"]]
        risk = g["risk_label"].fillna(0).astype(int)
        return pd.Series({
            "sequence": seq,
            "length": len(seq),
            "risk_flag": int(risk.max()) if len(risk) else 0,
            "n_risky": int(risk.sum()),
            "avg_duration": float(g["delivery_duration_min"].mean()),
            "duration_p90": float(g["delivery_duration_min"].quantile(0.9)),
            "n_orders": len(g),
        })

    grouped = clean.groupby(["city", "courier_id", "ds"], as_index=False).apply(_agg, include_groups=False)
    return grouped.reset_index(drop=True)


def main(processed_path: str, city_code: str) -> pd.DataFrame:
    with ModuleRun(log, module="MODULE 03 - SEQUENCE BUILD") as run:
        df = pd.read_parquet(processed_path)
        input_rows = len(df)
        seqs = build_sequences(df)
        run.record(
            city=city_code,
            input_rows=input_rows,
            sequences_built=len(seqs),
            avg_sequence_length=round(float(seqs["length"].mean()), 2),
            max_sequence_length=int(seqs["length"].max()),
            risk_positive_sequences=int(seqs["risk_flag"].sum()),
        )
        return seqs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    main(args.input, args.city)
