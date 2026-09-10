"""
NexusFlow — top-level pipeline orchestrator (Section 88).

Usage:
    python run_pipeline.py --mode sample --city sh
    python run_pipeline.py --mode full --city sh
    python run_pipeline.py --mode full --city all

--mode sample: subsamples the real processed delivery data (documented,
    deterministic, seeded) to run the full Module 2-8 flow quickly for
    development. This is NOT synthetic data — it's a real subset of the
    actual LaDe-D city file — and per Section 72, sample-mode results must
    never be reported as final findings.
--mode full: runs every module against the complete real dataset for the
    selected city/cities, exactly as was done for all 5 cities in this
    project's actual build.

This script calls the same underlying functions already validated
module-by-module earlier in the project (src/preprocessing,
src/sequential, src/spatial, src/graph, src/models, src/explainability,
src/decision) — it does not reimplement any logic, only sequences it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402
from run_city_pipeline import run_module3, run_module4, run_module5, run_module6, run_module7, run_module8, CITY_NAMES  # noqa: E402

log = get_module_logger("RUN_PIPELINE")

ALL_CITIES = ["sh", "cq", "hz", "jl", "yt"]
SAMPLE_ROWS = 5000
SAMPLE_SEED = 42


def get_processed_df(city_code: str, mode: str) -> pd.DataFrame:
    path = f"data/processed/deliveries_processed_{city_code}.parquet"
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run Module 2 first: "
            f"python -m src.preprocessing.run_module2 --input data/raw/delivery/delivery_{city_code}.parquet --city {city_code}"
        )
    df = pd.read_parquet(path)
    if mode == "sample":
        n = min(SAMPLE_ROWS, len(df))
        df = df.sample(n=n, random_state=SAMPLE_SEED).reset_index(drop=True)
        log.warning(
            f"[{city_code}] SAMPLE MODE: using a real but deterministic "
            f"{n}-row subset (seed={SAMPLE_SEED}). Per Section 72, do not "
            f"report these results as final findings."
        )
    return df


def run_pipeline_for_city(city_code: str, mode: str, skip_modules: set) -> dict:
    city_name = CITY_NAMES[city_code]
    log.info(f"===== PIPELINE START: city={city_code} ({city_name}) mode={mode} =====")
    t0 = time.time()

    df = get_processed_df(city_code, mode)

    # Sample mode must apply to EVERY module, not just the ones that take
    # `df` directly in-memory — Module 6 reads its own file from disk, so
    # a sampled run needs its own on-disk copy or it silently falls back
    # to the full dataset (a real bug caught while testing this script).
    if mode == "sample":
        processed_path = f"data/interim/_sample_processed_{city_code}.parquet"
        Path("data/interim").mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_path, index=False)
    else:
        processed_path = f"data/processed/deliveries_processed_{city_code}.parquet"

    results = {}
    if "3" not in skip_modules:
        results["patterns"] = run_module3(df, city_code)
    if "4" not in skip_modules:
        results["spatial"] = run_module4(df, city_code, city_name)
    if "5" not in skip_modules:
        results["graph"] = run_module5(df, city_code, city_name)
    if "6" not in skip_modules:
        results["m6"] = run_module6(processed_path, city_code, city_name)
        model = results["m6"]["train_out"]["models"]["lightgbm"]
        feat = results["m6"]["feat"]
        if "7" not in skip_modules:
            results["m7"] = run_module7(feat, model, city_code)
        if "8" not in skip_modules:
            results["m8"] = run_module8(feat, model, city_code)

    log.info(f"===== PIPELINE COMPLETE: city={city_code} runtime={time.time()-t0:.1f}s =====")
    return results


def main():
    parser = argparse.ArgumentParser(description="NexusFlow pipeline orchestrator")
    parser.add_argument("--mode", choices=["sample", "full"], required=True)
    parser.add_argument("--city", default="sh", help="City code (sh/cq/hz/jl/yt) or 'all'")
    parser.add_argument("--skip-modules", default="", help="Comma-separated module numbers to skip, e.g. '5,7'")
    args = parser.parse_args()

    cities = ALL_CITIES if args.city == "all" else [args.city]
    skip = set(args.skip_modules.split(",")) if args.skip_modules else set()

    with ModuleRun(log, module="RUN_PIPELINE") as run:
        all_results = {}
        for city_code in cities:
            all_results[city_code] = run_pipeline_for_city(city_code, args.mode, skip)
        run.record(mode=args.mode, cities=cities, skipped_modules=list(skip) or "none")

    return all_results


if __name__ == "__main__":
    main()
