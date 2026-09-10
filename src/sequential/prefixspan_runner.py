"""
Module 3 — PrefixSpan mining and pattern scoring (Sections 25-27).

SCORING PERFORMANCE NOTE: an earlier version of this scorer used a pure
Python generator-based subsequence check (`all(item in it for item in
pattern)`) applied via pandas .apply across all mined patterns. At the
spec's configured parameters (min_support=1%, maxlen=6) on Shanghai's real
70,253 sequences, that produces ~4,363 patterns and would require ~306M
individual subsequence checks — empirically this did not complete in
reasonable time on this sandbox's 1 CPU. A regex-based alternative was also
tested and found to suffer pathological backtracking given aoi_type's
extreme skew (75% of all tokens are a single symbol). The current
implementation instead precomputes, once per sequence, a per-symbol sorted
list of token positions, and scores each pattern with `bisect`-based lookups
— empirically 4,363 patterns over 70,253 sequences scores in ~114s.
"""
from __future__ import annotations

import bisect
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from prefixspan import PrefixSpan

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402

log = get_module_logger("MODULE_03_PREFIXSPAN")


def mine_patterns(
    sequences_df: pd.DataFrame,
    min_support_rate: float = 0.01,
    max_pattern_length: int = 6,
    min_pattern_length: int = 2,
):
    """Runs real PrefixSpan over the courier-day sequences. Returns
    (results, runtime) where results is a list of (support_count, pattern).
    min_support_rate is a FRACTION of the total sequence count (Section 25
    config `min_support`), converted here to the absolute count PrefixSpan's
    API expects."""
    db = sequences_df["sequence"].tolist()
    n = len(db)
    min_support_count = max(1, int(np.ceil(min_support_rate * n)))

    ps = PrefixSpan(db)
    ps.minlen = min_pattern_length
    ps.maxlen = max_pattern_length

    t0 = time.time()
    results = ps.frequent(min_support_count)
    runtime = time.time() - t0
    log.info(
        f"PrefixSpan: {len(results)} patterns found, n_sequences={n}, "
        f"min_support_count={min_support_count} ({min_support_rate:.1%}), runtime={runtime:.2f}s"
    )
    return results, runtime


def _build_position_index(sequences: list[list[str]]) -> list[dict]:
    """Once per sequence: map each symbol to its sorted list of positions,
    enabling O(pattern_length * log n) subsequence checks via bisect
    instead of an O(len(seq)) scan per check."""
    index = []
    for seq in sequences:
        d = defaultdict(list)
        for i, tok in enumerate(seq):
            d[tok].append(i)
        index.append(d)
    return index


def _contains_subseq_indexed(pos_dict: dict, pattern) -> bool:
    cur = -1
    for tok in pattern:
        lst = pos_dict.get(tok)
        if not lst:
            return False
        i = bisect.bisect_right(lst, cur)
        if i == len(lst):
            return False
        cur = lst[i]
    return True


def score_patterns(sequences_df: pd.DataFrame, mined, algorithm: str) -> pd.DataFrame:
    """Computes the full Section-10/27 pattern table for each mined pattern,
    by actually re-scanning which real sequences contain it (not estimated),
    using the bisect-indexed subsequence check for tractable runtime.
    """
    n_total = len(sequences_df)
    all_durations = sequences_df["avg_duration"]
    dur_min, dur_max = all_durations.min(), all_durations.max()

    pos_index = _build_position_index(sequences_df["sequence"].tolist())
    risk_flags = sequences_df["risk_flag"].to_numpy()
    avg_durations = sequences_df["avg_duration"].to_numpy()

    rows = []
    for i, (support_count, pattern) in enumerate(mined):
        match_mask = np.array([_contains_subseq_indexed(d, pattern) for d in pos_index])
        occurrence_count = int(match_mask.sum())
        risk_count = int(risk_flags[match_mask].sum()) if occurrence_count else 0
        risk_rate = risk_count / occurrence_count if occurrence_count else 0.0
        avg_duration = float(avg_durations[match_mask].mean()) if occurrence_count else 0.0
        duration_p90 = float(np.percentile(avg_durations[match_mask], 90)) if occurrence_count else 0.0
        normalized_excess_duration = (
            (avg_duration - dur_min) / (dur_max - dur_min) if dur_max > dur_min else 0.0
        )
        support_rate = occurrence_count / n_total
        pattern_risk_score = support_rate * risk_rate * normalized_excess_duration

        rows.append({
            "pattern_id": f"{algorithm}_{i:05d}",
            "pattern": " -> ".join(pattern),
            "pattern_length": len(pattern),
            "support_count": support_count,
            "support_rate": round(support_rate, 6),
            "occurrence_count": occurrence_count,
            "risk_count": risk_count,
            "risk_rate": round(risk_rate, 4),
            "average_duration": round(avg_duration, 2),
            "duration_p90": round(duration_p90, 2),
            "pattern_risk_score": round(pattern_risk_score, 6),
            "algorithm": algorithm,
        })
    return pd.DataFrame(rows)


def main(sequences_path: str, city_code: str, min_support: float, max_len: int) -> pd.DataFrame:
    with ModuleRun(log, module="MODULE 03 - PREFIXSPAN") as run:
        seqs = pd.read_parquet(sequences_path)
        mined, runtime = mine_patterns(seqs, min_support_rate=min_support, max_pattern_length=max_len)
        scored = score_patterns(seqs, mined, algorithm="prefixspan")
        scored = scored.sort_values("pattern_risk_score", ascending=False)

        run.record(
            city=city_code,
            n_sequences=len(seqs),
            patterns_found=len(mined),
            mining_runtime_s=round(runtime, 2),
            top_pattern_risk_score=float(scored["pattern_risk_score"].iloc[0]) if len(scored) else None,
        )
        return scored


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--max-len", type=int, default=6)
    args = parser.parse_args()
    result = main(args.sequences, args.city, args.min_support, args.max_len)
    print(result.head(20).to_string())
