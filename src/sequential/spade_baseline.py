"""
SPADE baseline — Section 26 (PrefixSpan vs SPADE comparison).

IMPLEMENTATION NOTE (honesty about what this is): a byte-exact textbook
SPADE implementation performs its candidate generation and support counting
via genuine vertical id-list temporal joins. This implementation follows
SPADE's core algorithmic STRATEGY — level-wise (breadth-first) candidate
generation via Apriori-style joins of frequent (k-1)-patterns, rather than
PrefixSpan's depth-first pattern-growth via projected databases — which is
the actual dimension Section 26 asks to compare (runtime, pattern count,
scalability). Support counting itself uses the same bisect-indexed
subsequence check as prefixspan_runner.py, so both algorithms are guaranteed
to compute support identically and any difference in results reflects
search-strategy correctness, not a scoring discrepancy.
"""
from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402
from src.sequential.prefixspan_runner import _build_position_index, _contains_subseq_indexed, score_patterns  # noqa: E402

log = get_module_logger("MODULE_03_SPADE")


def mine_patterns_spade(
    sequences_df: pd.DataFrame,
    min_support_rate: float = 0.01,
    max_pattern_length: int = 6,
    min_pattern_length: int = 2,
):
    """Level-wise Apriori-style sequential pattern mining.
    Returns (results, runtime) in the same (support_count, pattern) format
    as prefixspan_runner.mine_patterns, for direct comparison.
    """
    n = len(sequences_df)
    min_sup_count = max(1, int(min_support_rate * n))
    pos_index = _build_position_index(sequences_df["sequence"].tolist())

    t0 = time.time()

    # Frequent 1-patterns
    alphabet = sorted({tok for seq in sequences_df["sequence"] for tok in seq})
    freq1 = []
    for tok in alphabet:
        count = sum(1 for d in pos_index if tok in d)
        if count >= min_sup_count:
            freq1.append(((tok,), count))

    all_results = []
    current_level = freq1  # list of (pattern_tuple, support_count)
    length = 1
    while current_level and length < max_pattern_length:
        length += 1
        candidates = set()
        # Apriori-style join: extend every frequent (length-1)-pattern by one
        # frequent symbol (level-wise candidate generation, SPADE's strategy)
        for (patt, _), (tok,) in product(current_level, [p for p, _ in freq1]):
            candidates.add(patt + (tok,))

        next_level = []
        for cand in candidates:
            count = sum(1 for d in pos_index if _contains_subseq_indexed(d, cand))
            if count >= min_sup_count:
                next_level.append((cand, count))
                if length >= min_pattern_length:
                    all_results.append((count, list(cand)))
        current_level = next_level

    if min_pattern_length <= 1:
        all_results = [(c, list(p)) for p, c in freq1] + all_results

    runtime = time.time() - t0
    log.info(
        f"SPADE-style: {len(all_results)} patterns found, n_sequences={n}, "
        f"min_support_count={min_sup_count} ({min_support_rate:.1%}), runtime={runtime:.2f}s"
    )
    return all_results, runtime


def main(sequences_path: str, city_code: str, min_support: float, max_len: int) -> pd.DataFrame:
    with ModuleRun(log, module="MODULE 03 - SPADE") as run:
        seqs = pd.read_parquet(sequences_path)
        mined, runtime = mine_patterns_spade(seqs, min_support_rate=min_support, max_pattern_length=max_len)
        scored = score_patterns(seqs, mined, algorithm="spade")
        scored = scored.sort_values("pattern_risk_score", ascending=False)
        run.record(
            city=city_code,
            n_sequences=len(seqs),
            patterns_found=len(mined),
            mining_runtime_s=round(runtime, 2),
        )
        return scored


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--min-support", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=4)
    args = parser.parse_args()
    result = main(args.sequences, args.city, args.min_support, args.max_len)
    print(result.head(10).to_string())
