import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sequential.sequence_builder import build_sequences
from src.sequential.prefixspan_runner import mine_patterns, score_patterns, _contains_subseq_indexed, _build_position_index
from src.sequential.spade_baseline import mine_patterns_spade


def _toy_deliveries():
    """A tiny, hand-constructed courier-day dataset with a known, verifiable
    pattern structure — courier 1's day has aoi_type 1,1,2 repeated across
    2 days; courier 2 has a different pattern entirely."""
    return pd.DataFrame({
        "city": ["sh"] * 9,
        "courier_id": [1, 1, 1, 1, 1, 1, 2, 2, 2],
        "ds": [501, 501, 501, 502, 502, 502, 501, 501, 501],
        "delivery_time": pd.to_datetime([
            "2022-05-01 08:00", "2022-05-01 09:00", "2022-05-01 10:00",
            "2022-05-02 08:00", "2022-05-02 09:00", "2022-05-02 10:00",
            "2022-05-01 08:00", "2022-05-01 09:00", "2022-05-01 10:00",
        ]),
        "aoi_type": [1, 1, 2, 1, 1, 2, 3, 3, 3],
        "risk_label": [0, 0, 1, 0, 0, 1, 0, 0, 0],
        "delivery_duration_min": [20, 25, 90, 22, 24, 85, 15, 15, 15],
    })


class TestSequenceBuilder:
    def test_builds_one_sequence_per_courier_day(self):
        seqs = build_sequences(_toy_deliveries())
        assert len(seqs) == 3

    def test_sequence_content_matches_real_events(self):
        seqs = build_sequences(_toy_deliveries())
        c1_day1 = seqs[(seqs["courier_id"] == 1) & (seqs["ds"] == 501)].iloc[0]
        assert c1_day1["sequence"] == ["aoi_type_1", "aoi_type_1", "aoi_type_2"]
        assert c1_day1["risk_flag"] == 1
        assert c1_day1["n_risky"] == 1

    def test_empty_input(self):
        empty = pd.DataFrame({
            "city": [], "courier_id": [], "ds": [], "delivery_time": pd.to_datetime([]),
            "aoi_type": [], "risk_label": [], "delivery_duration_min": [],
        })
        seqs = build_sequences(empty)
        assert len(seqs) == 0

    def test_null_delivery_time_excluded(self):
        df = _toy_deliveries()
        df.loc[0, "delivery_time"] = pd.NaT
        seqs = build_sequences(df)
        c1_day1 = seqs[(seqs["courier_id"] == 1) & (seqs["ds"] == 501)].iloc[0]
        assert c1_day1["length"] == 2


class TestPrefixSpanScoring:
    def test_subsequence_containment_indexed(self):
        pos_index = _build_position_index([["a", "b", "c"], ["a", "c"], ["b", "b"]])
        assert _contains_subseq_indexed(pos_index[0], ["a", "c"])
        assert _contains_subseq_indexed(pos_index[1], ["a", "c"])
        assert not _contains_subseq_indexed(pos_index[2], ["a", "c"])

    def test_mine_and_score_real_toy_data(self):
        seqs = build_sequences(_toy_deliveries())
        mined, _ = mine_patterns(seqs, min_support_rate=0.3, max_pattern_length=3, min_pattern_length=2)
        assert len(mined) > 0
        scored = score_patterns(seqs, mined, algorithm="prefixspan")
        for _, row in scored.iterrows():
            assert 0 < row["occurrence_count"] <= len(seqs)
            assert 0 <= row["risk_rate"] <= 1
            assert 0 <= row["support_rate"] <= 1


class TestPrefixSpanVsSpadeCrossValidation:
    def test_identical_patterns_at_matching_params(self):
        seqs = build_sequences(_toy_deliveries())
        ps_mined, _ = mine_patterns(seqs, min_support_rate=0.3, max_pattern_length=3, min_pattern_length=2)
        spade_mined, _ = mine_patterns_spade(seqs, min_support_rate=0.3, max_pattern_length=3, min_pattern_length=2)

        ps_patterns = {tuple(p) for _, p in ps_mined}
        spade_patterns = {tuple(p) for _, p in spade_mined}
        assert ps_patterns == spade_patterns

        ps_support = {tuple(p): c for c, p in ps_mined}
        spade_support = {tuple(p): c for c, p in spade_mined}
        for pattern in ps_patterns:
            assert ps_support[pattern] == spade_support[pattern]
