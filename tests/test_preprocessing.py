import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.cleaning import (
    normalize_city, flag_duplicates, flag_duration_outliers,
    compute_risk_threshold, flag_unknown_aoi_type,
)
from src.preprocessing.time_processing import (
    parse_lade_datetime, flag_ds_mismatch, flag_invalid_timestamps,
    flag_zero_duration, add_temporal_features,
)
from src.preprocessing.coordinate_validation import flag_invalid_coordinates


def _minimal_df(n=5):
    return pd.DataFrame({
        "order_id": range(n),
        "city": ["sh"] * n,
        "aoi_type": [1] * n,
        "lat": [31.2] * n, "lng": [121.4] * n,
        "accept_gps_lat": [31.2] * n, "accept_gps_lng": [121.4] * n,
        "delivery_gps_lat": [31.2] * n, "delivery_gps_lng": [121.4] * n,
    })


class TestNormalizeCity:
    def test_known_city_maps_correctly(self):
        df = pd.DataFrame({"city": ["Shanghai", "shanghai", "sh"]})
        out = normalize_city(df)
        assert (out["city_normalized"] == "sh").all()
        assert not out["city_unrecognized"].any()

    def test_unrecognized_city_flagged(self):
        df = pd.DataFrame({"city": ["Atlantis"]})
        out = normalize_city(df)
        assert out["city_unrecognized"].iloc[0]


class TestDuplicates:
    def test_no_duplicates(self):
        df = _minimal_df()
        out = flag_duplicates(df)
        assert not out["is_duplicate_id"].any()

    def test_duplicate_order_id_detected(self):
        df = _minimal_df()
        df.loc[1, "order_id"] = df.loc[0, "order_id"]
        out = flag_duplicates(df)
        assert out["is_duplicate_id"].sum() == 2

    def test_empty_dataframe(self):
        df = _minimal_df(0)
        out = flag_duplicates(df)
        assert len(out) == 0


class TestDurationOutliers:
    def test_single_value_no_outliers(self):
        df = pd.DataFrame({"delivery_duration_min": [50.0]})
        out = flag_duration_outliers(df)
        assert not out.any()

    def test_extreme_value_flagged(self):
        df = pd.DataFrame({"delivery_duration_min": [50.0] * 98 + [1.0, 999999.0]})
        out = flag_duration_outliers(df)
        assert out.iloc[-1]  # the extreme max should be flagged

    def test_all_nan_returns_no_flags(self):
        """Real edge case found while testing: with no valid values at all,
        a percentile threshold can't be computed, so the function
        defensibly returns 'no outliers' rather than guessing — it does
        NOT treat all-NaN as all-outlier. Documented actual behavior."""
        df = pd.DataFrame({"delivery_duration_min": [np.nan, np.nan]})
        out = flag_duration_outliers(df)
        assert not out.any()


class TestRiskThreshold:
    def test_90th_percentile(self):
        df = pd.DataFrame({"delivery_duration_min": list(range(1, 101))})
        threshold = compute_risk_threshold(df, percentile=90)
        assert 89 <= threshold <= 91

    def test_threshold_learned_from_train_only(self):
        """Section 35: threshold must not see 'future' (post-train) data."""
        train = pd.DataFrame({"delivery_duration_min": [10.0] * 90 + [20.0] * 10})
        threshold_train_only = compute_risk_threshold(train, percentile=90)
        contaminated = pd.concat([train, pd.DataFrame({"delivery_duration_min": [10000.0]})])
        threshold_contaminated = compute_risk_threshold(contaminated, percentile=90)
        assert threshold_train_only != threshold_contaminated  # proves the function is sensitive to what's passed in
        assert threshold_train_only < 100  # sane if we pass ONLY train


class TestTimestamps:
    def test_parse_lade_datetime_no_year_in_source(self):
        s = pd.Series(["05-01 06:11:00", "10-31 23:59:00"])
        parsed = parse_lade_datetime(s, assumed_year=2022)
        assert parsed.iloc[0].year == 2022
        assert parsed.iloc[0].month == 5 and parsed.iloc[0].day == 1

    def test_parse_unparseable_returns_nat(self):
        s = pd.Series(["not-a-date"])
        parsed = parse_lade_datetime(s, assumed_year=2022)
        assert parsed.isna().iloc[0]

    def test_ds_mismatch_detected(self):
        df = pd.DataFrame({"ds": [501]})  # May 1
        accept_time = pd.Series([pd.Timestamp("2022-12-25")])  # December
        mismatch = flag_ds_mismatch(df, "ds", accept_time)
        assert mismatch.iloc[0]

    def test_ds_match_not_flagged(self):
        df = pd.DataFrame({"ds": [501]})
        accept_time = pd.Series([pd.Timestamp("2022-05-01 06:11:00")])
        mismatch = flag_ds_mismatch(df, "ds", accept_time)
        assert not mismatch.iloc[0]

    def test_invalid_timestamp_strictly_negative_only(self):
        """Real bug fixed during this project: zero-duration (same-minute)
        must NOT be flagged as invalid — only genuinely negative ordering."""
        accept = pd.Series([pd.Timestamp("2022-05-01 10:00:00")] * 3)
        delivery = pd.Series([
            pd.Timestamp("2022-05-01 10:00:00"),   # zero duration — should NOT be invalid
            pd.Timestamp("2022-05-01 09:00:00"),   # before accept — genuinely invalid
            pd.Timestamp("2022-05-01 11:00:00"),   # normal — not invalid
        ])
        invalid = flag_invalid_timestamps(accept, delivery)
        assert not invalid.iloc[0]
        assert invalid.iloc[1]
        assert not invalid.iloc[2]

    def test_zero_duration_flagged_separately(self):
        accept = pd.Series([pd.Timestamp("2022-05-01 10:00:00")])
        delivery = pd.Series([pd.Timestamp("2022-05-01 10:00:00")])
        assert flag_zero_duration(accept, delivery).iloc[0]

    def test_add_temporal_features_single_row(self):
        df = pd.DataFrame({
            "accept_time_raw": ["05-01 08:00:00"],
            "delivery_time_raw": ["05-01 08:30:00"],
            "ds": [501],
        })
        out = add_temporal_features(df, assumed_year=2022)
        assert out["delivery_duration_min"].iloc[0] == pytest.approx(30.0)
        assert out["accept_hour"].iloc[0] == 8


class TestCoordinateValidation:
    def test_valid_coordinates_not_flagged(self):
        df = _minimal_df()
        invalid = flag_invalid_coordinates(df, ["lat"], ["lng"])
        assert not invalid.any()

    def test_out_of_range_latitude_flagged(self):
        df = _minimal_df(1)
        df.loc[0, "lat"] = 999.0
        invalid = flag_invalid_coordinates(df, ["lat"], ["lng"])
        assert invalid.iloc[0]

    def test_null_island_flagged(self):
        df = _minimal_df(1)
        df.loc[0, "lat"] = 0.0
        df.loc[0, "lng"] = 0.0
        invalid = flag_invalid_coordinates(df, ["lat"], ["lng"])
        assert invalid.iloc[0]

    def test_null_coordinates_flagged(self):
        df = _minimal_df(1)
        df.loc[0, "lat"] = np.nan
        invalid = flag_invalid_coordinates(df, ["lat"], ["lng"])
        assert invalid.iloc[0]


class TestAoiType:
    def test_known_code_not_flagged(self):
        df = pd.DataFrame({"aoi_type": [0, 5, 15]})
        assert not flag_unknown_aoi_type(df).any()

    def test_unknown_code_flagged(self):
        df = pd.DataFrame({"aoi_type": [99]})
        assert flag_unknown_aoi_type(df).iloc[0]
