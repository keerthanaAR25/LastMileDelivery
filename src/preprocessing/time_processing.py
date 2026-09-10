"""
Time processing — Section 19/21.

Timestamp conversion, invalid-ordering flags, and derived temporal features
(delivery_duration_min, accept_hour, weekday, is_weekend, time_period).

IMPORTANT — real LaDe-D specifics (verified 2026-09-05 against actual files,
see data/raw/README.md):
  - accept_time / delivery_time / accept_gps_time / delivery_gps_time are
    strings "MM-DD HH:MM:SS" with NO YEAR embedded anywhere in the source.
  - The year (e.g. 2022) is an external assumption from the HF dataset card,
    NOT a fact derivable from the data. It is passed explicitly as
    `assumed_year` to every parsing function below so it is never silently
    baked in.
  - `ds` (format MMDD, e.g. 501 = May 1) usually matches accept_time's own
    month/day, but NOT always — 13/931,351 rows disagree in the real
    Chongqing file, and those same rows produce impossible orderings
    (delivery before accept). `flag_ds_mismatch` surfaces this for audit.
"""
from __future__ import annotations

import pandas as pd


def parse_lade_datetime(series: pd.Series, assumed_year: int) -> pd.Series:
    """Parses LaDe-D's 'MM-DD HH:MM:SS' strings using an EXPLICIT assumed
    year (never a silent default — Section 91: don't assume undocumented
    facts). Returns NaT for unparseable values rather than raising."""
    with_year = str(assumed_year) + "-" + series.astype(str)
    return pd.to_datetime(with_year, format="%Y-%m-%d %H:%M:%S", errors="coerce")


def flag_ds_mismatch(df: pd.DataFrame, ds_col: str, accept_time_parsed: pd.Series) -> pd.Series:
    """True if the `ds` partition key (MMDD) disagrees with the month/day
    actually embedded in accept_time. Real, observed data-quality signal —
    do not delete these rows, only flag (Section 18)."""
    ds_str = df[ds_col].astype(str).str.zfill(4)
    ds_month = pd.to_numeric(ds_str.str[:2], errors="coerce")
    ds_day = pd.to_numeric(ds_str.str[2:], errors="coerce")
    return (ds_month != accept_time_parsed.dt.month) | (ds_day != accept_time_parsed.dt.day)


def flag_invalid_timestamps(accept: pd.Series, delivery: pd.Series) -> pd.Series:
    """True only for GENUINELY impossible ordering: null timestamps, or
    delivery_time strictly BEFORE accept_time (Section 18).

    Deliberately uses strict '<' rather than '<=': real Chongqing data has
    4,850 rows with delivery_time == accept_time to the minute (plausible
    given 1-minute timestamp granularity — a same-minute delivery, not an
    impossible one). Those are flagged separately via
    flag_zero_duration so they aren't conflated with the 3 rows that are
    genuinely negative (delivery strictly before accept)."""
    return accept.isna() | delivery.isna() | (delivery < accept)


def flag_zero_duration(accept: pd.Series, delivery: pd.Series) -> pd.Series:
    """True if delivery_time == accept_time exactly. Not necessarily wrong
    (1-minute granularity can round very fast deliveries to 0) but worth
    tracking separately from impossible ordering."""
    return (delivery == accept) & accept.notna()


def time_period_bucket(hour: int) -> str:
    if 5 <= hour < 11:
        return "MORNING"
    if 11 <= hour < 14:
        return "MIDDAY"
    if 14 <= hour < 18:
        return "AFTERNOON"
    if 18 <= hour < 22:
        return "EVENING"
    return "NIGHT"


def add_temporal_features(
    df: pd.DataFrame,
    accept_col_raw: str = "accept_time_raw",
    delivery_col_raw: str = "delivery_time_raw",
    ds_col: str = "ds",
    assumed_year: int = 2022,
) -> pd.DataFrame:
    """Builds real timestamps + derived temporal features from LaDe-D's
    year-less raw strings. Also flags ds/accept_time disagreement.
    """
    out = df.copy()
    out["assumed_year"] = assumed_year
    out["accept_time"] = parse_lade_datetime(out[accept_col_raw], assumed_year)
    out["delivery_time"] = parse_lade_datetime(out[delivery_col_raw], assumed_year)

    out["is_malformed_ds"] = flag_ds_mismatch(out, ds_col, out["accept_time"])

    out["delivery_duration_min"] = (
        (out["delivery_time"] - out["accept_time"]).dt.total_seconds() / 60.0
    )
    out["accept_hour"] = out["accept_time"].dt.hour
    out["delivery_hour"] = out["delivery_time"].dt.hour
    out["weekday"] = out["accept_time"].dt.dayofweek  # 0=Monday
    out["is_weekend"] = out["weekday"].isin([5, 6])
    out["time_period"] = out["accept_hour"].apply(
        lambda h: time_period_bucket(int(h)) if pd.notna(h) else None
    )
    return out
