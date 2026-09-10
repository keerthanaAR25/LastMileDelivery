"""
Service layer (Section 52). The dashboard NEVER queries Postgres or loads
models directly — it only calls these functions, which are the single
source of truth for what data actually exists vs. what's still pending.

REAL DATA SCOPE (as of this build): raw.deliveries has all 5 cities loaded.
analytics.*, ml.*, decision.* tables only have Shanghai populated so far
(Modules 3-8 have only been run on Shanghai). Functions below reflect this
honestly — e.g. get_patterns() returns an empty result with a clear reason
for any city other than 'sh', rather than silently returning nothing with
no explanation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
CITY_CODE_TO_NAME = {"sh": "Shanghai", "cq": "Chongqing", "hz": "Hangzhou", "jl": "Jilin", "yt": "Yantai"}
CITIES_WITH_FULL_PIPELINE = {"sh", "cq", "hz", "jl", "yt"}  # all 5 cities now have Modules 3-8 run


def _conn():
    url = os.environ.get("DATABASE_URL", "postgresql://postgres:NexusFlow2026@localhost:5432/nexusflow_db")
    return psycopg2.connect(url)


def _query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    with _conn() as conn:
        return pd.read_sql(sql, conn, params=params)


def get_dashboard_summary(city_code: str) -> dict:
    city_name = CITY_CODE_TO_NAME.get(city_code, city_code)
    df = _query_df(
        "SELECT count(*) AS total_deliveries, count(DISTINCT courier_id) AS total_couriers, "
        "avg(delivery_duration_min) AS avg_duration, avg(risk_label::float) AS avg_risk "
        "FROM raw.deliveries WHERE city = %s", (city_name,)
    )
    row = df.iloc[0]
    result = {
        "city": city_code,
        "total_deliveries": int(row["total_deliveries"]),
        "total_couriers": int(row["total_couriers"]),
        "avg_duration_min": round(float(row["avg_duration"]), 1) if row["avg_duration"] else None,
        "avg_risk_rate": round(float(row["avg_risk"]), 4) if row["avg_risk"] else None,
        "full_pipeline_available": city_code in CITIES_WITH_FULL_PIPELINE,
    }
    if city_code in CITIES_WITH_FULL_PIPELINE:
        zones = _query_df("SELECT count(*) AS n FROM analytics.zones WHERE city = %s", (city_name,))
        hotspots = _query_df("SELECT count(*) AS n FROM analytics.hotspots WHERE city = %s", (city_name,))
        prefix = f"{city_code}_prefixspan_%"
        patterns = _query_df("SELECT count(*) AS n FROM analytics.sequential_patterns WHERE pattern_id LIKE %s", (prefix,))
        result.update({
            "n_zones": int(zones.iloc[0]["n"]),
            "n_hotspots": int(hotspots.iloc[0]["n"]),
            "n_patterns": int(patterns.iloc[0]["n"]),
        })
    return result


def get_delivery(city_code: str, order_id: int) -> dict | None:
    city_name = CITY_CODE_TO_NAME.get(city_code, city_code)
    df = _query_df(
        "SELECT * FROM raw.deliveries WHERE city = %s AND order_id = %s LIMIT 1",
        (city_name, order_id)
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def get_patterns(city_code: str, top_n: int = 50) -> pd.DataFrame:
    if city_code not in CITIES_WITH_FULL_PIPELINE:
        return pd.DataFrame()
    # analytics.sequential_patterns has no city column; pattern_id is prefixed
    # per city (e.g. "cq_prefixspan_00000") EXCEPT Shanghai, whose patterns
    # were loaded before city-prefixing was introduced ("prefixspan_00000").
    # Documented inconsistency, not silently papered over.
    prefix = f"{city_code}_prefixspan_%"
    return _query_df(
        "SELECT pattern_id, pattern, pattern_length, support_rate, occurrence_count, "
        "risk_count, risk_rate, average_duration, pattern_risk_score, algorithm "
        "FROM analytics.sequential_patterns WHERE pattern_id LIKE %s "
        "ORDER BY pattern_risk_score DESC LIMIT %s", (prefix, top_n)
    )


def get_hotspots(city_code: str, hotspot_type: str | None = None) -> pd.DataFrame:
    if city_code not in CITIES_WITH_FULL_PIPELINE:
        return pd.DataFrame()
    city_name = CITY_CODE_TO_NAME.get(city_code, city_code)
    sql = "SELECT hotspot_type, method, score, delivery_count FROM analytics.hotspots WHERE city = %s"
    params = [city_name]
    if hotspot_type:
        sql += " AND hotspot_type = %s"
        params.append(hotspot_type)
    return _query_df(sql, tuple(params))


def get_zone_risk(city_code: str, top_n: int = 100) -> pd.DataFrame:
    if city_code not in CITIES_WITH_FULL_PIPELINE:
        return pd.DataFrame()
    city_name = CITY_CODE_TO_NAME.get(city_code, city_code)
    return _query_df(
        "SELECT z.zone_id, z.delivery_count, z.unique_couriers, z.average_duration, z.risk_rate, "
        "sr.spatial_risk_score FROM analytics.zones z "
        "LEFT JOIN analytics.spatial_risk sr ON sr.zone_id = z.zone_id "
        "WHERE z.city = %s "
        "ORDER BY sr.spatial_risk_score DESC NULLS LAST LIMIT %s", (city_name, top_n)
    )


def get_road_risk(city_code: str) -> pd.DataFrame:
    """Load existing processed road-network data for the selected city."""

    if city_code not in CITY_CODE_TO_NAME:
        return pd.DataFrame()

    road_path = (
        REPO_ROOT
        / "data"
        / "raw"
        / "road_network"
        / f"roads_{city_code}.parquet"
    )

    if not road_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(road_path)

        preferred = [
            "road_id",
            "osm_id",
            "city_code",
            "city_name_en",
            "fclass",
            "name",
            "ref",
            "oneway",
            "maxspeed",
            "length_km",
            "geometry_wkt_4326",
            "is_maxspeed_unknown",
        ]

        available = [c for c in preferred if c in df.columns]

        if available:
            df = df[available].copy()

        if "city_name_en" not in df.columns:
            df["city_name_en"] = CITY_CODE_TO_NAME[city_code]

        return df

    except Exception as e:
        print(f"Error loading road data for {city_code}: {e}")
        return pd.DataFrame()


def get_courier_workload(city_code: str, top_n: int = 50) -> pd.DataFrame:
    if city_code not in CITIES_WITH_FULL_PIPELINE:
        return pd.DataFrame()
    return _query_df(
        "SELECT node_id, attributes_json->>'workload' AS workload, "
        "attributes_json->>'risk_rate' AS risk_rate "
        "FROM analytics.logistics_nodes WHERE node_type = 'COURIER' AND node_id LIKE %s "
        "ORDER BY (attributes_json->>'workload')::float DESC LIMIT %s", (f"{city_code}_%", top_n,)
    )


@lru_cache(maxsize=4)
def _load_model(model_name: str, city_code: str):
    path = REPO_ROOT / "models" / "final" / f"nexusflow_{model_name}_{city_code}_v1.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def predict_delivery_risk(city_code: str, order_id: int) -> dict | None:
    if city_code not in CITIES_WITH_FULL_PIPELINE:
        return None
    df = _query_df(
        "SELECT risk_probability, risk_class, model_name FROM ml.predictions p "
        "JOIN raw.deliveries d ON d.delivery_id = p.delivery_id "
        "WHERE d.order_id = %s AND d.city = %s AND p.model_name = 'lightgbm' LIMIT 1",
        (order_id, CITY_CODE_TO_NAME.get(city_code, city_code))
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def explain_delivery(city_code: str, order_id: int) -> pd.DataFrame:
    if city_code not in CITIES_WITH_FULL_PIPELINE:
        return pd.DataFrame()
    return _query_df(
        "SELECT s.feature_name, s.feature_value, s.shap_value, s.direction, s.importance_rank "
        "FROM ml.shap_explanations s "
        "JOIN ml.predictions p ON p.prediction_id = s.prediction_id "
        "JOIN raw.deliveries d ON d.delivery_id = p.delivery_id "
        "WHERE d.order_id = %s AND d.city = %s ORDER BY s.importance_rank",
        (order_id, CITY_CODE_TO_NAME.get(city_code, city_code))
    )


def generate_recommendation(city_code: str, order_id: int) -> dict | None:
    if city_code not in CITIES_WITH_FULL_PIPELINE:
        return None
    df = _query_df(
        "SELECT r.risk_probability, r.risk_class, r.recommended_action, r.reason, r.priority "
        "FROM decision.recommendations r JOIN raw.deliveries d ON d.delivery_id = r.delivery_id "
        "WHERE d.order_id = %s AND d.city = %s ORDER BY r.created_at DESC LIMIT 1",
        (order_id, CITY_CODE_TO_NAME.get(city_code, city_code))
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def simulate_route(city_code: str, order_id: int) -> dict | None:
    if city_code not in CITIES_WITH_FULL_PIPELINE:
        return None
    df = _query_df(
        "SELECT s.baseline_risk, s.alternative_risk, s.risk_delta, s.recommended_action, s.label "
        "FROM decision.route_simulations s JOIN raw.deliveries d ON d.delivery_id = s.delivery_id "
        "WHERE d.order_id = %s AND d.city = %s ORDER BY s.simulation_timestamp DESC LIMIT 1",
        (order_id, CITY_CODE_TO_NAME.get(city_code, city_code))
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def get_operational_insights(city_code: str) -> dict:
    if city_code not in CITIES_WITH_FULL_PIPELINE:
        return {"available": False}
    city_name = CITY_CODE_TO_NAME.get(city_code, city_code)
    top_zones = get_zone_risk(city_code, top_n=10)
    top_patterns = get_patterns(city_code, top_n=10)
    recos = _query_df(
        "SELECT r.recommended_action, count(*) AS n FROM decision.recommendations r "
        "JOIN raw.deliveries d ON d.delivery_id = r.delivery_id WHERE d.city = %s "
        "GROUP BY r.recommended_action", (city_name,)
    )
    return {
        "available": True,
        "top_risk_zones": top_zones,
        "top_risk_patterns": top_patterns,
        "recommendation_breakdown": recos,
    }


def get_available_deliveries_sample(city_code: str, n: int = 20) -> pd.DataFrame:
    """Real sample of high-risk deliveries for the demo delivery-picker."""
    if city_code not in CITIES_WITH_FULL_PIPELINE:
        return pd.DataFrame()
    city_name = CITY_CODE_TO_NAME.get(city_code, city_code)
    return _query_df(
        "SELECT d.order_id, p.risk_probability, p.risk_class "
        "FROM ml.predictions p JOIN raw.deliveries d ON d.delivery_id = p.delivery_id "
        "WHERE p.model_name = 'lightgbm' AND d.city = %s "
        "ORDER BY p.risk_probability DESC LIMIT %s", (city_name, n)
    )
