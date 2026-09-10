"""
Loads Module 3-8 artifacts for one city into the live database. Reuses the
same patterns as the Shanghai loads done earlier in this session, just
parameterized by city_code/city_name so it isn't rewritten five times.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402

log = get_module_logger("LOAD_CITY_TO_DB")

CITY_NAMES = {"sh": "Shanghai", "cq": "Chongqing", "hz": "Hangzhou", "jl": "Jilin", "yt": "Yantai"}


def load_city(city_code: str):
    city_name = CITY_NAMES[city_code]
    conn = psycopg2.connect("postgresql://postgres:NexusFlow2026@localhost:5432/nexusflow_db")
    counts = {}

    # --- patterns ---
    patt_path = Path(f"artifacts/patterns/sequential_patterns_prefixspan_{city_code}_maxlen6_sup1pct.csv")
    if patt_path.exists():
        df = pd.read_csv(patt_path)
        df["pattern_id"] = city_code + "_" + df["pattern_id"]  # avoid cross-city pattern_id collisions
        df["spatial_coverage"] = None
        cols = ["pattern_id", "pattern", "pattern_length", "support_count", "support_rate",
                "occurrence_count", "risk_count", "risk_rate", "average_duration",
                "duration_p90", "pattern_risk_score", "algorithm", "spatial_coverage"]
        rows = [tuple(r[c] if pd.notna(r[c]) else None for c in cols) for _, r in df.iterrows()]
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, f"INSERT INTO analytics.sequential_patterns ({', '.join(cols)}) VALUES %s "
                     f"ON CONFLICT (pattern_id) DO NOTHING", rows, page_size=2000)
        conn.commit()
        counts["patterns"] = len(rows)

    # --- spatial: zones + spatial_risk + hotspots ---
    grid_path = Path(f"artifacts/spatial/spatial_risk_score_{city_code}.parquet")
    if grid_path.exists():
        grid = pd.read_parquet(grid_path)
        grid["grid_id"] = city_code + "_" + grid["grid_id"]  # namespace zone_ids per city
        with conn.cursor() as cur:
            zone_rows = [
                (r.grid_id, "GRID", city_name, int(r.delivery_count), int(r.unique_couriers),
                 float(r.average_duration), float(r.duration_p90),
                 float(r.risk_rate) if pd.notna(r.risk_rate) else None, None, None, None)
                for r in grid.itertuples()
            ]
            psycopg2.extras.execute_values(cur, """
                INSERT INTO analytics.zones (zone_id, zone_type, city, delivery_count, unique_couriers,
                    average_duration, duration_p90, risk_rate, average_speed, congestion_index, pattern_count)
                VALUES %s ON CONFLICT (zone_id) DO NOTHING
            """, zone_rows, page_size=2000)

            sr_rows = [
                (r.grid_id, float(r.historical_risk_component) if pd.notna(r.historical_risk_component) else None,
                 None, float(r.density_component), None, float(r.spatial_risk_score),
                 json.dumps({"historical_risk_rate": 0.35 / 0.55, "delivery_density": 0.20 / 0.55}))
                for r in grid.itertuples()
            ]
            psycopg2.extras.execute_values(cur, """
                INSERT INTO analytics.spatial_risk (zone_id, historical_risk_rate, congestion_component,
                    density_component, pattern_risk_component, spatial_risk_score, weights_json)
                VALUES %s
            """, sr_rows, page_size=2000)
        conn.commit()
        counts["zones"] = len(zone_rows)

        hot_rows = []
        for fname, htype, method in [
            (f"hotspots_density_grid_{city_code}.parquet", "DENSITY", "grid"),
            (f"hotspots_risk_grid_{city_code}.parquet", "RISK", "grid"),
            (f"hotspots_density_dbscan_{city_code}.parquet", "DENSITY", "dbscan"),
        ]:
            p = Path("artifacts/spatial") / fname
            if p.exists():
                hdf = pd.read_parquet(p)
                for r in hdf.itertuples():
                    hot_rows.append((htype, city_name, method, float(r.score), int(r.delivery_count)))
        if hot_rows:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO analytics.hotspots (hotspot_type, city, method, score, delivery_count)
                    VALUES %s
                """, hot_rows, page_size=2000)
            conn.commit()
        counts["hotspots"] = len(hot_rows)

    # --- graph ---
    nodes_path = Path(f"artifacts/graph/graph_nodes_{city_code}.parquet")
    edges_path = Path(f"artifacts/graph/graph_edges_{city_code}.parquet")
    if nodes_path.exists():
        nodes = pd.read_parquet(nodes_path)
        edges = pd.read_parquet(edges_path)
        nodes["node_id"] = city_code + "_" + nodes["node_id"]  # namespace per city
        edges["source"] = city_code + "_" + edges["source"]
        edges["target"] = city_code + "_" + edges["target"]

        with conn.cursor() as cur:
            node_rows = []
            for r in nodes.itertuples():
                attrs = {k: (None if pd.isna(v) else v) for k, v in r._asdict().items()
                         if k not in ("Index", "node_id", "node_type")}
                node_rows.append((r.node_id, r.node_type, json.dumps(attrs, default=str)))
            psycopg2.extras.execute_values(cur, """
                INSERT INTO analytics.logistics_nodes (node_id, node_type, attributes_json)
                VALUES %s ON CONFLICT (node_id) DO NOTHING
            """, node_rows, page_size=2000)

            edge_rows = []
            for r in edges.itertuples():
                edge_rows.append((
                    r.source, r.target, r.edge_type,
                    int(r.frequency) if hasattr(r, "frequency") and pd.notna(r.frequency) else None,
                    float(r.duration) if hasattr(r, "duration") and pd.notna(r.duration) else None,
                    float(r.risk) if hasattr(r, "risk") and pd.notna(r.risk) else None,
                ))
            psycopg2.extras.execute_values(cur, """
                INSERT INTO analytics.logistics_edges (source_node_id, target_node_id, edge_type, frequency, duration, risk)
                VALUES %s
            """, edge_rows, page_size=2000)
        conn.commit()
        counts["graph_nodes"] = len(node_rows)
        counts["graph_edges"] = len(edge_rows)

    # --- predictions (lightgbm + xgboost) ---
    feat_path = Path(f"data/features/features_{city_code}.parquet")
    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        test_part = feat[feat["split"] == "test"].reset_index(drop=True)
        for model_name in ["lightgbm", "xgboost"]:
            import joblib
            from src.models.feature_engineering import FEATURE_COLUMNS
            model_path = Path(f"models/final/nexusflow_{model_name}_{city_code}_v1.joblib")
            if not model_path.exists():
                continue
            model = joblib.load(model_path)
            X = test_part[FEATURE_COLUMNS].copy()
            X["previous_delivery_duration"] = X["previous_delivery_duration"].fillna(0)
            probs = model.predict_proba(X)[:, 1]
            risk_class = pd.cut(probs, bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
                                 labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
            rows = [(int(oid), model_name, "v1", float(p), str(rc))
                    for oid, p, rc in zip(test_part["order_id"], probs, risk_class)]
            with conn.cursor() as cur:
                cur.execute("CREATE TEMP TABLE tmp_pred (order_id BIGINT, model_name TEXT, "
                            "model_version TEXT, risk_probability FLOAT, risk_class TEXT) ON COMMIT DROP;")
                psycopg2.extras.execute_values(cur, "INSERT INTO tmp_pred VALUES %s", rows, page_size=5000)
                cur.execute("""
                    INSERT INTO ml.predictions (delivery_id, model_name, model_version, risk_probability, risk_class)
                    SELECT d.delivery_id, t.model_name, t.model_version, t.risk_probability, t.risk_class
                    FROM tmp_pred t JOIN raw.deliveries d ON d.order_id = t.order_id AND d.city = %s
                """, (city_name,))
            conn.commit()
            counts[f"predictions_{model_name}"] = len(rows)

    # --- shap local examples ---
    shap_path = Path(f"artifacts/explanations/shap_local_examples_{city_code}.csv")
    if shap_path.exists():
        local_df = pd.read_csv(shap_path)
        with conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE tmp_shap (order_id BIGINT, feature_name TEXT, feature_value FLOAT, "
                        "shap_value FLOAT, direction TEXT, importance_rank INT) ON COMMIT DROP;")
            rows = [(int(r.order_id), r.feature, float(r.feature_value), float(r.shap_value),
                     r.direction, int(r.importance_rank)) for r in local_df.itertuples()]
            psycopg2.extras.execute_values(cur, "INSERT INTO tmp_shap VALUES %s", rows, page_size=1000)
            cur.execute("""
                INSERT INTO ml.shap_explanations (prediction_id, feature_name, feature_value, shap_value, direction, importance_rank)
                SELECT p.prediction_id, t.feature_name, t.feature_value, t.shap_value, t.direction, t.importance_rank
                FROM tmp_shap t
                JOIN raw.deliveries d ON d.order_id = t.order_id AND d.city = %s
                JOIN ml.predictions p ON p.delivery_id = d.delivery_id AND p.model_name = 'lightgbm'
            """, (city_name,))
        conn.commit()
        counts["shap_rows"] = len(rows)

    # --- recommendations + simulations ---
    reco_path = Path(f"artifacts/recommendations/recommendations_{city_code}.csv")
    sim_path = Path(f"artifacts/recommendations/simulations_{city_code}.csv")
    if reco_path.exists():
        recos = pd.read_csv(reco_path)
        with conn.cursor() as cur:
            reco_rows = [(int(r.order_id), float(r.risk_probability), r.risk_class, None,
                          r.recommended_action, r.reason, r.priority) for r in recos.itertuples()]
            cur.execute("""
                CREATE TEMP TABLE tmp_reco (order_id BIGINT, risk_probability FLOAT, risk_class TEXT,
                    detected_pattern TEXT, recommended_action TEXT, reason TEXT, priority TEXT) ON COMMIT DROP;
            """)
            psycopg2.extras.execute_values(cur, "INSERT INTO tmp_reco VALUES %s", reco_rows, page_size=2000)
            cur.execute("""
                INSERT INTO decision.recommendations (delivery_id, risk_probability, risk_class, detected_pattern,
                    recommended_action, reason, priority)
                SELECT d.delivery_id, t.risk_probability, t.risk_class, t.detected_pattern,
                    t.recommended_action, t.reason, t.priority
                FROM tmp_reco t JOIN raw.deliveries d ON d.order_id = t.order_id AND d.city = %s
            """, (city_name,))
        conn.commit()
        counts["recommendations"] = len(reco_rows)

        if sim_path.exists():
            sims = pd.read_csv(sim_path)
            with conn.cursor() as cur:
                sim_rows = [(int(r.order_id), None, None, None, None, float(r.baseline_risk),
                             float(r.alternative_risk), float(r.risk_delta), r.simulation_kind, r.label)
                            for r in sims.itertuples()]
                cur.execute("""
                    CREATE TEMP TABLE tmp_sim (order_id BIGINT, baseline_distance FLOAT, alternative_distance FLOAT,
                        baseline_duration FLOAT, alternative_duration FLOAT, baseline_risk FLOAT,
                        alternative_risk FLOAT, risk_delta FLOAT, recommended_action TEXT, label TEXT) ON COMMIT DROP;
                """)
                psycopg2.extras.execute_values(cur, "INSERT INTO tmp_sim VALUES %s", sim_rows, page_size=2000)
                cur.execute("""
                    INSERT INTO decision.route_simulations (delivery_id, baseline_distance, alternative_distance,
                        baseline_duration, alternative_duration, baseline_risk, alternative_risk, risk_delta,
                        recommended_action, label)
                    SELECT d.delivery_id, t.baseline_distance, t.alternative_distance, t.baseline_duration,
                        t.alternative_duration, t.baseline_risk, t.alternative_risk, t.risk_delta,
                        t.recommended_action, t.label
                    FROM tmp_sim t JOIN raw.deliveries d ON d.order_id = t.order_id AND d.city = %s
                """, (city_name,))
            conn.commit()
            counts["simulations"] = len(sim_rows)

    conn.close()
    log.info(f"[{city_code}] DB load complete: {counts}")
    return counts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    load_city(args.city)
