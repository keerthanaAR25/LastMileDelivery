"""
Experiment D — Ablation Study (Section 66).

MODEL A (ML only):      temporal + courier features only
MODEL B (+ spatial):    A + AOI/region/grid risk-rate features
MODEL C (+ sequential): A + prior_stops_count, previous_delivery_duration
MODEL D (+ spatial+sequential): B + C combined
MODEL E (Full NexusFlow): D + a real graph-derived feature (courier's
    degree_centrality from the Module 5 logistics graph) — the one
    dimension D is missing relative to the full system.

All models are LightGBM (the spec's designated primary model), trained on
the same Shanghai chronological split used throughout, so the ONLY thing
that changes between A-E is the feature set — isolating whether sequential/
spatial/graph signals add real value, per Section 66's stated purpose.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import psycopg2
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402

log = get_module_logger("EXPERIMENT_D_ABLATION")

RANDOM_SEED = 42

BASE_FEATURES = ["accept_hour", "weekday", "is_weekend", "aoi_type",
                  "courier_risk_rate_train", "courier_workload_train"]
SPATIAL_FEATURES = ["aoi_risk_rate_train", "aoi_delivery_count_train",
                     "region_risk_rate_train", "grid_risk_rate_train", "grid_delivery_count_train"]
SEQUENTIAL_FEATURES = ["prior_stops_count", "previous_delivery_duration"]

MODEL_VARIANTS = {
    "A": BASE_FEATURES,
    "B": BASE_FEATURES + SPATIAL_FEATURES,
    "C": BASE_FEATURES + SEQUENTIAL_FEATURES,
    "D": BASE_FEATURES + SPATIAL_FEATURES + SEQUENTIAL_FEATURES,
    "E": BASE_FEATURES + SPATIAL_FEATURES + SEQUENTIAL_FEATURES + ["courier_degree_centrality"],
}


def add_graph_feature(feat: pd.DataFrame, city_code: str) -> pd.DataFrame:
    """Joins real courier degree_centrality from the Module 5 graph (loaded
    in analytics.logistics_nodes) onto the feature table by courier_id."""
    conn = psycopg2.connect("postgresql://postgres:NexusFlow2026@localhost:5432/nexusflow_db")
    nodes = pd.read_sql(
        "SELECT node_id, (attributes_json->>'degree_centrality')::float AS courier_degree_centrality "
        "FROM analytics.logistics_nodes WHERE node_type = 'COURIER' AND node_id LIKE %s",
        conn, params=(f"{city_code}_COURIER_%",)
    )
    conn.close()
    nodes["courier_id"] = nodes["node_id"].str.replace(f"{city_code}_COURIER_", "", regex=False).astype(int)
    out = feat.merge(nodes[["courier_id", "courier_degree_centrality"]], on="courier_id", how="left")
    out["courier_degree_centrality"] = out["courier_degree_centrality"].fillna(0)
    return out


def train_and_eval(feat: pd.DataFrame, feature_cols: list[str]) -> dict:
    train = feat[feat["split"] == "train"]
    test = feat[feat["split"] == "test"]

    X_train = train[feature_cols].copy()
    if "previous_delivery_duration" in X_train.columns:
        X_train["previous_delivery_duration"] = X_train["previous_delivery_duration"].fillna(0)
    y_train = train["risk_label"].astype(int)

    X_test = test[feature_cols].copy()
    if "previous_delivery_duration" in X_test.columns:
        X_test["previous_delivery_duration"] = X_test["previous_delivery_duration"].fillna(0)
    y_test = test["risk_label"].astype(int)

    t0 = time.time()
    model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                random_state=RANDOM_SEED, class_weight="balanced", n_jobs=1, verbose=-1)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probs)), 4),
        "train_time_s": round(time.time() - t0, 2),
        "n_features": len(feature_cols),
    }


def main(city_code: str = "sh"):
    with ModuleRun(log, module="EXPERIMENT D - ABLATION") as run:
        feat = pd.read_parquet(f"data/features/features_{city_code}.parquet")
        feat = add_graph_feature(feat, city_code)

        results = []
        for variant, cols in MODEL_VARIANTS.items():
            metrics = train_and_eval(feat, cols)
            metrics["model_variant"] = variant
            metrics["feature_groups_included"] = ", ".join(cols)
            results.append(metrics)
            log.info(f"Model {variant}: {metrics}")

        result_df = pd.DataFrame(results)[["model_variant", "n_features", "f1", "roc_auc", "pr_auc",
                                            "train_time_s", "feature_groups_included"]]
        out_dir = Path("reports/experiment_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_dir / f"experiment_D_ablation_{city_code}.csv", index=False)

        run.record(city=city_code, variants=list(MODEL_VARIANTS.keys()),
                   pr_auc_by_variant={r["model_variant"]: r["pr_auc"] for r in results})
        return result_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="sh")
    args = parser.parse_args()
    print(main(args.city).to_string())
