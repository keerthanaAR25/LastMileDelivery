"""
Runs Modules 3-8 for one city end-to-end, reusing intermediate results
across steps (e.g. grid stats computed once, not per hotspot/temporal call)
to keep runtime reasonable on this sandbox's 1 CPU / 3.9GB RAM.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.logging_config import get_module_logger  # noqa: E402
from src.spatial.grid import assign_grid_cell, flag_geo_outliers, aggregate_zone  # noqa: E402
from src.spatial.hotspots import density_hotspots_grid, risk_hotspots_grid, density_hotspots_dbscan  # noqa: E402
from src.spatial.temporal_and_risk_score import hour_weekday_heatmap, daily_risk_trend, compute_spatial_risk_score  # noqa: E402
from src.sequential.sequence_builder import build_sequences  # noqa: E402
from src.sequential.prefixspan_runner import mine_patterns, score_patterns  # noqa: E402
from src.graph.build_graph import build_graph, compute_graph_analytics  # noqa: E402
from src.models.feature_engineering import build_feature_table, FEATURE_COLUMNS  # noqa: E402
from src.models.train_baselines import train_all_models  # noqa: E402
from src.explainability.shap_explainer import compute_shap, global_importance, local_explanation  # noqa: E402
from src.decision.recommendation_engine import apply_rules, priority_from_prob  # noqa: E402
from src.decision.route_simulation import simulate_reassignment, simulate_reschedule  # noqa: E402

log = get_module_logger("CITY_PIPELINE")

CITY_NAMES = {"sh": "Shanghai", "cq": "Chongqing", "hz": "Hangzhou", "jl": "Jilin", "yt": "Yantai"}


def run_module3(df: pd.DataFrame, city_code: str) -> pd.DataFrame:
    t0 = time.time()
    seqs = build_sequences(df)
    seqs.to_parquet(f"data/interim/sequences_{city_code}.parquet", index=False)
    log.info(f"[{city_code}] M3 sequences: {len(seqs)} built in {time.time()-t0:.1f}s")

    t0 = time.time()
    mined, mine_rt = mine_patterns(seqs, min_support_rate=0.01, max_pattern_length=6)
    scored = score_patterns(seqs, mined, algorithm="prefixspan")
    scored = scored.sort_values("pattern_risk_score", ascending=False)
    Path("artifacts/patterns").mkdir(parents=True, exist_ok=True)
    scored.to_csv(f"artifacts/patterns/sequential_patterns_prefixspan_{city_code}_maxlen6_sup1pct.csv", index=False)
    log.info(f"[{city_code}] M3 patterns: {len(scored)} mined+scored in {time.time()-t0:.1f}s (mining {mine_rt:.1f}s)")
    return scored


def run_module4(df: pd.DataFrame, city_code: str, city_name: str) -> dict:
    t0 = time.time()
    df = df.copy()
    df["is_geo_outlier"] = flag_geo_outliers(df, city_name)
    clean = df[~df["is_geo_outlier"]].copy()
    clean["grid_id"] = assign_grid_cell(clean, 1.0)

    grid_stats = aggregate_zone(clean, "grid_id")
    aoi_stats = aggregate_zone(clean, "aoi_id")
    region_stats = aggregate_zone(clean, "region_id")

    density_hot = density_hotspots_grid(grid_stats)
    risk_hot = risk_hotspots_grid(grid_stats)
    dbscan_hot, n_noise, n_scanned = density_hotspots_dbscan(clean)
    heatmap = hour_weekday_heatmap(clean)
    trend = daily_risk_trend(clean)
    scored_grid = compute_spatial_risk_score(grid_stats)

    out_dir = Path("artifacts/spatial")
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_grid.to_parquet(out_dir / f"spatial_risk_score_{city_code}.parquet", index=False)
    density_hot.to_parquet(out_dir / f"hotspots_density_grid_{city_code}.parquet", index=False)
    risk_hot.to_parquet(out_dir / f"hotspots_risk_grid_{city_code}.parquet", index=False)
    dbscan_hot.to_parquet(out_dir / f"hotspots_density_dbscan_{city_code}.parquet", index=False)
    heatmap.to_parquet(out_dir / f"hour_weekday_heatmap_{city_code}.parquet", index=False)

    log.info(
        f"[{city_code}] M4 done in {time.time()-t0:.1f}s: {len(grid_stats)} grid cells, "
        f"{len(aoi_stats)} AOIs, {len(density_hot)} density hotspots, {len(risk_hot)} risk hotspots, "
        f"{len(dbscan_hot)} dbscan clusters (noise={n_noise}/{n_scanned})"
    )
    return {"grid": scored_grid, "density_hot": density_hot, "risk_hot": risk_hot,
            "dbscan_hot": dbscan_hot, "clean_df": clean}


def run_module5(df: pd.DataFrame, city_code: str, city_name: str) -> dict:
    t0 = time.time()
    G, build_stats = build_graph(df, city_name)
    node_df = compute_graph_analytics(G)
    edge_rows = [{"source": u, "target": v, **attrs} for u, v, attrs in G.edges(data=True)]
    edge_df = pd.DataFrame(edge_rows)

    out_dir = Path("artifacts/graph")
    out_dir.mkdir(parents=True, exist_ok=True)
    node_df.to_parquet(out_dir / f"graph_nodes_{city_code}.parquet", index=False)
    edge_df.to_parquet(out_dir / f"graph_edges_{city_code}.parquet", index=False)
    log.info(f"[{city_code}] M5 graph in {time.time()-t0:.1f}s: {build_stats}")
    return {"nodes": node_df, "edges": edge_df, "stats": build_stats}


def run_module6(processed_path: str, city_code: str, city_name: str) -> dict:
    t0 = time.time()
    feat = build_feature_table(processed_path, city_name)
    feat.to_parquet(f"data/features/features_{city_code}.parquet", index=False)
    log.info(f"[{city_code}] M6 features in {time.time()-t0:.1f}s: {len(feat)} rows, "
             f"splits={feat['split'].value_counts().to_dict()}")

    t0 = time.time()
    out = train_all_models(feat, city_code)
    model_dir = Path("models/final")
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in out["models"].items():
        joblib.dump(model, model_dir / f"nexusflow_{name}_{city_code}_v1.joblib")
    results_df = pd.DataFrame(out["results"]).T.reset_index().rename(columns={"index": "model_name"})
    results_df["city"] = city_code
    Path("reports/experiment_results").mkdir(parents=True, exist_ok=True)
    results_df.to_csv(f"reports/experiment_results/experiment_B_model_comparison_{city_code}.csv", index=False)
    log.info(f"[{city_code}] M6 models trained in {time.time()-t0:.1f}s:\n{results_df.to_string()}")
    return {"feat": feat, "train_out": out}


def run_module7(feat: pd.DataFrame, model, city_code: str) -> dict:
    from src.models.train_baselines import prepare_xy
    t0 = time.time()
    X_test, y_test = prepare_xy(feat, "test")
    shap_values, expected_value = compute_shap(model, X_test)
    global_imp = global_importance(shap_values, FEATURE_COLUMNS)
    out_dir = Path("artifacts/explanations")
    out_dir.mkdir(parents=True, exist_ok=True)
    global_imp.to_csv(out_dir / f"shap_global_importance_{city_code}.csv", index=False)

    probs = model.predict_proba(X_test)[:, 1]
    top_idx = np.argsort(-probs)[:5]
    test_meta = feat[feat["split"] == "test"].reset_index(drop=True)
    local_rows = []
    for idx in top_idx:
        expl = local_explanation(shap_values, X_test.reset_index(drop=True), idx)
        expl["order_id"] = test_meta.iloc[idx]["order_id"]
        expl["risk_probability"] = probs[idx]
        expl["actual_risk_label"] = int(y_test.iloc[idx])
        local_rows.append(expl)
    local_df = pd.concat(local_rows, ignore_index=True)
    local_df.to_csv(out_dir / f"shap_local_examples_{city_code}.csv", index=False)
    log.info(f"[{city_code}] M7 SHAP in {time.time()-t0:.1f}s: top feature = {global_imp.iloc[0]['feature']}")
    return {"global_importance": global_imp, "local_examples": local_df, "probs": probs, "X_test": X_test}


def run_module8(feat: pd.DataFrame, model, city_code: str, n_sample: int = 2000) -> dict:
    t0 = time.time()
    test = feat[feat["split"] == "test"].reset_index(drop=True)
    courier_lookup = test.groupby("courier_id").agg(
        courier_workload_train=("courier_workload_train", "first"),
        courier_risk_rate_train=("courier_risk_rate_train", "first"),
    ).reset_index()

    X = test[FEATURE_COLUMNS].copy()
    X["previous_delivery_duration"] = X["previous_delivery_duration"].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    sample_idx = np.argsort(-probs)[:n_sample]

    records, sims = [], []
    for i in sample_idx:
        row = test.iloc[i]
        rule_out = apply_rules(row, probs[i], courier_lookup)
        rec = {"order_id": int(row["order_id"]), "courier_id": int(row["courier_id"]),
               "risk_probability": round(float(probs[i]), 4), "risk_class": priority_from_prob(probs[i]), **rule_out}
        records.append(rec)

        X_row = X.iloc[[i]]
        if rec["recommended_action"] == "REASSIGN_COURIER" and rec["evidence"]:
            sim = simulate_reassignment(model, X_row, rec["evidence"]["candidate_workload"], rec["evidence"]["candidate_risk_rate"])
            sim["simulation_kind"] = "REASSIGN_COURIER"
        elif rec["recommended_action"] == "RESCHEDULE":
            current_hour = int(X_row["accept_hour"].iloc[0])
            alt_hour = 13 if current_hour in (6, 7, 8, 9) else current_hour
            sim = simulate_reschedule(model, X_row, alt_hour)
            sim["simulation_kind"] = "RESCHEDULE"
        else:
            sim = None
        if sim:
            sim["order_id"] = int(row["order_id"])
            sim["label"] = "SIMULATED_INTERVENTION_IMPACT"
            sims.append(sim)

    result = pd.DataFrame(records)
    sims_df = pd.DataFrame(sims)
    out_dir = Path("artifacts/recommendations")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / f"recommendations_{city_code}.csv", index=False)
    sims_df.to_csv(out_dir / f"simulations_{city_code}.csv", index=False)
    log.info(f"[{city_code}] M8 decisions in {time.time()-t0:.1f}s: "
             f"{result['recommended_action'].value_counts().to_dict()}")
    return {"recommendations": result, "simulations": sims_df}


def run_city(city_code: str):
    city_name = CITY_NAMES[city_code]
    log.info(f"===== STARTING PIPELINE FOR {city_code} ({city_name}) =====")
    processed_path = f"data/processed/deliveries_processed_{city_code}.parquet"
    df = pd.read_parquet(processed_path)

    patterns = run_module3(df, city_code)
    spatial = run_module4(df, city_code, city_name)
    graph = run_module5(df, city_code, city_name)
    m6 = run_module6(processed_path, city_code, city_name)
    lgb_model = m6["train_out"]["models"]["lightgbm"]
    m7 = run_module7(m6["feat"], lgb_model, city_code)
    m8 = run_module8(m6["feat"], lgb_model, city_code)

    log.info(f"===== FINISHED PIPELINE FOR {city_code} =====")
    return {"patterns": patterns, "spatial": spatial, "graph": graph, "m6": m6, "m7": m7, "m8": m8}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--step", choices=["3", "4", "5", "6", "7", "8", "all"], default="all")
    args = parser.parse_args()

    city_code = args.city
    city_name = CITY_NAMES[city_code]
    processed_path = f"data/processed/deliveries_processed_{city_code}.parquet"

    if args.step == "all":
        run_city(city_code)
    else:
        df = pd.read_parquet(processed_path)
        if args.step == "3":
            run_module3(df, city_code)
        elif args.step == "4":
            run_module4(df, city_code, city_name)
        elif args.step == "5":
            run_module5(df, city_code, city_name)
        elif args.step == "6":
            run_module6(processed_path, city_code, city_name)
        elif args.step in ("7", "8"):
            feat = pd.read_parquet(f"data/features/features_{city_code}.parquet")
            model = joblib.load(f"models/final/nexusflow_lightgbm_{city_code}_v1.joblib")
            if args.step == "7":
                run_module7(feat, model, city_code)
            else:
                run_module8(feat, model, city_code)
