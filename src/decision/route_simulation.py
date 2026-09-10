"""
Module 8 — Route/Intervention simulation (Section 45).

Labelled SIMULATED_INTERVENTION_IMPACT throughout (Section 45: "Never claim
actual operational savings"). For REASSIGN_COURIER recommendations, this
swaps in the REAL candidate courier's train-derived stats and re-runs the
actual trained model — a genuine counterfactual re-scoring, not an invented
number. For RESCHEDULE, it swaps accept_hour. Distance is NOT simulated
(no road-network data — see recommendation_engine.py docstring); only
predicted-risk deltas are reported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402
from src.models.feature_engineering import FEATURE_COLUMNS  # noqa: E402

log = get_module_logger("MODULE_08_SIMULATION")


def simulate_reassignment(model, X_row: pd.DataFrame, candidate_workload: float, candidate_risk_rate: float) -> dict:
    baseline_risk = float(model.predict_proba(X_row)[:, 1][0])
    alt_row = X_row.copy()
    alt_row["courier_workload_train"] = candidate_workload
    alt_row["courier_risk_rate_train"] = candidate_risk_rate
    alt_risk = float(model.predict_proba(alt_row)[:, 1][0])
    return {"baseline_risk": round(baseline_risk, 4), "alternative_risk": round(alt_risk, 4),
            "risk_delta": round(alt_risk - baseline_risk, 4)}


def simulate_reschedule(model, X_row: pd.DataFrame, alt_hour: int) -> dict:
    baseline_risk = float(model.predict_proba(X_row)[:, 1][0])
    alt_row = X_row.copy()
    alt_row["accept_hour"] = alt_hour
    alt_risk = float(model.predict_proba(alt_row)[:, 1][0])
    return {"baseline_risk": round(baseline_risk, 4), "alternative_risk": round(alt_risk, 4),
            "risk_delta": round(alt_risk - baseline_risk, 4)}


def main(recommendations_path: str, features_path: str, model_path: str, city_code: str):
    with ModuleRun(log, module="MODULE 08 - SIMULATION") as run:
        recos = pd.read_csv(recommendations_path)
        feat = pd.read_parquet(features_path)
        model = joblib.load(model_path)
        test = feat[feat["split"] == "test"]

        sims = []
        for _, reco in recos.iterrows():
            row = test[test["order_id"] == reco["order_id"]]
            if row.empty:
                continue
            X_row = row[FEATURE_COLUMNS].copy()
            X_row["previous_delivery_duration"] = X_row["previous_delivery_duration"].fillna(0)

            if reco["recommended_action"] == "REASSIGN_COURIER":
                evidence = eval(reco["evidence"]) if isinstance(reco["evidence"], str) else reco["evidence"]
                if not evidence:
                    continue
                sim = simulate_reassignment(model, X_row, evidence["candidate_workload"], evidence["candidate_risk_rate"])
                sim["simulation_kind"] = "REASSIGN_COURIER"
            elif reco["recommended_action"] == "RESCHEDULE":
                current_hour = int(X_row["accept_hour"].iloc[0])
                alt_hour = 13 if current_hour in (6, 7, 8, 9) else current_hour
                sim = simulate_reschedule(model, X_row, alt_hour)
                sim["simulation_kind"] = "RESCHEDULE"
            else:
                continue

            sim["order_id"] = int(reco["order_id"])
            sim["label"] = "SIMULATED_INTERVENTION_IMPACT"
            sims.append(sim)

        result = pd.DataFrame(sims)
        out_dir = Path("artifacts/recommendations")
        result.to_csv(out_dir / f"simulations_{city_code}.csv", index=False)

        run.record(
            city=city_code,
            n_simulations=len(result),
            mean_risk_delta=round(float(result["risk_delta"].mean()), 4) if len(result) else None,
            pct_simulations_reducing_risk=round(float((result["risk_delta"] < 0).mean()), 4) if len(result) else None,
            artifact_location=str(out_dir),
        )
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    result = main(args.recommendations, args.features, args.model, args.city)
    print(result.groupby("simulation_kind")["risk_delta"].describe())
