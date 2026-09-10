"""
Module 7 — Counterfactual scenario explanations (Section 41).

Per Section 41: these are MODEL-BASED SCENARIOS, never causal evidence.
Only ACTIONABLE factors are perturbed (e.g. accept_hour, prior_stops_count)
— not target-encoded historical rates, which aren't something an operator
can "change" for a given delivery.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402
from src.models.feature_engineering import FEATURE_COLUMNS  # noqa: E402

log = get_module_logger("MODULE_07_COUNTERFACTUAL")


def counterfactual_scenario(model, X_row: pd.DataFrame, factor: str, new_value) -> dict:
    baseline_prob = float(model.predict_proba(X_row)[:, 1][0])
    scenario_row = X_row.copy()
    scenario_row[factor] = new_value
    scenario_prob = float(model.predict_proba(scenario_row)[:, 1][0])
    return {
        "factor": factor,
        "baseline_value": X_row[factor].iloc[0],
        "scenario_value": new_value,
        "baseline_risk_probability": round(baseline_prob, 4),
        "scenario_risk_probability": round(scenario_prob, 4),
        "risk_difference": round(scenario_prob - baseline_prob, 4),
        "label": "MODEL_BASED_SCENARIO",
    }


def main(features_path: str, model_path: str, city_code: str, order_id: int):
    import joblib
    with ModuleRun(log, module="MODULE 07 - COUNTERFACTUAL") as run:
        feat = pd.read_parquet(features_path)
        model = joblib.load(model_path)
        test = feat[feat["split"] == "test"]
        row = test[test["order_id"] == order_id]
        if row.empty:
            raise ValueError(f"order_id {order_id} not found in test split")

        X_row = row[FEATURE_COLUMNS].copy()
        X_row["previous_delivery_duration"] = X_row["previous_delivery_duration"].fillna(0)

        scenarios = []
        current_hour = int(X_row["accept_hour"].iloc[0])
        for alt_hour in [max(0, current_hour - 3), min(23, current_hour + 3)]:
            scenarios.append(counterfactual_scenario(model, X_row, "accept_hour", alt_hour))
        current_stops = int(X_row["prior_stops_count"].iloc[0])
        scenarios.append(counterfactual_scenario(model, X_row, "prior_stops_count", max(0, current_stops - 5)))

        result = pd.DataFrame(scenarios)
        result["order_id"] = order_id
        out_dir = Path("artifacts/explanations")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"counterfactual_{city_code}_{order_id}.csv"
        result.to_csv(out_path, index=False)

        run.record(city=city_code, order_id=order_id, n_scenarios=len(scenarios), artifact_location=str(out_path))
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--order-id", type=int, required=True)
    args = parser.parse_args()
    print(main(args.features, args.model, args.city, args.order_id))
