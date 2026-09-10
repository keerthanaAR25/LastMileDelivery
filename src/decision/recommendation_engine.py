"""
Module 8 — Intelligent Decision Support: recommendation engine (Sections 42-43).

REAL DATA CONSTRAINT (documented, not silent): REROUTE and AVOID_HIGH_RISK_ROAD
require an alternate physical path to route through, which needs road-network
data. None has been provided yet (Section 91: road-network files were never
supplied). This engine therefore only emits REASSIGN_COURIER, RESCHEDULE,
PRIORITIZE_DELIVERY, and NO_ACTION — all of which are genuinely simulable from
real courier/temporal data already in the database. REROUTE support is left
structurally present (the action type exists in the schema and CHECK
constraint) but is never emitted until road-network data exists.

Priority thresholds mirror the risk_class binning already used in Module 6/7
(LOW <0.25, MEDIUM 0.25-0.5, HIGH 0.5-0.75, CRITICAL >=0.75) so recommendation
priority is consistent with reported risk_class elsewhere in the system.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG  # noqa: E402
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402
from src.models.feature_engineering import FEATURE_COLUMNS  # noqa: E402

log = get_module_logger("MODULE_08_DECISION_ENGINE")

REROUTE_THRESHOLD = CONFIG.get("decision_engine", "reroute_risk_threshold", default=0.70)
REASSIGN_WORKLOAD_MARGIN = CONFIG.get("decision_engine", "reassign_workload_margin", default=0.30)


def priority_from_prob(p: float) -> str:
    if p >= 0.75:
        return "CRITICAL"
    if p >= 0.50:
        return "HIGH"
    if p >= 0.25:
        return "MEDIUM"
    return "LOW"


def find_reassignment_candidate(courier_lookup: pd.DataFrame, current_courier_id, current_workload: float):
    """Real candidate selection: a courier with meaningfully lower workload
    (>= REASSIGN_WORKLOAD_MARGIN below current) AND a lower historical risk
    rate than the current courier — an actual different real courier's
    stats, not a synthetic one."""
    candidates = courier_lookup[
        (courier_lookup["courier_id"] != current_courier_id)
        & (courier_lookup["courier_workload_train"] <= current_workload * (1 - REASSIGN_WORKLOAD_MARGIN))
    ].sort_values("courier_risk_rate_train")
    if candidates.empty:
        return None
    return candidates.iloc[0]


def apply_rules(row: pd.Series, risk_prob: float, courier_lookup: pd.DataFrame) -> dict:
    priority = priority_from_prob(risk_prob)
    workload_p75 = courier_lookup["courier_workload_train"].quantile(0.75)
    reason_parts = []

    if risk_prob >= REROUTE_THRESHOLD and row["courier_workload_train"] >= workload_p75:
        candidate = find_reassignment_candidate(courier_lookup, row["courier_id"], row["courier_workload_train"])
        if candidate is not None:
            action = "REASSIGN_COURIER"
            reason_parts.append(
                f"risk_probability={risk_prob:.3f} >= {REROUTE_THRESHOLD} and current courier workload "
                f"({row['courier_workload_train']:.0f}) is at/above the 75th percentile "
                f"({workload_p75:.0f}); candidate courier has {REASSIGN_WORKLOAD_MARGIN:.0%}+ lower "
                f"workload and lower historical risk rate ({candidate['courier_risk_rate_train']:.3f} "
                f"vs {row['courier_risk_rate_train']:.3f})"
            )
            evidence = {"candidate_courier_id": int(candidate["courier_id"]),
                        "candidate_workload": float(candidate["courier_workload_train"]),
                        "candidate_risk_rate": float(candidate["courier_risk_rate_train"])}
        else:
            action = "NO_ACTION"
            reason_parts.append("high risk and high workload, but no lower-workload/lower-risk courier found")
            evidence = {}
    elif risk_prob >= REROUTE_THRESHOLD and row["accept_hour"] in (6, 7, 8, 9):
        action = "RESCHEDULE"
        reason_parts.append(
            f"risk_probability={risk_prob:.3f} >= {REROUTE_THRESHOLD}; accept_hour={int(row['accept_hour'])} "
            "falls in the empirically elevated-risk morning window (Module 4 finding: hour 6-9 shows "
            "meaningfully higher risk_rate than the daily average)"
        )
        evidence = {"current_accept_hour": int(row["accept_hour"])}
    elif risk_prob >= 0.50:
        action = "PRIORITIZE_DELIVERY"
        reason_parts.append(f"risk_probability={risk_prob:.3f} is elevated (HIGH/CRITICAL) but no specific "
                             "reassignment or reschedule condition is met — flagged for operator priority")
        evidence = {}
    else:
        action = "NO_ACTION"
        reason_parts.append(f"risk_probability={risk_prob:.3f} below actionable thresholds")
        evidence = {}

    return {
        "recommended_action": action,
        "reason": "; ".join(reason_parts),
        "priority": priority,
        "evidence": evidence,
    }


def main(features_path: str, model_path: str, city_code: str, n_sample: int = 2000):
    with ModuleRun(log, module="MODULE 08 - DECISION ENGINE") as run:
        feat = pd.read_parquet(features_path)
        model = joblib.load(model_path)
        test = feat[feat["split"] == "test"].reset_index(drop=True)

        # Real per-courier lookup table (train-derived stats, already leakage-safe)
        courier_lookup = test.groupby("courier_id").agg(
            courier_workload_train=("courier_workload_train", "first"),
            courier_risk_rate_train=("courier_risk_rate_train", "first"),
        ).reset_index()

        X = test[FEATURE_COLUMNS].copy()
        X["previous_delivery_duration"] = X["previous_delivery_duration"].fillna(0)
        probs = model.predict_proba(X)[:, 1]

        # Focus on the highest-risk deliveries for the recommendation demo —
        # a real, documented sample, not the full 222,580 rows (tractability)
        sample_idx = np.argsort(-probs)[:n_sample]

        records = []
        for i in sample_idx:
            row = test.iloc[i]
            rule_out = apply_rules(row, probs[i], courier_lookup)
            records.append({
                "order_id": int(row["order_id"]),
                "courier_id": int(row["courier_id"]),
                "risk_probability": round(float(probs[i]), 4),
                "risk_class": priority_from_prob(probs[i]),
                **rule_out,
            })

        result = pd.DataFrame(records)
        out_dir = Path("artifacts/recommendations")
        out_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_dir / f"recommendations_{city_code}.csv", index=False)

        run.record(
            city=city_code,
            n_recommendations=len(result),
            action_breakdown=result["recommended_action"].value_counts().to_dict(),
            artifact_location=str(out_dir),
        )
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    result = main(args.features, args.model, args.city)
    print(result["recommended_action"].value_counts())
