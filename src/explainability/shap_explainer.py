"""
Module 7 — Explainable AI: SHAP (Section 39).

Uses LightGBM as the primary model (per Section 36's designation), with
TreeSHAP — exact and polynomial-time for tree ensembles, so full-test-set
global importance is tractable even on this sandbox's 1 CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402
from src.models.feature_engineering import FEATURE_COLUMNS  # noqa: E402
from src.models.train_baselines import prepare_xy  # noqa: E402

log = get_module_logger("MODULE_07_SHAP")


def compute_shap(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Binary classifiers: TreeExplainer can return a single 2D array (positive
    # class) or a list [neg, pos] depending on version/model. Handle both.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return shap_values, explainer.expected_value


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    out = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_explanation(shap_values: np.ndarray, X: pd.DataFrame, row_idx: int, top_n: int = 5) -> pd.DataFrame:
    row_shap = shap_values[row_idx]
    row_vals = X.iloc[row_idx]
    df = pd.DataFrame({
        "feature": X.columns,
        "feature_value": row_vals.values,
        "shap_value": row_shap,
    })
    df["direction"] = np.where(df["shap_value"] > 0, "INCREASES_RISK", "DECREASES_RISK")
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    df["importance_rank"] = range(1, len(df) + 1)
    return df.drop(columns="abs_shap")


def main(features_path: str, model_path: str, city_code: str, n_local_examples: int = 5):
    with ModuleRun(log, module="MODULE 07 - SHAP") as run:
        feat = pd.read_parquet(features_path)
        model = joblib.load(model_path)
        X_test, y_test = prepare_xy(feat, "test")

        shap_values, expected_value = compute_shap(model, X_test)
        global_imp = global_importance(shap_values, FEATURE_COLUMNS)

        out_dir = Path("artifacts/explanations")
        out_dir.mkdir(parents=True, exist_ok=True)
        global_imp.to_csv(out_dir / f"shap_global_importance_{city_code}.csv", index=False)

        # Local examples: pick a few real high-risk-probability test deliveries
        probs = model.predict_proba(X_test)[:, 1]
        top_risk_idx = np.argsort(-probs)[:n_local_examples]
        test_meta = feat[feat["split"] == "test"].reset_index(drop=True)

        local_rows = []
        for idx in top_risk_idx:
            expl = local_explanation(shap_values, X_test.reset_index(drop=True), idx)
            expl["order_id"] = test_meta.iloc[idx]["order_id"]
            expl["risk_probability"] = probs[idx]
            expl["actual_risk_label"] = int(y_test.iloc[idx])
            local_rows.append(expl)
        local_df = pd.concat(local_rows, ignore_index=True)
        local_df.to_csv(out_dir / f"shap_local_examples_{city_code}.csv", index=False)

        run.record(
            city=city_code,
            n_test_rows_explained=len(X_test),
            expected_value=float(expected_value if np.isscalar(expected_value) else expected_value[0]),
            top_global_feature=global_imp.iloc[0]["feature"],
            n_local_examples=n_local_examples,
            artifact_location=str(out_dir),
        )
        return {"global_importance": global_imp, "local_examples": local_df,
                "shap_values": shap_values, "X_test": X_test, "test_meta": test_meta, "probs": probs}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    result = main(args.features, args.model, args.city)
    print(result["global_importance"])
