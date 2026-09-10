"""
Experiment F — Cross-City Generalization (Section 68).

Train on Shanghai + Hangzhou (combined chronological TRAIN splits, each
city's own train-derived aggregate features already computed independently
— no cross-city leakage since each city's courier/AOI/region/grid stats
were learned only from that city's own training data).

Test on Chongqing's TEST split.

Compares against Chongqing's own in-city model (already trained in Module 6)
to show real performance degradation, per Section 68's explicit requirement
("Report in-city performance, cross-city performance, performance
degradation... Do not claim universal generalization").
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402
from src.models.feature_engineering import FEATURE_COLUMNS  # noqa: E402

log = get_module_logger("EXPERIMENT_F_CROSS_CITY")

RANDOM_SEED = 42


def prepare_xy(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    X["previous_delivery_duration"] = X["previous_delivery_duration"].fillna(0)
    y = df["risk_label"].astype(int)
    return X, y


def evaluate(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
    }


def main(train_cities=("sh", "hz"), test_city="cq"):
    with ModuleRun(log, module="EXPERIMENT F - CROSS CITY") as run:
        train_dfs = []
        for c in train_cities:
            feat = pd.read_parquet(f"data/features/features_{c}.parquet")
            train_dfs.append(feat[feat["split"] == "train"])
        combined_train = pd.concat(train_dfs, ignore_index=True)
        del train_dfs

        # Real memory constraint: this sandbox has 3.9GB RAM and the naive
        # 2.34M-row combined training set OOM-killed the process. Using a
        # documented deterministic sample (Section 71/72's FAST MODE
        # principle) rather than silently failing or under-reporting.
        MAX_TRAIN_ROWS = 800_000
        sampled = len(combined_train) > MAX_TRAIN_ROWS
        if sampled:
            combined_train = combined_train.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_SEED)
        log.info(f"Combined train ({'+'.join(train_cities)}): {len(combined_train)} rows "
                 f"(sampled={sampled} from >{MAX_TRAIN_ROWS if sampled else len(combined_train)})")

        test_feat = pd.read_parquet(f"data/features/features_{test_city}.parquet")
        test_part = test_feat[test_feat["split"] == "test"]

        X_train, y_train = prepare_xy(combined_train)
        X_test, y_test = prepare_xy(test_part)

        t0 = time.time()
        model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                    random_state=RANDOM_SEED, class_weight="balanced", n_jobs=1, verbose=-1)
        model.fit(X_train, y_train)
        cross_city_probs = model.predict_proba(X_test)[:, 1]
        cross_city_metrics = evaluate(y_test, cross_city_probs)
        cross_city_metrics["train_time_s"] = round(time.time() - t0, 2)
        log.info(f"Cross-city (train {train_cities} -> test {test_city}): {cross_city_metrics}")

        # In-city baseline: test_city's own model comparison CSV (already trained in Module 6)
        in_city_path = Path(f"reports/experiment_results/experiment_B_model_comparison_{test_city}.csv")
        in_city_metrics = None
        if in_city_path.exists():
            in_city_df = pd.read_csv(in_city_path)
            in_city_row = in_city_df[in_city_df["model_name"] == "lightgbm"].iloc[0]
            in_city_metrics = {"f1": in_city_row["f1"], "roc_auc": in_city_row["roc_auc"], "pr_auc": in_city_row["pr_auc"]}
            log.info(f"In-city baseline ({test_city} trained+tested on itself): {in_city_metrics}")

        result = {
            "train_cities": "+".join(train_cities),
            "test_city": test_city,
            "cross_city_f1": cross_city_metrics["f1"],
            "cross_city_roc_auc": cross_city_metrics["roc_auc"],
            "cross_city_pr_auc": cross_city_metrics["pr_auc"],
            "in_city_f1": in_city_metrics["f1"] if in_city_metrics else None,
            "in_city_roc_auc": in_city_metrics["roc_auc"] if in_city_metrics else None,
            "in_city_pr_auc": in_city_metrics["pr_auc"] if in_city_metrics else None,
        }
        if in_city_metrics:
            result["pr_auc_degradation"] = round(in_city_metrics["pr_auc"] - cross_city_metrics["pr_auc"], 4)
            result["roc_auc_degradation"] = round(in_city_metrics["roc_auc"] - cross_city_metrics["roc_auc"], 4)

        result_df = pd.DataFrame([result])
        out_dir = Path("reports/experiment_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_dir / f"experiment_F_cross_city_{'_'.join(train_cities)}_to_{test_city}.csv", index=False)

        run.record(**{k: v for k, v in result.items() if v is not None})
        return result_df


if __name__ == "__main__":
    print(main().to_string())
