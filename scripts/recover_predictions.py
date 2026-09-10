from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import psycopg2

from src.models.train_baselines import prepare_xy
from src.decision.recommendation_engine import priority_from_prob


CITY_CODE = "sh"
CITY_NAME = "Shanghai"

FEATURE_PATH = Path("data/features/features_sh.parquet")
MODEL_PATH = Path("models/final/nexusflow_lightgbm_sh_v1.joblib")
PREDICTION_FILE = Path("data/processed/recovered_predictions_sh.csv")


def main():

    print("1. Loading features...")
    feat = pd.read_parquet(FEATURE_PATH)

    print("2. Loading existing LightGBM model...")
    model = joblib.load(MODEL_PATH)

    print("3. Preparing test data...")
    X_test, _ = prepare_xy(feat, "test")
    test_meta = feat[feat["split"] == "test"].reset_index(drop=True)

    print("4. Generating predictions...")
    probs = model.predict_proba(X_test)[:, 1]

    predictions = pd.DataFrame({
        "order_id": test_meta["order_id"].astype("int64"),
        "risk_probability": probs.astype(float),
        "risk_class": [
            priority_from_prob(float(p))
            for p in probs
        ],
    })

    print("Predictions generated:", len(predictions))

    # Save predictions first so they are not lost
    predictions.to_csv(PREDICTION_FILE, index=False)

    print("5. Saved:", PREDICTION_FILE)

    print("6. Connecting to PostgreSQL...")

    conn = psycopg2.connect(
        "postgresql://postgres:NexusFlow2026@localhost:5432/nexusflow_db"
    )

    try:
        cur = conn.cursor()

        print("7. Creating temporary table...")

        cur.execute("""
            CREATE TEMP TABLE tmp_predictions (
                order_id BIGINT,
                risk_probability DOUBLE PRECISION,
                risk_class TEXT
            )
        """)

        print("8. Loading predictions into PostgreSQL...")

        with open(PREDICTION_FILE, "r", encoding="utf-8") as f:
            next(f)  # skip CSV header
            cur.copy_expert(
                """
                COPY tmp_predictions
                (order_id, risk_probability, risk_class)
                FROM STDIN
                WITH CSV
                """,
                f,
            )

        print("9. Creating temporary index...")

        cur.execute("""
            CREATE INDEX tmp_predictions_order_idx
            ON tmp_predictions(order_id)
        """)

        print("10. Inserting into ml.predictions...")

        cur.execute("""
            INSERT INTO ml.predictions
            (
                delivery_id,
                model_name,
                model_version,
                risk_probability,
                risk_class
            )
            SELECT
                d.delivery_id,
                'lightgbm',
                'v1',
                t.risk_probability,
                t.risk_class
            FROM tmp_predictions t
            JOIN raw.deliveries d
              ON d.order_id = t.order_id
             AND d.city = %s
            WHERE NOT EXISTS (
                SELECT 1
                FROM ml.predictions p
                WHERE p.delivery_id = d.delivery_id
                  AND p.model_name = 'lightgbm'
            )
        """, (CITY_NAME,))

        print("Rows inserted:", cur.rowcount)

        conn.commit()

        cur.execute("""
            SELECT COUNT(*)
            FROM ml.predictions
            WHERE model_name = 'lightgbm'
        """)

        total = cur.fetchone()[0]

        print("TOTAL LIGHTGBM PREDICTIONS:", total)

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    print("DONE.")


if __name__ == "__main__":
    main()