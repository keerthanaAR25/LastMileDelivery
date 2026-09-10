from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import psycopg2

from src.models.train_baselines import prepare_xy
from src.decision.recommendation_engine import priority_from_prob


CITIES = {
    "cq": "Chongqing",
    "hz": "Hangzhou",
    "jl": "Jilin",
    "yt": "Yantai",
}

DB_URL = "postgresql://postgres:NexusFlow2026@localhost:5432/nexusflow_db"


def generate_city_predictions(city_code: str, city_name: str, conn):
    feature_path = Path(f"data/features/features_{city_code}.parquet")
    model_path = Path(
        f"models/final/nexusflow_lightgbm_{city_code}_v1.joblib"
    )
    prediction_file = Path(
        f"data/processed/recovered_predictions_{city_code}.csv"
    )

    print(f"\n===== {city_code.upper()} — {city_name} =====")

    print("1. Loading features...")
    feat = pd.read_parquet(feature_path)

    print("2. Loading existing LightGBM model...")
    model = joblib.load(model_path)

    print("3. Preparing test data...")
    X_test, _ = prepare_xy(feat, "test")

    test_meta = feat[feat["split"] == "test"].reset_index(drop=True)

    print("Test rows:", len(test_meta))

    print("4. Generating predictions...")
    probs = model.predict_proba(X_test)[:, 1]

    predictions = pd.DataFrame(
        {
            "order_id": test_meta["order_id"].astype("int64"),
            "risk_probability": probs.astype(float),
            "risk_class": [
                priority_from_prob(float(p))
                for p in probs
            ],
        }
    )

    print("Predictions generated:", len(predictions))

    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_file, index=False)

    print("5. Saved:", prediction_file)

    cur = conn.cursor()

    # Important: remove the temporary table from the previous city
    print("6. Creating temporary table...")
    cur.execute("DROP TABLE IF EXISTS tmp_predictions")

    cur.execute("""
        CREATE TEMP TABLE tmp_predictions (
            order_id BIGINT,
            risk_probability DOUBLE PRECISION,
            risk_class TEXT
        )
    """)

    print("7. Loading predictions into PostgreSQL...")

    with open(prediction_file, "r", encoding="utf-8") as f:
        next(f)

        cur.copy_expert(
            """
            COPY tmp_predictions
            (order_id, risk_probability, risk_class)
            FROM STDIN
            WITH CSV
            """,
            f,
        )

    print("8. Creating temporary index...")

    cur.execute("""
        CREATE INDEX tmp_predictions_order_idx
        ON tmp_predictions(order_id)
    """)

    print("9. Inserting into ml.predictions...")

    cur.execute(
        """
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
        """,
        (city_name,),
    )

    inserted = cur.rowcount

    print("Rows inserted:", inserted)

    conn.commit()

    cur.close()


def main():
    print("NexusFlow — Recovering ML predictions for remaining cities")
    print("Shanghai is intentionally skipped.\n")

    conn = psycopg2.connect(DB_URL)

    try:
        for city_code, city_name in CITIES.items():
            try:
                generate_city_predictions(
                    city_code,
                    city_name,
                    conn,
                )
            except Exception:
                conn.rollback()
                print(
                    f"\nERROR in {city_code.upper()}. "
                    "Stopping safely so the problem can be inspected."
                )
                raise

        cur = conn.cursor()

        print("\n===== FINAL DATABASE CHECK =====")

        cur.execute("""
            SELECT
                d.city,
                COUNT(*) AS predictions
            FROM ml.predictions p
            JOIN raw.deliveries d
              ON d.delivery_id = p.delivery_id
            WHERE p.model_name = 'lightgbm'
            GROUP BY d.city
            ORDER BY d.city
        """)

        for row in cur.fetchall():
            print(row)

        cur.close()

    finally:
        conn.close()

    print("\nDONE.")


if __name__ == "__main__":
    main()