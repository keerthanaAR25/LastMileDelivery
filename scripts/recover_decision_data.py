import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import psycopg2
import psycopg2.extras


DB = "postgresql://postgres:NexusFlow2026@localhost:5432/nexusflow_db"
CITY = "Shanghai"


def main():
    conn = psycopg2.connect(DB)

    try:
        # ---------------- SHAP ----------------
        shap_path = Path("artifacts/explanations/shap_local_examples_sh.csv")
        shap = pd.read_csv(shap_path)

        print("SHAP rows:", len(shap))

        with conn.cursor() as cur:
            rows = [
                (
                    int(r.order_id),
                    r.feature,
                    float(r.feature_value),
                    float(r.shap_value),
                    r.direction,
                    int(r.importance_rank),
                )
                for r in shap.itertuples()
            ]

            cur.execute("""
                CREATE TEMP TABLE tmp_shap (
                    order_id BIGINT,
                    feature_name TEXT,
                    feature_value FLOAT,
                    shap_value FLOAT,
                    direction TEXT,
                    importance_rank INT
                )
            """)

            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO tmp_shap VALUES %s",
                rows,
                page_size=1000,
            )

            cur.execute("""
                INSERT INTO ml.shap_explanations
                (
                    prediction_id,
                    feature_name,
                    feature_value,
                    shap_value,
                    direction,
                    importance_rank
                )
                SELECT
                    p.prediction_id,
                    t.feature_name,
                    t.feature_value,
                    t.shap_value,
                    t.direction,
                    t.importance_rank
                FROM tmp_shap t
                JOIN raw.deliveries d
                  ON d.order_id = t.order_id
                 AND d.city = %s
                JOIN ml.predictions p
                  ON p.delivery_id = d.delivery_id
                 AND p.model_name = 'lightgbm'
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM ml.shap_explanations s
                    WHERE s.prediction_id = p.prediction_id
                      AND s.feature_name = t.feature_name
                )
            """, (CITY,))

        conn.commit()

        # ---------------- RECOMMENDATIONS ----------------
        reco_path = Path(
            "artifacts/recommendations/recommendations_sh.csv"
        )
        recos = pd.read_csv(reco_path)

        print("Recommendation rows:", len(recos))

        with conn.cursor() as cur:
            rows = [
                (
                    int(r.order_id),
                    float(r.risk_probability),
                    r.risk_class,
                    None,
                    r.recommended_action,
                    r.reason,
                    r.priority,
                )
                for r in recos.itertuples()
            ]

            cur.execute("""
                CREATE TEMP TABLE tmp_reco (
                    order_id BIGINT,
                    risk_probability FLOAT,
                    risk_class TEXT,
                    detected_pattern TEXT,
                    recommended_action TEXT,
                    reason TEXT,
                    priority TEXT
                )
            """)

            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO tmp_reco VALUES %s",
                rows,
                page_size=2000,
            )

            cur.execute("""
                INSERT INTO decision.recommendations
                (
                    delivery_id,
                    risk_probability,
                    risk_class,
                    detected_pattern,
                    recommended_action,
                    reason,
                    priority
                )
                SELECT
                    d.delivery_id,
                    t.risk_probability,
                    t.risk_class,
                    t.detected_pattern,
                    t.recommended_action,
                    t.reason,
                    t.priority
                FROM tmp_reco t
                JOIN raw.deliveries d
                  ON d.order_id = t.order_id
                 AND d.city = %s
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM decision.recommendations r
                    WHERE r.delivery_id = d.delivery_id
                )
            """, (CITY,))

        conn.commit()

        # ---------------- SIMULATIONS ----------------
        sim_path = Path(
            "artifacts/recommendations/simulations_sh.csv"
        )
        sims = pd.read_csv(sim_path)

        print("Simulation rows:", len(sims))

        if not sims.empty:
            with conn.cursor() as cur:
                rows = [
                    (
                        int(r.order_id),
                        None,
                        None,
                        None,
                        None,
                        float(r.baseline_risk),
                        float(r.alternative_risk),
                        float(r.risk_delta),
                        r.simulation_kind,
                        r.label,
                    )
                    for r in sims.itertuples()
                ]

                cur.execute("""
                    CREATE TEMP TABLE tmp_sim (
                        order_id BIGINT,
                        baseline_distance FLOAT,
                        alternative_distance FLOAT,
                        baseline_duration FLOAT,
                        alternative_duration FLOAT,
                        baseline_risk FLOAT,
                        alternative_risk FLOAT,
                        risk_delta FLOAT,
                        recommended_action TEXT,
                        label TEXT
                    )
                """)

                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO tmp_sim VALUES %s",
                    rows,
                    page_size=2000,
                )

                cur.execute("""
                    INSERT INTO decision.route_simulations
                    (
                        delivery_id,
                        baseline_distance,
                        alternative_distance,
                        baseline_duration,
                        alternative_duration,
                        baseline_risk,
                        alternative_risk,
                        risk_delta,
                        recommended_action,
                        label
                    )
                    SELECT
                        d.delivery_id,
                        t.baseline_distance,
                        t.alternative_distance,
                        t.baseline_duration,
                        t.alternative_duration,
                        t.baseline_risk,
                        t.alternative_risk,
                        t.risk_delta,
                        t.recommended_action,
                        t.label
                    FROM tmp_sim t
                    JOIN raw.deliveries d
                      ON d.order_id = t.order_id
                     AND d.city = %s
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM decision.route_simulations s
                        WHERE s.delivery_id = d.delivery_id
                    )
                """, (CITY,))

            conn.commit()

        # ---------------- VERIFY ----------------
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ml.shap_explanations")
            print("SHAP IN DB:", cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM decision.recommendations")
            print("RECOMMENDATIONS IN DB:", cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM decision.route_simulations")
            print("SIMULATIONS IN DB:", cur.fetchone()[0])

    finally:
        conn.close()

    print("DONE.")


if __name__ == "__main__":
    main()