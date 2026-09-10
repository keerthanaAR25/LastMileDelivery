from pathlib import Path

import pandas as pd
import psycopg2


DB_URL = "postgresql://postgres:NexusFlow2026@localhost:5432/nexusflow_db"

CITIES = {
    "cq": "Chongqing",
    "hz": "Hangzhou",
    "jl": "Jilin",
    "sh": "Shanghai",
    "yt": "Yantai",
}

BASE = Path(__file__).resolve().parents[1]


def main():

    conn = psycopg2.connect(DB_URL)

    try:
        cur = conn.cursor()

        for code, city in CITIES.items():

            print("\n" + "=" * 60)
            print(f"{code.upper()} - {city}")
            print("=" * 60)

            # ======================================================
            # 1. SHAP
            # ======================================================

            shap_file = (
                BASE
                / "artifacts"
                / "explanations"
                / f"shap_local_examples_{code}.csv"
            )

            if shap_file.exists():

                shap = pd.read_csv(shap_file)

                print("SHAP rows:", len(shap))
                print("SHAP columns:", shap.columns.tolist())

                inserted = 0
                skipped = 0

                required = {
                    "feature",
                    "feature_value",
                    "shap_value",
                    "direction",
                    "importance_rank",
                    "order_id",
                }

                if not required.issubset(shap.columns):

                    print(
                        "SHAP skipped: required columns are missing."
                    )

                else:

                    for _, r in shap.iterrows():

                        # Find the prediction belonging to this order
                        cur.execute(
                            """
                            SELECT p.prediction_id
                            FROM ml.predictions p
                            JOIN raw.deliveries d
                                ON d.delivery_id = p.delivery_id
                            WHERE d.order_id = %s
                              AND d.city = %s
                              AND p.model_name = 'lightgbm'
                            ORDER BY p.prediction_id DESC
                            LIMIT 1
                            """,
                            (
                                int(r["order_id"]),
                                city,
                            ),
                        )

                        prediction = cur.fetchone()

                        if prediction is None:

                            skipped += 1
                            continue

                        prediction_id = prediction[0]

                        # Insert SHAP explanation
                        cur.execute(
                            """
                            INSERT INTO ml.shap_explanations
                            (
                                prediction_id,
                                feature_name,
                                feature_value,
                                shap_value,
                                direction,
                                importance_rank
                            )
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (
                                prediction_id,
                                str(r["feature"]),
                                float(r["feature_value"]),
                                float(r["shap_value"]),
                                str(r["direction"]),
                                int(r["importance_rank"]),
                            ),
                        )

                        inserted += cur.rowcount

                    print("SHAP inserted:", inserted)
                    print("SHAP skipped:", skipped)

            else:

                print("SHAP file not found:", shap_file)

            # ======================================================
            # 2. RECOMMENDATIONS
            # ======================================================

            reco_file = (
                BASE
                / "artifacts"
                / "recommendations"
                / f"recommendations_{code}.csv"
            )

            if reco_file.exists():

                reco = pd.read_csv(reco_file)

                print("Recommendation rows:", len(reco))
                print(
                    "Recommendation columns:",
                    reco.columns.tolist(),
                )

                inserted = 0
                skipped = 0

                required = {
                    "order_id",
                    "recommended_action",
                    "priority",
                    "reason",
                }

                if not required.issubset(reco.columns):

                    print(
                        "Recommendations skipped: "
                        "required columns are missing."
                    )

                else:

                    for _, r in reco.iterrows():

                        cur.execute(
                            """
                            INSERT INTO decision.recommendations
                            (
                                delivery_id,
                                recommended_action,
                                priority,
                                reason
                            )
                            SELECT
                                d.delivery_id,
                                %s,
                                %s,
                                %s
                            FROM raw.deliveries d
                            WHERE d.order_id = %s
                              AND d.city = %s
                            ON CONFLICT DO NOTHING
                            """,
                            (
                                str(r["recommended_action"]),
                                str(r["priority"]),
                                str(r["reason"]),
                                int(r["order_id"]),
                                city,
                            ),
                        )

                        if cur.rowcount:
                            inserted += 1
                        else:
                            skipped += 1

                    print(
                        "Recommendations inserted:",
                        inserted,
                    )
                    print(
                        "Recommendations skipped:",
                        skipped,
                    )

            else:

                print(
                    "Recommendation file not found:",
                    reco_file,
                )

            # ======================================================
            # 3. SIMULATIONS
            # ======================================================

            sim_file = (
                BASE
                / "artifacts"
                / "recommendations"
                / f"simulations_{code}.csv"
            )

            if sim_file.exists():

                sim = pd.read_csv(sim_file)

                print("Simulation rows:", len(sim))
                print(
                    "Simulation columns:",
                    sim.columns.tolist(),
                )

                inserted = 0
                skipped = 0

                required = {
                    "baseline_risk",
                    "alternative_risk",
                    "risk_delta",
                    "simulation_kind",
                    "order_id",
                    "label",
                }

                if not required.issubset(sim.columns):

                    print(
                        "Simulations skipped: "
                        "required columns are missing."
                    )

                elif code == "sh":

                    print(
                        "Shanghai simulations already loaded - skipping."
                    )

                else:

                    for _, r in sim.iterrows():

                        cur.execute(
                            """
                            SELECT d.delivery_id
                            FROM raw.deliveries d
                            WHERE d.order_id = %s
                              AND d.city = %s
                            LIMIT 1
                            """,
                            (
                                int(r["order_id"]),
                                city,
                            ),
                        )

                        delivery = cur.fetchone()

                        if delivery is None:

                            skipped += 1
                            continue

                        delivery_id = delivery[0]

                        cur.execute(
                            """
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
                                distance_delta,
                                duration_delta,
                                recommended_action,
                                simulation_timestamp,
                                label
                            )
                            VALUES
                            (
                                %s,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                %s,
                                %s,
                                %s,
                                NULL,
                                NULL,
                                %s,
                                NOW(),
                                %s
                            )
                            """,
                            (
                                delivery_id,
                                float(r["baseline_risk"]),
                                float(r["alternative_risk"]),
                                float(r["risk_delta"]),
                                str(r["simulation_kind"]),
                                str(r["label"]),
                            ),
                        )

                        inserted += cur.rowcount

                    print(
                        "Simulations inserted:",
                        inserted,
                    )
                    print(
                        "Simulations skipped:",
                        skipped,
                    )

            else:

                print(
                    "Simulation file not found:",
                    sim_file,
                )

            # Commit this city
            conn.commit()

        # ==========================================================
        # FINAL TOTALS
        # ==========================================================

        print("\n" + "=" * 60)
        print("FINAL DATABASE TOTALS")
        print("=" * 60)

        cur.execute(
            "SELECT COUNT(*) FROM ml.shap_explanations"
        )
        print("SHAP:", cur.fetchone()[0])

        cur.execute(
            "SELECT COUNT(*) FROM decision.recommendations"
        )
        print(
            "RECOMMENDATIONS:",
            cur.fetchone()[0],
        )

        cur.execute(
            "SELECT COUNT(*) FROM decision.route_simulations"
        )
        print(
            "SIMULATIONS:",
            cur.fetchone()[0],
        )

        # ==========================================================
        # CITY-WISE SHAP
        # ==========================================================

        cur.execute(
            """
            SELECT
                d.city,
                COUNT(*)
            FROM ml.shap_explanations s
            JOIN ml.predictions p
                ON p.prediction_id = s.prediction_id
            JOIN raw.deliveries d
                ON d.delivery_id = p.delivery_id
            GROUP BY d.city
            ORDER BY d.city
            """
        )

        print("\nSHAP BY CITY:")
        print(cur.fetchall())

        # ==========================================================
        # CITY-WISE RECOMMENDATIONS
        # ==========================================================

        cur.execute(
            """
            SELECT
                d.city,
                COUNT(*)
            FROM decision.recommendations r
            JOIN raw.deliveries d
                ON d.delivery_id = r.delivery_id
            GROUP BY d.city
            ORDER BY d.city
            """
        )

        print("\nRECOMMENDATIONS BY CITY:")
        print(cur.fetchall())

        # ==========================================================
        # CITY-WISE SIMULATIONS
        # ==========================================================

        cur.execute(
            """
            SELECT
                d.city,
                COUNT(*)
            FROM decision.route_simulations s
            JOIN raw.deliveries d
                ON d.delivery_id = s.delivery_id
            GROUP BY d.city
            ORDER BY d.city
            """
        )

        print("\nSIMULATIONS BY CITY:")
        print(cur.fetchall())

        cur.close()

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()


if __name__ == "__main__":
    main()