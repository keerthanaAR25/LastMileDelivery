from pathlib import Path
import argparse

import pandas as pd
import psycopg2
import psycopg2.extras
from pyarrow import parquet as pq


CITY_NAMES = {
    "sh": "Shanghai",
    "cq": "Chongqing",
    "hz": "Hangzhou",
    "jl": "Jilin",
    "yt": "Yantai",
}


def load_city(city_code):
    if city_code not in CITY_NAMES:
        raise ValueError(f"Unknown city: {city_code}")

    city_name = CITY_NAMES[city_code]

    path = Path(
        f"data/processed/deliveries_processed_{city_code}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"\nLoading {city_name}")
    print(f"File: {path}")

    conn = psycopg2.connect(
        "postgresql://postgres:NexusFlow2026@localhost:5432/nexusflow_db"
    )

    parquet_file = pq.ParquetFile(path)

    total = 0

    insert_sql = """
        INSERT INTO raw.deliveries (
            order_id,
            region_id,
            city,
            courier_id,
            lng,
            lat,
            aoi_id,
            aoi_type,
            accept_time_raw,
            accept_gps_time_raw,
            accept_gps_lng,
            accept_gps_lat,
            delivery_time_raw,
            delivery_gps_time_raw,
            delivery_gps_lng,
            delivery_gps_lat,
            ds,
            assumed_year,
            accept_time,
            delivery_time,
            delivery_duration_min,
            is_malformed_ds,
            risk_label,
            geometry_point,
            geometry_accept_gps,
            geometry_delivery_gps,
            source_file
        )
        VALUES %s
    """

    template = """
        (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,

            CASE
                WHEN %s IS NOT NULL AND %s IS NOT NULL
                THEN ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                ELSE NULL
            END,

            CASE
                WHEN %s IS NOT NULL AND %s IS NOT NULL
                THEN ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                ELSE NULL
            END,

            CASE
                WHEN %s IS NOT NULL AND %s IS NOT NULL
                THEN ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                ELSE NULL
            END,

            %s
        )
    """

    try:
        for batch_no, batch in enumerate(
            parquet_file.iter_batches(batch_size=10000),
            start=1
        ):
            df = batch.to_pandas()

            rows = []

            for r in df.itertuples(index=False):

                # Delivery location
                lng = float(r.lng) if pd.notna(r.lng) else None
                lat = float(r.lat) if pd.notna(r.lat) else None

                # Accept GPS
                accept_lng = (
                    float(r.accept_gps_lng)
                    if pd.notna(r.accept_gps_lng)
                    else None
                )

                accept_lat = (
                    float(r.accept_gps_lat)
                    if pd.notna(r.accept_gps_lat)
                    else None
                )

                # Delivery GPS
                delivery_lng = (
                    float(r.delivery_gps_lng)
                    if pd.notna(r.delivery_gps_lng)
                    else None
                )

                delivery_lat = (
                    float(r.delivery_gps_lat)
                    if pd.notna(r.delivery_gps_lat)
                    else None
                )

                rows.append(
                    (
                        int(r.order_id),
                        int(r.region_id) if pd.notna(r.region_id) else None,
                        city_name,
                        int(r.courier_id),

                        lng,
                        lat,

                        int(r.aoi_id) if pd.notna(r.aoi_id) else None,
                        int(r.aoi_type) if pd.notna(r.aoi_type) else None,

                        r.accept_time_raw,
                        r.accept_gps_time_raw,

                        accept_lng,
                        accept_lat,

                        r.delivery_time_raw,
                        r.delivery_gps_time_raw,

                        delivery_lng,
                        delivery_lat,

                        int(r.ds) if pd.notna(r.ds) else None,

                        int(r.assumed_year)
                        if pd.notna(r.assumed_year)
                        else None,

                        r.accept_time,
                        r.delivery_time,

                        float(r.delivery_duration_min)
                        if pd.notna(r.delivery_duration_min)
                        else None,

                        bool(r.is_malformed_ds)
                        if pd.notna(r.is_malformed_ds)
                        else False,

                        int(r.risk_label)
                        if pd.notna(r.risk_label)
                        else None,

                        # geometry_point coordinates
                        lng,
                        lat,
                        lng,
                        lat,

                        # geometry_accept_gps coordinates
                        accept_lng,
                        accept_lat,
                        accept_lng,
                        accept_lat,

                        # geometry_delivery_gps coordinates
                        delivery_lng,
                        delivery_lat,
                        delivery_lng,
                        delivery_lat,

                        str(path),
                    )
                )

            if rows:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(
                        cur,
                        insert_sql,
                        rows,
                        template=template,
                        page_size=2000,
                    )

                conn.commit()

                total += len(rows)

            print(
                f"{city_name}: batch {batch_no} "
                f"→ {total:,} rows loaded"
            )

            del df
            del rows

        print("\n======================================")
        print(f"{city_name} COMPLETE")
        print(f"Rows loaded: {total:,}")
        print("======================================")

    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--city",
        required=True,
        choices=["sh", "cq", "hz", "jl", "yt"],
    )

    args = parser.parse_args()

    load_city(args.city)