import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sqlalchemy.engine import make_url

# Load .env
load_dotenv(override=True)

# Read DATABASE_URL from .env
database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise RuntimeError("DATABASE_URL was not found in .env")

print("DATABASE_URL found: YES")

# Convert SQLAlchemy URL to a psycopg2-compatible PostgreSQL URL
url = make_url(database_url)

# Remove SQLAlchemy driver such as +psycopg2
url = url.set(drivername="postgresql")

print("Database:", url.database)
print("Username:", url.username)
print("Host:", url.host)
print("Port:", url.port)
print("Password supplied:", "YES" if url.password else "NO")

if not url.password:
    raise RuntimeError("DATABASE_URL does not contain a password.")

conn = psycopg2.connect(
    host=url.host or "localhost",
    port=url.port or 5432,
    dbname=url.database,
    user=url.username,
    password=url.password
)

conn.autocommit = False
cur = conn.cursor()

files = {
    "sh": "data/raw/road_network/roads_sh.parquet",
    "cq": "data/raw/road_network/roads_cq.parquet",
    "hz": "data/raw/road_network/roads_hz.parquet",
    "jl": "data/raw/road_network/roads_jl.parquet",
    "yt": "data/raw/road_network/roads_yt.parquet",
}

try:
    cur.execute("TRUNCATE TABLE raw.road_segments;")
    conn.commit()

    total = 0

    for city, path in files.items():

        print()
        print("=" * 60)
        print(f"LOADING ROAD NETWORK: {city}")
        print("=" * 60)

        df = pd.read_parquet(path)

        for _, r in df.iterrows():

            road_id = str(r["road_id"])

            road_type = (
                str(r["fclass"])
                if pd.notna(r["fclass"])
                else None
            )

            length_km = (
                float(r["length_km"])
                if pd.notna(r["length_km"])
                else None
            )

            maxspeed = r["maxspeed"]

            historical_speed = (
                float(maxspeed)
                if pd.notna(maxspeed) and float(maxspeed) > 0
                else None
            )

            geometry = r["geometry_wkt_4326"]

            cur.execute(
                """
                INSERT INTO raw.road_segments
                (
                    road_id,
                    city,
                    road_type,
                    length_km,
                    geometry,
                    historical_speed,
                    congestion_index,
                    risk_score,
                    source_file
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    ST_GeomFromText(%s, 4326),
                    %s,
                    0,
                    0,
                    %s
                )
                ON CONFLICT (road_id) DO NOTHING;
                """,
                (
                    road_id,
                    city,
                    road_type,
                    length_km,
                    geometry,
                    historical_speed,
                    path
                )
            )

        conn.commit()

        count = len(df)
        total += count

        print(f"Loaded: {count:,}")
        print(f"Running total: {total:,}")

    print()
    print("=" * 60)
    print("ROAD NETWORK LOAD COMPLETE")
    print("=" * 60)
    print(f"Total processed: {total:,}")

except Exception as e:
    conn.rollback()
    print()
    print("ERROR:", e)
    raise

finally:
    cur.close()
    conn.close()
