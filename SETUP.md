# NexusFlow — Setup Guide

This is the complete, step-by-step path from an unzipped repo to a running
system. It reflects exactly what was built and tested during this project
— nothing here is aspirational. If a step wasn't validated in the build
environment (Docker, the trajectory step), that's stated explicitly.

Before you start, read `reports/COMPLETION_REPORT.md` once — it tells you
honestly what's COMPLETE, what's PARTIAL (real but scoped down, reason
named), and what's NOT DONE. This guide gets you running the COMPLETE and
PARTIAL parts; it won't make the NOT DONE parts appear.

---

## Prerequisites

- Python 3.11+ (3.12 was used throughout this build)
- PostgreSQL 16 + PostGIS 3.4 (or Docker, for the DB only)
- ~8GB free RAM recommended if you also want to process the trajectory
  file (Step 6) — the delivery/road-network pipeline itself (Steps 1-5,
  7-11) comfortably fits in far less
- Your own copies of the real source files: 5 LaDe-D city parquet files
  (`delivery_sh`, `delivery_cq`, `delivery_hz`, `delivery_jl`,
  `delivery_yt`), `roads.csv`, and `courier_detailed_trajectory_20s.pkl.xz`

---

## Step 1 — Unzip and install dependencies

```bash
unzip NexusFlow.zip && cd NexusFlow
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or with conda: `conda env create -f environment.yml`

---

## Step 2 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` with real values, e.g.:

```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/nexusflow_db
```

Never commit the real `.env` — only `.env.example` is tracked.

---

## Step 3 — Database

### Option A — Docker

```bash
docker compose up -d postgres
```

**Note:** this was written and syntax-validated but never actually run in
the build sandbox (no Docker daemon there). Validate `docker compose up`
on your own machine before relying on it.

### Option B — local PostgreSQL install

```bash
createdb nexusflow_db
psql "$DATABASE_URL" -f database/init.sql
psql "$DATABASE_URL" -f database/schema.sql
psql "$DATABASE_URL" -f database/indexes.sql
```

### Verify

```bash
psql "$DATABASE_URL" -c "SELECT PostGIS_Version();"
```

This DDL was validated for real during the build: 17 tables, 53 indexes,
FK constraints and PostGIS `ST_Distance`/`ST_DWithin` queries all confirmed
against a live instance.

---

## Step 4 — Place your real delivery data

```bash
mkdir -p data/raw/delivery
cp /path/to/delivery_sh*.parquet data/raw/delivery/delivery_sh.parquet
cp /path/to/delivery_cq*.parquet data/raw/delivery/delivery_cq.parquet
cp /path/to/delivery_hz*.parquet data/raw/delivery/delivery_hz.parquet
cp /path/to/delivery_jl*.parquet data/raw/delivery/delivery_jl.parquet
cp /path/to/delivery_yt*.parquet data/raw/delivery/delivery_yt.parquet
```

The confirmed real schema for these files (17 columns, verified column
names, known data-quality issues like the `ds`/`accept_time` mismatch in
Chongqing) is documented in `data/raw/README.md` — read it before writing
any code against a different copy of this dataset, since field names can
look similar but differ (e.g. this project's real files use `order_id`,
not `package_id`).

---

## Step 5 — Module 2: preprocessing + audit, per city

```bash
for city in sh cq hz jl yt; do
  python -m src.preprocessing.run_module2 \
    --input data/raw/delivery/delivery_${city}.parquet --city ${city}
done
```

Check output:
- `data/processed/deliveries_processed_{city}.parquet`
- `reports/data_report/data_quality_report_{city}.csv`

---

## Step 6 — Road network + trajectory

### Road network (fast, runs anywhere)

```bash
python -m src.preprocessing.road_network_ingestion /path/to/roads.csv
```

Real, verified behavior: detects the source CRS is EPSG:3857 (not
lat/lng), reprojects to EPSG:4326, computes real geodesic segment lengths,
and writes `data/raw/road_network/roads_{city}.parquet` per city.

### Trajectory (needs a machine with more RAM)

The trajectory file (~2.18GB decompressed) could not be loaded in the
build sandbox — confirmed by directly monitoring peak memory at 3.77GB,
against a 3.9GB ceiling. Run this step on a machine with more headroom:

```bash
python process_trajectory_locally.py \
  --trajectory /path/to/courier_detailed_trajectory_20s.pkl.xz \
  --roads /path/to/roads.csv \
  --output congestion_by_road_hour.parquet
```

It prints the trajectory file's actual column names on load. If they
differ from the documented `courier_id`/`gps_time`/`lat`/`lng`, rerun with
`--courier-col` / `--time-col` / `--lat-col` / `--lon-col` overrides — do
not assume the names match without checking the printed output first.

If your machine also has less than ~8GB RAM, use `--sample-frac 0.3` (or
lower) to subsample before processing.

The output is small (a few hundred thousand `(road_id, hour)` rows, not
millions) — bring it back and it can be loaded directly into
`raw.road_segments.congestion_index` / `analytics.spatial_risk`, finally
filling in the congestion component that is NULL everywhere without it.

---

## Step 7 — Run the pipeline (Modules 3-8)

```bash
# quick smoke test on a real (not synthetic) 5,000-row sample:
python run_pipeline.py --mode sample --city sh

# the full run, per city, exactly as done during the build:
python run_pipeline.py --mode full --city sh
python run_pipeline.py --mode full --city all   # all 5 cities
```

Real per-city runtimes from the build (1 CPU / 3.9GB sandbox — expect
faster on typical hardware): Jilin ~18s, Yantai ~114s, Chongqing and
Shanghai several minutes each, Hangzhou (largest, 1.86M rows) longest.
If a run hits your machine's own time/memory limits, use
`--skip-modules` to split it into stages, e.g.:

```bash
python run_pipeline.py --mode full --city hz --skip-modules 6,7,8
python run_pipeline.py --mode full --city hz --skip-modules 3,4,5
```

(Module 6 depends on nothing from 3-5 directly reading from disk, so
Modules 3-5 and 6-8 can be run as two separate passes if needed — see
`run_city_pipeline.py` for exactly what each module step needs.)

---

## Step 8 — Load results into the database

```bash
for city in sh cq hz jl yt; do
  python load_city_to_db.py --city ${city}
done
```

Verify real counts landed:

```bash
psql "$DATABASE_URL" -c "
SELECT 'patterns', count(*) FROM analytics.sequential_patterns
UNION ALL SELECT 'zones', count(*) FROM analytics.zones
UNION ALL SELECT 'predictions', count(*) FROM ml.predictions
UNION ALL SELECT 'recommendations', count(*) FROM decision.recommendations;
"
```

---

## Step 9 — Experiments

```bash
python run_experiment_ablation.py --city sh        # Section 66 (Models A-E)
python run_experiment_cross_city.py                 # Section 68 (SH+HZ -> CQ)
```

Results land in `reports/experiment_results/`.

---

## Step 10 — Run the tests

```bash
pytest tests/ -v
```

Expect **52 passed**. The 9 database tests (`tests/test_database.py`)
need Postgres running (Step 3) — they'll skip cleanly if it isn't.

---

## Step 11 — Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`. Use the city selector in the sidebar — all
5 cities show real data; a city only shows the full pattern/risk/decision
pages once you've run Steps 7-8 for it (before that, it shows real raw
delivery stats with an honest "not yet processed" notice, not fabricated
numbers).

---

## Where to look when something doesn't match

| Question | Look here |
|---|---|
| Is X actually done, or scoped down, or missing? | `reports/COMPLETION_REPORT.md` |
| What real bugs were found and how were they fixed? | `reports/final_validation_report.md` |
| What's the real, confirmed schema of the source data? | `data/raw/README.md` |
| What does a specific module function actually do? | docstring at the top of the relevant file in `src/` — every one explains its real scope and any documented limitation inline |

## Quick end-to-end sanity check

If you just want to confirm the whole thing wires together without
running the full multi-hour pipeline:

```bash
python run_pipeline.py --mode sample --city jl   # smallest city, ~10-20s
python load_city_to_db.py --city jl
pytest tests/ -v
streamlit run app/streamlit_app.py
```
