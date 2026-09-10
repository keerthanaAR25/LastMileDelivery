# NexusFlow

**Intelligent Sequential–Spatial Pattern Discovery for Last-Mile Delivery Decision Support**

NexusFlow is an integrated pattern-to-decision intelligence framework for
last-mile delivery. It combines sequential pattern mining, spatial-temporal
analytics, logistics graph modeling, machine-learning risk prediction,
explainable AI, and decision support to identify recurring delivery-risk
patterns and recommend operational interventions.

```
DATA → PREPROCESSING → SEQUENTIAL MINING → SPATIAL-TEMPORAL MINING
     → GRAPH MODEL → ML → XAI → DECISION SUPPORT → ROUTE SIMULATION
     → DATABASE → SERVICE LAYER → DASHBOARD
```

**Before anything else, read `reports/COMPLETION_REPORT.md`.** It states
plainly what is COMPLETE (executed and validated), PARTIAL (real but
intentionally scoped down, with the reason named), and NOT DONE. This
project follows the rule that nothing is called "100% working" unless it
actually ran — that report is the honest source of truth, not this README.

**For full step-by-step setup, see `SETUP.md`.** This README covers the
short version; `SETUP.md` has every command in order, with real timings,
troubleshooting notes, and exactly what to check at each step.

## What's real vs. what needs your machine

This was built inside a sandboxed environment with **1 CPU / 3.9GB RAM** and
no internet access beyond package registries. That shaped two real,
disclosed gaps:

1. **Trajectory data** (`courier_detailed_trajectory_20s.pkl.xz`, 2.18GB
   decompressed) cannot be loaded here — confirmed by directly monitoring
   peak memory (3.77GB, right at the sandbox's ceiling) across multiple
   attempts. Run `process_trajectory_locally.py` on a machine with more
   RAM (6-8GB recommended), then bring the small output file back.
2. **Docker** was written and never run (no Docker daemon in the sandbox).
   `docker compose up` needs to be validated on your machine.

Everything else — 4,514,661 real delivery records across 5 cities, 531,280
real road segments, all 9 modules, the database, the dashboard, the test
suite — was actually executed and is verifiable by running the commands
below yourself.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DB credentials
```

### 1. Database

```bash
docker compose up -d postgres
# or, without Docker, against a local PostgreSQL 16 + PostGIS 3.4:
psql "$DATABASE_URL" -f database/init.sql
psql "$DATABASE_URL" -f database/schema.sql
psql "$DATABASE_URL" -f database/indexes.sql
```

### 2. Data

The real LaDe-D dataset (5 cities) and `roads.csv` must be placed under
`data/raw/delivery/` and provided to `src/preprocessing/road_network_ingestion.py`
respectively — see `data/raw/README.md` for exactly which columns are real
vs. assumed, and how the CRS was verified (not assumed).

```bash
# Module 2: preprocessing + audit, per city
python -m src.preprocessing.run_module2 --input data/raw/delivery/delivery_sh.parquet --city sh

# Road network (real CRS-verified ingestion)
python -m src.preprocessing.road_network_ingestion
```

### 3. Full pipeline (Modules 3-8)

```bash
# Quick, real (not synthetic) sample for development:
python run_pipeline.py --mode sample --city sh

# Full run, exactly as done for all 5 cities in this project:
python run_pipeline.py --mode full --city sh
python run_pipeline.py --mode full --city all   # all 5 cities, will take a while — see COMPLETION_REPORT for real per-city timings
```

Then load results into the database:

```bash
python load_city_to_db.py --city sh
```

### 4. Experiments

```bash
python run_experiment_ablation.py --city sh       # Section 66
python run_experiment_cross_city.py                # Section 68 (SH+HZ -> CQ)
```

### 5. Dashboard

```bash
streamlit run app/streamlit_app.py
# or: docker compose up -d
```

### 6. Tests

```bash
pytest tests/ -v
```

52 real tests (preprocessing, sequential mining incl. a live PrefixSpan-vs-
SPADE cross-validation, spatial/hotspot logic, decision-engine rules, and 9
database integration tests against a live Postgres instance). All 52 pass
as of this build — rerun them yourself to confirm, don't take it on faith.

## Project structure

- `src/` — all pipeline logic, one subpackage per module.
- `run_pipeline.py` — top-level orchestrator (`--mode sample|full`).
- `run_city_pipeline.py` — the actual Module 3-8 implementations, called by `run_pipeline.py`.
- `load_city_to_db.py` — loads a city's artifacts into Postgres.
- `run_experiment_*.py` — Experiments D and F.
- `process_trajectory_locally.py` — run this on your own machine (see above).
- `database/` — DDL (`init.sql`, `schema.sql`, `indexes.sql`), all validated against a live instance.
- `app/` — Streamlit dashboard + service layer (`app/services/data_access.py` — the dashboard never queries the DB directly).
- `reports/COMPLETION_REPORT.md` — the honest, module-by-module status.
- `reports/final_validation_report.md` — the full build log, including every real bug found and fixed along the way.
- `tests/` — the real pytest suite described above.

## Academic honesty

This system never reports fabricated accuracy, fabricated delay-reduction
percentages, or invented dataset fields. Every derived value in the
database is tagged with its provenance (RAW / DERIVED / PREDICTED /
SIMULATED / RECOMMENDED). Route-simulation results are explicitly labelled
`SIMULATED_INTERVENTION_IMPACT` and are never presented as realized
operational savings. Where a component is genuinely incomplete (trajectory
data, OR-Tools route optimization, notebooks, FastAPI), it is named
specifically in `reports/COMPLETION_REPORT.md` rather than glossed over.
