"""
Road network ingestion (Section 9, real file uploaded 2026-09-06).

CRS VERIFICATION (not assumed — Section 91): the raw geometry values are
NOT lat/lng. Transformed a real coordinate (13373499.14, 3538916.29) via
EPSG:3857 -> EPSG:4326 and got (120.1362, 30.2748), which matches
Hangzhou's real location almost exactly (the source row's city column says
杭州市 = Hangzhou) — confirms the source CRS is EPSG:3857 (Web Mercator).

DATA QUALITY NOTE: 524,280 of 531,280 rows (98.7%) have maxspeed=0, which
is OSM's null-sentinel for "unknown", not a real 0 km/h speed limit. Only
7,000 rows (1.3%) have genuine tagged speeds. This confirms maxspeed tags
are NOT a usable congestion signal on their own — consistent with the
spec's original design intent that congestion must come from real
trajectory GPS speeds (Section 23), not static tags.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pyproj import Geod, Transformer
from shapely import wkt
from shapely.ops import transform as shapely_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402

log = get_module_logger("ROAD_NETWORK_INGESTION")

CITY_NAME_MAP = {
    "上海市": ("sh", "Shanghai"),
    "杭州市": ("hz", "Hangzhou"),
    "重庆市": ("cq", "Chongqing"),
    "吉林市": ("jl", "Jilin"),
    "烟台市": ("yt", "Yantai"),
}

_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
_geod = Geod(ellps="WGS84")


def _transform_geom(geom_wkt: str):
    """Parses WKT (EPSG:3857) and reprojects to WGS84 (EPSG:4326)."""
    geom = wkt.loads(geom_wkt)
    return shapely_transform(_transformer.transform, geom)


def _geodesic_length_km(geom_4326) -> float:
    """Real geodesic length via pyproj's C-implemented geometry_length
    (GeographicLib), not a manual point-pair Python loop (which measured
    ~1.26ms/row — too slow for 531K rows in this sandbox)."""
    return _geod.geometry_length(geom_4326) / 1000.0


def load_and_transform(csv_path: str, sample_n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="\t")
    if sample_n:
        df = df.sample(n=sample_n, random_state=42)

    df["city_code"] = df["city"].map(lambda c: CITY_NAME_MAP.get(c, (None, None))[0])
    df["city_name_en"] = df["city"].map(lambda c: CITY_NAME_MAP.get(c, (None, None))[1])
    unmapped = df["city_code"].isna().sum()
    if unmapped:
        log.warning(f"{unmapped} rows have unrecognized city names — dropping (not silently keeping)")
    df = df[df["city_code"].notna()].copy()

    df["road_id"] = "road_" + df["osm_id"].astype(str)
    df["is_maxspeed_unknown"] = df["maxspeed"] == 0

    geoms_4326 = df["geometry"].apply(_transform_geom)
    df["length_km"] = geoms_4326.apply(_geodesic_length_km)
    df["geometry_wkt_4326"] = geoms_4326.apply(lambda g: g.wkt)

    return df


def main(csv_path: str):
    """csv_path has no sandbox-specific default — pass your own path to
    roads.csv explicitly, e.g.:
        python -m src.preprocessing.road_network_ingestion /path/to/roads.csv
    """
    with ModuleRun(log, module="ROAD NETWORK INGESTION") as run:
        df = load_and_transform(csv_path)

        out_dir = Path("data/raw/road_network")
        out_dir.mkdir(parents=True, exist_ok=True)
        for city_code, group in df.groupby("city_code"):
            group.to_parquet(out_dir / f"roads_{city_code}.parquet", index=False)

        run.record(
            total_rows=len(df),
            by_city=df["city_code"].value_counts().to_dict(),
            pct_maxspeed_unknown=round(100 * df["is_maxspeed_unknown"].mean(), 1),
            total_length_km=round(df["length_km"].sum(), 1),
            artifact_location=str(out_dir),
        )
        return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to roads.csv")
    args = parser.parse_args()
    result = main(args.csv_path)
    print(result.groupby("city_code")["length_km"].agg(["count", "sum", "mean"]))
