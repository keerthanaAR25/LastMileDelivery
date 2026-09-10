"""
Module 1 — Multi-Source Data Acquisition (Section 17).

Downloads the real Cainiao LaDe-D / LaDe-P dataset from Hugging Face.

CONFIRMED (via the actual Hugging Face dataset cards, not assumed):
  - Repo:   Cainiao-AI/LaDe   (also split as Cainiao-AI/LaDe-D, Cainiao-AI/LaDe-P)
  - Files under it include, per city (sh/hz/cq/jl/yt):
        delivery/delivery_<city>.csv  (or data/delivery_<city>-*.parquet)
        pickup/pickup_<city>.csv
        road-network/<city>/...
        data_with_trajectory_20s/courier_detailed_trajectory_20s.pkl.xz
  - Confirmed LaDe-D fields: package_id, lng, lat, city, region_id, aoi_id,
    aoi_type, courier_id, accept_time, accept_gps_time, accept_gps_lng/lat,
    delivery_time, delivery_gps_time(implied), ds.
  - Confirmed trajectory fields: ds, courier_id (labelled postman_id in one
    sample), gps_time, lat, lng.

KNOWN ENVIRONMENT CONSTRAINT: this script requires network access to
huggingface.co. In a network-restricted sandbox (no huggingface.co in the
egress allowlist) this will fail with a clear, honest error rather than
silently producing empty/fake data — see `_check_network` below.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CONFIG
from src.logging_config import get_module_logger, ModuleRun

log = get_module_logger("MODULE_01_ACQUISITION")

REPO_ID = "Cainiao-AI/LaDe"
CITY_CODES = {"sh": "Shanghai", "hz": "Hangzhou", "cq": "Chongqing", "jl": "Jilin", "yt": "Yantai"}


def _check_network() -> bool:
    """Real connectivity probe — never assume, always test."""
    import urllib.request

    try:
        urllib.request.urlopen("https://huggingface.co", timeout=8)
        return True
    except Exception as e:
        log.warning(f"huggingface.co unreachable from this environment: {e}")
        return False


def download_city(city_code: str, raw_dir: Path) -> dict:
    """Download one city's delivery + pickup CSVs via huggingface_hub.

    Returns a dict describing what actually happened (never fabricated).
    """
    if city_code not in CITY_CODES:
        raise ValueError(f"Unknown city code '{city_code}'. Known: {list(CITY_CODES)}")

    result = {"city": city_code, "delivery_file": None, "pickup_file": None, "status": "NOT_ATTEMPTED"}

    if not _check_network():
        result["status"] = "FAILED_NO_NETWORK"
        return result

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.error("huggingface_hub is not installed. `pip install huggingface_hub`.")
        result["status"] = "FAILED_MISSING_DEPENDENCY"
        return result

    try:
        delivery_path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
            filename=f"delivery/delivery_{city_code}.csv",
            local_dir=str(raw_dir / "delivery"),
        )
        result["delivery_file"] = delivery_path
    except Exception as e:
        log.warning(f"Could not fetch delivery_{city_code}.csv: {e}")

    try:
        pickup_path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
            filename=f"pickup/pickup_{city_code}.csv",
            local_dir=str(raw_dir / "pickup"),
        )
        result["pickup_file"] = pickup_path
    except Exception as e:
        log.warning(f"Could not fetch pickup_{city_code}.csv: {e}")

    result["status"] = "OK" if (result["delivery_file"] or result["pickup_file"]) else "FAILED"
    return result


def main(cities: list[str] | None = None) -> list[dict]:
    raw_dir = CONFIG.path(CONFIG.get("dataset", "raw_dir", default="data/raw"))
    cities = cities or CONFIG.get("dataset", "cities", "develop", default=["sh"])

    with ModuleRun(log, module="MODULE 01") as run:
        results = [download_city(c, raw_dir) for c in cities]
        ok = sum(1 for r in results if r["status"] == "OK")
        run.record(
            cities_attempted=len(results),
            cities_succeeded=ok,
            artifact_location=str(raw_dir),
        )
        for r in results:
            log.info(f"  city={r['city']} status={r['status']}")
        return results


if __name__ == "__main__":
    main()
