import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.spatial.grid import assign_grid_cell, flag_geo_outliers, CITY_BBOX
from src.spatial.hotspots import density_hotspots_grid, risk_hotspots_grid
from src.decision.recommendation_engine import priority_from_prob, find_reassignment_candidate, apply_rules


class TestGrid:
    def test_geo_outlier_detection_shanghai(self):
        df = pd.DataFrame({"lng": [121.4, 102.0], "lat": [31.2, 26.7]})
        out = flag_geo_outliers(df, "Shanghai")
        assert not out.iloc[0]
        assert out.iloc[1]

    def test_unknown_city_no_outliers_flagged(self):
        df = pd.DataFrame({"lng": [999.0], "lat": [999.0]})
        out = flag_geo_outliers(df, "Atlantis")
        assert not out.any()

    def test_grid_cell_assignment_is_deterministic(self):
        df = pd.DataFrame({"lat": [31.2, 31.2], "lng": [121.4, 121.4]})
        cells = assign_grid_cell(df, cell_size_km=1.0)
        assert cells.iloc[0] == cells.iloc[1]

    def test_grid_cell_differs_for_distant_points(self):
        df = pd.DataFrame({"lat": [31.2, 32.5], "lng": [121.4, 122.5]})
        cells = assign_grid_cell(df, cell_size_km=1.0)
        assert cells.iloc[0] != cells.iloc[1]


class TestHotspots:
    def _grid_stats(self):
        return pd.DataFrame({
            "grid_id": ["a", "b", "c", "d"],
            "delivery_count": [1000, 500, 10, 5],
            "risk_rate": [0.02, 0.05, 0.9, 0.95],
        })

    def test_density_hotspot_favors_high_volume(self):
        hot = density_hotspots_grid(self._grid_stats(), top_pct=0.5)
        assert "a" in hot["grid_id"].values

    def test_risk_hotspot_requires_minimum_volume(self):
        grid = self._grid_stats()
        hot = risk_hotspots_grid(grid, min_deliveries=30)
        assert "c" not in hot["grid_id"].values
        assert "d" not in hot["grid_id"].values

    def test_density_and_risk_hotspots_can_differ(self):
        grid = self._grid_stats()
        density = set(density_hotspots_grid(grid, top_pct=0.5)["grid_id"])
        risk = set(risk_hotspots_grid(grid, min_deliveries=1)["grid_id"])
        assert density != risk


class TestDecisionEngine:
    def test_priority_bucketing(self):
        assert priority_from_prob(0.9) == "CRITICAL"
        assert priority_from_prob(0.6) == "HIGH"
        assert priority_from_prob(0.3) == "MEDIUM"
        assert priority_from_prob(0.1) == "LOW"

    def test_priority_boundary_values(self):
        assert priority_from_prob(0.75) == "CRITICAL"
        assert priority_from_prob(0.5) == "HIGH"
        assert priority_from_prob(0.25) == "MEDIUM"
        assert priority_from_prob(0.2499) == "LOW"

    def test_reassignment_candidate_found(self):
        lookup = pd.DataFrame({
            "courier_id": [1, 2, 3],
            "courier_workload_train": [1000, 200, 900],
            "courier_risk_rate_train": [0.3, 0.05, 0.25],
        })
        candidate = find_reassignment_candidate(lookup, current_courier_id=1, current_workload=1000)
        assert candidate is not None
        assert candidate["courier_id"] == 2

    def test_no_candidate_when_none_qualify(self):
        lookup = pd.DataFrame({
            "courier_id": [1, 2],
            "courier_workload_train": [1000, 950],
            "courier_risk_rate_train": [0.3, 0.1],
        })
        candidate = find_reassignment_candidate(lookup, current_courier_id=1, current_workload=1000)
        assert candidate is None

    def test_apply_rules_no_action_for_low_risk(self):
        row = pd.Series({"courier_id": 1, "courier_workload_train": 500, "accept_hour": 14})
        lookup = pd.DataFrame({"courier_id": [1, 2], "courier_workload_train": [500, 100],
                                "courier_risk_rate_train": [0.1, 0.05]})
        result = apply_rules(row, risk_prob=0.1, courier_lookup=lookup)
        assert result["recommended_action"] == "NO_ACTION"

    def test_apply_rules_never_emits_reroute(self):
        lookup = pd.DataFrame({"courier_id": [1, 2], "courier_workload_train": [1000, 100],
                                "courier_risk_rate_train": [0.3, 0.05]})
        for prob in [0.1, 0.4, 0.6, 0.8, 0.95]:
            for hour in [2, 8, 14, 20]:
                row = pd.Series({"courier_id": 1, "courier_workload_train": 1000,
                                  "courier_risk_rate_train": 0.3, "accept_hour": hour})
                result = apply_rules(row, risk_prob=prob, courier_lookup=lookup)
                assert result["recommended_action"] != "REROUTE"
                assert result["recommended_action"] != "AVOID_HIGH_RISK_ROAD"
