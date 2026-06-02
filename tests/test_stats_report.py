"""Tests for src/shared/stats_report.py (US-07)."""
from __future__ import annotations

import math

import pytest

from src.shared.stats_report import (
    aggregate_seeds,
    detect_outliers,
    format_stats_table,
    t_critical,
)


class TestTCritical:
    def test_known_df5(self):
        assert t_critical(6) == pytest.approx(2.571)

    def test_large_n_returns_z(self):
        assert t_critical(100) == pytest.approx(1.96)

    def test_n_one_returns_nan(self):
        assert math.isnan(t_critical(1))


class TestAggregateSeeds:
    def test_empty_returns_empty(self):
        assert aggregate_seeds([]) == {}

    def test_single_seed_passthrough(self):
        result = aggregate_seeds([{"foot_sliding": 0.5, "label": "foo"}])
        assert result["foot_sliding"]["mean"] == pytest.approx(0.5)
        assert math.isnan(result["foot_sliding"]["std"])
        assert result["label"] == "foo"

    def test_multiple_seeds_mean(self):
        rows = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        result = aggregate_seeds(rows)
        assert result["x"]["mean"] == pytest.approx(2.0)

    def test_multiple_seeds_std(self):
        rows = [{"x": 1.0}, {"x": 3.0}]
        result = aggregate_seeds(rows)
        assert result["x"]["std"] == pytest.approx(math.sqrt(2), rel=1e-4)

    def test_ci95_width_positive(self):
        rows = [{"x": float(i)} for i in range(10)]
        result = aggregate_seeds(rows)
        assert result["x"]["ci95_high"] > result["x"]["ci95_low"]

    def test_non_numeric_key_passes_through(self):
        rows = [{"name": "alice"}, {"name": "bob"}]
        result = aggregate_seeds(rows)
        assert result["name"] == "alice"

    def test_n_recorded(self):
        rows = [{"v": float(i)} for i in range(5)]
        result = aggregate_seeds(rows)
        assert result["v"]["n"] == 5

    def test_ci_bounds_bracket_mean(self):
        rows = [{"x": float(i)} for i in range(20)]
        result = aggregate_seeds(rows)
        assert result["x"]["ci95_low"] < result["x"]["mean"] < result["x"]["ci95_high"]


class TestDetectOutliers:
    def test_no_outliers_in_clean_data(self):
        rows = [{"x": float(i)} for i in range(10)]
        assert detect_outliers(rows) == []

    def test_detects_obvious_outlier(self):
        # 19 values tightly clustered near 1.0; one extreme value at 10000.0 — clearly > 3σ
        rows = [{"x": 1.0 + i * 0.001} for i in range(19)] + [{"x": 10000.0}]
        outliers = detect_outliers(rows)
        assert any(key == "x" for _, key in outliers)

    def test_returns_seed_id(self):
        # 19 values near 1.0, one extreme at 10000.0
        rows = [{"x": 1.0 + i * 0.001} for i in range(19)] + [{"x": 10000.0}]
        outliers = detect_outliers(rows, seed_ids=list(range(19)) + [99])
        seed_id_list = [s for s, _ in outliers]
        assert 99 in seed_id_list

    def test_fewer_than_3_returns_empty(self):
        rows = [{"x": 1.0}, {"x": 2.0}]
        assert detect_outliers(rows) == []

    def test_constant_series_no_outliers(self):
        rows = [{"x": 5.0}] * 10
        assert detect_outliers(rows) == []


class TestFormatStatsTable:
    def test_returns_string(self):
        aggregated = {"foot_sliding": {"mean": 0.5, "std": 0.1, "ci95_low": 0.3, "ci95_high": 0.7, "n": 5}}
        out = format_stats_table(aggregated)
        assert isinstance(out, str)

    def test_contains_metric_name(self):
        aggregated = {"my_metric": {"mean": 1.0, "std": 0.0, "ci95_low": 1.0, "ci95_high": 1.0, "n": 3}}
        out = format_stats_table(aggregated)
        assert "my_metric" in out

    def test_skips_non_numeric(self):
        aggregated = {"label": "no_m2", "foot": {"mean": 0.5, "std": 0.1, "ci95_low": 0.3, "ci95_high": 0.7, "n": 2}}
        out = format_stats_table(aggregated)
        assert "foot" in out
