"""Configuration boundaries and repeated-sampling statistics (no estimator implementation mirror)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paper1.efficiency_experiment.metrics import paired_variance_ratio, summarize
from paper1.efficiency_experiment.schema import from_dict, load_config


class ConfigTests(unittest.TestCase):
    def test_strict_tables_types_and_cross_field_checks(self):
        config = from_dict({"source": {"mode": "generate", "generator_config": "generator.toml"}})
        self.assertEqual(config.source.sample_sizes, ())
        self.assertEqual([a.name for a in config.arms], ["O", "FULL", "SHARED"])
        self.assertEqual(config.execution.reference_arm, "O")
        bad = [
            {"unknown": {}},
            {"source": {"mode": "saved", "path": "samples", "n_replicates": True}},
            {"source": {"mode": "saved", "path": "samples", "graph_ids": [0, 0]}},
            {"source": {"mode": "saved", "path": "samples", "sample_sizes": [0]}},
            {"source": {"mode": "generate"}},
            {"source": {"mode": "legacy_d3", "n_replicates": 3, "sample_sizes": [100], "graph_ids": [48]}},
            {"source": {"mode": "saved", "path": "samples"}, "execution": {"seed": -1}},
            {"source": {"mode": "saved", "path": "samples"}, "execution": {"workers": False}},
            {"source": {"mode": "saved", "path": "samples"}, "execution": {"allow_non_admissible": "false"}},
            {"source": {"mode": "saved", "path": "samples"}, "output": {"per_replicate": "yes"}},
            {"source": {"mode": "saved", "path": "samples"}, "endpoints": {"equality_lower": 2, "equality_upper": 1}},
            {"source": {"mode": "saved", "path": "samples"}, "arms": [{"name": "O", "CM": ["C"]}]},
            {"source": {"mode": "saved", "path": "samples"}, "arms": [{"name": "O"}, {"name": "O"}]},
            {"source": {"mode": "saved", "path": "samples"}, "arms": [{"name": "O", "kind": "custom"}]},
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                from_dict(raw)

    def test_direct_config_section_types_raise_clear_errors(self):
        valid = from_dict({"source": {"mode": "saved", "path": "samples"}})
        for name in ("execution", "output", "endpoints"):
            with self.subTest(section=name), self.assertRaisesRegex(ValueError, name + " must be"):
                replace(valid, **{name: {}}).validate()

    def test_toml_custom_arm_and_roundtrip(self):
        text = '''[source]
mode = "legacy_d3"
sample_sizes = [100, 500]
n_replicates = 20
graph_ids = [0, 47]
[[arms]]
name = "O"
[[arms]]
name = "EMPTY"
kind = "custom"
CM = []
CY = []
[estimator]
name = "linear_ols"
se_method = "none"
[endpoints]
enabled = false
'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.toml"
            path.write_text(text)
            config = load_config(path)
        self.assertEqual(config.source.sample_sizes, (100, 500))
        self.assertEqual(config.arms[1].kind, "custom")
        self.assertEqual(from_dict(config.to_dict()).to_dict(), config.to_dict())


def make_rows(values, *, graph_id=0, arm="O", truth=2.0, n=100):
    return [dict(graph_id=graph_id, n=n, arm=arm, replicate=i, status="ok", admissible=True,
                 estimator="linear_ols", NIE=float(x), NDE=float(x / 2), TE=float(x * 1.5), truth_NIE=truth, truth_NDE=truth / 2, truth_TE=truth * 1.5,
                 se_NIE=0.5, se_NDE=0.25, ci_NIE=[float(x - 1), float(x + 1)],
                 ci_NDE=[float(x / 2 - .5), float(x / 2 + .5)]) for i, x in enumerate(values)]


class MetricsTests(unittest.TestCase):
    def test_known_bias_variance_mse_coverage_and_mcse(self):
        report = summarize(make_rows([1, 2, 3]))
        result = report["groups"][0]["arms"]["O"]["results"]["NIE"]
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["bias"], 0)
        self.assertEqual(result["variance"], 1)
        self.assertAlmostEqual(result["mse"], 2 / 3)
        self.assertAlmostEqual(result["mcse_mean"], 1 / np.sqrt(3))
        self.assertEqual(result["coverage"], 1)
        self.assertEqual(result["mean_interval_width"], 2)
        json.dumps(report, allow_nan=False)

    def test_graphs_and_sample_sizes_are_never_pooled(self):
        rows = make_rows([1, 2, 3]) + make_rows([101, 102, 103], graph_id=1, truth=102)
        rows += make_rows([1, 2, 3], n=500)
        report = summarize(rows)
        self.assertEqual(len(report["groups"]), 3)
        self.assertEqual([g["arms"]["O"]["results"]["NIE"]["variance"] for g in report["groups"]], [1, 1, 1])

    def test_paired_ratio_accounts_for_common_random_numbers(self):
        baseline = make_rows([1, 2, 3, 4])
        comparison = make_rows([2, 4, 6, 8], arm="FULL")
        result = paired_variance_ratio(baseline + comparison, "FULL", "O", "NIE")
        self.assertEqual(result["n_paired"], 4)
        self.assertAlmostEqual(result["variance_ratio"], 4)
        # FULL is exactly 2*O: the paired ratio has no Monte Carlo fluctuation.
        self.assertAlmostEqual(result["variance_ratio_mcse"], 0)
        self.assertAlmostEqual(result["variance_reduction_reference_vs_arm"], .75)

    def test_failed_fit_excluded_but_recorded_and_pair_uses_overlap(self):
        rows = make_rows([1, 2, 3]) + make_rows([2, 4, 6], arm="FULL")
        rows[-1].update(status="failed", error="rank_deficient", NIE=None, NDE=None)
        group = summarize(rows)["groups"][0]
        self.assertEqual(group["arms"]["FULL"]["n_success"], 2)
        self.assertAlmostEqual(group["arms"]["FULL"]["failure_rate"], 1 / 3)
        ratio = [r for r in group["comparisons"] if r["effect"] == "NIE"][0]
        self.assertEqual(ratio["n_paired"], 2)
        self.assertEqual(ratio["variance_ratio"], 4)

    def test_missing_single_zero_variance_remain_explicit(self):
        report = summarize(make_rows([2]))
        result = report["groups"][0]["arms"]["O"]["results"]["NIE"]
        self.assertIsNone(result["variance"])
        self.assertIsNone(result["mcse_mean"])
        rows = make_rows([2, 2]) + make_rows([3, 3], arm="FULL")
        result = paired_variance_ratio(rows, "FULL", "O", "NIE")
        self.assertEqual(result["status"], "zero_reference_variance")
        self.assertIsNone(result["variance_ratio"])
        json.dumps(summarize(rows), allow_nan=False)

    def test_duplicate_fit_and_changing_truth_are_errors(self):
        rows = make_rows([1, 2])
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]])
        rows[1]["truth_NIE"] = 3
        with self.assertRaises(ValueError):
            summarize(rows)
        with self.assertRaises(ValueError):
            summarize(make_rows([1, 2], arm="O") + make_rows([1, 2], arm="FULL", truth=9))

    def test_confidence_levels_must_not_mix_and_diagnostics_are_summarized(self):
        rows = make_rows([1, 2, 3])
        for i, row in enumerate(rows):
            row.update(confidence_level=.90, seconds=i + 1., n_replicates_planned=5,
                       diagnostics={"mediator": {"n_predictors": 2, "n_coefficients": 3,
                                    "condition_number": i + 2., "df_residual": 97},
                                    "outcome": {"n_predictors": 3, "n_coefficients": 4,
                                    "condition_number": i + 3., "df_residual": 96}})
        group = summarize(rows)["groups"][0]
        self.assertEqual(group["confidence_level"], .90)
        self.assertEqual(group["n_replicates"], 5)
        self.assertEqual(group["n_replicates_observed"], 3)
        arm = group["arms"]["O"]
        self.assertEqual(arm["seconds_total"], 6.)
        self.assertEqual(arm["seconds_median"], 2.)
        self.assertEqual(arm["numerical_diagnostics"]["outcome"]["max_condition_number"], 5.)
        rows[0]["confidence_level"] = .95
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_success_status_without_finite_effect_is_counted_as_failure(self):
        rows = make_rows([1, 2, 3])
        rows[-1]["NIE"] = None
        arm = summarize(rows)["groups"][0]["arms"]["O"]
        self.assertEqual(arm["n_success"], 2)
        self.assertEqual(arm["n_failed"], 1)
        self.assertEqual(arm["results"]["NIE"]["n"], 2)
        self.assertIn("missing_or_non_finite_effect", arm["non_success_reasons"])

    def test_skipped_arm_is_reported_separately_from_fit_failure(self):
        rows = make_rows([1, 2, 3])
        rows[-1].update(status="skipped", reason="inadmissible")
        arm = summarize(rows)["groups"][0]["arms"]["O"]
        self.assertEqual(arm["n_failed"], 0)
        self.assertEqual(arm["n_skipped"], 1)
        self.assertAlmostEqual(arm["exclusion_rate"], 1 / 3)

    def test_unavailable_arm_is_excluded_without_counting_as_fit_failure(self):
        rows = make_rows([1, 2, 3], arm="OMISS")
        for row in rows:
            row.update(status="unavailable", reason="no_weak_parents")
        arm = summarize(rows)["groups"][0]["arms"]["OMISS"]
        self.assertEqual(arm["n_success"], 0)
        self.assertEqual(arm["n_failed"], 0)
        self.assertEqual(arm["n_skipped"], 3)
        self.assertEqual(arm["failure_rate"], 0)
        self.assertEqual(arm["exclusion_rate"], 1)
        self.assertEqual(arm["non_success_reasons"], {"no_weak_parents": 3})

    def test_diagnostic_model_scope_flags_survive_summary(self):
        rows = make_rows([1, 2, 3])
        rows[1]["diagnostic_misspecification"] = True
        rows[2]["diagnostic_non_admissible"] = True
        group = summarize(rows)["groups"][0]
        self.assertTrue(group["diagnostic_misspecification"])
        self.assertTrue(group["diagnostic_non_admissible"])
        plain = summarize(make_rows([1, 2, 3]))["groups"][0]
        self.assertFalse(plain["diagnostic_misspecification"])
        self.assertFalse(plain["diagnostic_non_admissible"])

    def test_two_repeats_do_not_imply_zero_variance_mcse(self):
        rows = make_rows([1, 2]) + make_rows([2, 5], arm="FULL")
        group = summarize(rows)["groups"][0]
        self.assertIsNone(group["arms"]["O"]["results"]["NIE"]["mcse_variance"])
        ratio = paired_variance_ratio(rows, "FULL", "O", "NIE")
        self.assertEqual(ratio["variance_ratio"], 9)
        self.assertIsNone(ratio["variance_ratio_mcse"])
        self.assertEqual(ratio["mcse_note"], "too_few_repeats_for_variance_mcse")

    def test_ratio_delta_mcse_matches_manual_paired_formula(self):
        x = np.asarray([1., 2., 4., 7., 8.]); y = np.asarray([1., 3., 3., 5., 8.])
        result = paired_variance_ratio(make_rows(y) + make_rows(x, arm="FULL"), "FULL", "O", "NIE")
        n = len(x); vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
        ratio = vx / vy
        influence = n / (n - 1) * ((x - x.mean()) ** 2 - ratio * (y - y.mean()) ** 2) / vy
        self.assertAlmostEqual(result["variance_ratio_mcse"], np.std(influence, ddof=1) / np.sqrt(n))


if __name__ == "__main__":
    unittest.main()
