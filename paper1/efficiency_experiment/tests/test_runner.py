"""Integration and adapter checks for the generated-data → adjustment → fit contract."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from paper1.efficiency_experiment import legacy_d3 as designs
from paper1.data_generation.generator import (build_design, derive_seed, diagnose, simulate, structural_truth)
from paper1.data_generation.schema import (CalibrationConfig, ExperimentConfig as GenerationExperiment,
                                          GenerationConfig, GraphConfig, MechanismConfig)
from paper1.efficiency_experiment.arms import build_arms
from paper1.efficiency_experiment.io import file_hash, load_inputs, read_npz
from paper1.efficiency_experiment.run import run_experiment
from paper1.efficiency_experiment.schema import from_dict


def product_estimate(values, CM, CY, X, M, Y):
    """Independent reference for the frozen paper1.estimation product formula.

    Keep this tiny reference local so the exported generator/experiment package
    does not need the full CHARM pipeline merely to run its integration tests.
    """
    zm = np.column_stack((np.ones(len(values)), values[:, [X] + list(CM)]))
    zy = np.column_stack((np.ones(len(values)), values[:, [X, M] + list(CY)]))
    a = np.linalg.lstsq(zm, values[:, M], rcond=None)[0][1]
    outcome = np.linalg.lstsq(zy, values[:, Y], rcond=None)[0]
    return a * outcome[2], outcome[1]


def tiny_generation(*, middle=False, nonlinear=False):
    return GenerationConfig(
        graph=GraphConfig(n_baseline=7, baseline_mean_parents=0.5, n_shared=1,
                          n_x_extra=1, n_m_extra=1, n_y_extra=1, n_inst=1, n_pm=1,
                          n_py=1, n_irrelevant=0, n_auxiliary_baseline_parents=1,
                          n_middle_confounders=int(middle), allow_middle_confounding=middle),
        mechanism=(MechanismConfig(mode="additive_nonlinear", edge_functions=("id", "tanh"),
                                   protected_function="tanh", nonlinear_fraction=0.5, outcome_interaction=0.1)
                   if nonlinear else MechanismConfig()),
        calibration=CalibrationConfig(enabled=False),
        experiment=GenerationExperiment(n_samples=120, n_graphs=1, n_replicates=3))


def make_saved(root, *, middle=False, nonlinear=False):
    root.mkdir()
    cfg = tiny_generation(middle=middle, nonlinear=nonlinear)
    design = build_design(cfg, 11, 12, 13)
    truth = structural_truth(design, cfg.mechanism, cfg.experiment, 14)
    metadata = dict(design.metadata(), query=diagnose(design, cfg.mechanism), truth=truth)
    directory = root / "graph_000"
    directory.mkdir()
    (directory / "design.json").write_text(json.dumps(metadata), encoding="utf-8")
    replicates = []
    for rep in range(3):
        values, columns = simulate(design, 120, 100 + rep, cfg.mechanism)
        path = directory / f"data_r{rep:03d}.npz"
        np.savez_compressed(path, values=values, columns=columns)
        replicates.append(dict(replicate=rep, file=path.name, shape=list(values.shape),
                               data_seed=100 + rep, sha256=file_hash(path)))
    manifest = dict(schema_version=1, status="complete", config=cfg.to_dict(),
                    graphs=[dict(graph_id=0, directory="graph_000", replicates=replicates)])
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return metadata, directory


def rows(path):
    return [json.loads(line) for line in (path / "replicates.jsonl").read_text(encoding="utf-8").splitlines()]


class RunnerTests(unittest.TestCase):
    def test_saved_relative_path_and_effect_parity(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            metadata, directory = make_saved(base / "input")
            cfg = from_dict(dict(source=dict(mode="saved", path="input", sample_sizes=[80, 120]),
                                 arms=[dict(name=name) for name in ("O", "FULL", "SHARED", "SHAREDm")],
                                 output=dict(directory="output")))
            result = run_experiment(cfg, config_directory=base)
            output = result.parent
            r = rows(output)
            self.assertEqual(len(r), 3 * 2 * 4)
            self.assertTrue(all(row["status"] == "ok" for row in r))
            data, columns = read_npz(directory / "data_r000.npz")
            ix = {c: i for i, c in enumerate(columns)}
            cm, cy = metadata["query"]["arms"]["O"]
            expected = product_estimate(data, [ix[c] for c in cm], [ix[c] for c in cy], ix["X"], ix["M"], ix["Y"])
            actual = next(row for row in r if row["n"] == 120 and row["replicate"] == 0 and row["arm"] == "O")
            np.testing.assert_allclose([actual["NIE"], actual["NDE"]], expected, rtol=1e-12)
            self.assertFalse(actual["truth_used"])
            self.assertFalse(actual["truth_used_for_fitting"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["groups"]), 2)
            self.assertTrue(all(g["n_replicates"] == 3 for g in summary["groups"]))
            for name in ("summary.csv", "comparisons.csv", "endpoints.json", "REPORT.md", "graph_000/arms.json"):
                self.assertTrue((output / name).is_file(), name)
            with self.assertRaises(FileExistsError):
                run_experiment(cfg, config_directory=base)

    def test_saved_hash_shape_order_and_requested_repeats_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            metadata, directory = make_saved(base / "input")
            file = directory / "data_r000.npz"
            cfg = from_dict(dict(source=dict(mode="saved", path="input", n_replicates=4)))
            with self.assertRaisesRegex(ValueError, "only 3 saved"):
                load_inputs(cfg, base)
            cfg = from_dict(dict(source=dict(mode="saved", path="input", sample_sizes=[200])))
            with self.assertRaisesRegex(ValueError, "sample size exceeds"):
                load_inputs(cfg, base)
            values, columns = read_npz(file)
            with self.assertRaisesRegex(ValueError, "column order"):
                read_npz(file, expected_columns=columns[::-1])
            with self.assertRaisesRegex(ValueError, "shape differs"):
                read_npz(file, expected_shape=(119, values.shape[1]))
            cfg = from_dict(dict(source=dict(mode="saved", path="input")))
            graphs, _ = load_inputs(cfg, base)
            np.savez_compressed(file, values=values + 1, columns=columns)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                graphs[0].sample(0)

    def test_artifact_paths_cannot_escape_source(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            make_saved(base / "input")
            p = base / "input/manifest.json"
            manifest = json.loads(p.read_text())
            manifest["graphs"][0]["directory"] = "../outside"
            p.write_text(json.dumps(manifest))
            cfg = from_dict(dict(source=dict(mode="saved", path="input")))
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_inputs(cfg, base)

    def test_boundary_invalid_arms_are_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            make_saved(base / "input", middle=True)
            cfg = from_dict(dict(source=dict(mode="saved", path="input", n_replicates=2), output=dict(directory="out")))
            with patch("paper1.efficiency_experiment.run.fit_estimate") as fit:
                path = run_experiment(cfg, config_directory=base)
            fit.assert_not_called()
            r = rows(path.parent)
            self.assertEqual({row["status"] for row in r}, {"skipped_non_admissible"})
            self.assertTrue(all(row["admissible"] is False for row in r))
            summary = json.loads((path.parent / "summary.json").read_text())
            self.assertTrue(all(comp["n_paired"] == 0 for g in summary["groups"] for comp in g["comparisons"]))

    def test_workers_and_sample_sizes_preserve_estimates(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            make_saved(base / "input")
            common = dict(source=dict(mode="saved", path="input", sample_sizes=[80, 120]))
            a = run_experiment(from_dict(common), config_directory=base, output=base / "a")
            b = run_experiment(from_dict(dict(common, execution=dict(workers=2))), config_directory=base, output=base / "b")
            ra, rb = rows(a.parent), rows(b.parent)
            self.assertEqual([(r["graph_id"], r["n"], r["replicate"], r["arm"]) for r in ra],
                             [(r["graph_id"], r["n"], r["replicate"], r["arm"]) for r in rb])
            for x, y in zip(ra, rb):
                np.testing.assert_array_equal([x[k] for k in ("NIE", "NDE", "TE")], [y[k] for k in ("NIE", "NDE", "TE")])

    def test_legacy_adapter_retains_original_weak_ablation(self):
        cfg = from_dict(dict(source=dict(mode="legacy_d3", graph_ids=[7], sample_sizes=[80, 120], n_replicates=2),
                             arms=[dict(name=name) for name in ("O", "FULL", "SHARED", "SHAREDm", "OMISS")]))
        graphs, source = load_inputs(cfg, Path.cwd())
        self.assertEqual(len(graphs), 1)
        graph = graphs[0]
        old = designs.make_design(**designs.cells()[7])
        seed = derive_seed(cfg.execution.seed, 7, 0, 44)
        expected, cols = designs.simulate(old, 120, np.random.default_rng(seed))
        actual, actual_cols, provenance = graph.sample(0)
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(tuple(cols), actual_cols)
        self.assertEqual(provenance["data_seed"], seed)
        arms, roles, factors, prediction = build_arms(graph.metadata, cfg.arms)
        by_name = {arm.name: arm for arm in arms}
        self.assertTrue(by_name["OMISS"].available)
        self.assertNotIn("WM", by_name["OMISS"].CM)
        self.assertNotIn("WY", by_name["OMISS"].CY)
        self.assertIn("WM", by_name["O"].CM)
        self.assertTrue(by_name["OMISS"].admissible)
        self.assertEqual(source["mode"], "legacy_d3")
        self.assertIsInstance(prediction["predicted_equal"], bool)

    def test_fit_failure_is_recorded_and_raise_policy_is_honored(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            make_saved(base / "input")
            cfg = from_dict(dict(source=dict(mode="saved", path="input", n_replicates=1, sample_sizes=[3]),
                                 output=dict(directory="out")))
            result = run_experiment(cfg, config_directory=base)
            self.assertEqual({row["status"] for row in rows(result.parent)}, {"error"})
            cfg = replace(cfg, execution=replace(cfg.execution, on_error="raise"))
            with self.assertRaises(ValueError):
                run_experiment(cfg, config_directory=base, output=base / "raise")
            manifest = json.loads((base / "raise/manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")

    def test_custom_arm_and_missing_weak_role_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            metadata, _ = make_saved(base / "input")
            cfg = from_dict(dict(source=dict(mode="saved", path="input"),
                                 arms=[dict(name="O"), dict(name="OMISS"),
                                       dict(name="USER", kind="custom", CM=[], CY=[])]))
            arms, *_ = build_arms(metadata, cfg.arms)
            by_name = {arm.name: arm for arm in arms}
            self.assertFalse(by_name["OMISS"].available)
            self.assertIsNone(by_name["OMISS"].admissible)
            self.assertFalse(by_name["USER"].admissible)
            metadata["query"]["arms"]["O"] = ([], metadata["query"]["arms"]["O"][1])
            with self.assertRaisesRegex(ValueError, "disagrees"):
                build_arms(metadata, cfg.arms)

    def test_nonlinear_input_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            make_saved(base / "input", nonlinear=True)
            cfg = from_dict(dict(source=dict(mode="saved", path="input"), output=dict(directory="output")))
            with self.assertRaisesRegex(ValueError, "linear_ols requires identity"):
                run_experiment(cfg, config_directory=base)
            self.assertFalse((base / "output").exists())

    def test_misspecification_flag_requires_an_actual_mechanism_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for nonlinear in (False, True):
                name = "nonlinear" if nonlinear else "linear"
                make_saved(base / name, nonlinear=nonlinear)
                cfg = from_dict(dict(source=dict(mode="saved", path=name, n_replicates=2),
                                     estimator=dict(allow_misspecification=True),
                                     output=dict(directory=name + "_output")))
                result = run_experiment(cfg, config_directory=base)
                fitted = rows(result.parent)
                self.assertEqual({r["status"] for r in fitted}, {"ok"})
                self.assertTrue(all(r["diagnostic_misspecification"] is nonlinear for r in fitted))

    def test_multigraph_summary_and_optional_raw_output(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = from_dict(dict(source=dict(mode="legacy_d3", graph_ids=[1, 7],
                                            sample_sizes=[60], n_replicates=2),
                                 arms=[dict(name=name) for name in ("O", "FULL", "SHARED", "SHAREDm", "OMISS")],
                                 output=dict(per_replicate=False)))
            result = run_experiment(cfg, output=Path(folder) / "out")
            summary = json.loads((result.parent / "summary.json").read_text())
            self.assertEqual(summary["n_records"], 20)
            self.assertEqual({g["graph_id"] for g in summary["groups"]}, {1, 7})
            self.assertFalse((result.parent / "replicates.jsonl").exists())
            self.assertEqual(len(summary["groups"]), 2)
            self.assertTrue(all(g["n_replicates"] == 2 for g in summary["groups"]))

    def test_bootstrap_seed_is_paired_across_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            make_saved(base / "input")
            cfg = from_dict(dict(source=dict(mode="saved", path="input", n_replicates=2,
                                            sample_sizes=[80, 120]),
                                 estimator=dict(se_method="bootstrap", bootstrap_replicates=5)))
            result = run_experiment(cfg, config_directory=base, output=base / "out")
            fitted = rows(result.parent)
            self.assertTrue(all(r["status"] == "ok" for r in fitted))
            keys = {(r["replicate"], r["n"]) for r in fitted}
            all_seeds = []
            for key in keys:
                seeds = {r["bootstrap_seed"] for r in fitted if (r["replicate"], r["n"]) == key}
                self.assertEqual(len(seeds), 1)
                all_seeds.extend(seeds)
            self.assertEqual(len(set(all_seeds)), len(keys))

    def test_generate_mode_builds_one_fixed_design_with_fresh_replicates(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            # Use the real review config as a public generator contract; n is overridden.
            original = Path(__file__).resolve().parents[2] / "data_generation/examples/review_linear.toml"
            cfg = from_dict(dict(source=dict(mode="generate", generator_config=str(original),
                                            graph_ids=[0], sample_sizes=[50, 100], n_replicates=2)))
            graphs, source = load_inputs(cfg, base)
            graph = graphs[0]
            a, ca, pa = graph.sample(0)
            b, cb, pb = graph.sample(0)
            c, cc, pc = graph.sample(1)
            np.testing.assert_array_equal(a, b)
            self.assertFalse(np.array_equal(a, c))
            self.assertEqual(a.shape[0], 100)
            self.assertEqual(graph.sample_sizes, (50, 100))
            self.assertEqual(pa["data_seed"], pb["data_seed"])
            self.assertNotEqual(pa["data_seed"], pc["data_seed"])
            self.assertEqual(graph.metadata["truth"]["kind"], "truth_exact_linear")


if __name__ == "__main__":
    unittest.main()
