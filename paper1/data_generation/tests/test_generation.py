"""Behavioral checks for the configured continuous structural generator.

Run with ``python -m unittest discover -s paper1/data_generation/tests -v``.
Reference effects below come from path algebra or independent Gaussian quadrature;
they do not reuse the generator's truth implementation.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json
import contextlib
import io
import tempfile
import unittest

import networkx as nx
import numpy as np

from paper1.data_generation import generator
from paper1.data_generation.schema import (
    CalibrationConfig,
    ExperimentConfig,
    GenerationConfig,
    GraphConfig,
    MechanismConfig,
    OutputConfig,
    load_config,
)
from paper1.graphs import graph_pair, intermediate_confounders, pair_admissible_set, pool_of


def small_config(**graph_changes):
    """An explicit three-node base; optional structures can be enabled individually."""
    graph = GraphConfig(
        n_baseline=0, baseline_mean_parents=0.0, baseline_max_parents=0,
        n_shared=0, n_x_extra=0, n_m_extra=0, n_y_extra=0,
        n_inst=0, n_pm=0, n_py=0, n_irrelevant=0,
        xm_chain_nodes=0, my_chain_nodes=0, n_parallel=0,
        n_post_descendants=0, n_middle_confounders=0, n_auxiliary_baseline_parents=0,
        allow_parallel_mediators=False, allow_post_descendants=False,
        allow_middle_confounding=False, direct_edge=True,
    )
    return GenerationConfig(
        graph=replace(graph, **graph_changes),
        mechanism=MechanismConfig(mode="linear", nonlinear_fraction=0.0),
        calibration=CalibrationConfig(enabled=False, n_samples=1024),
        experiment=ExperimentConfig(n_samples=50, n_graphs=1, n_replicates=1,
                                    truth_samples=30000, chunk_size=257),
        output=OutputConfig(csv=True, plot=False),
    )


def rich_config():
    return small_config(
        n_baseline=18, baseline_mean_parents=2.0, baseline_max_parents=3,
        n_shared=2, n_x_extra=3, n_m_extra=4, n_y_extra=3,
        n_inst=2, n_pm=2, n_py=2, n_irrelevant=1,
        xm_chain_nodes=1, my_chain_nodes=2, n_parallel=2,
        n_post_descendants=1, n_auxiliary_baseline_parents=2, allow_parallel_mediators=True,
        allow_post_descendants=True,
    )


def build(config, graph_seed=71, mechanism_seed=83, calibration_seed=97):
    return generator.build_design(config, graph_seed, mechanism_seed, calibration_seed)


def truth(design, config, seed=107):
    # public function is the stable entry point after the generator refactor.
    if hasattr(generator, "structural_truth"):
        return generator.structural_truth(design, config.mechanism, config.experiment, seed)
    if config.mechanism.mode == "linear":
        return generator._linear_truth(design, config.experiment.x0, config.experiment.x1)
    return generator._mc_truth(design, config.mechanism, config.experiment, seed)


class StructuralSimulationTests(unittest.TestCase):
    def test_minimal_equations_and_interventions(self):
        config = small_config()
        design = build(config)
        columns = list(design.order)
        noise = np.zeros((2, len(columns)))
        noise[:, columns.index("X")] = [1.0, -2.0]
        noise[:, columns.index("M")] = [0.2, -0.1]
        noise[:, columns.index("Y")] = [0.3, 0.4]
        observed, cols = generator.simulate(design, 2, 1, config.mechanism, noise=noise)
        x = np.array([1.0, -2.0])
        m = 0.6 * x + np.array([0.2, -0.1])
        y = 0.2 * x + 0.5 * m + np.array([0.3, 0.4])
        np.testing.assert_allclose(observed[:, cols.index("M")], m)
        np.testing.assert_allclose(observed[:, cols.index("Y")], y)
        # do(X=4) replaces the full X equation, including the supplied X noise.
        changed, _ = generator.simulate(design, 2, 1, config.mechanism,
                                        noise=noise, interventions={"X": 4.0})
        np.testing.assert_array_equal(changed[:, cols.index("X")], [4.0, 4.0])
        np.testing.assert_allclose(changed[:, cols.index("M")], [2.6, 2.3])
        # The caller's exogenous disturbances are never mutated by simulation.
        np.testing.assert_array_equal(noise[:, cols.index("X")], x)

    def test_disabled_optional_structures_leave_only_main_route(self):
        config = small_config(direct_edge=False)
        design = build(config)
        self.assertEqual(set(design.graph), {"X", "M", "Y"})
        self.assertEqual(set(design.graph.edges), {("X", "M"), ("M", "Y")})
        result = truth(design, config)
        self.assertAlmostEqual(result["NIE"], 0.3)
        self.assertAlmostEqual(result["NDE"], 0.0)
        self.assertAlmostEqual(result["TE"], 0.3)

    def test_chains_and_parallel_paths_change_truth_as_path_algebra_predicts(self):
        config = small_config(xm_chain_nodes=2, my_chain_nodes=1, n_parallel=2,
                              allow_parallel_mediators=True)
        config = replace(config, experiment=replace(config.experiment, x0=-0.5, x1=1.5))
        design = build(config)
        result = truth(design, config)
        # Direct protected edges remain, and each extra chain adds another path.
        first_stage = 0.6 + 0.6 ** 3
        second_stage = 0.5 + 0.6 ** 2
        expected_nie = 2.0 * first_stage * second_stage
        expected_nde = 2.0 * (0.2 + 2 * 0.5 * 0.4)
        self.assertAlmostEqual(result["NIE"], expected_nie, places=12)
        self.assertAlmostEqual(result["NDE"], expected_nde, places=12)
        self.assertAlmostEqual(result["TE"], expected_nie + expected_nde, places=12)

    def test_linear_truth_matches_same_noise_nested_interventions_per_individual(self):
        config = rich_config()
        config = replace(config, mechanism=replace(config.mechanism, intercept_sd=0.3))
        design = build(config)
        n = 37
        noise = np.random.default_rng(241).normal(size=(n, len(design.order)))
        x0, x1 = config.experiment.x0, config.experiment.x1
        d0, columns = generator.simulate(design, n, 0, config.mechanism, noise=noise,
                                         interventions={"X": x0})
        d1, _ = generator.simulate(design, n, 0, config.mechanism, noise=noise,
                                   interventions={"X": x1})
        dm, _ = generator.simulate(design, n, 0, config.mechanism, noise=noise,
                                   interventions={"X": x1, "M": d0[:, columns.index("M")]})
        y = columns.index("Y")
        expected = truth(design, config)
        np.testing.assert_allclose(d1[:, y] - dm[:, y], expected["NIE"], atol=2e-13)
        np.testing.assert_allclose(dm[:, y] - d0[:, y], expected["NDE"], atol=2e-13)
        np.testing.assert_allclose(d1[:, y] - d0[:, y], expected["TE"], atol=2e-13)

    def test_input_noise_shape_and_unknown_interventions_fail_explicitly(self):
        config = small_config()
        design = build(config)
        for bad_noise in (np.zeros((2, 2)), np.full((2, 3), np.nan)):
            with self.subTest(noise_shape=bad_noise.shape):
                with self.assertRaises(ValueError):
                    generator.simulate(design, 2, 0, config.mechanism, noise=bad_noise)
        with self.assertRaises(ValueError):
            generator.simulate(design, 2, 0, config.mechanism, interventions={"unknown": 0})


class ReproducibilityAndCalibrationTests(unittest.TestCase):
    def test_graph_and_mechanism_seeds_are_separate(self):
        config = rich_config()
        first = build(config, graph_seed=71, mechanism_seed=83)
        new_mechanism = build(config, graph_seed=71, mechanism_seed=84)
        same = build(config, graph_seed=71, mechanism_seed=83)
        self.assertEqual(set(first.graph.edges), set(new_mechanism.graph.edges))
        self.assertEqual(first.order, new_mechanism.order)
        self.assertEqual(first.roles, new_mechanism.roles)
        self.assertEqual(first.metadata(), same.metadata())
        edges = list(first.graph.edges)
        weights = [first.graph[u][v]["weight"] for u, v in edges]
        changed = [new_mechanism.graph[u][v]["weight"] for u, v in edges]
        self.assertFalse(np.array_equal(weights, changed))
        new_graph = build(config, graph_seed=72, mechanism_seed=83)
        self.assertNotEqual(set(first.graph.edges), set(new_graph.graph.edges))

    def test_data_seed_changes_samples_not_the_structural_model(self):
        config = rich_config()
        first = generator.generate(config, data_seed=201)
        repeated = generator.generate(config, data_seed=201)
        new_data = generator.generate(config, data_seed=202)
        np.testing.assert_array_equal(first["data"], repeated["data"])
        self.assertFalse(np.array_equal(first["data"], new_data["data"]))
        self.assertEqual(set(first["design"].graph.edges), set(new_data["design"].graph.edges))
        self.assertEqual(first["truth"], new_data["truth"])
        for u, v, edge in first["design"].graph.edges(data=True):
            self.assertEqual(edge["weight"], new_data["design"].graph[u][v]["weight"])

    def test_calibration_preserves_protected_effect_paths_and_records_scales(self):
        raw_config = rich_config()
        calibrated_config = replace(raw_config, calibration=CalibrationConfig(
            enabled=True, n_samples=2048, target_signal_sd=0.9))
        raw = build(raw_config)
        calibrated = build(calibrated_config)
        self.assertEqual(set(raw.graph.edges), set(calibrated.graph.edges))
        protected_roles = {"query_XM", "query_MY", "query_direct", "parallel_XM",
                           "parallel_MY", "parallel_mediator"}
        for u, v, edge in raw.graph.edges(data=True):
            if edge.get("protected") or edge["edge_role"] in protected_roles:
                self.assertEqual(edge["weight"], calibrated.graph[u][v]["weight"])
        self.assertGreater(len(calibrated.calibration["nodes"]), 0)
        for u, v, edge in calibrated.graph.edges(data=True):
            self.assertEqual(edge["raw_weight"], raw.graph[u][v]["weight"])
            self.assertAlmostEqual(edge["weight"], edge["raw_weight"] * edge["calibration_scale"], places=12)
        for node, record in calibrated.calibration["nodes"].items():
            self.assertAlmostEqual(record["after_sd"], calibrated_config.calibration.target_signal_sd, places=12)
            for parent in record["parents"]:
                self.assertAlmostEqual(calibrated.graph[parent][node]["weight"],
                                       raw.graph[parent][node]["weight"] * record["scale"], places=12)
        self.assertAlmostEqual(truth(raw, raw_config)["NIE"],
                               truth(calibrated, calibrated_config)["NIE"], places=12)
        self.assertAlmostEqual(truth(raw, raw_config)["NDE"],
                               truth(calibrated, calibrated_config)["NDE"], places=12)

    def test_exported_metadata_reconstructs_the_exact_nonlinear_mechanism(self):
        config = rich_config()
        config = replace(config, mechanism=replace(config.mechanism,
                         mode="additive_nonlinear", protected_function="tanh",
                         nonlinear_fraction=.7, outcome_interaction=.2, intercept_sd=.3),
                         calibration=CalibrationConfig(enabled=True, n_samples=1024))
        design = build(config)
        metadata = json.loads(json.dumps(design.metadata()))
        graph = nx.DiGraph()
        graph.add_nodes_from(metadata["columns"])
        functions = {}
        for edge in metadata["edges"]:
            attributes = dict(edge)
            u, v = attributes.pop("source"), attributes.pop("target")
            functions[u, v] = attributes.pop("function")
            graph.add_edge(u, v, **attributes)
        restored = generator.ContinuousDesign(
            graph, tuple(metadata["columns"]), metadata["roles"], functions,
            metadata["config"], metadata["seeds"], metadata["calibration"],
            metadata["identification_status"],
        )
        # TOML arrays become tuples in the schema; normalize this schema field after JSON loading.
        restored.config["mechanism"]["edge_functions"] = tuple(restored.config["mechanism"]["edge_functions"])
        observed, columns = generator.simulate(design, 25, 149, config.mechanism)
        reconstructed, reconstructed_columns = generator.simulate(restored, 25, 149, config.mechanism)
        self.assertEqual(columns, reconstructed_columns)
        # JSON preserves values; rebuilding a DiGraph can reorder parent additions.
        np.testing.assert_allclose(observed, reconstructed, rtol=0, atol=2e-14)

    def test_linear_population_moments_match_closed_form_three_node_model(self):
        config = small_config()
        design = build(config)
        mean, cov = generator.analytic_moments(design, config.mechanism)
        order = [design.order.index(v) for v in ("X", "M", "Y")]
        covariance = np.asarray(cov)[np.ix_(order, order)]
        # X=eX; M=.6eX+eM; Y=.5eX+.5eM+eY; all noises have variance 1.
        mixing = np.array([[1, 0, 0], [.6, 1, 0], [.5, .5, 1]])
        np.testing.assert_allclose(np.asarray(mean)[order], np.zeros(3), atol=1e-14)
        np.testing.assert_allclose(covariance, mixing @ mixing.T, atol=1e-14)


class TruthAndIdentificationTests(unittest.TestCase):
    def test_nonlinear_truth_matches_independent_gaussian_quadrature(self):
        config = small_config()
        config = replace(config, mechanism=replace(config.mechanism,
                         mode="additive_nonlinear", protected_function="tanh",
                         nonlinear_fraction=1.0),
                         experiment=replace(config.experiment, truth_samples=40000, chunk_size=311))
        config.validate()
        design = build(config)
        actual = truth(design, config)
        # M(x)=.6*tanh(x)+eM, Y(x,m)=.2*tanh(x)+.5*tanh(m)+eY.
        # Independent Gauss-Hermite integration is a reference for the population NIE.
        nodes, weights = np.polynomial.hermite.hermgauss(60)
        standard_normal = np.sqrt(2) * nodes
        x0, x1 = config.experiment.x0, config.experiment.x1
        contrast = np.tanh(.6 * np.tanh(x1) + standard_normal) - np.tanh(.6 * np.tanh(x0) + standard_normal)
        expected_nie = .5 * float(weights @ contrast) / np.sqrt(np.pi)
        expected_nde = .2 * (np.tanh(x1) - np.tanh(x0))
        allowance = max(6.0 * actual["mcse_NIE"], 0.001)
        self.assertAlmostEqual(actual["NIE"], expected_nie, delta=allowance)
        self.assertAlmostEqual(actual["NDE"], expected_nde, places=12)
        self.assertAlmostEqual(actual["TE"], actual["NIE"] + actual["NDE"], places=12)
        # The protected path is actually nonlinear, rather than nonlinear only in nuisance nodes.
        self.assertNotAlmostEqual(actual["NIE"], .6 * .5, places=2)

    def test_nonlinear_truth_is_invariant_to_chunking(self):
        config = small_config()
        config = replace(config, mechanism=replace(config.mechanism,
                         mode="additive_nonlinear", protected_function="tanh",
                         outcome_interaction=0.3),
                         experiment=replace(config.experiment, truth_samples=1207, chunk_size=137))
        design = build(config)
        first = truth(design, config, seed=137)
        whole_config = replace(config, experiment=replace(config.experiment, chunk_size=1207))
        whole = truth(design, whole_config, seed=137)
        for effect in ("TE", "NIE", "NDE", "mcse_TE", "mcse_NIE", "mcse_NDE"):
            self.assertAlmostEqual(first[effect], whole[effect], places=12)

    def test_simulation_rejects_a_mechanism_changed_after_design_freeze(self):
        config = small_config()
        design = build(config)
        with self.assertRaises(ValueError):
            generator.simulate(design, 10, 23, replace(config.mechanism, noise_sd=2.0))

    def test_three_adjustment_arms_are_valid_without_intermediate_confounding(self):
        config = rich_config()
        for seed in (7, 29, 53):
            design = build(config, graph_seed=seed)
            graph = design.graph
            self.assertTrue(nx.is_directed_acyclic_graph(graph))
            self.assertEqual(intermediate_confounders(graph, "X", "M", "Y"), [])
            om, oy = graph_pair(graph, "X", ("M",), "Y")
            full = pool_of([graph], "X", "M", "Y")
            shared = set(om) | set(oy)
            for name, q, r in (("O", om, oy), ("FULL", full, full),
                               ("SHARED", shared, shared)):
                with self.subTest(seed=seed, arm=name):
                    self.assertTrue(pair_admissible_set(graph, "X", ("M",), "Y", q, r))

    def test_middle_confounder_is_recorded_and_natural_effect_adjustment_fails(self):
        config = small_config(n_middle_confounders=1, allow_middle_confounding=True)
        design = build(config)
        graph = design.graph
        middle = intermediate_confounders(graph, "X", "M", "Y")
        self.assertEqual(middle, ["L001"])
        om, oy = graph_pair(graph, "X", ("M",), "Y")
        full = pool_of([graph], "X", "M", "Y")
        self.assertFalse(pair_admissible_set(graph, "X", ("M",), "Y", om, oy))
        self.assertFalse(pair_admissible_set(graph, "X", ("M",), "Y", full, full))
        # A fully specified simulator still defines counterfactual truth at this boundary.
        effects = truth(design, config)
        self.assertTrue(np.isfinite([effects[k] for k in ("NIE", "NDE", "TE")]).all())
        self.assertAlmostEqual(effects["TE"], effects["NIE"] + effects["NDE"])


class ConfigurationTests(unittest.TestCase):
    def test_unknown_toml_section_and_key_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            for contents in ('[typo]\nx=1\n', '[graph]\nn_baselines=10\n'):
                path.write_text(contents, encoding="utf-8")
                with self.subTest(contents=contents):
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_impossible_structure_switches_are_rejected(self):
        for changes in (
            {"n_parallel": 1, "allow_parallel_mediators": False},
            {"n_post_descendants": 1, "allow_post_descendants": False},
            {"n_middle_confounders": 1, "allow_middle_confounding": False},
            {"n_baseline": 1, "n_shared": 2},
            {"n_inst": 1, "n_baseline": 0},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    small_config(**changes).validate()

    def test_nonfinite_and_invalid_numeric_settings_are_rejected(self):
        original = small_config()
        invalid = [
            replace(original, mechanism=replace(original.mechanism, noise_sd=0)),
            replace(original, mechanism=replace(original.mechanism, noise_sd=float("nan"))),
            replace(original, mechanism=replace(original.mechanism, weight_max=float("inf"))),
            replace(original, mechanism=replace(original.mechanism, noise_distribution="invalid")),
            replace(original, mechanism=replace(original.mechanism, noise_distribution="student_t", noise_df=2)),
            replace(original, experiment=replace(original.experiment, n_replicates=0)),
            replace(original, experiment=replace(original.experiment, n_samples=-1)),
            replace(original, experiment=replace(original.experiment, x1=float("nan"))),
            replace(original, experiment=replace(original.experiment, chunk_size=0)),
            replace(original, calibration=replace(original.calibration, enabled=True, target_signal_sd=0)),
        ]
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    config.validate()

    def test_shipped_examples_load_and_use_continuous_models(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        paths = sorted(examples.glob("*.toml"))
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertIn(config.mechanism.mode, {"linear", "additive_nonlinear"})
                self.assertGreater(config.mechanism.noise_sd, 0)


class ExportRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from paper1.data_generation.run import run
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / "result"
        config = rich_config()
        cls.config = replace(config, experiment=replace(config.experiment,
                             n_samples=31, n_graphs=2, n_replicates=2),
                             output=replace(config.output, directory=str(cls.output), preview_rows=4))
        cls.config_path = cls.root / "config.toml"
        lines = []
        for section, values in asdict(cls.config).items():
            lines.append(f"[{section}]")
            for name, value in values.items():
                lines.append(f"{name} = {json.dumps(value)}")
            lines.append("")
        cls.config_path.write_text("\n".join(lines), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            cls.manifest_path = run(cls.config_path)
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    def test_all_graphs_and_replicates_have_distinct_reproducible_samples(self):
        self.assertEqual(self.manifest["status"], "complete")
        self.assertEqual(len(self.manifest["graphs"]), 2)
        data_seeds = []
        for row in self.manifest["graphs"]:
            graph_dir = self.output / row["directory"]
            self.assertEqual(len(row["replicates"]), 2)
            seeds = row["seeds"]
            design = build(self.config, seeds["graph"], seeds["mechanism"], seeds["calibration"])
            previous = None
            for replica in row["replicates"]:
                data_seeds.append(replica["data_seed"])
                with np.load(graph_dir / replica["file"], allow_pickle=False) as exported:
                    values, columns = exported["values"], exported["columns"].tolist()
                expected, expected_columns = generator.simulate(design, 31,
                                               replica["data_seed"], self.config.mechanism)
                np.testing.assert_array_equal(values, expected)
                self.assertEqual(columns, expected_columns)
                self.assertEqual(list(values.shape), replica["shape"])
                if previous is not None:
                    self.assertFalse(np.array_equal(previous, values))
                previous = values
                csv_path = graph_dir / (Path(replica["file"]).stem + ".csv")
                csv_values = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                np.testing.assert_array_equal(csv_values, values)
                preview = np.loadtxt(graph_dir / f"preview_r{replica['replicate']:03d}.csv",
                                     delimiter=",", skiprows=1)
                np.testing.assert_array_equal(preview, values[:4])
        self.assertEqual(len(data_seeds), len(set(data_seeds)))

    def test_snapshot_and_mathematical_exports_are_self_contained(self):
        self.assertEqual((self.output / "config.toml").read_bytes(), self.config_path.read_bytes())
        resolved = json.loads((self.output / "resolved_config.json").read_text(encoding="utf-8"))
        self.assertEqual(resolved, json.loads(json.dumps(self.config.to_dict())))
        self.assertTrue((self.output / "REVIEW.md").is_file())
        for row in self.manifest["graphs"]:
            graph_dir = self.output / row["directory"]
            metadata = json.loads((graph_dir / "design.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["truth"], row["truth"])
            self.assertEqual(metadata["query"], row["query"])
            equations = (graph_dir / "equations.md").read_text(encoding="utf-8")
            self.assertGreater(len(equations), 100)
            self.assertTrue(all(node in equations for node in metadata["columns"]))
            self.assertEqual(metadata["population_moments"]["status"], "computed")
            for name, projection in metadata["population_moments"]["population_ols"].items():
                with self.subTest(graph=row["graph_id"], arm=name):
                    self.assertTrue(projection["admissible"])
                    self.assertAlmostEqual(projection["bias_NIE"], 0, places=10)
                    self.assertAlmostEqual(projection["bias_NDE"], 0, places=10)

    def test_existing_output_is_not_overwritten(self):
        from paper1.data_generation.run import run
        before = self.manifest_path.read_bytes()
        with self.assertRaises(FileExistsError):
            run(self.config_path)
        self.assertEqual(self.manifest_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
