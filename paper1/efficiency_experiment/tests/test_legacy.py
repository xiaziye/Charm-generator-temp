"""Exact compatibility and independent-import checks for the original D3 panel."""
from pathlib import Path
import subprocess
import sys
import unittest

import networkx as nx
import numpy as np

from paper1.efficiency_experiment import legacy_d3

try:
    from paper1.bench import designs as original
except ModuleNotFoundError:
    # The exported generator-only repository intentionally does not include bench.
    original = None


class LegacyD3Tests(unittest.TestCase):
    def test_fixed_original_48_cell_factorial_panel(self):
        panel = legacy_d3.cells()
        self.assertEqual(len(panel), 48)
        self.assertEqual(panel[0], dict(base="split", n_inst=0, n_pm=0, n_py=0, n_weak=0))
        self.assertEqual(panel[-1], dict(base="common", n_inst=3, n_pm=2, n_py=2, n_weak=2))
        self.assertEqual(len({tuple(cell.items()) for cell in panel}), 48)

    def test_all_graphs_have_declared_route_truth_and_gaussian_samples(self):
        for cell in legacy_d3.cells():
            with self.subTest(cell=cell):
                design = legacy_d3.make_design(**cell)
                graph = design["G"]
                self.assertTrue(nx.is_directed_acyclic_graph(graph))
                self.assertEqual(design["truth"], {"NIE": .3, "NDE": .4})
                self.assertEqual(graph["X"]["M"]["weight"], .6)
                self.assertEqual(graph["M"]["Y"]["weight"], .5)
                self.assertNotIn("M2", design["pool"])
                self.assertNotIn("D", design["pool"])
                values, columns = legacy_d3.simulate(design, 7, np.random.default_rng(7))
                self.assertEqual(values.shape, (7, len(graph)))
                self.assertEqual(columns, design["order"])
                self.assertTrue(np.isfinite(values).all())

    @unittest.skipIf(original is None, "Full repository's bench is absent in standalone export")
    def test_exact_edges_roles_metadata_and_same_seed_samples_for_every_original_cell(self):
        self.assertEqual(legacy_d3.cells(), original.cells())
        for i, cell in enumerate(original.cells()):
            with self.subTest(cell=i):
                actual = legacy_d3.make_design(**cell)
                expected = original.make_design(**cell)
                self.assertEqual(list(actual["G"].nodes), list(expected["G"].nodes))
                self.assertEqual(list(actual["G"].edges(data=True)), list(expected["G"].edges(data=True)))
                for field in ("order", "roles", "truth", "pool", "factors"):
                    self.assertEqual(actual[field], expected[field])
                # Exact array equality also checks operation order and RNG consumption.
                rng_a, rng_b = np.random.default_rng([13, i]), np.random.default_rng([13, i])
                for n in (1, 17):
                    a, ca = legacy_d3.simulate(actual, n, rng_a)
                    b, cb = original.simulate(expected, n, rng_b)
                    self.assertEqual(ca, cb)
                    np.testing.assert_array_equal(a, b)

    @unittest.skipIf(original is None, "Full repository's bench is absent in standalone export")
    def test_optional_paths_and_nondefault_weights_retain_legacy_behavior(self):
        for m2 in (False, True):
            for dpost in (False, True):
                with self.subTest(m2=m2, dpost=dpost):
                    kwargs = dict(base="common", n_inst=2, n_pm=3, n_py=1, n_weak=2,
                                  m2=m2, dpost=dpost, inst=.9, pred=-.4, weak=.05)
                    actual, expected = legacy_d3.make_design(**kwargs), original.make_design(**kwargs)
                    self.assertEqual(list(actual["G"].edges(data=True)), list(expected["G"].edges(data=True)))
                    a, ca = legacy_d3.simulate(actual, 31, np.random.default_rng(19))
                    b, cb = original.simulate(expected, 31, np.random.default_rng(19))
                    self.assertEqual(ca, cb)
                    np.testing.assert_array_equal(a, b)

    def test_module_imports_without_any_paper1_package_or_config(self):
        path = Path(legacy_d3.__file__).resolve()
        script = '''
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("standalone_d3", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert len(module.cells()) == 48
assert module.make_design()["truth"]["NIE"] == .3
assert not any(name == "paper1" or name.startswith("paper1.") for name in sys.modules)
'''
        result = subprocess.run([sys.executable, "-c", script, str(path)],
                                cwd="/tmp", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
