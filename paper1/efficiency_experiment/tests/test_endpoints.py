"""Endpoint tests use hand-built summaries so decisions are independently known."""
from copy import deepcopy
import unittest

from paper1.efficiency_experiment.endpoints import EndpointThresholds, evaluate_endpoints


def group(graph_id=0, *, n=1000, reps=200, instrument=True, equal=False,
          ratios=None, bias=0.001, mcse=0.002):
    ratios = ratios or {"FULL": 1.5, "SHARED": 1.2, "SHAREDm": 1.1}
    names = ("O", *ratios)
    result = {
        "graph_id": graph_id, "n": n, "estimator": "linear_ols", "n_replicates": reps,
        "factors": {"n_inst": 1 if instrument else 0, "n_pm": 2},
        "prediction": {"predicted_equal": equal},
        "arms": {
            arm: {"admissible": True, "n_attempted": reps, "n_success": reps,
                  "results": {effect: {"n": reps, "truth": .3, "bias": bias, "mcse_mean": mcse,
                                       "variance": reps * mcse ** 2}
                              for effect in ("NIE", "NDE")}}
            for arm in names},
        "comparisons": [
            {"arm": arm, "reference_arm": "O", "effect": effect,
             "n_paired": reps, "variance_ratio": ratio}
            for arm, ratio in ratios.items() for effect in ("NIE", "NDE")],
    }
    return result


def effect_endpoints(output, effect="NIE", index=0):
    return output["scopes"][index]["effects"][effect]


class EndpointTests(unittest.TestCase):
    def test_original_five_endpoint_arithmetic_on_known_summaries(self):
        # The instrument cell has FULL/O=1.5, giving a 1/3 reduction.
        # The explicitly equal cell has FULL/O=1.0 and must be selected for E3.
        rows = [group(), group(1, instrument=False, equal=True,
                              ratios={"FULL": 1., "SHARED": 1., "SHAREDm": 1.})]
        result = evaluate_endpoints({"groups": rows})
        self.assertFalse(result["confirmation"])
        self.assertEqual(result["purpose"], "exploratory_endpoint_diagnostics")
        endpoints = effect_endpoints(result)
        self.assertEqual(len(endpoints), 5)
        self.assertTrue(all(row["status"] == "passed" for row in endpoints.values()))
        self.assertEqual(endpoints["E1_full_vs_o"]["statistic"], 1.)
        self.assertAlmostEqual(endpoints["E2_instrument_reduction"]["statistic"], 1/3)
        self.assertEqual(endpoints["E3_predicted_equality"]["range"], [1., 1.])
        self.assertEqual(endpoints["E4_better_shared"]["statistic"], 1.)
        self.assertEqual(endpoints["E5_admissibility_and_oracle_bias"]["statistic"], .5)

    def test_minimum_not_mean_controls_full_endpoint(self):
        rows = [group(), group(1, ratios={"FULL": .94, "SHARED": 2., "SHAREDm": 2.})]
        endpoints = effect_endpoints(evaluate_endpoints(rows))
        self.assertEqual(endpoints["E1_full_vs_o"]["status"], "failed")
        self.assertAlmostEqual(endpoints["E1_full_vs_o"]["statistic"], .94)

    def test_shared_endpoint_uses_better_shared_arm(self):
        rows = [group(ratios={"FULL": 2., "SHARED": .9, "SHAREDm": 1.8})]
        endpoint = effect_endpoints(evaluate_endpoints(rows))["E4_better_shared"]
        self.assertEqual(endpoint["status"], "failed")
        self.assertAlmostEqual(endpoint["statistic"], 1/.9)

    def test_shared_endpoint_only_uses_original_pure_m_panel(self):
        row = group()
        row["factors"]["n_pm"] = 0
        endpoint = effect_endpoints(evaluate_endpoints([row]))["E4_better_shared"]
        self.assertEqual(endpoint["status"], "not_evaluated")
        self.assertEqual(endpoint["n_used"], 0)
        del row["factors"]["n_pm"]
        endpoint = effect_endpoints(evaluate_endpoints([row]))["E4_better_shared"]
        self.assertEqual(endpoint["exclusions"][0]["reason"], "missing_pure_m_annotation")

    def test_equality_is_not_guessed_from_no_declared_instruments(self):
        row = group(instrument=False, equal=True)
        del row["prediction"]
        endpoint = effect_endpoints(evaluate_endpoints([row]))["E3_predicted_equality"]
        self.assertEqual(endpoint["status"], "not_evaluated")
        self.assertEqual(endpoint["n_used"], 0)
        self.assertEqual(endpoint["exclusions"][0]["reason"], "missing_equality_prediction")

    def test_graph_predictions_from_frozen_prototype_are_supported(self):
        row = group(instrument=False, ratios={"FULL": 1.})
        row["prediction"] = {"out_inst": False, "med_inst": False}
        endpoint = effect_endpoints(evaluate_endpoints([row]))["E3_predicted_equality"]
        self.assertEqual(endpoint["status"], "passed")
        row["prediction"]["out_inst"] = True
        endpoint = effect_endpoints(evaluate_endpoints([row]))["E3_predicted_equality"]
        self.assertEqual(endpoint["status"], "not_evaluated")

    def test_effect_specific_equality_annotations_are_respected(self):
        row = group(ratios={"FULL": 1.})
        row["prediction"]["predicted_equal"] = {"NIE": True, "NDE": False}
        output = evaluate_endpoints([row])
        self.assertEqual(effect_endpoints(output, "NIE")["E3_predicted_equality"]["status"], "passed")
        self.assertEqual(effect_endpoints(output, "NDE")["E3_predicted_equality"]["status"], "not_evaluated")
        # A missing effect-specific prediction is unknown, never guessed.
        del row["prediction"]["predicted_equal"]["NDE"]
        endpoint = effect_endpoints(evaluate_endpoints([row]), "NDE")["E3_predicted_equality"]
        self.assertEqual(endpoint["exclusions"][0]["reason"], "missing_equality_prediction")

    def test_small_smoke_run_is_inconclusive_not_passed(self):
        endpoints = effect_endpoints(evaluate_endpoints([group(reps=2, equal=True)]))
        self.assertTrue(all(item["status"] == "inconclusive" for item in endpoints.values()))
        output = evaluate_endpoints([group(reps=2)], thresholds={"min_replicates": 2})
        self.assertEqual(effect_endpoints(output)["E1_full_vs_o"]["status"], "passed")
        self.assertFalse(output["confirmation"])

    def test_complete_successful_paired_replicates_required(self):
        row = group()
        row["comparisons"][0]["n_paired"] = 199
        endpoint = effect_endpoints(evaluate_endpoints([row]))["E1_full_vs_o"]
        self.assertEqual(endpoint["status"], "inconclusive")
        self.assertEqual(endpoint["exclusions"][0]["reason"], "incomplete_paired_repetitions")

    def test_uses_paired_variance_not_unpaired_arm_metrics(self):
        row = group()
        row["arms"]["FULL"]["results"]["NIE"]["variance"] = 1e8
        endpoint = effect_endpoints(evaluate_endpoints([row]))["E1_full_vs_o"]
        self.assertEqual(endpoint["statistic"], 1.5)
        row["comparisons"] = []
        endpoint = effect_endpoints(evaluate_endpoints([row]))["E1_full_vs_o"]
        self.assertEqual(endpoint["status"], "not_evaluated")

    def test_missing_cells_prevent_unqualified_pass(self):
        row = group(1)
        del row["factors"]
        endpoint = effect_endpoints(evaluate_endpoints([group(), row]))["E2_instrument_reduction"]
        self.assertEqual(endpoint["status"], "inconclusive")
        self.assertEqual(endpoint["n_used"], 1)
        self.assertEqual(endpoint["n_excluded"], 1)

    def test_legality_and_bias_check_cannot_pass_when_main_arm_invalid(self):
        row = group()
        row["arms"]["FULL"]["admissible"] = False
        endpoints = effect_endpoints(evaluate_endpoints([row]))
        self.assertEqual(endpoints["E1_full_vs_o"]["status"], "not_evaluated")
        self.assertEqual(endpoints["E5_admissibility_and_oracle_bias"]["status"], "failed")
        self.assertEqual(endpoints["E5_admissibility_and_oracle_bias"]["invalid_main_arms"][0]["arm"], "FULL")

    def test_diagnostic_omission_arm_does_not_redefine_main_legality(self):
        row = group()
        row["arms"]["OMISS"] = {"admissible": False, "n_attempted": 200, "n_success": 200}
        endpoint = effect_endpoints(evaluate_endpoints([row]))["E5_admissibility_and_oracle_bias"]
        self.assertEqual(endpoint["status"], "passed")

    def test_bias_gate_uses_monte_carlo_standard_error(self):
        endpoint = effect_endpoints(evaluate_endpoints([group(bias=.007, mcse=.002)]))["E5_admissibility_and_oracle_bias"]
        self.assertEqual(endpoint["status"], "failed")
        self.assertEqual(endpoint["statistic"], 3.5)
        row = group()
        row["arms"]["O"]["results"]["NIE"]["mcse_mean"] = None
        endpoint = effect_endpoints(evaluate_endpoints([row]))["E5_admissibility_and_oracle_bias"]
        self.assertEqual(endpoint["status"], "not_evaluated")

    def test_effect_missing_from_successful_fit_prevents_bias_pass(self):
        row = group()
        row["arms"]["O"]["results"]["NIE"]["n"] = 199
        output = evaluate_endpoints([row])
        endpoint = effect_endpoints(output)["E5_admissibility_and_oracle_bias"]
        self.assertEqual(endpoint["status"], "inconclusive")
        self.assertEqual(effect_endpoints(output, "NDE")["E5_admissibility_and_oracle_bias"]["status"], "passed")

    def test_different_sample_sizes_and_estimators_are_not_pooled(self):
        first, second, third = group(), group(n=5000), group()
        third["estimator"] = "future_estimator"
        output = evaluate_endpoints([first, second, third])
        self.assertEqual(len(output["scopes"]), 3)
        self.assertTrue(all(scope["n_models"] == 1 for scope in output["scopes"]))

    def test_different_panel_ids_are_not_pooled(self):
        first, second = group(), group()
        first["panel_id"], second["panel_id"] = "original", "stress"
        self.assertEqual(len(evaluate_endpoints([first, second])["scopes"]), 2)

    def test_invalid_nonfinite_or_zero_variance_ratios_never_pass(self):
        for value in (None, float("nan"), float("inf"), 0., -1.):
            row = group()
            row["comparisons"][0]["variance_ratio"] = value
            with self.subTest(value=value):
                endpoint = effect_endpoints(evaluate_endpoints([row]))["E1_full_vs_o"]
                self.assertEqual(endpoint["status"], "not_evaluated")

    def test_deliberately_misspecified_linear_fit_cannot_pass_endpoints(self):
        row = group(equal=True)
        row["diagnostic_misspecification"] = True
        endpoints = effect_endpoints(evaluate_endpoints([row]))
        self.assertTrue(all(item["status"] == "not_evaluated" for item in endpoints.values()))
        self.assertTrue(all(item["exclusions"][0]["reason"] == "outside_linear_model_scope"
                            for item in endpoints.values()))

    def test_disabled_empty_and_strict_threshold_interfaces(self):
        output = evaluate_endpoints([], thresholds={"enabled": False})
        self.assertEqual(output["status"], "disabled")
        self.assertEqual(output["scopes"], [])
        self.assertEqual(evaluate_endpoints([])["scopes"], [])
        invalid = ({"unknown": 1}, {"min_replicates": True}, {"min_replicates": 1},
                   {"full_min_ratio": float("nan")}, {"equality_lower": 2.},
                   {"instrument_min_reduction": 1.}, {"max_bias_z": 0.})
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                evaluate_endpoints([], thresholds=values)
        with self.assertRaises(ValueError):
            evaluate_endpoints({"groups": "incorrect"})
        with self.assertRaises(ValueError):
            evaluate_endpoints([3])

    def test_evaluator_does_not_modify_input(self):
        data = {"groups": [group()]}
        before = deepcopy(data)
        evaluate_endpoints(data, thresholds=EndpointThresholds())
        self.assertEqual(data, before)


if __name__ == "__main__":
    unittest.main()
