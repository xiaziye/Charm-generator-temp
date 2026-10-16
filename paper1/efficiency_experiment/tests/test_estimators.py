"""Numerical and statistical checks of the estimator boundary."""
from dataclasses import replace
import unittest

import numpy as np

from paper1.efficiency_experiment.estimators import (
    EstimationError, EstimatorConfig, fit_estimate, register_estimator,
)
try:
    from paper1.estimation import product_estimate
except ModuleNotFoundError:
    # Independent reference from the frozen product estimator; keep standalone
    # tests usable when the rest of the paper-1 discovery package is absent.
    def product_estimate(values, CM, CY, X, M, Y):
        def slopes(target, predictors):
            design = np.column_stack((np.ones(len(target)), predictors))
            return np.linalg.lstsq(design, target, rcond=None)[0][1:]
        a = slopes(values[:, M], values[:, [X] + CM])[0]
        d, b = slopes(values[:, Y], values[:, [X, M] + CY])[:2]
        return a * b, d


def sample(n=300, seed=81, heteroskedastic=False):
    rng = np.random.default_rng(seed)
    c, x_noise, m_noise, y_noise = rng.normal(size=(4, n))
    x = .4 * c + x_noise
    m = .6 * x + .5 * c + m_noise
    scale = .3 + np.abs(x) if heteroskedastic else 1.
    y = .2 * x + .5 * m + .8 * c + scale * y_noise
    return np.column_stack((c, x, m, y)), ['C', 'X', 'M', 'Y']


class EstimatorTests(unittest.TestCase):
    def test_parity_with_previous_product_estimator_and_contrast(self):
        values, columns = sample()
        old_nie, old_nde = product_estimate(values, [0], [0], 1, 2, 3)
        unit = fit_estimate(values, columns, ['C'], ['C'])
        contrast = fit_estimate(values, columns, ['C'], ['C'], x0=2., x1=-1.)
        self.assertAlmostEqual(unit['NIE'], old_nie, places=12)
        self.assertAlmostEqual(unit['NDE'], old_nde, places=12)
        for effect in ('NIE', 'NDE', 'TE'):
            self.assertAlmostEqual(contrast[effect], -3 * unit[effect], places=12)
            self.assertAlmostEqual(contrast['se_' + effect], 3 * unit['se_' + effect], places=12)
            self.assertLess(contrast['ci_' + effect][0], contrast['ci_' + effect][1])
        self.assertFalse(unit['truth_used'])

    def test_sandwich_agrees_with_independent_delete_one_jackknife(self):
        values, columns = sample(n=170, heteroskedastic=True)
        estimate = fit_estimate(values, columns, ['C'], ['C'])
        point_only = EstimatorConfig(se_method='none')
        leave_one_out = []
        for k in range(len(values)):
            retained = np.delete(values, k, axis=0)
            fit = fit_estimate(retained, columns, ['C'], ['C'], config=point_only)
            leave_one_out.append([fit[key] for key in ('NIE', 'NDE', 'TE')])
        jackknife = np.asarray(leave_one_out)
        jackknife -= jackknife.mean(axis=0)
        jackknife_cov = (len(values) - 1) / len(values) * jackknife.T @ jackknife
        sandwich = np.asarray(estimate['effect_covariance'])
        np.testing.assert_allclose(sandwich, jackknife_cov, rtol=.12, atol=.0005)
        # The TE error depends on both effects and their joint covariance.
        self.assertAlmostEqual(sandwich[2, 2], sandwich[0, 0] + sandwich[1, 1] + 2 * sandwich[0, 1], places=13)

    def test_standard_errors_track_repeated_sampling_sd(self):
        estimates, standard_errors = [], []
        for seed in range(140):
            values, columns = sample(n=350, seed=seed, heteroskedastic=True)
            fit = fit_estimate(values, columns, ['C'], ['C'])
            estimates.append([fit[k] for k in ('NIE', 'NDE')])
            standard_errors.append([fit['se_' + k] for k in ('NIE', 'NDE')])
        empirical = np.std(estimates, axis=0, ddof=1)
        reported = np.mean(standard_errors, axis=0)
        self.assertTrue(np.all(np.abs(reported / empirical - 1) < .20), (reported, empirical))
        np.testing.assert_allclose(np.mean(estimates, axis=0), [.3, .2], atol=.025)

    def test_bootstrap_reproducible_and_custom_confidence(self):
        values, columns = sample(n=100)
        config = EstimatorConfig(se_method='bootstrap', bootstrap_replicates=39, confidence_level=.8)
        a = fit_estimate(values, columns, ['C'], ['C'], config=config)
        b = fit_estimate(values, columns, ['C'], ['C'], config=config)
        self.assertEqual(a['ci_NIE'], b['ci_NIE'])
        self.assertEqual(a['se_NDE'], b['se_NDE'])
        self.assertNotIn('ci95_NIE', a)
        self.assertGreater(a['se_NIE'], 0)
        none = fit_estimate(values, columns, ['C'], ['C'], config=EstimatorConfig(se_method='none'))
        self.assertIsNone(none['ci_NIE'])
        self.assertIsNone(none['se_NIE'])

    def test_refuses_invalid_data_and_underdetermined_regression(self):
        values, columns = sample(n=10)
        bad = values.copy()
        bad[:, 0] = 1
        for data, names, cm, cy in (
            (bad, columns, ['C'], ['C']),
            (values[:4], columns, ['C'], ['C']),
            (values, columns, ['C', 'C'], []),
            (values, columns, ['missing'], []),
            (values, columns, ['X'], []),
            (values, ['C', 'X', 'M', 'M'], [], []),
        ):
            with self.subTest(names=names, cm=cm), self.assertRaises(EstimationError):
                fit_estimate(data, names, cm, cy)
        with self.assertRaises(EstimationError):
            fit_estimate(values, columns, [], [], x0=1, x1=1)

    def test_external_estimator_registration_and_contract(self):
        def external(data, columns, CM, CY, **kwargs):
            self.assertNotIn('truth', kwargs)
            return dict(NIE=.3, NDE=.2, TE=.5)
        name = 'test_external_no_truth'
        register_estimator(name, external)
        values, columns = sample(n=20)
        fit = fit_estimate(values, columns, [], [], config=EstimatorConfig(name=name))
        self.assertEqual(fit['TE'], .5)
        self.assertFalse(fit['truth_used'])
        with self.assertRaises(ValueError):
            register_estimator(name, external)
        with self.assertRaises(ValueError):
            fit_estimate(values, columns, [], [], config=EstimatorConfig(name='not_registered'))

    def test_plugin_with_non_json_result_fails_at_estimator_boundary(self):
        register_estimator('test_non_json', lambda *args, **kwargs: dict(NIE=.3, NDE=.2, TE=.5, extra=np.array([1, 2])))
        values, columns = sample(n=20)
        with self.assertRaisesRegex(EstimationError, 'JSON-compatible'):
            fit_estimate(values, columns, [], [], config=EstimatorConfig(name='test_non_json'))

    def test_plugin_options_are_available_without_schema_changes(self):
        def plugin(data, columns, CM, CY, **kwargs):
            self.assertEqual(kwargs['config'].options['degree'], 3)
            return dict(NIE=.3, NDE=.2, TE=.5)
        register_estimator('test_plugin_options', plugin)
        values, columns = sample(n=20)
        result = fit_estimate(values, columns, [], [], config=EstimatorConfig(name='test_plugin_options', options={'degree':3}))
        self.assertEqual(result['NIE'], .3)

    def test_config_validation(self):
        for changes in ({'confidence_level':1.}, {'max_condition_number':float('nan')},
                        {'bootstrap_seed':True}, {'bootstrap_replicates':1},
                        {'allow_misspecification':'yes'}, {'se_method':'unknown'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(EstimatorConfig(), **changes).validate()


if __name__ == '__main__':
    unittest.main()
