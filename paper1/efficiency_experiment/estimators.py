"""Extensible effect-estimation interface for fixed, known-graph experiments.

An estimator sees observational data and the chosen adjustment pair. It never
receives the generating coefficients, graph or truth. Built-in OLS retains the
cross-stage covariance when forming product-effect standard errors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from statistics import NormalDist
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np


class EstimationError(ValueError):
    """A fit is not identified numerically under the requested estimator."""


@dataclass(frozen=True)
class EstimatorConfig:
    name: str = 'linear_ols'
    se_method: str = 'sandwich'  # sandwich | none | bootstrap
    confidence_level: float = 0.95
    max_condition_number: float = 1e12
    bootstrap_replicates: int = 199
    bootstrap_seed: int = 20261016
    allow_misspecification: bool = False
    options: dict = field(default_factory=dict)  # JSON-compatible plugin-specific parameters

    def validate(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('estimator.name must be a nonempty string')
        if self.se_method not in ('sandwich', 'none', 'bootstrap'):
            raise ValueError('estimator.se_method must be sandwich, none or bootstrap')
        for field in ('confidence_level', 'max_condition_number'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'estimator.{field} must be finite')
        if not 0 < self.confidence_level < 1:
            raise ValueError('estimator.confidence_level must be strictly between 0 and 1')
        if self.max_condition_number <= 1:
            raise ValueError('estimator.max_condition_number must exceed 1')
        if isinstance(self.bootstrap_replicates, bool) or not isinstance(self.bootstrap_replicates, int) or self.bootstrap_replicates < 2:
            raise ValueError('estimator.bootstrap_replicates must be an integer >= 2')
        if isinstance(self.bootstrap_seed, bool) or not isinstance(self.bootstrap_seed, int) or self.bootstrap_seed < 0:
            raise ValueError('estimator.bootstrap_seed must be a nonnegative integer')
        if not isinstance(self.allow_misspecification, bool):
            raise ValueError('estimator.allow_misspecification must be boolean')
        if not isinstance(self.options, dict) or any(not isinstance(k, str) for k in self.options):
            raise ValueError('estimator.options must be a table with string keys')
        try:
            json.dumps(self.options, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('estimator.options must contain JSON-compatible finite values') from exc
        if self.name == 'linear_ols' and self.options:
            raise ValueError('linear_ols has no estimator.options; use the documented common settings')
        return self

    def to_dict(self):
        return asdict(self)


_ESTIMATORS: dict[str, Callable] = {}


def register_estimator(name: str, function: Callable | None = None):
    """Register a callable, directly or as a decorator; existing names cannot be replaced.

    The callable has the same keyword contract as ``fit_estimate`` and returns a
    JSON-ready dict containing NIE, NDE and TE. Import the plugin before invoking
    the runner; a new method name keeps old results reproducible.
    """
    if not isinstance(name, str) or not name:
        raise ValueError('estimator name must be a nonempty string')

    def install(fn):
        if name in _ESTIMATORS:
            raise ValueError(f'estimator already registered: {name}')
        if not callable(fn):
            raise TypeError('estimator implementation must be callable')
        _ESTIMATORS[name] = fn
        return fn
    return install if function is None else install(function)


def available_estimators():
    return tuple(sorted(_ESTIMATORS))


def _inputs(data, columns, CM, CY, x, m, y, x0, x1):
    values = np.asarray(data, dtype=float)
    columns = tuple(columns)
    if not columns or any(not isinstance(c, str) or not c for c in columns) or len(set(columns)) != len(columns):
        raise EstimationError('column names must be unique nonempty strings')
    if values.ndim != 2 or values.shape[1] != len(columns) or values.shape[0] < 2:
        raise EstimationError('data must have shape (n >= 2, len(columns))')
    if len({x, m, y}) != 3 or not {x, m, y} <= set(columns):
        raise EstimationError('X, M and Y must name three different data columns')
    for label, stage in (('CM', CM), ('CY', CY)):
        if len(stage) != len(set(stage)):
            raise EstimationError(f'{label} has duplicate columns')
        if not set(stage) <= set(columns):
            raise EstimationError(f'{label} names unknown columns')
        if set(stage) & {x, m, y}:
            raise EstimationError(f'{label} cannot include X, M or Y')
    # Unused columns may be huge; only variables used by this fit must be finite.
    ix = {c: i for i, c in enumerate(columns)}
    used = sorted({ix[c] for c in (x, m, y, *CM, *CY)})
    for start in range(0, len(used), 128):
        if not np.isfinite(values[:, used[start:start + 128]]).all():
            raise EstimationError('fitting variables contain non-finite values')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in (x0, x1)):
        raise EstimationError('x0 and x1 must be finite numbers')
    if x0 == x1:
        raise EstimationError('x0 and x1 must specify different exposure levels')
    return values, ix


def _ols(y, predictors, max_condition_number, coefficient_indices):
    design = np.column_stack((np.ones(len(y)), predictors))
    n, k = design.shape
    if n <= k:
        raise EstimationError(f'OLS requires n > number of coefficients including intercept; got {n} <= {k}')
    u, singular, vt = np.linalg.svd(design, full_matrices=False)
    tolerance = singular[0] * max(design.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > tolerance))
    if rank < k:
        raise EstimationError(f'rank-deficient regression: rank={rank}, coefficients={k}')
    condition = float(singular[0] / singular[-1])
    if condition > max_condition_number:
        raise EstimationError(f'regression condition number {condition:.6g} exceeds {max_condition_number:.6g}')
    beta = vt.T @ ((u.T @ y) / singular)
    residual = y - design @ beta
    # For coefficient q, each row contribution is residual_i times the ith
    # element of (Z'Z)^(-1) Z' in row q. Its sum is zero at the OLS solution.
    contributions = np.column_stack([
        (u @ (vt[:, j] / singular)) * residual * np.sqrt(n / (n - k))
        for j in coefficient_indices
    ])
    diag = dict(n=n, n_predictors=k - 1, n_coefficients=k, rank=rank,
                df_residual=n - k, condition_number=condition,
                residual_sd=float(np.sqrt(residual @ residual / (n - k))))
    return beta, contributions, diag


def _linear_fit(values, ix, CM, CY, x, m, y, delta, settings):
    mediator, ia, dm = _ols(values[:, ix[m]], values[:, [ix[x]] + [ix[c] for c in CM]],
                            settings.max_condition_number, (1,))
    outcome, idb, dy = _ols(values[:, ix[y]], values[:, [ix[x], ix[m]] + [ix[c] for c in CY]],
                            settings.max_condition_number, (1, 2))
    a, d, b = float(mediator[1]), float(outcome[1]), float(outcome[2])
    estimates = np.array([delta * a * b, delta * d, delta * (a * b + d)])
    if not np.isfinite(estimates).all():
        raise EstimationError('effect estimates are non-finite')
    coefficients = dict(a=a, b=b, d=d, intercept_M=float(mediator[0]), intercept_Y=float(outcome[0]),
                        mediator_adjustment={c: float(v) for c, v in zip(CM, mediator[2:])},
                        outcome_adjustment={c: float(v) for c, v in zip(CY, outcome[3:])})
    # Each equation has its own HC1 degrees-of-freedom correction. Stacking
    # row contributions preserves their cross-stage empirical covariance.
    influence_nie = delta * (b * ia[:, 0] + a * idb[:, 1])
    influence_nde = delta * idb[:, 0]
    influences = np.column_stack((influence_nie, influence_nde, influence_nie + influence_nde))
    return estimates, influences, coefficients, dict(mediator=dm, outcome=dy)


@register_estimator('linear_ols')
def _estimate_linear(data, columns, CM, CY, *, x='X', m='M', y='Y', x0=0., x1=1., config=None):
    settings = config or EstimatorConfig()
    values, ix = _inputs(data, columns, CM, CY, x, m, y, x0, x1)
    delta = float(x1 - x0)
    estimates, influences, coefficients, diagnostics = _linear_fit(values, ix, CM, CY, x, m, y, delta, settings)
    covariance = None
    intervals = None
    if settings.se_method == 'sandwich':
        covariance = influences.T @ influences
        se = np.sqrt(np.maximum(np.diag(covariance), 0.))
        z = NormalDist().inv_cdf((1. + settings.confidence_level) / 2.)
        intervals = np.column_stack((estimates - z * se, estimates + z * se))
        method = 'stacked HC1 sandwich; product delta method including cross-stage covariance; normal CI'
    elif settings.se_method == 'bootstrap':
        rng = np.random.default_rng(settings.bootstrap_seed)
        draws = np.empty((settings.bootstrap_replicates, 3))
        for j in range(settings.bootstrap_replicates):
            indices = rng.integers(len(values), size=len(values))
            try:
                draws[j] = _linear_fit(values[indices], ix, CM, CY, x, m, y, delta, settings)[0]
            except EstimationError as exc:
                # Do not silently condition on successful bootstrap fits.
                raise EstimationError(f'bootstrap fit {j} failed: {exc}') from exc
        covariance = np.cov(draws, rowvar=False, ddof=1)
        se = np.sqrt(np.maximum(np.diag(covariance), 0.))
        tail = (1. - settings.confidence_level) / 2.
        intervals = np.quantile(draws, (tail, 1. - tail), axis=0).T
        method = 'paired-row nonparametric bootstrap; percentile CI'
        diagnostics['bootstrap'] = dict(replicates=settings.bootstrap_replicates, seed=settings.bootstrap_seed)
    else:
        se = [None, None, None]
        method = None
    result = dict(estimator=settings.name, n=len(values), CM=list(CM), CY=list(CY),
                  coefficients=coefficients, diagnostics=diagnostics, se_method=method,
                  confidence_level=settings.confidence_level, contrast=dict(x0=float(x0), x1=float(x1), delta_x=delta),
                  effect_order=['NIE', 'NDE', 'TE'],
                  effect_covariance=None if covariance is None else covariance.tolist(), truth_used=False)
    for j, effect in enumerate(('NIE', 'NDE', 'TE')):
        result[effect] = float(estimates[j])
        result['se_' + effect] = None if se[j] is None else float(se[j])
        result['ci_' + effect] = None if intervals is None else intervals[j].tolist()
        if settings.confidence_level == .95:
            result['ci95_' + effect] = result['ci_' + effect]
    return result


def fit_estimate(data, columns: Sequence[str], CM: Sequence[str], CY: Sequence[str], *,
                 x='X', m='M', y='Y', x0=0., x1=1., config: EstimatorConfig | Mapping | None = None):
    """Fit one adjustment pair on one sample, with explicit units and diagnostics.

    A plugin is selected by ``config.name``; register it before calling. The
    adapter validates the common inputs and required JSON result fields for all
    estimators. Generated graph, mechanism and effect truth are absent by design.
    """
    settings = EstimatorConfig(**config) if isinstance(config, Mapping) else config or EstimatorConfig()
    if not isinstance(settings, EstimatorConfig):
        raise TypeError('config must be EstimatorConfig, a mapping, or None')
    settings.validate()
    if settings.name not in _ESTIMATORS:
        raise ValueError(f'unknown estimator {settings.name!r}; available: {available_estimators()}')
    CM, CY = tuple(CM), tuple(CY)
    if settings.name == 'linear_ols':
        shape = np.shape(data)
        if len(shape) == 2 and (shape[0] <= len(CM) + 2 or shape[0] <= len(CY) + 3):
            raise EstimationError('OLS requires more rows than coefficients including intercept in both stages')
    values, _ = _inputs(data, columns, CM, CY, x, m, y, x0, x1)
    started = perf_counter()
    result = _ESTIMATORS[settings.name](values, tuple(columns), CM, CY,
                                       x=x, m=m, y=y, x0=x0, x1=x1, config=settings)
    if not isinstance(result, dict):
        raise TypeError('estimator must return a dict')
    for effect in ('NIE', 'NDE', 'TE'):
        value = result.get(effect)
        if value is None or not np.isscalar(value) or not np.isfinite(value):
            raise EstimationError(f'estimator returned missing/non-finite {effect}')
    if not np.isclose(result['TE'], result['NIE'] + result['NDE'], rtol=1e-10, atol=1e-12):
        raise EstimationError('estimator must use the same natural-effect decomposition: TE = NIE + NDE')
    if result.get('truth_used', False):
        raise EstimationError('estimator must not use effect truth')
    result.update(estimator=settings.name, truth_used=False, seconds=perf_counter() - started)
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EstimationError('estimator result must be JSON-compatible and finite (use None for missing values)') from exc
    return result
