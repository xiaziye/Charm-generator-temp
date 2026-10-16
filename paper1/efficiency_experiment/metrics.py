"""Summarize repeated sampling conditional on one frozen graph and mechanism.

No estimate from one graph is pooled into another graph's sampling variance. A paired
variance ratio uses exactly the successful replicate ids shared by the two arms.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable, Mapping

import numpy as np

EFFECTS = ("NIE", "NDE", "TE")
SUCCESS_STATUSES = frozenset(("ok", "success", "complete"))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def _as_float(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _mcse(values: np.ndarray) -> float | None:
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) >= 2 else None


def _variance(values: np.ndarray) -> tuple[float | None, float | None]:
    """Sample variance and its plug-in influence-function Monte Carlo SE."""
    n = len(values)
    if n < 2:
        return None, None
    variance = float(np.var(values, ddof=1))
    squared = (values - np.mean(values)) ** 2 * (n / (n - 1))
    return variance, _mcse(squared) if n >= 4 else None


def _result(record: Mapping[str, Any]) -> Mapping[str, Any]:
    # Flat records are the file schema; nested results are convenient for API clients.
    return record.get("estimate") or record.get("estimates") or record


def _truth(record: Mapping[str, Any], effect: str) -> float | None:
    if _finite(record.get("truth_" + effect)):
        return float(record["truth_" + effect])
    truth = record.get("truth", {})
    if isinstance(truth, Mapping):
        return _as_float(truth.get(effect))
    return None


def _success(record: Mapping[str, Any]) -> bool:
    return (record.get("status", "ok") in SUCCESS_STATUSES
            and all(_finite(_result(record).get(effect)) for effect in EFFECTS))


def _effect_summary(records: list[Mapping[str, Any]], effect: str) -> dict[str, Any]:
    usable = [r for r in records if _success(r) and _finite(_result(r).get(effect))]
    values = np.asarray([_result(r)[effect] for r in usable], dtype=float)
    truths = [_truth(r, effect) for r in usable]
    finite_truths = [t for t in truths if t is not None]
    if finite_truths and any(not np.isclose(t, finite_truths[0], rtol=1e-12, atol=1e-12) for t in finite_truths):
        raise ValueError(f"{effect} truth changed within one frozen-model summary group")
    truth = finite_truths[0] if finite_truths and len(finite_truths) == len(truths) else None
    variance, variance_mcse = _variance(values)
    result = dict(n=len(values), truth=truth, mean=float(values.mean()) if len(values) else None,
                  bias=float(values.mean() - truth) if len(values) and truth is not None else None,
                  mcse_mean=_mcse(values), mcse_bias=_mcse(values), variance=variance,
                  mcse_variance=variance_mcse, sd=math.sqrt(variance) if variance is not None else None,
                  mse=None, mcse_mse=None, rmse=None, mean_abs_error=None, mae=None, mcse_mae=None,
                  mean_se=None, mcse_mean_se=None, n_se=0,
                  coverage=None, mcse_coverage=None, n_intervals=0,
                  mean_interval_width=None, mcse_interval_width=None)
    if len(values) and truth is not None:
        losses = (values - truth) ** 2
        absolute = np.abs(values - truth)
        result.update(mse=float(losses.mean()), mcse_mse=_mcse(losses), rmse=float(np.sqrt(losses.mean())),
                      mean_abs_error=float(absolute.mean()), mae=float(absolute.mean()), mcse_mae=_mcse(absolute))
    ses = np.asarray([_result(r)["se_" + effect] for r in usable
                      if _finite(_result(r).get("se_" + effect)) and _result(r)["se_" + effect] >= 0], dtype=float)
    if len(ses):
        result.update(mean_se=float(ses.mean()), mcse_mean_se=_mcse(ses), n_se=len(ses))
    intervals, indicators = [], []
    for r in usable:
        fitted = _result(r)
        interval = fitted.get("ci_" + effect, fitted.get("ci95_" + effect))
        if isinstance(interval, (list, tuple, np.ndarray)) and len(interval) == 2 and all(_finite(v) for v in interval):
            lo, hi = map(float, interval)
            if lo <= hi:
                intervals.append(hi - lo)
                target = _truth(r, effect)
                if target is not None:
                    indicators.append(float(lo <= target <= hi))
    if intervals:
        widths = np.asarray(intervals)
        result.update(n_intervals=len(widths), mean_interval_width=float(widths.mean()), mcse_interval_width=_mcse(widths))
    if indicators:
        coverage = float(np.mean(indicators))
        result.update(coverage=coverage, mcse_coverage=math.sqrt(coverage * (1 - coverage) / len(indicators)),
                      n_coverage=len(indicators))
    else:
        result["n_coverage"] = 0
    return result


def paired_variance_ratio(records: Iterable[Mapping[str, Any]], arm: str, reference_arm: str,
                          effect: str) -> dict[str, Any]:
    """Variance(arm)/Variance(reference), conditional on both arms succeeding.

    The MCSE accounts for the dependence induced by using the same generated data
    for both arms. It is a first-order delta approximation, not a confidence proof.
    """
    selected: dict[str, dict[Any, float]] = {arm: {}, reference_arm: {}}
    for row in records:
        name = row.get("arm")
        if name not in selected or not _success(row) or not _finite(_result(row).get(effect)):
            continue
        replicate = row.get("replicate", row.get("replicate_id"))
        if replicate is None:
            raise ValueError("paired variance ratios require a replicate id")
        if replicate in selected[name]:
            raise ValueError(f"duplicate replicate {replicate!r} for arm {name!r}")
        selected[name][replicate] = float(_result(row)[effect])
    ids = sorted(selected[arm].keys() & selected[reference_arm].keys(), key=repr)
    x = np.asarray([selected[arm][r] for r in ids]); y = np.asarray([selected[reference_arm][r] for r in ids])
    output = dict(arm=arm, reference_arm=reference_arm, effect=effect, n_paired=len(ids),
                  variance_arm=None, variance_reference=None, variance_ratio=None,
                  variance_ratio_mcse=None, variance_reduction_reference_vs_arm=None,
                  status="insufficient_paired_replicates")
    if len(ids) < 2:
        return output
    n = len(ids)
    vx, _ = _variance(x); vy, _ = _variance(y)
    output.update(variance_arm=vx, variance_reference=vy)
    if vy == 0:
        output["status"] = "zero_reference_variance"
        return output
    ratio = vx / vy
    ux = (x - x.mean()) ** 2 * n / (n - 1)
    uy = (y - y.mean()) ** 2 * n / (n - 1)
    influence = (ux - ratio * uy) / vy
    output.update(variance_ratio=float(ratio), variance_ratio_mcse=_mcse(influence) if n >= 4 else None,
                  variance_reduction_reference_vs_arm=(float(1 - 1 / ratio) if ratio > 0 else None), status="ok",
                  mcse_note=("too_few_repeats_for_variance_mcse" if n < 4 else "paired first-order delta approximation"))
    return output


def summarize(records: Iterable[Mapping[str, Any]], reference_arm: str = "O", *, min_replicates: int = 30) -> dict[str, Any]:
    """Aggregate raw fit records; return JSON-safe fixed-model groups and comparisons.

    Required identity: graph_id, n, arm, replicate. Successful rows contain NIE/NDE,
    optional se_NIE/ci_NIE, and truth_NIE/truth_NDE. Missing effects stay missing.
    ``status`` is ok/success for fits and failed/skipped otherwise.
    """
    if isinstance(min_replicates, bool) or not isinstance(min_replicates, int) or min_replicates < 2:
        raise ValueError("min_replicates must be an integer >= 2")
    rows = list(records)
    buckets = defaultdict(list)
    seen = set()
    for row in rows:
        for key in ("graph_id", "n", "arm"):
            if key not in row:
                raise ValueError(f"summary row is missing {key!r}")
        replicate = row.get("replicate", row.get("replicate_id"))
        if replicate is None:
            raise ValueError("summary row is missing 'replicate'")
        estimator = row.get("estimator", _result(row).get("estimator", "unknown"))
        key = (row["graph_id"], row["n"], estimator)
        identity = key + (row["arm"], replicate)
        if identity in seen:
            raise ValueError(f"duplicate fit record: {identity!r}")
        seen.add(identity)
        buckets[key].append(row)
    groups = []
    for (graph_id, n, estimator), group_rows in sorted(buckets.items(), key=lambda kv: repr(kv[0])):
        for effect in EFFECTS:
            targets = [_truth(r, effect) for r in group_rows if _truth(r, effect) is not None]
            if targets and any(not np.isclose(t, targets[0], rtol=1e-12, atol=1e-12) for t in targets):
                raise ValueError(f"{effect} truth changed between arms/repetitions of one frozen-model group")
        levels = {_result(r).get("confidence_level") for r in group_rows if _success(r)}
        if len(levels) > 1:
            raise ValueError("confidence_level changed within one fixed-model summary group")
        confidence_level = next(iter(levels)) if levels else None
        if confidence_level is not None and (not _finite(confidence_level) or not 0 < confidence_level < 1):
            raise ValueError("confidence_level must be strictly between 0 and 1")
        planned_values = {r["n_replicates_planned"] for r in group_rows if "n_replicates_planned" in r}
        if len(planned_values) > 1:
            raise ValueError("n_replicates_planned changed within one fixed-model summary group")
        observed_replicates = len({r.get("replicate", r.get("replicate_id")) for r in group_rows})
        planned = next(iter(planned_values)) if planned_values else observed_replicates
        if isinstance(planned, bool) or not isinstance(planned, int) or planned < observed_replicates:
            raise ValueError("n_replicates_planned must be an integer >= observed repetitions")
        arm_rows = defaultdict(list)
        for row in group_rows:
            arm_rows[row["arm"]].append(row)
        arms = {}
        for name, entries in sorted(arm_rows.items()):
            validities = {r.get("admissible") for r in entries if r.get("admissible") is not None}
            if len(validities) > 1:
                raise ValueError(f"admissibility changed within arm {name!r} of graph {graph_id!r}")
            statuses = Counter(r.get("status", "ok") for r in entries)
            successes = sum(_success(r) for r in entries)
            skipped = sum(r.get("status", "").startswith("skip") or r.get("status") == "unavailable" for r in entries)
            failures = len(entries) - successes - skipped
            reasons = Counter(str(r.get("reason", r.get("error", "missing_or_non_finite_effect" if r.get("status", "ok") in SUCCESS_STATUSES else "unspecified")))
                              for r in entries if not _success(r))
            times = np.asarray([_result(r).get("seconds", r.get("seconds")) for r in entries
                                if _finite(_result(r).get("seconds", r.get("seconds")))], dtype=float)
            diagnostics = {}
            for stage in ("mediator", "outcome"):
                entries_diag = [_result(r).get("diagnostics", {}).get(stage, {}) for r in entries if _success(r)]
                diagnostics[stage] = {}
                for source, target, reducer in (("condition_number", "max_condition_number", max),
                                                ("n_predictors", "max_n_predictors", max),
                                                ("n_coefficients", "max_n_coefficients", max),
                                                ("df_residual", "min_df_residual", min)):
                    values = [d[source] for d in entries_diag if _finite(d.get(source))]
                    diagnostics[stage][target] = float(reducer(values)) if values else None
            arms[name] = dict(admissible=next(iter(validities)) if validities else None,
                              n_attempted=len(entries), n_success=successes, n_failed=failures, n_skipped=skipped,
                              failure_rate=failures / len(entries), exclusion_rate=skipped / len(entries),
                              non_success_rate=(len(entries) - successes) / len(entries), statuses=dict(statuses),
                              non_success_reasons=dict(reasons), confidence_level=confidence_level,
                              seconds_total=float(times.sum()) if len(times) else None,
                              seconds_mean=float(times.mean()) if len(times) else None,
                              seconds_median=float(np.median(times)) if len(times) else None,
                              numerical_diagnostics=diagnostics,
                              results={effect: _effect_summary(entries, effect) for effect in EFFECTS})
        comparisons = [paired_variance_ratio(group_rows, arm, reference_arm, effect)
                       for arm in sorted(arms) if arm != reference_arm for effect in EFFECTS]
        groups.append(dict(graph_id=graph_id, n=n, estimator=estimator,
                           n_replicates=planned, n_replicates_observed=observed_replicates,
                           confidence_level=confidence_level,
                           diagnostic_misspecification=any(bool(r.get("diagnostic_misspecification", False)) for r in group_rows),
                           diagnostic_non_admissible=any(bool(r.get("diagnostic_non_admissible", False)) for r in group_rows),
                           precision_note=("insufficient_repetitions_for_precision_guard" if planned < min_replicates
                                           else "Monte Carlo SEs remain necessary; the repetition guard is not a precision guarantee"),
                           factors=group_rows[0].get("factors", {}), prediction=group_rows[0].get("prediction", {}),
                           arms=arms, comparisons=comparisons))
        for field in ("design_id", "panel_id"):
            if field in group_rows[0]:
                groups[-1][field] = group_rows[0][field]
    return dict(schema_version=1, reference_arm=reference_arm, n_records=len(rows), groups=groups,
                interpretation="Conditional on each frozen graph/mechanism; failed fits are excluded from estimates and paired ratios, with failure rates reported separately. Variance-ratio MCSE is a paired delta approximation.")
