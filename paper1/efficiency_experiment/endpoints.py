"""Exploratory checks of the five original known-DAG efficiency endpoints.

The unit of comparison is a fixed graph/mechanism and sample size.  Results from
unlike graph truths must never be pooled to calculate a sampling variance.
Passing these numerical checks does not constitute a preregistered confirmation
or a verification of the paper's efficient-influence-function variance theorem.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import math
from statistics import median
from typing import Any

EFFECTS = ("NIE", "NDE")
MAIN_ARMS = ("O", "FULL", "SHARED", "SHAREDm")


@dataclass(frozen=True)
class EndpointThresholds:
    """The original A2 numerical cutoffs, with an explicit repetition guard.

    ``min_replicates`` is a diagnostic precision guard, not a promise that this
    repetition count suffices for every design.  A formal study must freeze its
    full panel, repetition count, seeds and criteria before running it.
    """

    enabled: bool = True
    min_replicates: int = 100
    full_min_ratio: float = 0.95
    instrument_min_reduction: float = 0.20
    equality_lower: float = 0.95
    equality_upper: float = 1.05
    shared_max_o_ratio: float = 1.05
    max_bias_z: float = 3.0

    def validate(self) -> "EndpointThresholds":
        if not isinstance(self.enabled, bool):
            raise ValueError("endpoints.enabled must be boolean")
        if (isinstance(self.min_replicates, bool)
                or not isinstance(self.min_replicates, int)
                or self.min_replicates < 2):
            raise ValueError("endpoints.min_replicates must be an integer >= 2")
        for field in fields(self):
            if field.name in {"enabled", "min_replicates"}:
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"endpoints.{field.name} must be a finite number")
        if self.full_min_ratio <= 0 or self.shared_max_o_ratio <= 0:
            raise ValueError("Variance-ratio bounds must be positive")
        if not 0 <= self.instrument_min_reduction < 1:
            raise ValueError("instrument_min_reduction must lie in [0, 1)")
        if not 0 < self.equality_lower <= self.equality_upper:
            raise ValueError("Require 0 < equality_lower <= equality_upper")
        if self.max_bias_z <= 0:
            raise ValueError("max_bias_z must be positive")
        return self


def _thresholds(value: EndpointThresholds | Mapping[str, Any] | None) -> EndpointThresholds:
    if value is None:
        return EndpointThresholds().validate()
    if isinstance(value, EndpointThresholds):
        return value.validate()
    if not isinstance(value, Mapping):
        raise ValueError("endpoints thresholds must be a mapping or EndpointThresholds")
    allowed = {field.name for field in fields(EndpointThresholds)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown endpoint settings: {sorted(unknown)}")
    return EndpointThresholds(**dict(value)).validate()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _group_id(group: Mapping[str, Any]) -> dict[str, Any]:
    return {key: group.get(key) for key in ("design_id", "graph_id", "n", "estimator") if key in group}


def _annotation(group: Mapping[str, Any], name: str) -> Any:
    if name in group:
        return group[name]
    for section in ("prediction", "annotations", "factors"):
        values = group.get(section)
        if isinstance(values, Mapping) and name in values:
            return values[name]
    return None


def _has_instrument(group: Mapping[str, Any]) -> bool | None:
    explicit = _annotation(group, "has_instrument")
    if isinstance(explicit, bool):
        return explicit
    count = _number(_annotation(group, "n_inst"))
    return None if count is None or count < 0 else count >= 1


def _has_pure_m(group: Mapping[str, Any]) -> bool | None:
    explicit = _annotation(group, "has_pure_m_predictor")
    if isinstance(explicit, bool):
        return explicit
    count = _number(_annotation(group, "n_pm"))
    return None if count is None or count < 0 else count >= 1


def _predicted_equal(group: Mapping[str, Any], effect: str) -> bool | None:
    explicit = _annotation(group, "predicted_equal")
    if isinstance(explicit, Mapping):
        value = explicit.get(effect)
        return value if isinstance(value, bool) else None
    if isinstance(explicit, bool):
        return explicit
    out_inst = _annotation(group, "out_inst")
    med_inst = _annotation(group, "med_inst")
    if isinstance(out_inst, bool) and isinstance(med_inst, bool):
        return not out_inst and not med_inst
    return None


def _arms(group: Mapping[str, Any]) -> Mapping[str, Any]:
    arms = group.get("arms", {})
    return arms if isinstance(arms, Mapping) else {}


def _comparison(group: Mapping[str, Any], arm: str, effect: str) -> Mapping[str, Any] | None:
    for item in group.get("comparisons", ()):
        if (isinstance(item, Mapping) and item.get("arm") == arm
                and item.get("reference_arm", "O") == "O" and item.get("effect") == effect):
            return item
    return None


def _ratio(group: Mapping[str, Any], arm: str, effect: str,
           config: EndpointThresholds) -> tuple[float | None, str | None]:
    """Use paired successful repetitions; require a complete planned panel."""
    if group.get("diagnostic_misspecification") is True:
        return None, "outside_linear_model_scope"
    arms = _arms(group)
    for name in ("O", arm):
        if name not in arms:
            return None, f"missing_arm:{name}"
        if arms[name].get("admissible") is not True:
            return None, f"inadmissible_or_unknown:{name}"
    comparison = _comparison(group, arm, effect)
    if comparison is None:
        return None, "missing_paired_comparison"
    value = _number(comparison.get("variance_ratio"))
    if value is None or value <= 0:
        return None, "undefined_variance_ratio"
    paired = _number(comparison.get("n_paired"))
    planned = _number(group.get("n_replicates"))
    if paired is None or paired < config.min_replicates:
        return None, "insufficient_repetitions"
    if planned is None:
        return None, "missing_planned_repetitions"
    if paired != planned:
        return None, "incomplete_paired_repetitions"
    return value, None


def _result(name: str, observations: list[dict[str, Any]], exclusions: list[dict[str, Any]],
            statistic: float | None, threshold: dict[str, Any], passed: bool | None,
            explanation: str) -> dict[str, Any]:
    if observations:
        # An observed violation remains a violation even if other cells are absent.
        status = "failed" if passed is False else "inconclusive" if exclusions else "passed"
    else:
        precision_reasons = {"insufficient_repetitions", "incomplete_paired_repetitions"}
        status = ("inconclusive" if any(row["reason"] in precision_reasons for row in exclusions)
                  else "not_evaluated")
    return dict(name=name, status=status, statistic=statistic, threshold=threshold,
                n_used=len(observations), n_excluded=len(exclusions), observations=observations,
                exclusions=exclusions, interpretation=explanation)


def _variance_endpoint(groups: Sequence[Mapping[str, Any]], effect: str, name: str,
                       config: EndpointThresholds) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for group in groups:
        ident = _group_id(group)
        if name == "E2_instrument_reduction":
            annotation = _has_instrument(group)
            if annotation is False:
                continue
            if annotation is None:
                exclusions.append(dict(**ident, reason="missing_instrument_annotation"))
                continue
        if name == "E3_predicted_equality":
            annotation = _predicted_equal(group, effect)
            if annotation is False:
                continue
            if annotation is None:
                exclusions.append(dict(**ident, reason="missing_equality_prediction"))
                continue
        if name == "E4_better_shared":
            # Original A2 applies this comparison to the pure-M panel, where both
            # shared alternatives exist.  Ratios need the same complete reps.
            if "SHAREDm" not in _arms(group):
                continue
            annotation = _has_pure_m(group)
            if annotation is False:
                continue
            if annotation is None:
                exclusions.append(dict(**ident, reason="missing_pure_m_annotation"))
                continue
            first, first_reason = _ratio(group, "SHARED", effect, config)
            second, second_reason = _ratio(group, "SHAREDm", effect, config)
            reason = first_reason or second_reason
            if reason:
                exclusions.append(dict(**ident, reason=reason))
                continue
            observations.append(dict(**ident, value=1.0 / min(first, second),
                                     shared_over_o=first, sharedm_over_o=second))
        else:
            ratio, reason = _ratio(group, "FULL", effect, config)
            if reason:
                exclusions.append(dict(**ident, reason=reason))
                continue
            value = 1.0 - 1.0 / ratio if name == "E2_instrument_reduction" else ratio
            observations.append(dict(**ident, value=value, full_over_o=ratio))
    values = [row["value"] for row in observations]
    if name == "E1_full_vs_o":
        statistic = min(values) if values else None
        return _result(name, observations, exclusions, statistic, {"minimum": config.full_min_ratio},
                       None if statistic is None else statistic >= config.full_min_ratio,
                       "Minimum within-model Var(FULL)/Var(O); finite-sample OLS diagnostic.")
    if name == "E2_instrument_reduction":
        statistic = median(values) if values else None
        return _result(name, observations, exclusions, statistic,
                       {"minimum_median_reduction": config.instrument_min_reduction},
                       None if statistic is None else statistic >= config.instrument_min_reduction,
                       "Median of 1-Var(O)/Var(FULL) over explicitly annotated instrument cells.")
    if name == "E3_predicted_equality":
        low, high = (min(values), max(values)) if values else (None, None)
        result = _result(name, observations, exclusions, None,
                         {"lower": config.equality_lower, "upper": config.equality_upper},
                         None if not values else low >= config.equality_lower and high <= config.equality_upper,
                         "Equality cells must be predicted from the graph before viewing variance estimates.")
        result["range"] = None if not values else [low, high]
        return result
    statistic = max(values) if values else None
    return _result(name, observations, exclusions, statistic, {"maximum": config.shared_max_o_ratio},
                   None if statistic is None else statistic <= config.shared_max_o_ratio,
                   "Maximum Var(O)/min(Var(SHARED),Var(SHAREDm)); applies where both shared arms exist.")


def _validity_and_bias(groups: Sequence[Mapping[str, Any]], effect: str,
                       config: EndpointThresholds) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for group in groups:
        ident = _group_id(group)
        arms = _arms(group)
        if group.get("diagnostic_misspecification") is True:
            exclusions.append(dict(**ident, reason="outside_linear_model_scope"))
            continue
        if "O" not in arms:
            exclusions.append(dict(**ident, reason="missing_arm:O"))
            continue
        for name in MAIN_ARMS:
            if name not in arms:
                continue
            if arms[name].get("admissible") is False:
                invalid.append(dict(**ident, arm=name))
            elif arms[name].get("admissible") is not True:
                exclusions.append(dict(**ident, reason=f"unknown_admissibility:{name}"))
        arm = arms["O"]
        metrics = arm.get("results", {}).get(effect, {})
        bias = _number(metrics.get("bias"))
        mcse = _number(metrics.get("mcse_mean"))
        count = _number(metrics.get("n"))
        planned = _number(group.get("n_replicates"))
        if count is None or count < config.min_replicates:
            exclusions.append(dict(**ident, reason="insufficient_repetitions"))
            continue
        if planned is None or count != planned:
            exclusions.append(dict(**ident, reason="incomplete_paired_repetitions"))
            continue
        if bias is None or mcse is None or mcse < 0 or (mcse == 0 and bias != 0):
            exclusions.append(dict(**ident, reason="undefined_bias_mcse"))
            continue
        observations.append(dict(**ident, bias=bias, mcse_mean=mcse,
                                 value=abs(bias) / mcse if mcse else 0.0))
    statistic = max((row["value"] for row in observations), default=None)
    passed = None if statistic is None and not invalid else not invalid and statistic is not None and statistic <= config.max_bias_z
    result = _result("E5_admissibility_and_oracle_bias", observations, exclusions, statistic,
                     {"max_abs_bias_over_mcse": config.max_bias_z, "invalid_main_arms": 0}, passed,
                     "Checks main-arm graph validity and O's mean bias against Monte Carlo SE. This is not proof of unbiasedness; OMISS is a separate omission diagnostic.")
    result["invalid_main_arms"] = invalid
    if invalid:
        result["status"] = "failed"
    result["multiplicity_note"] = "The original max-|bias|/MCSE cutoff is descriptive across many cells; it has no family-wise error calibration."
    return result


def evaluate_endpoints(summary_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]], *,
                       thresholds: EndpointThresholds | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate original A2 endpoints per sample size, estimator and effect.

    Accept the metrics module's summary dictionary (``groups`` key), or its
    sequence of fixed-model groups.  Annotation values must be supplied by the
    true graph/configuration; missing predictions are never inferred from the
    observed effect or variance.  All returned decisions are exploratory.
    """
    config = _thresholds(thresholds)
    raw_groups = summary_rows.get("groups", ()) if isinstance(summary_rows, Mapping) else summary_rows
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
        raise ValueError("Endpoint input must contain a sequence of fixed-model groups")
    if not all(isinstance(group, Mapping) for group in raw_groups):
        raise ValueError("Every endpoint group must be a mapping")
    output: dict[str, Any] = dict(
        purpose="exploratory_endpoint_diagnostics", confirmation=False,
        reference="Original D3/A2 five endpoints; b3_summarize_a2.py and doc/实验一_1011.md §4",
        thresholds=asdict(config), status="disabled" if not config.enabled else "complete", scopes=[])
    if not config.enabled:
        return output
    scopes: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = {}
    for group in raw_groups:
        # Designs with different data-generating families may use the same graph
        # number; panel_id/design_family distinguishes their endpoint panels.
        key = (group.get("panel_id", group.get("design_family", "default")),
               group.get("n"), group.get("estimator"))
        scopes.setdefault(key, []).append(group)
    for (panel, n, estimator), groups in sorted(scopes.items(), key=lambda item: tuple(str(x) for x in item[0])):
        effects: dict[str, Any] = {}
        for effect in EFFECTS:
            entries = {name: _variance_endpoint(groups, effect, name, config) for name in (
                "E1_full_vs_o", "E2_instrument_reduction", "E3_predicted_equality", "E4_better_shared")}
            entries["E5_admissibility_and_oracle_bias"] = _validity_and_bias(groups, effect, config)
            effects[effect] = entries
        output["scopes"].append(dict(panel_id=panel, n=n, estimator=estimator,
                                     n_models=len(groups), effects=effects))
    return output
