"""Strict TOML contract for known-graph repeated-sampling efficiency experiments.

The experiment consumes generated samples or generates repeats from a fixed model.
Estimator configuration lives beside the estimator implementation so registered
estimators can extend it without changing the data adapter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 / 3.10
    import tomli as tomllib


def _estimator_default():
    from .estimators import EstimatorConfig
    return EstimatorConfig()


@dataclass(frozen=True)
class SourceConfig:
    mode: str = "saved"  # saved | generate | legacy_d3
    path: str = ""
    generator_config: str = ""
    sample_sizes: tuple[int, ...] = ()
    n_replicates: int = 0  # saved: zero = all stored replicates
    graph_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ArmConfig:
    name: str
    kind: str = "query"  # named arm from the graph adapter | explicit custom sets
    CM: tuple[str, ...] = ()
    CY: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionConfig:
    seed: int = 20261016
    workers: int = 1
    on_error: str = "continue"
    allow_non_admissible: bool = False
    min_replicates: int = 30
    reference_arm: str = "O"


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "paper1_out/efficiency_experiment"
    per_replicate: bool = True


@dataclass(frozen=True)
class EndpointsConfig:
    enabled: bool = True
    min_replicates: int = 100
    full_min_ratio: float = 0.95
    instrument_min_reduction: float = 0.20
    equality_lower: float = 0.95
    equality_upper: float = 1.05
    shared_max_o_ratio: float = 1.05
    max_bias_z: float = 3.0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    estimator: Any = field(default_factory=_estimator_default)
    arms: tuple[ArmConfig, ...] = field(default_factory=lambda: tuple(ArmConfig(s) for s in ("O", "FULL", "SHARED")))
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    endpoints: EndpointsConfig = field(default_factory=EndpointsConfig)

    def validate(self):
        s, x, o, ep = self.source, self.execution, self.output, self.endpoints
        for name, value, cls in (("execution", x, ExecutionConfig), ("output", o, OutputConfig),
                                 ("endpoints", ep, EndpointsConfig)):
            if not isinstance(value, cls):
                raise ValueError(f"{name} must be a {cls.__name__}")
        if not isinstance(s, SourceConfig):
            raise ValueError("source must be a SourceConfig")
        if s.mode not in {"saved", "generate", "legacy_d3"}:
            raise ValueError("source.mode must be saved, generate, or legacy_d3")
        _string(s.path, "source.path", allow_empty=True)
        _string(s.generator_config, "source.generator_config", allow_empty=True)
        if s.mode == "saved" and not s.path.strip():
            raise ValueError("saved source requires source.path")
        if s.mode == "generate" and not s.generator_config.strip():
            raise ValueError("generate source requires source.generator_config")
        if s.mode != "saved" and s.path:
            raise ValueError("source.path is only used by mode=saved")
        if s.mode != "generate" and s.generator_config:
            raise ValueError("source.generator_config is only used by mode=generate")
        _integer(s.n_replicates, "source.n_replicates")
        if s.mode == "legacy_d3" and not s.n_replicates:
            raise ValueError("legacy_d3 requires source.n_replicates > 0")
        for name, values, minimum in (("sample_sizes", s.sample_sizes, 1), ("graph_ids", s.graph_ids, 0)):
            if not isinstance(values, tuple):
                raise ValueError(f"source.{name} must be a tuple")
            for value in values:
                _integer(value, f"source.{name} entry", minimum=minimum)
            if len(values) != len(set(values)):
                raise ValueError(f"source.{name} must not contain duplicates")
        if s.mode == "legacy_d3" and any(i >= 48 for i in s.graph_ids):
            raise ValueError("legacy_d3 graph_ids must be in 0..47")
        if s.mode == "legacy_d3" and not s.sample_sizes:
            raise ValueError("legacy_d3 requires source.sample_sizes")
        if not isinstance(self.arms, tuple) or not self.arms:
            raise ValueError("at least one arm is required")
        names = []
        for arm in self.arms:
            if not isinstance(arm, ArmConfig):
                raise ValueError("arms must contain ArmConfig instances")
            _string(arm.name, "arms.name")
            if arm.kind not in {"query", "custom"}:
                raise ValueError("arms.kind must be query or custom")
            for name, values in (("CM", arm.CM), ("CY", arm.CY)):
                if not isinstance(values, tuple) or any(not isinstance(v, str) or not v.strip() for v in values):
                    raise ValueError(f"arms.{name} must be an array of non-empty column names")
                if len(values) != len(set(values)):
                    raise ValueError(f"arms.{name} contains duplicate columns")
            if arm.kind == "custom" and arm.name in {"O", "FULL", "SHARED", "SHAREDm", "OMISS"}:
                raise ValueError("custom arms cannot use reserved paper-1 strategy names")
            if arm.kind == "query" and (arm.CM or arm.CY):
                raise ValueError("query arms read their sets from the adapter; CM/CY require kind=custom")
            names.append(arm.name)
        if len(names) != len(set(names)):
            raise ValueError("arm names must be unique")
        _integer(x.seed, "execution.seed")
        _integer(x.workers, "execution.workers", minimum=1)
        _integer(x.min_replicates, "execution.min_replicates", minimum=2)
        if x.on_error not in {"continue", "raise"}:
            raise ValueError("execution.on_error must be continue or raise")
        _boolean(x.allow_non_admissible, "execution.allow_non_admissible")
        _string(x.reference_arm, "execution.reference_arm")
        if x.reference_arm not in names:
            raise ValueError("execution.reference_arm must name a configured arm")
        _string(o.directory, "output.directory")
        _boolean(o.per_replicate, "output.per_replicate")
        _boolean(ep.enabled, "endpoints.enabled")
        _integer(ep.min_replicates, "endpoints.min_replicates", minimum=2)
        for name in ("full_min_ratio", "equality_lower", "equality_upper", "shared_max_o_ratio", "max_bias_z"):
            _number(getattr(ep, name), "endpoints." + name, minimum=0, exclusive=True)
        _number(ep.instrument_min_reduction, "endpoints.instrument_min_reduction", minimum=0, maximum=1)
        if ep.instrument_min_reduction == 1:
            raise ValueError("endpoints.instrument_min_reduction must be < 1")
        if ep.equality_lower > ep.equality_upper:
            raise ValueError("endpoints.equality_lower must not exceed equality_upper")
        if ep.enabled and x.reference_arm != "O":
            raise ValueError("the paper-1 endpoints require execution.reference_arm='O'")
        if not hasattr(self.estimator, "validate"):
            raise ValueError("estimator must implement validate()")
        self.estimator.validate()
        return self

    def to_dict(self):
        return asdict(self)


def _string(value, name, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a {'string' if allow_empty else 'non-empty string'}")


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name, minimum=None, maximum=None, exclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and (value <= minimum if exclusive else value < minimum):
        raise ValueError(f"{name} must be {'>' if exclusive else '>='} {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _construct(cls, raw, section, tuple_fields=()):
    if not isinstance(raw, Mapping):
        raise ValueError(f"[{section}] must be a TOML table")
    unknown = set(raw) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"unknown keys in [{section}]: {sorted(unknown)}")
    values = dict(raw)
    for name in tuple_fields:
        if name in values:
            if not isinstance(values[name], (tuple, list)):
                raise ValueError(f"{section}.{name} must be an array")
            values[name] = tuple(values[name])
    return cls(**values)


def from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Parse an in-memory TOML-shaped mapping with the same strictness as a file."""
    from .estimators import EstimatorConfig
    if not isinstance(raw, Mapping):
        raise ValueError("experiment configuration must be a mapping")
    unknown = set(raw) - {"source", "estimator", "arms", "execution", "output", "endpoints"}
    if unknown:
        raise ValueError(f"unknown TOML sections: {sorted(unknown)}")
    arm_values = raw.get("arms")
    if arm_values is not None and (not isinstance(arm_values, (tuple, list)) or not arm_values):
        raise ValueError("[[arms]] must be a non-empty array of TOML tables")
    arms = tuple(_construct(ArmConfig, arm, "arms", ("CM", "CY")) for arm in arm_values) if arm_values is not None else tuple(ArmConfig(s) for s in ("O", "FULL", "SHARED"))
    return ExperimentConfig(
        source=_construct(SourceConfig, raw.get("source", {}), "source", ("sample_sizes", "graph_ids")),
        estimator=_construct(EstimatorConfig, raw.get("estimator", {}), "estimator"),
        arms=arms,
        execution=_construct(ExecutionConfig, raw.get("execution", {}), "execution"),
        output=_construct(OutputConfig, raw.get("output", {}), "output"),
        endpoints=_construct(EndpointsConfig, raw.get("endpoints", {}), "endpoints"),
    ).validate()


def load_config(path: str | Path) -> ExperimentConfig:
    """Load TOML. Relative path resolution is performed by the runner at its boundary."""
    with Path(path).open("rb") as stream:
        return from_dict(tomllib.load(stream))
