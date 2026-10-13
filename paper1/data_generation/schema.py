"""TOML schema for the continuous paper-1 generator.

The five sections are intentionally separate: graph structure, numerical mechanism, calibration,
experiment seeds/sizes, and output.  A TOML file is the reviewable experiment contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9 compatibility
    import tomli as tomllib  # type: ignore


@dataclass(frozen=True)
class GraphConfig:
    n_baseline: int = 24
    baseline_mean_parents: float = 2.0
    baseline_max_parents: int = 3
    n_shared: int = 1
    n_x_extra: int = 3
    n_m_extra: int = 4
    n_y_extra: int = 3
    n_inst: int = 2
    n_pm: int = 2
    n_py: int = 2
    n_irrelevant: int = 2
    xm_chain_nodes: int = 1
    my_chain_nodes: int = 1
    n_parallel: int = 1
    n_post_descendants: int = 1
    n_middle_confounders: int = 0
    n_auxiliary_baseline_parents: int = 2
    allow_parallel_mediators: bool = True
    allow_post_descendants: bool = True
    allow_middle_confounding: bool = False
    direct_edge: bool = True


@dataclass(frozen=True)
class MechanismConfig:
    mode: str = "linear"  # linear | additive_nonlinear
    edge_functions: tuple[str, ...] = ("id", "tanh", "sin")
    protected_function: str = "id"
    nonlinear_fraction: float = 0.0
    negative_edge_fraction: float = 0.15
    weak_fraction: float = 0.15
    weak_multiplier: float = 0.2
    weight_min: float = 0.25
    weight_max: float = 0.75
    noise_distribution: str = "normal"  # normal | laplace | student_t
    noise_df: float = 5.0
    noise_sd: float = 1.0
    intercept_sd: float = 0.0
    xm_weight: float = 0.6
    my_weight: float = 0.5
    direct_weight: float = 0.2
    chain_weight: float = 0.6
    parallel_x_weight: float = 0.5
    parallel_y_weight: float = 0.4
    instrument_weight: float = 0.7
    predictor_weight: float = 0.5
    outcome_interaction: float = 0.0  # coefficient of X*M in Y equation


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = True
    n_samples: int = 4096
    target_signal_sd: float = 0.8
    scale_floor: float = 1e-8


@dataclass(frozen=True)
class ExperimentConfig:
    n_samples: int = 2000
    n_graphs: int = 2
    n_replicates: int = 1
    graph_seed: int = 20261012
    mechanism_seed: int = 20261022
    calibration_seed: int = 20261032
    data_seed: int = 20261042
    truth_seed: int = 20261052
    truth_samples: int = 20000
    chunk_size: int = 1000
    x0: float = 0.0
    x1: float = 1.0


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "paper1_out/data_generation_review"
    csv: bool = True
    plot: bool = False
    preview_rows: int = 8
    max_plot_nodes: int = 80
    max_covariance_nodes: int = 200


@dataclass(frozen=True)
class GenerationConfig:
    graph: GraphConfig = GraphConfig()
    mechanism: MechanismConfig = MechanismConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    output: OutputConfig = OutputConfig()

    def validate(self) -> "GenerationConfig":
        for name, cls in (
            ("graph", GraphConfig), ("mechanism", MechanismConfig),
            ("calibration", CalibrationConfig), ("experiment", ExperimentConfig),
            ("output", OutputConfig),
        ):
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f"{name} must be a {cls.__name__}")

        g, m, c, e, o = self.graph, self.mechanism, self.calibration, self.experiment, self.output
        for name in (
            "n_baseline", "baseline_max_parents", "n_shared", "n_x_extra", "n_m_extra",
            "n_y_extra", "n_inst", "n_pm", "n_py", "n_irrelevant", "xm_chain_nodes",
            "my_chain_nodes", "n_parallel", "n_post_descendants", "n_middle_confounders",
            "n_auxiliary_baseline_parents",
        ):
            _integer(getattr(g, name), f"graph.{name}")
        _number(g.baseline_mean_parents, "graph.baseline_mean_parents", minimum=0.0)
        for name in (
            "allow_parallel_mediators", "allow_post_descendants",
            "allow_middle_confounding", "direct_edge",
        ):
            _boolean(getattr(g, name), f"graph.{name}")
        reserved = g.n_inst + g.n_pm + g.n_py + g.n_irrelevant
        n_core = g.n_baseline - reserved
        if n_core < 0:
            raise ValueError("n_baseline must include specialized baseline nodes")
        if g.n_shared + max(g.n_x_extra, g.n_m_extra, g.n_y_extra) > n_core:
            raise ValueError("shared and endpoint-specific parents exceed the ordinary baseline pool")
        if g.n_auxiliary_baseline_parents > n_core:
            raise ValueError("n_auxiliary_baseline_parents exceeds the ordinary baseline pool")
        for count, enabled in (
            ("n_middle_confounders", "allow_middle_confounding"),
            ("n_parallel", "allow_parallel_mediators"),
            ("n_post_descendants", "allow_post_descendants"),
        ):
            if getattr(g, count) and not getattr(g, enabled):
                raise ValueError(f"{count}>0 requires {enabled}=true")

        if not isinstance(m.mode, str) or m.mode not in {"linear", "additive_nonlinear"}:
            raise ValueError("mechanism.mode must be linear or additive_nonlinear")
        valid_functions = {"id", "tanh", "sin"}
        if not isinstance(m.edge_functions, tuple) or not m.edge_functions:
            raise ValueError("edge_functions must be a non-empty tuple of function names")
        if any(not isinstance(fn, str) or fn not in valid_functions for fn in m.edge_functions):
            raise ValueError("edge_functions must contain only id, tanh, or sin")
        if not isinstance(m.protected_function, str) or m.protected_function not in valid_functions:
            raise ValueError("protected_function must be id, tanh, or sin")
        for name in ("nonlinear_fraction", "negative_edge_fraction", "weak_fraction"):
            _number(getattr(m, name), f"mechanism.{name}", minimum=0.0, maximum=1.0)
        _number(m.weak_multiplier, "mechanism.weak_multiplier", minimum=0.0, maximum=1.0, exclusive_min=True)
        for name in ("weight_min", "weight_max", "noise_sd", "noise_df"):
            _number(getattr(m, name), f"mechanism.{name}", minimum=0.0, exclusive_min=True)
        if m.weight_min > m.weight_max:
            raise ValueError("weight_min cannot exceed weight_max")
        _number(m.intercept_sd, "mechanism.intercept_sd", minimum=0.0)
        for name in (
            "xm_weight", "my_weight", "chain_weight", "parallel_x_weight",
            "parallel_y_weight", "instrument_weight", "predictor_weight",
        ):
            _number(getattr(m, name), f"mechanism.{name}")
            if getattr(m, name) == 0:
                raise ValueError(f"mechanism.{name} must be nonzero")
        _number(m.direct_weight, "mechanism.direct_weight")
        _number(m.outcome_interaction, "mechanism.outcome_interaction")
        if not isinstance(m.noise_distribution, str) or m.noise_distribution not in {"normal", "laplace", "student_t"}:
            raise ValueError("noise_distribution must be normal, laplace, or student_t")
        if m.noise_distribution == "student_t" and m.noise_df <= 2:
            raise ValueError("student_t noise requires noise_df > 2 for finite variance")
        if m.mode == "linear" and (m.nonlinear_fraction != 0 or m.outcome_interaction != 0 or m.protected_function != "id"):
            raise ValueError("linear mode requires nonlinear_fraction=0, outcome_interaction=0, and protected_function=id")
        if m.outcome_interaction != 0 and (not g.direct_edge or m.direct_weight == 0):
            raise ValueError("outcome_interaction requires direct_edge=true and direct_weight!=0")

        _boolean(c.enabled, "calibration.enabled")
        _integer(c.n_samples, "calibration.n_samples", minimum=2)
        _number(c.target_signal_sd, "calibration.target_signal_sd", minimum=0.0, exclusive_min=True)
        _number(c.scale_floor, "calibration.scale_floor", minimum=0.0, exclusive_min=True)
        for name in ("n_samples", "n_graphs", "n_replicates", "chunk_size"):
            _integer(getattr(e, name), f"experiment.{name}", minimum=1)
        _integer(e.truth_samples, "experiment.truth_samples", minimum=2)
        for name in ("graph_seed", "mechanism_seed", "calibration_seed", "data_seed", "truth_seed"):
            _integer(getattr(e, name), f"experiment.{name}")
        _number(e.x0, "experiment.x0")
        _number(e.x1, "experiment.x1")
        if e.x0 == e.x1:
            raise ValueError("experiment.x0 and experiment.x1 must differ")
        if not isinstance(o.directory, str) or not o.directory.strip():
            raise ValueError("output.directory must be a non-empty string")
        _boolean(o.csv, "output.csv")
        _boolean(o.plot, "output.plot")
        _integer(o.preview_rows, "output.preview_rows", minimum=1)
        _integer(o.max_plot_nodes, "output.max_plot_nodes", minimum=3)
        _integer(o.max_covariance_nodes, "output.max_covariance_nodes")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _boolean(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _integer(value: Any, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None, exclusive_min: bool = False) -> None:
    finite = False
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            pass
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and (value <= minimum if exclusive_min else value < minimum):
        relation = ">" if exclusive_min else ">="
        raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _construct(cls, raw: Mapping[str, Any], section: str):
    if not isinstance(raw, Mapping):
        raise ValueError(f"[{section}] must be a TOML table")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in [{section}]: {unknown}")
    values = dict(raw)
    if cls is MechanismConfig and "edge_functions" in values:
        if not isinstance(values["edge_functions"], (list, tuple)):
            raise ValueError("edge_functions must be an array of function names")
        values["edge_functions"] = tuple(values["edge_functions"])
    return cls(**values)


def load_config(path: str | Path) -> GenerationConfig:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if not isinstance(raw, Mapping):
        raise ValueError("configuration must be a mapping of TOML sections")
    sections = {"graph", "mechanism", "calibration", "experiment", "output"}
    unknown = sorted(set(raw) - sections)
    if unknown:
        raise ValueError(f"unknown TOML sections: {unknown}")
    config = GenerationConfig(
        graph=_construct(GraphConfig, raw.get("graph", {}), "graph"),
        mechanism=_construct(MechanismConfig, raw.get("mechanism", {}), "mechanism"),
        calibration=_construct(CalibrationConfig, raw.get("calibration", {}), "calibration"),
        experiment=_construct(ExperimentConfig, raw.get("experiment", {}), "experiment"),
        output=_construct(OutputConfig, raw.get("output", {}), "output"),
    )
    return config.validate()
