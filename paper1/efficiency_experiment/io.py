"""Validated inputs for fixed-model efficiency experiments.

Saved artifacts are consumed without changing them.  Generate mode freezes one design
per graph and draws fresh observational samples lazily; no replicate tables are retained.
Paths in an experiment TOML are relative to that TOML, never silently to the shell cwd.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..data_generation.generator import build_design, derive_seed, diagnose, simulate, structural_truth
from ..data_generation.schema import load_config as load_generator_config


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def contained_path(root: Path, relative: str) -> Path:
    """Manifest-relative paths must stay within their input artifact directory."""
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes its input directory: {relative}")
    return target


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_npz(path: Path, *, expected_columns=None, expected_shape=None, sha256=None):
    """Read a numeric sample without enabling pickle, checking its manifest contract."""
    if sha256 and file_hash(path) != sha256:
        raise ValueError(f"Sample SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"values", "columns"}:
            raise ValueError(f"Sample must contain exactly values and columns: {path}")
        values = np.asarray(payload["values"], dtype=float)
        raw_columns = payload["columns"]
        if raw_columns.ndim != 1 or raw_columns.dtype.kind not in "US":
            raise ValueError(f"columns must be a one-dimensional string array: {path}")
        columns = tuple(str(c) for c in raw_columns.tolist())
    if values.ndim != 2 or values.shape[1] != len(columns) or values.shape[0] < 1:
        raise ValueError(f"Sample shape is inconsistent with columns: {path}")
    if len(set(columns)) != len(columns) or any(not c for c in columns):
        raise ValueError(f"Sample column names must be nonempty and unique: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Sample contains non-finite values: {path}")
    if expected_columns is not None and tuple(expected_columns) != columns:
        raise ValueError(f"Sample column order differs from design: {path}")
    if expected_shape is not None and tuple(expected_shape) != values.shape:
        raise ValueError(f"Sample shape differs from manifest: {path}")
    return values, columns


@dataclass
class GraphInput:
    graph_id: int
    metadata: dict[str, Any]
    default_n: int
    replicate_ids: tuple[int, ...]
    provenance: dict[str, Any]
    sample_sizes: tuple[int, ...]
    _saved: tuple[tuple[Path, dict], ...] = ()
    _design: Any = None
    _generation: Any = None
    _data_seed: int = 0
    _legacy: Any = None
    _replicate_map: dict[int, tuple[Path, dict]] = field(init=False, repr=False)

    def __post_init__(self):
        self._replicate_map = {int(row["replicate"]): (path, row) for path, row in self._saved}

    def sample(self, replicate: int):
        """Return one maximum-size sample; callers use prefixes for configured sample sizes."""
        if replicate not in self.replicate_ids:
            raise ValueError(f"Unknown replicate {replicate} for graph {self.graph_id}")
        if self._saved:
            path, descriptor = self._replicate_map[replicate]
            values, columns = read_npz(path, expected_columns=self.metadata["columns"],
                                       expected_shape=descriptor.get("shape"), sha256=descriptor.get("sha256"))
            data_seed = descriptor.get("data_seed")
            source = {"path": str(path), "sha256": descriptor.get("sha256") or file_hash(path)}
        else:
            data_seed = derive_seed(self._data_seed, self.graph_id, replicate, 44)
            if self._legacy is not None:
                from .legacy_d3 import simulate as simulate_legacy
                values, columns = simulate_legacy(self._legacy, max(self.sample_sizes), np.random.default_rng(data_seed))
            else:
                values, columns = simulate(self._design, max(self.sample_sizes), data_seed, self._generation.mechanism)
            columns = tuple(columns)
            source = {"path": None, "sha256": None}
        if max(self.sample_sizes) > len(values):
            raise ValueError(f"Requested n={max(self.sample_sizes)} exceeds {len(values)} saved rows for graph {self.graph_id}")
        return values, tuple(columns), {"data_seed": data_seed, **source}


def _selected_graphs(all_ids, wanted):
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Input graph identifiers must be unique")
    if wanted:
        absent = set(wanted) - set(all_ids)
        if absent:
            raise ValueError(f"Requested graph IDs are absent: {sorted(absent)}")
        return set(wanted)
    return set(all_ids)


def _load_saved(config, base):
    root = resolve_path(config.source.path, base)
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("schema_version") != 1:
        raise ValueError("Saved generator input must have a complete schema-version-1 manifest")
    graph_rows = manifest.get("graphs")
    if not isinstance(graph_rows, list) or not graph_rows:
        raise ValueError("Generator manifest contains no graphs")
    ids = [int(g["graph_id"]) for g in graph_rows]
    selected = _selected_graphs(ids, config.source.graph_ids)
    graphs = []
    for descriptor in graph_rows:
        gid = int(descriptor["graph_id"])
        if gid not in selected:
            continue
        directory = contained_path(root, descriptor["directory"])
        metadata_path = directory / "design.json"
        metadata = read_json(metadata_path)
        reps = descriptor.get("replicates")
        if not isinstance(reps, list) or not reps:
            raise ValueError(f"Graph {gid} has no saved replicate descriptors")
        reps = sorted(reps, key=lambda r: int(r["replicate"]))
        rep_ids = [int(r["replicate"]) for r in reps]
        if len(set(rep_ids)) != len(rep_ids):
            raise ValueError(f"Graph {gid} has duplicate replicate IDs")
        count = config.source.n_replicates or len(reps)
        if count > len(reps):
            raise ValueError(f"Requested {count} replicates but graph {gid} has only {len(reps)} saved replicates")
        reps = reps[:count]
        n_available = min(int(r["shape"][0]) for r in reps)
        sizes = config.source.sample_sizes or (n_available,)
        if max(sizes) > n_available:
            raise ValueError(f"Requested sample size exceeds saved rows for graph {gid}")
        paths = tuple((contained_path(directory, r["file"]), r) for r in reps)
        graphs.append(GraphInput(gid, metadata, n_available, tuple(int(r["replicate"]) for r in reps),
                                 {"mode": "saved", "manifest": str(manifest_path),
                                  "manifest_sha256": file_hash(manifest_path), "design": str(metadata_path),
                                  "design_sha256": file_hash(metadata_path)}, tuple(sizes), paths))
    return graphs, {"mode": "saved", "path": str(root), "manifest_sha256": file_hash(manifest_path),
                    "generator_manifest_status": manifest["status"], "generator_config": manifest.get("config")}


def _load_generate(config, base):
    path = resolve_path(config.source.generator_config, base)
    generation = load_generator_config(path)
    e = generation.experiment
    ids = list(range(e.n_graphs))
    selected = _selected_graphs(ids, config.source.graph_ids)
    sizes = config.source.sample_sizes or (e.n_samples,)
    count = config.source.n_replicates or e.n_replicates
    graphs = []
    for gid in ids:
        if gid not in selected:
            continue
        seeds = {name: derive_seed(getattr(e, name + "_seed"), gid, stream=tag)
                 for tag, name in enumerate(("graph", "mechanism", "calibration", "truth"))}
        design = build_design(generation, seeds["graph"], seeds["mechanism"], seeds["calibration"])
        truth = structural_truth(design, generation.mechanism, e, seeds["truth"])
        metadata = design.metadata()
        metadata.update(truth=truth, query=diagnose(design, generation.mechanism), seeds=seeds)
        provenance = {"mode": "generate", "generator_config": str(path),
                      "generator_config_sha256": file_hash(path), "seeds": seeds}
        graphs.append(GraphInput(gid, metadata, e.n_samples, tuple(range(count)), provenance,
                                 tuple(sizes), _design=design, _generation=generation,
                                 _data_seed=config.execution.seed))
    return graphs, {"mode": "generate", "generator_config": str(path),
                    "generator_config_sha256": file_hash(path), "generator_config_resolved": generation.to_dict(),
                    "data_seed": config.execution.seed,
                    "rng_protocol": "SeedSequence([execution.seed, graph_id, replicate_id, 44]); all arms and n-prefixes share a replicate"}


def _load_legacy(config, base):
    from . import legacy_d3 as designs

    cells = designs.cells()
    selected = _selected_graphs(list(range(len(cells))), config.source.graph_ids)
    sizes = config.source.sample_sizes or (1000, 5000)
    count = config.source.n_replicates or 500
    graphs = []
    for gid, factors in enumerate(cells):
        if gid not in selected:
            continue
        design = designs.make_design(**factors)
        roles = dict(design["roles"])
        roles.update(X="exposure", M="mediator", Y="outcome")
        truth = dict(design["truth"], TE=design["truth"]["NIE"] + design["truth"]["NDE"],
                     kind="truth_exact_linear", x0=0.0, x1=1.0, delta_x=1.0)
        metadata = {"columns": list(design["order"]), "roles": roles,
                    "edges": [{"source": u, "target": v, "function": "id", "weight": float(d["weight"])}
                              for u, v, d in design["G"].edges(data=True)],
                    "truth": truth, "factors": dict(factors),
                    "config": {"mechanism": {"mode": "linear", "noise_distribution": "normal",
                                               "noise_sd": 1.0, "outcome_interaction": 0.0}},
                    "identification_status": "admissible_known_graph"}
        graphs.append(GraphInput(gid, metadata, 5000, tuple(range(count)),
                                 {"mode": "legacy_d3", "cell": gid, "factors": dict(factors),
                                  "adapter": "paper1.efficiency_experiment.legacy_d3.make_design/simulate"},
                                 tuple(sizes), _data_seed=config.execution.seed, _legacy=design))
    return graphs, {"mode": "legacy_d3", "n_cells_available": len(cells), "data_seed": config.execution.seed,
                    "rng_protocol": "SeedSequence([execution.seed, cell_id, replicate_id, 44]); n-prefixes and arms share samples",
                    "note": "Original 48 D3 factor cells; fresh experiment seeds, not the preregistered old random draws"}


def load_inputs(config, config_directory: Path):
    """Resolve input contracts and freeze graph-specific models without fitting any outcomes."""
    if config.source.mode == "saved":
        return _load_saved(config, config_directory)
    if config.source.mode == "generate":
        return _load_generate(config, config_directory)
    if config.source.mode == "legacy_d3":
        return _load_legacy(config, config_directory)
    raise ValueError(f"Unsupported source.mode: {config.source.mode}")
