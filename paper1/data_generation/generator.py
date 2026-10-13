"""Continuous DAG/SEM generation, independent calibration and structural effect truth.

This module generates a known data-generating process. It never learns a graph or estimates
an effect from its own observational data. The frozen graph criteria are used for diagnostics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import networkx as nx
import numpy as np

from ..graphs import graph_pair, intermediate_confounders, pair_admissible_set
from .schema import GenerationConfig

BASELINE_ROLES = {"baseline", "instrument", "pure_M_predictor", "pure_Y_predictor", "irrelevant"}
FUNCTIONS = {"id": lambda z: z, "tanh": np.tanh, "sin": np.sin}


def derive_seed(base: int, graph_id: int = 0, replicate: int = 0, stream: int = 0) -> int:
    """Separate random streams; independent of execution order and worker count."""
    return int(np.random.SeedSequence([base, graph_id, replicate, stream]).generate_state(1, dtype=np.uint64)[0])


@dataclass
class ContinuousDesign:
    graph: nx.DiGraph
    order: tuple[str, ...]
    roles: dict[str, str]
    edge_functions: dict[tuple[str, str], str]
    config: dict[str, Any]
    seeds: dict[str, int]
    calibration: dict[str, Any]
    identification_status: str = "unknown"

    @property
    def columns(self) -> tuple[str, ...]:
        return self.order

    def metadata(self) -> dict[str, Any]:
        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append(dict(source=u, target=v, function=self.edge_functions[(u, v)], **d))
        return {
            "columns": list(self.order), "roles": self.roles, "edges": edges,
            "config": self.config, "seeds": self.seeds, "calibration": self.calibration,
            "identification_status": self.identification_status,
            "n_nodes": len(self.graph), "n_edges": self.graph.number_of_edges(),
            "longest_path_edges": nx.dag_longest_path_length(self.graph, weight=None),
        }


def _sample_graph(config: GenerationConfig, graph_seed: int, mechanism_seed: int) -> ContinuousDesign:
    g, m = config.graph, config.mechanism
    grng, mrng = np.random.default_rng(graph_seed), np.random.default_rng(mechanism_seed)
    n_core = g.n_baseline - g.n_inst - g.n_pm - g.n_py - g.n_irrelevant
    core = [f"C{i+1:03d}" for i in range(n_core)]
    groups = {
        "instrument": [f"I{i+1:03d}" for i in range(g.n_inst)],
        "pure_M_predictor": [f"PM{i+1:03d}" for i in range(g.n_pm)],
        "pure_Y_predictor": [f"PY{i+1:03d}" for i in range(g.n_py)],
        "irrelevant": [f"U{i+1:03d}" for i in range(g.n_irrelevant)],
        "pre_mediator": [f"XM{i+1:03d}" for i in range(g.xm_chain_nodes)],
        "post_mediator": [f"MY{i+1:03d}" for i in range(g.my_chain_nodes)],
        "parallel_mediator": [f"M2_{i+1:03d}" for i in range(g.n_parallel)],
        "post_exposure": [f"D{i+1:03d}" for i in range(g.n_post_descendants)],
        "intermediate_confounder": [f"L{i+1:03d}" for i in range(g.n_middle_confounders)],
    }
    roles = {c: "baseline" for c in core}
    for role, nodes in groups.items():
        roles.update({node: role for node in nodes})
    roles.update(X="exposure", M="mediator", Y="outcome")
    G = nx.DiGraph()
    G.add_nodes_from(roles)
    functions = {}

    def choose(values, count, limit=None):
        n = len(values) if limit is None else limit
        return [values[int(i)] for i in grng.choice(n, size=count, replace=False)]

    def add(u, v, role, fixed=None, protected=False):
        if G.has_edge(u, v):
            raise RuntimeError(f"Duplicate edge {u} -> {v}")
        weak = False
        if fixed is None:
            weight = mrng.uniform(m.weight_min, m.weight_max)
            weight *= -1 if mrng.random() < m.negative_edge_fraction else 1
            weak = bool(mrng.random() < m.weak_fraction)
            if weak:
                weight *= m.weak_multiplier
        else:
            weight = fixed
        fn = "id"
        if protected:
            fn = m.protected_function
        elif m.mode != "linear" and mrng.random() < m.nonlinear_fraction:
            nonidentity = [name for name in m.edge_functions if name != "id"]
            fn = str(mrng.choice(nonidentity)) if nonidentity else "id"
        G.add_edge(u, v, weight=float(weight), raw_weight=float(weight), edge_role=role,
                   protected=bool(protected), weak_draw=weak, calibration_scale=1.0)
        functions[u, v] = fn

    permutation = grng.permutation(core).tolist()
    for i, target in enumerate(permutation):
        count = min(i, g.baseline_max_parents, int(grng.poisson(g.baseline_mean_parents)))
        for parent in choose(permutation, count, limit=i):
            add(parent, target, "baseline_internal")
    shared = choose(core, g.n_shared)
    shared_set = set(shared)
    remaining = [node for node in core if node not in shared_set]
    for target, count in (("X", g.n_x_extra), ("M", g.n_m_extra), ("Y", g.n_y_extra)):
        for parent in shared + choose(remaining, count):
            add(parent, target, "baseline_to_query")
    for role, target, weight in (("instrument", "X", m.instrument_weight),
                                  ("pure_M_predictor", "M", m.predictor_weight),
                                  ("pure_Y_predictor", "Y", m.predictor_weight)):
        for node in groups[role]:
            add(node, target, role, fixed=weight)
    add("X", "M", "query_XM", fixed=m.xm_weight, protected=True)
    add("M", "Y", "query_MY", fixed=m.my_weight, protected=True)
    if g.direct_edge and m.direct_weight != 0:
        add("X", "Y", "query_direct", fixed=m.direct_weight, protected=True)
    for source, role, target in (("X", "pre_mediator", "M"), ("M", "post_mediator", "Y")):
        chain = groups[role]
        if chain:
            route = [source] + chain + [target]
            for u, v in zip(route, route[1:]):
                add(u, v, role, fixed=m.chain_weight, protected=True)
    for node in groups["parallel_mediator"]:
        add("X", node, "parallel_mediator", fixed=m.parallel_x_weight, protected=True)
        add(node, "Y", "parallel_mediator", fixed=m.parallel_y_weight, protected=True)
    for node in groups["post_exposure"]:
        add("X", node, "post_exposure")
    for node in groups["intermediate_confounder"]:
        add("X", node, "intermediate_confounder")
        add(node, "M", "intermediate_confounder")
        add(node, "Y", "intermediate_confounder")
    for role in ("pre_mediator", "post_mediator", "parallel_mediator", "post_exposure", "intermediate_confounder"):
        for node in groups[role]:
            for parent in choose(core, g.n_auxiliary_baseline_parents):
                add(parent, node, "baseline_to_auxiliary")
    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("Generator constructed a cyclic graph")
    order = tuple(nx.topological_sort(G))
    # Intercepts use a substream, so drawing them does not reuse coefficient RNG values.
    irng = np.random.default_rng(derive_seed(mechanism_seed, stream=91))
    intercepts = {node: float(irng.normal(0, m.intercept_sd)) for node in order}
    return ContinuousDesign(G, order, roles, functions,
                            {"graph": asdict(g), "mechanism": asdict(m), "intercepts": intercepts,
                             "direct_shared_baseline_parents": shared},
                            {"graph": int(graph_seed), "mechanism": int(mechanism_seed)}, {})


def draw_noise(rng, shape, mechanism):
    """Independent, centered errors scaled to noise_sd (including t and Laplace)."""
    m = mechanism
    if m.noise_distribution == "normal":
        return rng.normal(0, m.noise_sd, shape)
    if m.noise_distribution == "laplace":
        return rng.laplace(0, m.noise_sd / np.sqrt(2), shape)
    if m.noise_distribution == "student_t":
        return rng.standard_t(m.noise_df, shape) * m.noise_sd * np.sqrt((m.noise_df - 2) / m.noise_df)
    raise ValueError(f"Unknown noise distribution {m.noise_distribution}")


def _node_value(design, node, values, index, errors, mechanism):
    result = errors + design.config["intercepts"][node]
    for parent in design.graph.predecessors(node):
        result = result + design.graph[parent][node]["weight"] * FUNCTIONS[design.edge_functions[parent, node]](values[:, index[parent]])
    if node == "Y" and mechanism.outcome_interaction:
        result = result + mechanism.outcome_interaction * values[:, index["X"]] * values[:, index["M"]]
    return result


def simulate(design: ContinuousDesign, n: int, data_seed: int, mech_cfg,
             *, noise: np.ndarray | None = None, interventions: Mapping[str, Any] | None = None):
    """Generate observations, or replace entire node equations by interventions.

    ``noise`` is actual scaled exogenous noise, not standard-normal random numbers. It is
    copied so the exact same errors can be passed to every counterfactual world.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if asdict(mech_cfg) != design.config["mechanism"]:
        raise ValueError("Simulation mechanism differs from the frozen design")
    shape = (n, len(design.order))
    eps = draw_noise(np.random.default_rng(data_seed), shape, mech_cfg) if noise is None else np.asarray(noise, dtype=float)
    if eps.shape != shape or not np.isfinite(eps).all():
        raise ValueError("noise must be a finite (n, p) matrix")
    values = np.empty(shape)
    index = {node: i for i, node in enumerate(design.order)}
    interventions = dict(interventions or {})
    if set(interventions) - set(index):
        raise ValueError("Unknown intervention node")
    for node in design.order:
        j = index[node]
        if node in interventions:
            val = np.asarray(interventions[node], dtype=float)
            if val.shape not in ((), (n,)) or not np.isfinite(val).all():
                raise ValueError("An intervention must be a finite scalar or length-n vector")
            values[:, j] = val
        else:
            values[:, j] = _node_value(design, node, values, index, eps[:, j], mech_cfg)
    if not np.isfinite(values).all():
        raise FloatingPointError("SEM produced non-finite values; inspect the configured weights and interactions")
    return values, list(design.order)


def _calibrate(design, m, c, seed):
    if not c.enabled:
        return {"enabled": False, "nodes": {}, "seed": int(seed)}
    eps = draw_noise(np.random.default_rng(seed), (c.n_samples, len(design.order)), m)
    values = np.empty_like(eps)
    index = {node: i for i, node in enumerate(design.order)}
    baseline = {node for node, role in design.roles.items() if role in BASELINE_ROLES}
    records = {}
    for node in design.order:
        parents = [parent for parent in design.graph.predecessors(node) if parent in baseline]
        if parents:
            signal = sum((design.graph[parent][node]["weight"] * FUNCTIONS[design.edge_functions[parent, node]](values[:, index[parent]])
                          for parent in parents), np.zeros(c.n_samples))
            before = float(signal.std(ddof=1))
            if not np.isfinite(before) or before <= c.scale_floor:
                raise ValueError(f"Calibration for {node} has near-zero/non-finite signal SD: {before}")
            scale = float(c.target_signal_sd / before)
            for parent in parents:
                design.graph[parent][node]["weight"] *= scale
                design.graph[parent][node]["calibration_scale"] = scale
            records[node] = {"parents": parents, "before_sd": before, "scale": scale,
                             "after_sd": float((signal * scale).std(ddof=1))}
        values[:, index[node]] = _node_value(design, node, values, index, eps[:, index[node]], m)
    if not np.isfinite(values).all():
        raise FloatingPointError("Calibration produced non-finite values")
    return {"enabled": True, "seed": int(seed), "n_samples": c.n_samples,
            "target_signal_sd": c.target_signal_sd, "scope": "sum of direct baseline-parent contributions per node", "nodes": records}


def is_linear(design, mechanism):
    return not mechanism.outcome_interaction and all(fn == "id" for fn in design.edge_functions.values())


def _linear_truth(design, x0, x1):
    def response(source, held=()):
        d = {}
        for node in design.order:
            if node == source:
                d[node] = 1.0
            elif node in held:
                d[node] = 0.0
            else:
                d[node] = sum(design.graph[parent][node]["weight"] * d[parent]
                              for parent in design.graph.predecessors(node))
        return d
    total = response("X")
    direct = response("X", {"M"})
    mediator = response("M")
    dx = x1 - x0
    return {"kind": "truth_exact_linear", "TE": float(dx * total["Y"]),
            "NIE": float(dx * (total["Y"] - direct["Y"])), "NDE": float(dx * direct["Y"]),
            "X_to_M": float(dx * total["M"]), "M_to_Y_per_unit": float(mediator["Y"]),
            "x0": x0, "x1": x1, "delta_x": dx}


def _mc_truth(design, mech_cfg, exp_cfg, seed):
    rng = np.random.default_rng(seed)
    total_n = 0
    mean, m2 = np.zeros(3), np.zeros(3)
    identity_error = 0.0
    ix = {c: i for i, c in enumerate(design.order)}
    while total_n < exp_cfg.truth_samples:
        size = min(exp_cfg.chunk_size, exp_cfg.truth_samples - total_n)
        errors = draw_noise(rng, (size, len(design.order)), mech_cfg)
        v0, _ = simulate(design, size, seed, mech_cfg, noise=errors, interventions={"X": exp_cfg.x0})
        v1, _ = simulate(design, size, seed, mech_cfg, noise=errors, interventions={"X": exp_cfg.x1})
        v10, _ = simulate(design, size, seed, mech_cfg, noise=errors,
                          interventions={"X": exp_cfg.x1, "M": v0[:, ix["M"]]})
        differences = np.column_stack((v1[:, ix["Y"]] - v0[:, ix["Y"]],
                                       v1[:, ix["Y"]] - v10[:, ix["Y"]],
                                       v10[:, ix["Y"]] - v0[:, ix["Y"]]))
        identity_error = max(identity_error, float(np.max(np.abs(differences[:, 0] - differences[:, 1] - differences[:, 2]))))
        batch_mean = differences.mean(axis=0)
        batch_m2 = np.square(differences - batch_mean).sum(axis=0)
        delta = batch_mean - mean
        new_n = total_n + size
        m2 += batch_m2 + delta * delta * (total_n * size / new_n)
        mean += delta * (size / new_n)
        total_n = new_n
    se = np.sqrt(m2 / (total_n - 1) / total_n)
    result = {"kind": "truth_mc_nonlinear", "truth_samples": total_n, "truth_seed": int(seed),
              "chunk_size": exp_cfg.chunk_size, "x0": exp_cfg.x0, "x1": exp_cfg.x1,
              "delta_x": exp_cfg.x1 - exp_cfg.x0, "same_noise_identity_max_abs": identity_error}
    for i, effect in enumerate(("TE", "NIE", "NDE")):
        result[effect] = float(mean[i])
        result["mcse_" + effect] = float(se[i])
    return result


def structural_truth(design, mech_cfg, exp_cfg, seed):
    if is_linear(design, mech_cfg):
        result = _linear_truth(design, exp_cfg.x0, exp_cfg.x1)
        # Independent intervention-based check of the exact path-response calculation.
        eps = draw_noise(np.random.default_rng(seed), (32, len(design.order)), mech_cfg)
        v0, columns = simulate(design, 32, seed, mech_cfg, noise=eps, interventions={"X": exp_cfg.x0})
        v1, _ = simulate(design, 32, seed, mech_cfg, noise=eps, interventions={"X": exp_cfg.x1})
        mi, yi = columns.index("M"), columns.index("Y")
        v10, _ = simulate(design, 32, seed, mech_cfg, noise=eps, interventions={"X": exp_cfg.x1, "M": v0[:, mi]})
        delta = np.column_stack((v1[:, yi] - v0[:, yi], v1[:, yi] - v10[:, yi], v10[:, yi] - v0[:, yi]))
        result["counterfactual_check_max_abs"] = float(np.max(np.abs(delta - [result["TE"], result["NIE"], result["NDE"]])))
        if result["counterfactual_check_max_abs"] > 1e-8 * max(1, abs(result["TE"]), abs(result["NIE"]), abs(result["NDE"])):
            raise ValueError("Linear path truth disagrees with structural counterfactuals")
        return result
    return _mc_truth(design, mech_cfg, exp_cfg, seed)


def analytic_moments(design, mech_cfg):
    """Exact observational mean/covariance for identity edges, no interaction."""
    if not is_linear(design, mech_cfg):
        raise ValueError("Analytic covariance requires a linear additive SEM")
    p = len(design.order)
    index = {node: i for i, node in enumerate(design.order)}
    transfer = np.eye(p)
    for node in design.order:
        j = index[node]
        for parent in design.graph.predecessors(node):
            transfer[j] += design.graph[parent][node]["weight"] * transfer[index[parent]]
    intercept = np.array([design.config["intercepts"][node] for node in design.order])
    return transfer @ intercept, mech_cfg.noise_sd ** 2 * (transfer @ transfer.T)


def diagnose(design, mech_cfg):
    G = design.graph
    pool = sorted(set(G) - nx.descendants(G, "X") - {"X", "M", "Y"})
    om, oy = graph_pair(G, "X", ("M",), "Y")
    union = sorted(set(om) | set(oy))
    arms = {"O": (list(om), list(oy)), "FULL": (pool, pool), "SHARED": (union, union)}
    pure = {node for node, role in design.roles.items() if role == "pure_M_predictor"}
    if pure:
        minus = sorted(set(union) - pure)
        arms["SHAREDm"] = (minus, minus)
    admissible = {name: bool(pair_admissible_set(G, "X", ("M",), "Y", q, r)) for name, (q, r) in arms.items()}
    middle = intermediate_confounders(G, "X", "M", "Y")
    status = "boundary_middle_confounding" if middle else "admissible_known_graph"
    if not middle and not all(admissible.values()):
        raise ValueError(f"Constructed main-design adjustment arms are invalid: {admissible}")
    design.identification_status = status
    sep = "verified_linear_gaussian" if is_linear(design, mech_cfg) and mech_cfg.noise_distribution == "normal" and not middle else "not_certified"
    return {"O_M": list(om), "O_Y": list(oy), "pool_non_descendants": pool,
            "mediator_only_O": sorted(set(om) - set(oy)), "outcome_only_O": sorted(set(oy) - set(om)),
            "shared_O": sorted(set(om) & set(oy)), "outside_both_O": sorted(set(pool) - set(union)),
            "arms": arms, "admissible": admissible, "O_pair_admissible": admissible["O"],
            "intermediate_confounders": middle, "identification_status": status,
            "marginal_separability": sep,
            "note": "Structural truth remains computable in boundary designs; graph-adjustment validity is reported separately."}


def build_design(generation_config, graph_seed, mechanism_seed, calibration_seed):
    config = generation_config.validate()
    design = _sample_graph(config, graph_seed, mechanism_seed)
    design.calibration = _calibrate(design, config.mechanism, config.calibration, calibration_seed)
    design.seeds["calibration"] = int(calibration_seed)
    design.config["calibration_config"] = asdict(config.calibration)
    diagnose(design, config.mechanism)
    return design


def generate(generation_config, *, graph_seed=None, mechanism_seed=None, calibration_seed=None, data_seed=None):
    c = generation_config.validate()
    e = c.experiment
    gs = e.graph_seed if graph_seed is None else graph_seed
    ms = e.mechanism_seed if mechanism_seed is None else mechanism_seed
    cs = e.calibration_seed if calibration_seed is None else calibration_seed
    ds = e.data_seed if data_seed is None else data_seed
    design = build_design(c, gs, ms, cs)
    data, columns = simulate(design, e.n_samples, ds, c.mechanism)
    truth = structural_truth(design, c.mechanism, e, e.truth_seed)
    return {"design": design, "data": data, "columns": columns, "truth": truth,
            "seeds": {"graph": gs, "mechanism": ms, "calibration": cs, "data": ds},
            "identification_status": design.identification_status}
