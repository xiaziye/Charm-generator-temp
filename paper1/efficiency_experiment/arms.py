"""Known-graph comparators inherited from the D3 efficiency experiment.

The oracle graph determines adjustment sets here; effect estimation receives only the
observations, the two sets and the exposure contrast.  This is not a graph-learning run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx

from ..graphs import d_sep, graph_pair, pair_admissible_set


@dataclass(frozen=True)
class AdjustmentArm:
    name: str
    CM: tuple[str, ...]
    CY: tuple[str, ...]
    admissible: bool | None
    kind: str = "query"
    available: bool = True
    reason: str | None = None

    def to_dict(self):
        return asdict(self)


def metadata_graph(metadata):
    columns = metadata.get("columns")
    if (not isinstance(columns, list) or not columns or any(not isinstance(c, str) or not c for c in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError("Design columns must be a nonempty list of unique node names")
    G = nx.DiGraph()
    G.add_nodes_from(columns)
    for edge in metadata.get("edges", ()):
        u, v = edge["source"], edge["target"]
        if u not in G or v not in G:
            raise ValueError("Design edge references a node absent from columns")
        if G.has_edge(u, v):
            raise ValueError("Design has duplicate directed edges")
        G.add_edge(u, v)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Known-graph experiment requires an acyclic design")
    return G


def query_roles(metadata):
    roles = metadata.get("roles", {})
    result = {}
    for key, role in (("x", "exposure"), ("m", "mediator"), ("y", "outcome")):
        matches = [node for node, value in roles.items() if value == role]
        if len(matches) != 1:
            raise ValueError(f"Design must identify exactly one {role}, got {matches}")
        result[key] = matches[0]
    if any(node not in metadata["columns"] for node in result.values()):
        raise ValueError("Query role is absent from design columns")
    return result


def build_arms(metadata, specs):
    """Build standard/custom adjustment pairs and independently verify their graph validity."""
    G = metadata_graph(metadata)
    r = query_roles(metadata)
    x, m, y = r["x"], r["m"], r["y"]
    om, oy = (set(v) for v in graph_pair(G, x, (m,), y))
    pool = set(G) - nx.descendants(G, x) - {x, m, y}
    union = om | oy
    roles = metadata.get("roles", {})
    pm = {node for node, role in roles.items() if role == "pure_M_predictor"}
    weak = {node for node, role in roles.items() if role in {"weak_M_parent", "weak_Y_parent"}}
    standard = {"O": (om, oy), "FULL": (pool, pool), "SHARED": (union, union),
                "SHAREDm": (union - pm, union - pm)}
    if weak:
        standard["OMISS"] = (om - weak, oy - weak)
    output = []
    for spec in specs:
        if spec.kind == "custom":
            cm, cy = set(spec.CM), set(spec.CY)
        elif spec.name in standard:
            cm, cy = standard[spec.name]
            # Saved query snapshots may contain additional arms; verify standard definitions,
            # rather than accepting stale labels after edits to a design.
            saved = metadata.get("query", {}).get("arms", {}).get(spec.name)
            if saved is not None and (set(saved[0]), set(saved[1])) != (cm, cy):
                raise ValueError(f"Saved {spec.name} adjustment pair disagrees with its known-graph definition")
        else:
            output.append(AdjustmentArm(spec.name, (), (), None, spec.kind, False,
                                        "OMISS requires explicitly labelled weak_M_parent/weak_Y_parent nodes"
                                        if spec.name == "OMISS" else "query arm is unavailable in this design"))
            continue
        missing = (cm | cy) - set(G)
        if missing:
            raise ValueError(f"Arm {spec.name} references unknown columns: {sorted(missing)}")
        if cm & {x, m, y} or cy & {x, m, y}:
            raise ValueError(f"Arm {spec.name} adjustment sets cannot include X, M or Y")
        valid = bool(pair_admissible_set(G, x, (m,), y, cm, cy))
        output.append(AdjustmentArm(spec.name, tuple(sorted(cm)), tuple(sorted(cy)), valid,
                                    spec.kind, True, None if valid else "graph-adjustment pair is not admissible"))
    prediction = {"med_inst": not d_sep(G, {x}, pool - om, om),
                  "out_inst": not d_sep(G, {x, m}, pool - oy, oy)}
    prediction["predicted_equal"] = not (prediction["med_inst"] or prediction["out_inst"])
    prediction["definition"] = "Graph d-separation conditions for O versus full adjustment; not a numerical equality test"
    factors = dict(metadata.get("factors", metadata.get("config", {}).get("graph", {})))
    for name, role in (("n_inst", "instrument"), ("n_pm", "pure_M_predictor"),
                       ("n_py", "pure_Y_predictor")):
        factors.setdefault(name, sum(value == role for value in roles.values()))
    factors.setdefault("n_weak", len(weak))
    return tuple(output), r, factors, prediction


def check_estimator_compatibility(metadata, estimator_config):
    """Reject unsupported mechanisms explicitly instead of treating a linear projection as an effect."""
    if estimator_config.name == "linear_ols":
        mechanism = metadata.get("config", {}).get("mechanism", {})
        nonlinear = any(edge.get("function", "id") != "id" for edge in metadata.get("edges", ()))
        if (nonlinear or mechanism.get("outcome_interaction", 0) != 0) and not estimator_config.allow_misspecification:
            raise ValueError("linear_ols requires identity edge functions and no X*M interaction; register an appropriate estimator for this mechanism")
