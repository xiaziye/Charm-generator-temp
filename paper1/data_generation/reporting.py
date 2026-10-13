"""Local mathematical and scientific views of a generated continuous SEM."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import networkx as nx
import numpy as np


_ROLE_COLORS = {
    "baseline": "#9ba6ac",
    "instrument": "#4a90c2",
    "pure_M_predictor": "#5cadb2",
    "pure_Y_predictor": "#b58abd",
    "irrelevant": "#d7d7d7",
    "exposure": "#dc6b5a",
    "mediator": "#59a76a",
    "outcome": "#e0b347",
    "pre_mediator": "#a2ca75",
    "post_mediator": "#bcce68",
    "parallel_mediator": "#70b4a4",
    "post_exposure": "#c39e78",
    "intermediate_confounder": "#be5b82",
}


def _pyplot():
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "charm-generator-mpl")
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return plt


def _endpoints(query: Mapping[str, Any]) -> tuple[str, str, str]:
    return tuple(str(query.get(name.lower(), name)) for name in ("X", "M", "Y"))


def _tex_node(name: str) -> str:
    return r"\mathrm{" + name.replace("_", r"\_") + "}"


def _number(value: float) -> str:
    return format(float(value), ".10g")


def _signed(value: float) -> str:
    return format(float(value), "+.10g")


def write_equations(design: Any, mechanism: Any, path: str | Path) -> dict[str, Any]:
    """Write every structural equation in topological order as Chinese Markdown."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    intercepts = design.config["intercepts"]
    lines = [
        "# 结构方程与生成机制", "",
        "这些方程是已知的数据生成机制，不是从观测数据拟合得到的回归式。变量按拓扑顺序生成；每条边使用校准后的实际权重。",
        "本文显示 10 位有效数字，`design.json` 保留实际使用的双精度参数，数值复现应以该文件为准。", "",
        f"节点数：{len(design.graph)}；边数：{design.graph.number_of_edges()}。",
        f"机制模式：`{mechanism.mode}`；识别诊断：`{design.identification_status}`。", "",
        r"函数定义：$\operatorname{id}(z)=z$，$\tanh(z)$ 和 $\sin(z)$ 分别为双曲正切与正弦。", "",
        "## 逐节点方程", "",
    ]
    for node in design.order:
        rhs = _number(intercepts[node])
        for parent in design.graph.predecessors(node):
            weight = design.graph[parent][node]["weight"]
            function = design.edge_functions[parent, node]
            rhs += (
                " " + _signed(weight) + r"\,\operatorname{" + function + "}("
                + _tex_node(parent) + ")"
            )
        if node == "Y" and mechanism.outcome_interaction:
            rhs += " " + _signed(mechanism.outcome_interaction) + r"\," + _tex_node("X") + r"\," + _tex_node("M")
        rhs += r" + \varepsilon_{" + _tex_node(node) + "}"
        lines.extend([
            f"### {node}（{design.roles.get(node, 'unknown')}）", "",
            "$$", _tex_node(node) + " = " + rhs, "$$", "",
        ])
    sd = _number(mechanism.noise_sd)
    lines.extend(["## 外生噪声", "", "各节点、各样本的外生噪声独立；噪声均值为零，标准差为 `noise_sd`。", ""])
    if mechanism.noise_distribution == "normal":
        lines.append(r"$\varepsilon_v \sim \mathcal{N}(0, " + sd + r"^2)$。")
    elif mechanism.noise_distribution == "laplace":
        lines.append(r"$\varepsilon_v \sim \operatorname{Laplace}(0, " + sd + r"/\sqrt{2})$，其中第二个参数是尺度。")
    else:
        df = _number(mechanism.noise_df)
        lines.append(r"$\varepsilon_v=" + sd + r"\,T_v/\sqrt{" + df + "/(" + df + r"-2)}$，$T_v\sim t_{" + df + r"}$，故噪声方差为 `noise_sd` 的平方。")
    lines.extend([
        "", "## 校准与干预", "",
        "校准只缩放一个节点的基线来源父点贡献之和，使其在独立校准样本中的标准差达到配置目标；它不固定总信噪比或总 R²。受保护的因果路径权重不会因此缩放。",
        "干预时，被干预节点的方程整体替换为指定值，其余方程和外生噪声保持同一套机制。中间混杂边界中的结构反事实仍可生成，但不等于自然效应已由观测数据识别。", "",
    ])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return {"status": "written", "path": str(destination)}


def _layers(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    layers = [sorted(layer) for layer in nx.topological_generations(graph)]
    positions = {}
    for depth, layer in enumerate(layers):
        for i, node in enumerate(layer):
            positions[node] = (float(depth), (len(layer) - 1) / 2.0 - i)
    return positions


def write_graph(design: Any, query: Mapping[str, Any], outdir: str | Path,
                max_nodes: int) -> dict[str, Any]:
    """Write a role-colored DAG, explicitly reporting any omitted nodes."""
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 3:
        raise ValueError("max_nodes must be an integer >= 3")
    original = design.graph
    if not nx.is_directed_acyclic_graph(original):
        raise ValueError("graph plotting requires a DAG")
    endpoints = _endpoints(query)
    if any(node not in original for node in endpoints):
        raise ValueError("query endpoints must occur in the graph")
    total = len(original)
    status = "written"
    graph = original
    reason = None
    if total > max_nodes:
        ancestors = set(endpoints)
        for endpoint in endpoints:
            ancestors.update(nx.ancestors(original, endpoint))
        if len(ancestors) > max_nodes:
            return {
                "status": "skipped", "reason": "query ancestors exceed max_nodes",
                "n_nodes_total": total, "n_nodes_plotted": 0,
                "n_query_ancestor_nodes": len(ancestors), "max_nodes": max_nodes,
            }
        graph = original.subgraph(ancestors).copy()
        status = "partial"
        reason = "only query endpoints and their ancestors are shown"
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    from matplotlib.patches import Patch

    layers = list(nx.topological_generations(graph))
    width = max(12.0, min(24.0, len(layers) * 1.8))
    height = max(7.0, min(20.0, max(map(len, layers)) * 0.44 + 2.5))
    figure, axis = plt.subplots(figsize=(width, height))
    nodes = list(graph)
    roles = design.roles
    colors = [_ROLE_COLORS.get(roles.get(node, "unknown"), "#cccccc") for node in nodes]
    positions = _layers(graph)
    for source, target, data in graph.edges(data=True):
        distance = positions[target][0] - positions[source][0]
        curvature = 0.12 if distance > 1 else 0.025
        nx.draw_networkx_edges(
            graph, positions, edgelist=[(source, target)],
            edge_color="#494949" if data.get("protected") else "#b1b6b9",
            width=1.0, arrows=True, arrowsize=13, node_size=760,
            connectionstyle=f"arc3,rad={curvature}", ax=axis,
        )
    nx.draw_networkx_nodes(graph, positions, nodelist=nodes, node_color=colors,
                           node_size=760, edgecolors="#555555", linewidths=0.6, ax=axis)
    nx.draw_networkx_labels(graph, positions, font_size=8, ax=axis)
    role_names = sorted({roles.get(node, "unknown") for node in nodes})
    legend = [Patch(facecolor=_ROLE_COLORS.get(role, "#cccccc"), label=role)
              for role in role_names]
    axis.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    qualifier = "DAG" if status == "written" else "PARTIAL DAG: query ancestors only"
    axis.set_title(
        f"{qualifier} | {len(graph)}/{total} nodes, "
        f"{graph.number_of_edges()}/{original.number_of_edges()} edges"
    )
    axis.set_axis_off()
    figure.tight_layout()
    png, svg = destination / "graph.png", destination / "graph.svg"
    try:
        figure.savefig(png, dpi=160, bbox_inches="tight")
        figure.savefig(svg, bbox_inches="tight")
    finally:
        plt.close(figure)
    return {
        "status": status, "png": str(png), "svg": str(svg), "reason": reason,
        "n_nodes_total": total, "n_nodes_plotted": len(graph),
        "n_edges_total": original.number_of_edges(), "n_edges_plotted": graph.number_of_edges(),
        "n_nodes_omitted": total - len(graph), "max_nodes": max_nodes,
    }


def write_data_plot(data: np.ndarray, columns: list[str], query: Mapping[str, Any],
                    outdir: str | Path) -> dict[str, Any]:
    """Write fixed-size distribution, association and selected-correlation panels."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(columns) or values.shape[0] == 0:
        raise ValueError("data must be a non-empty row-by-column matrix")
    if len(set(columns)) != len(columns):
        raise ValueError("column names must be unique")
    sample = values[:2000]
    if not np.isfinite(sample).all():
        raise ValueError("plot rows must contain only finite data")
    index = {name: i for i, name in enumerate(columns)}
    x, m, y = _endpoints(query)
    if any(name not in index for name in (x, m, y)):
        raise ValueError("query endpoints must occur in the data")
    selected = []
    for name in [x, m, y] + sorted(query.get("O_M", [])) + sorted(query.get("O_Y", [])):
        if name in index and name not in selected:
            selected.append(name)
        if len(selected) == 12:
            break
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, name, color in zip(axes[0], (x, m, y), ("#dc6b5a", "#59a76a", "#e0b347")):
        axis.hist(sample[:, index[name]], bins=35, color=color, alpha=0.8)
        axis.set(xlabel=name, ylabel="Count", title=f"{name} distribution")
    for axis, left, right in ((axes[1, 0], x, m), (axes[1, 1], m, y)):
        axis.scatter(sample[:, index[left]], sample[:, index[right]], s=9, alpha=0.3,
                     color="#347b9d", edgecolors="none", rasterized=True)
        axis.set(xlabel=left, ylabel=right, title=f"{left} vs {right}")
    subset = sample[:, [index[name] for name in selected]]
    centered = subset - subset.mean(axis=0)
    products = centered.T @ centered
    denom = np.sqrt(np.outer(np.diag(products), np.diag(products)))
    correlations = np.full_like(products, np.nan)
    np.divide(products, denom, out=correlations, where=denom > 0)
    axis = axes[1, 2]
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad("#eeeeee")
    heatmap = axis.imshow(np.ma.masked_invalid(correlations), vmin=-1, vmax=1,
                          cmap=colormap, interpolation="nearest")
    axis.set_xticks(range(len(selected)), selected, rotation=90, fontsize=8)
    axis.set_yticks(range(len(selected)), selected, fontsize=8)
    axis.set_title("Selected Pearson correlations")
    figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"Observational sample: first {len(sample)} of {len(values)} rows")
    png, svg = destination / "data_overview.png", destination / "data_overview.svg"
    try:
        figure.savefig(png, dpi=160)
        figure.savefig(svg)
    finally:
        plt.close(figure)
    return {
        "status": "written", "png": str(png), "svg": str(svg),
        "n_rows_plotted": len(sample), "n_rows_total": len(values),
        "correlation_columns": selected, "undefined_correlations": int(np.isnan(correlations).sum()),
    }
