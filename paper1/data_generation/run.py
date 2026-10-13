"""Create inspectable data and a full mathematical record from a TOML configuration."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from .generator import (analytic_moments, build_design, derive_seed, diagnose, is_linear,
                        simulate, structural_truth)
from .schema import load_config

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def dump(data, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_csv(path, data, columns):
    # Row iteration avoids a second full table of Python float objects.
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(data)


def _profile(data, columns, roles):
    return [{"column": node, "role": roles[node], "mean": float(data[:, i].mean()),
             "sd": float(data[:, i].std(ddof=1)) if len(data) > 1 else None,
             "min": float(data[:, i].min()), "max": float(data[:, i].max())}
            for i, node in enumerate(columns)]


def _population_ols(cov, columns, query, truth):
    """Population linear projections, not an EIF or a finite-sample efficiency claim."""
    index = {v: i for i, v in enumerate(columns)}
    x, m, y = (index[v] for v in ("X", "M", "Y"))
    output = {}
    for name, (q, r) in query["arms"].items():
        first = [x] + [index[node] for node in q]
        second = [x, m] + [index[node] for node in r]
        a = np.linalg.solve(cov[np.ix_(first, first)], cov[first, m])[0]
        beta = np.linalg.solve(cov[np.ix_(second, second)], cov[second, y])
        nie = float(a * beta[1] * truth["delta_x"])
        nde = float(beta[0] * truth["delta_x"])
        output[name] = {"NIE": nie, "NDE": nde, "bias_NIE": nie - truth["NIE"],
                        "bias_NDE": nde - truth["NDE"], "admissible": query["admissible"][name]}
    return output


def _review_report(manifest, path):
    lines = ["# 连续数据生成审阅结果", "", "这是生成器的配置与产物审阅记录，尚未进行方法优劣的确认实验。", "",
             f"- 状态：{manifest['status']}；配置快照：`config.toml`；完整参数：`resolved_config.json`。",
             f"- 源代码哈希、环境和每个随机流见 `manifest.json`；生成耗时 {manifest['seconds']:.3f} 秒。",
             "- 所有数值变量连续。图结构角色与 O 集归属分别记录；每张图固定后重新抽样生成多个重复。", "",
             "| 图 | 节点/边 | 样本数 × 重复 | NIE | NDE | TE | 真值方式 | O 对合法 |",
             "|---|---|---|---:|---:|---:|---|---|"]
    for row in manifest["graphs"]:
        t, q = row["truth"], row["query"]
        lines.append(f"| {row['graph_id']} | {row['n_nodes']}/{row['n_edges']} | {manifest['config']['experiment']['n_samples']} × {len(row['replicates'])} | "
                     f"{t['NIE']:.7g} | {t['NDE']:.7g} | {t['TE']:.7g} | {t['kind']} | {q['O_pair_admissible']} |")
    for row in manifest["graphs"]:
        directory = row["directory"]
        lines += ["", f"## 图 {row['graph_id']}", "",
                  f"- [逐节点方程]({directory}/equations.md)、[完整图与权重]({directory}/design.json)、[第一份数据预览]({directory}/preview_r000.csv)。",
                  f"- 第一个重复的完整数据为 `{directory}/data_r000.npz`；启用 CSV 时另有同名 `.csv`。",
                  f"- 只进中介段的 O 变量：`{', '.join(row['query']['mediator_only_O']) or '空'}`。",
                  f"- 只进结局段的 O 变量：`{', '.join(row['query']['outcome_only_O']) or '空'}`。",
                  f"- 两段共享 O 变量：`{', '.join(row['query']['shared_O']) or '空'}`。",
                  f"- 图上调整状态：`{row['query']['identification_status']}`；边际可分状态：`{row['query']['marginal_separability']}`。"]
        if row["truth"]["kind"] == "truth_mc_nonlinear":
            t = row["truth"]
            lines += [f"- 同噪声反事实样本 {t['truth_samples']} 个；NIE/NDE/TE 的 Monte Carlo 标准误："
                      f"{t['mcse_NIE']:.4g} / {t['mcse_NDE']:.4g} / {t['mcse_TE']:.4g}。"]
        plots = row.get("plots", {})
        if plots.get("graph", {}).get("status") in ("written", "partial"):
            lines += ["", f"![图结构]({directory}/graph.png)"]
        if plots.get("data", {}).get("status") == "written":
            lines += ["", f"![数据分布]({directory}/data_overview.png)"]
    lines += ["", "## 判读边界", "",
              "线性真值由路径响应递推，并用同噪声干预独立核对；非线性真值是明确标注标准误的 Monte Carlo 近似。",
              "中间混杂配置仍有结构反事实真值，但其调整对不合法，不能混入合法调整的效率比较。",
              "非线性或非高斯配置没有自动获得可分性证明；参数中没有交互也不能替代实际边际条件检验。",
              "每次生成的完整方程与数据都供审阅；这里未声称已经验证结论一的效率排序。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path, *, output=None):
    source = Path(config_path).resolve()
    config = load_config(source)
    if output is not None:
        config = replace(config, output=replace(config.output, directory=str(output)))
    config.validate()
    root = Path(config.output.directory).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, root / "config.toml")
    dump(config.to_dict(), root / "resolved_config.json")
    source_paths = sorted(HERE.rglob("*.py")) + [REPO / "paper1" / "graphs.py"]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False)
    manifest = {
        "schema_version": 1, "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(source), "source_config_sha256": file_hash(source),
        "config": config.to_dict(), "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": {str(p.relative_to(REPO)): file_hash(p) for p in source_paths},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "networkx": importlib.metadata.version("networkx")},
        "rng_protocol": "SeedSequence([base_seed, graph_id, replicate_id, stream_tag]); tags graph=0, mechanism=1, calibration=2, truth=3, data=4",
        "purpose": "continuous data-generator review, not a confirmatory efficiency experiment", "graphs": [],
    }
    manifest_path = root / "manifest.json"
    dump(manifest, manifest_path)
    started = time.monotonic()
    e, m, o = config.experiment, config.mechanism, config.output
    try:
        from .reporting import write_data_plot, write_equations, write_graph
        for graph_id in range(e.n_graphs):
            seeds = {name: derive_seed(getattr(e, name + "_seed"), graph_id, stream=tag)
                     for tag, name in enumerate(("graph", "mechanism", "calibration", "truth"))}
            design = build_design(config, seeds["graph"], seeds["mechanism"], seeds["calibration"])
            truth = structural_truth(design, m, e, seeds["truth"])
            query = diagnose(design, m)
            directory = root / f"graph_{graph_id:03d}"
            directory.mkdir()
            metadata = design.metadata()
            metadata.update(truth=truth, query=query, seeds=seeds)
            moment_result = {"status": "skipped", "reason": "nonlinear mechanism or covariance size limit"}
            if is_linear(design, m) and len(design.order) <= o.max_covariance_nodes:
                mean, cov = analytic_moments(design, m)
                np.savez_compressed(directory / "population_moments.npz", mean=mean, covariance=cov, columns=design.order)
                moment_result = {"status": "computed", "file": "population_moments.npz", "condition_number": float(np.linalg.cond(cov)),
                                 "population_ols": _population_ols(cov, list(design.order), query, truth)}
            metadata["population_moments"] = moment_result
            dump(metadata, directory / "design.json")
            write_equations(design, m, directory / "equations.md")
            plots = {"graph": {"status": "disabled"}, "data": {"status": "disabled"}}
            if o.plot:
                plots["graph"] = write_graph(design, query, directory, o.max_plot_nodes)
            row = {"graph_id": graph_id, "directory": directory.name, "n_nodes": len(design.order),
                   "n_edges": design.graph.number_of_edges(), "truth": truth, "query": query,
                   "seeds": seeds, "population_moments": moment_result, "replicates": [], "plots": plots}
            for rep in range(e.n_replicates):
                data_seed = derive_seed(e.data_seed, graph_id, rep, 4)
                data, columns = simulate(design, e.n_samples, data_seed, m)
                name = f"data_r{rep:03d}"
                np.savez_compressed(directory / (name + ".npz"), values=data, columns=columns)
                if o.csv:
                    _write_csv(directory / (name + ".csv"), data, columns)
                _write_csv(directory / f"preview_r{rep:03d}.csv", data[:o.preview_rows], columns)
                dump(_profile(data, columns, design.roles), directory / f"profile_r{rep:03d}.json")
                row["replicates"].append({"replicate": rep, "data_seed": data_seed, "shape": list(data.shape),
                                          "file": name + ".npz", "sha256": file_hash(directory / (name + ".npz"))})
                if o.plot and rep == 0:
                    plots["data"] = write_data_plot(data, columns, query, directory)
            manifest["graphs"].append(row)
            dump(manifest, manifest_path)
            print(f"graph {graph_id}: p={len(design.order)}, edges={design.graph.number_of_edges()}, "
                  f"replicates={e.n_replicates}, NIE={truth['NIE']:.6g}, NDE={truth['NDE']:.6g}, "
                  f"O admissible={query['O_pair_admissible']}", flush=True)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["seconds"] = round(time.monotonic() - started, 3)
        dump(manifest, manifest_path)
    _review_report(manifest, root / "REVIEW.md")
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, help="Override output.directory; must be new or empty")
    args = parser.parse_args(argv)
    print(run(args.config, output=args.out))


if __name__ == "__main__":
    main()
