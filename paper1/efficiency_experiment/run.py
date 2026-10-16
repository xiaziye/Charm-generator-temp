"""Run fixed-model repeated-sampling experiments with explicit estimator/arm interfaces."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

import numpy as np

from ..data_generation.generator import derive_seed
from .arms import build_arms, check_estimator_compatibility
from .endpoints import evaluate_endpoints
from .estimators import available_estimators, fit_estimate
from .io import file_hash, load_inputs, resolve_path
from .metrics import summarize
from .schema import ExperimentConfig, load_config

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def dump_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_summary_csv(summary, path):
    fields = ["graph_id", "n", "estimator", "arm", "admissible", "n_attempted", "n_success", "n_failed", "n_skipped", "failure_rate", "exclusion_rate", "non_success_rate",
              "confidence_level", "seconds_total", "seconds_mean", "seconds_median",
              "mediator_max_condition_number", "outcome_max_condition_number",
              "effect", "truth", "mean", "bias", "mcse_bias", "variance", "mcse_variance", "sd", "mse",
              "rmse", "mae", "mcse_mae", "mean_se", "coverage", "mcse_coverage", "n_intervals", "mean_interval_width"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for group in summary["groups"]:
            for arm, values in group["arms"].items():
                for effect, result in values["results"].items():
                    writer.writerow({"graph_id": group["graph_id"], "n": group["n"], "estimator": group["estimator"],
                                     "arm": arm, **{key: values.get(key) for key in ("admissible", "n_attempted", "n_success", "n_failed", "n_skipped",
                                     "failure_rate", "exclusion_rate", "non_success_rate", "confidence_level",
                                     "seconds_total", "seconds_mean", "seconds_median")},
                                     **{stage + "_max_condition_number": values.get("numerical_diagnostics", {}).get(stage, {}).get("max_condition_number")
                                        for stage in ("mediator", "outcome")},
                                     "effect": effect, **result})


def _write_comparisons_csv(summary, path):
    fields = ["graph_id", "n", "estimator", "arm", "reference_arm", "effect", "n_paired", "variance_arm",
              "variance_reference", "variance_ratio", "variance_ratio_mcse", "variance_reduction_reference_vs_arm", "status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for group in summary["groups"]:
            for comparison in group["comparisons"]:
                writer.writerow({"graph_id": group["graph_id"], "n": group["n"], "estimator": group["estimator"], **comparison})


def _ordered_parallel(function, identifiers, workers):
    """Bound resident data to O(workers) replicates and preserve deterministic output order."""
    if workers == 1:
        yield from map(function, identifiers)
        return
    iterator = iter(identifiers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="paper1-efficiency") as executor:
        pending = deque()
        for _ in range(workers * 2):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def _fit_replicate(graph, arms, roles, factors, prediction, config, replicate):
    common = {"graph_id": graph.graph_id, "replicate": replicate, "estimator": config.estimator.name,
              "factors": factors, "prediction": prediction, "confidence_level": config.estimator.confidence_level,
              "truth_used_for_fitting": False,
              "source_mode": graph.provenance["mode"], "n_replicates_planned": len(graph.replicate_ids)}
    truth = graph.metadata["truth"]
    common.update({"truth_" + effect: float(truth[effect]) for effect in ("NIE", "NDE", "TE")})
    try:
        values, columns, provenance = graph.sample(replicate)
    except Exception as exc:
        if config.execution.on_error == "raise":
            raise
        return [{**common, "n": n, "arm": arm.name, "admissible": arm.admissible,
                 "status": "error", "error_stage": "sample", "error": f"{type(exc).__name__}: {exc}"}
                for n in graph.sample_sizes for arm in arms]
    rows = []
    for n in graph.sample_sizes:
        estimator = config.estimator
        if estimator.se_method == "bootstrap":
            estimator = replace(estimator, bootstrap_seed=derive_seed(estimator.bootstrap_seed, graph.graph_id, replicate, n))
        for arm in arms:
            row = {**common, "n": n, "arm": arm.name, "admissible": arm.admissible,
                   "n_CM": len(arm.CM), "n_CY": len(arm.CY), "sample": provenance,
                   "diagnostic_non_admissible": arm.admissible is False,
                   "diagnostic_misspecification": (estimator.name == "linear_ols" and
                       (any(edge.get("function", "id") != "id" for edge in graph.metadata.get("edges", ())) or
                        graph.metadata.get("config", {}).get("mechanism", {}).get("outcome_interaction", 0) != 0)),
                   "bootstrap_seed": estimator.bootstrap_seed if estimator.se_method == "bootstrap" else None}
            if not arm.available:
                row.update(status="unavailable", reason=arm.reason)
            elif not arm.admissible and not config.execution.allow_non_admissible:
                row.update(status="skipped_non_admissible", reason=arm.reason)
            else:
                started = time.monotonic()
                try:
                    result = fit_estimate(values[:n], columns, arm.CM, arm.CY,
                                          **roles, x0=float(truth.get("x0", 0)), x1=float(truth.get("x1", 1)), config=estimator)
                    json.dumps(result, allow_nan=False)
                    row.update(result)
                    # Identity and truth fields are runner-owned, never overridable by an estimator plugin.
                    row.update(common, n=n, arm=arm.name, admissible=arm.admissible, status="ok")
                    json.dumps(row, allow_nan=False)
                except Exception as exc:
                    if config.execution.on_error == "raise":
                        raise
                    row.update(status="error", error_stage="fit", error=f"{type(exc).__name__}: {exc}",
                               seconds=time.monotonic() - started)
            rows.append(row)
    return rows


def _code_provenance():
    files = sorted(HERE.glob("*.py")) + [REPO / "paper1" / "graphs.py", REPO / "paper1" / "__init__.py"]
    files += sorted((REPO / "paper1" / "data_generation").glob("*.py"))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=False)
    return {"git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            "source_sha256": {str(path.relative_to(REPO)): file_hash(path) for path in files}}


def _report(manifest, summary, endpoints, path):
    lines = ["# 已知图效率实验结果", "", "本次运行按固定图与固定机制分别汇总；不同图之间不混算估计方差。",
             "这是可复核的实验执行结果，确认实验仍需事先冻结完整面板、随机种子和判据。", "",
             f"- 状态：`{manifest['status']}`；运行耗时：{manifest['seconds']:.3f} 秒。",
             f"- 输入模式：`{manifest['source']['mode']}`；拟合记录：{summary['n_records']} 条。",
             "- 配置：`config.toml`（文件方式运行时）、`resolved_config.json`；追溯信息：`manifest.json`。",
             "- 各图调整变量名单与合法性在 `graph_*/arms.json`；生成机制与真值在 `graph_*/design.json`。",
             "- `summary.csv` 提供偏差/方差/MSE/覆盖率，`comparisons.csv` 提供同一重复配对后的方差比及 Monte Carlo 标准误。",
             "- 真图只用于本已知图实验的方案构造/合法性判定；真值只用于拟合后的评分。", ""]
    if manifest["warnings"]:
        lines += ["## 本次运行的限制", ""] + ["- " + item for item in manifest["warnings"]] + [""]
    lines += ["| 图 | n | 方案 | 成功/尝试 | 合法 | NIE 偏差 | NIE 方差 | NDE 偏差 | NDE 方差 |",
              "|---|---:|---|---:|---|---:|---:|---:|---:|"]
    def number(value):
        return "—" if value is None else f"{value:.6g}"
    for group in summary["groups"]:
        for name, arm in group["arms"].items():
            a, b = arm["results"]["NIE"], arm["results"]["NDE"]
            lines.append(f"| {group['graph_id']} | {group['n']} | {name} | {arm['n_success']}/{arm['n_attempted']} | {arm['admissible']} | "
                         f"{number(a['bias'])} | {number(a['variance'])} | {number(b['bias'])} | {number(b['variance'])} |")
    lines += ["", "## 原实验终点", "", "完整判据与排除原因见 `endpoints.json`。小规模试运行的数值不构成效率排序结论。", "",
              "OLS 只有在样本量大于回归系数数量、设计矩阵满秩且条件数合格时才拟合；失败不会由伪逆或正则化静默替代。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(config: ExperimentConfig, *, config_directory=None, output=None, config_path=None):
    """Programmatic entry point; custom estimators can be registered before this call.

    Configured relative paths use ``config_directory`` (cwd by default). ``output``
    is an explicit CLI/API override relative to cwd. Existing nonempty output is refused.
    """
    config.validate()
    if config.estimator.name not in available_estimators():
        raise ValueError(f"Unknown estimator {config.estimator.name!r}; registered: {available_estimators()}")
    base = Path(config_directory or Path.cwd()).resolve()
    root = Path(output).expanduser().resolve() if output is not None else resolve_path(config.output.directory, base)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {root}")
    graphs, source = load_inputs(config, base)
    prepared = []
    for graph in graphs:
        # Reject malformed/non-finite metadata before creating any output files.
        json.dumps(graph.metadata, allow_nan=False)
        check_estimator_compatibility(graph.metadata, config.estimator)
        arms, roles, factors, prediction = build_arms(graph.metadata, config.arms)
        if "truth" not in graph.metadata or any(effect not in graph.metadata["truth"] for effect in ("NIE", "NDE", "TE")):
            raise ValueError(f"Graph {graph.graph_id} has no complete NIE/NDE/TE evaluation truth")
        truth = graph.metadata["truth"]
        if any(isinstance(truth[effect], bool) or not isinstance(truth[effect], (int, float, np.number))
               or not np.isfinite(truth[effect]) for effect in ("NIE", "NDE", "TE")):
            raise ValueError(f"Graph {graph.graph_id} evaluation truth must be finite numbers")
        if not np.isclose(truth["TE"], truth["NIE"] + truth["NDE"], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Graph {graph.graph_id} effect truth violates TE = NIE + NDE")
        x0, x1 = truth.get("x0", 0.0), truth.get("x1", 1.0)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
               for value in (x0, x1)) or x0 == x1:
            raise ValueError(f"Graph {graph.graph_id} truth must specify two distinct finite exposure levels")
        if "delta_x" in truth and not np.isclose(truth["delta_x"], x1 - x0, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Graph {graph.graph_id} truth contrast has inconsistent delta_x")
        prepared.append((graph, arms, roles, factors, prediction))
    root.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copyfile(config_path, root / "config.toml")
    dump_json(config.to_dict(), root / "resolved_config.json")
    manifest = {"schema_version": 1, "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "exploratory known-graph repeated-sampling efficiency experiment", "confirmation": False,
                "source": source, "config": config.to_dict(), "output": str(root),
                "config_sha256": file_hash(Path(config_path)) if config_path is not None else None,
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "networkx": importlib.metadata.version("networkx")},
                "warnings": [], "graphs": [], "rows_file": "replicates.jsonl" if config.output.per_replicate else None,
                "threads_note": "workers parallelize independent replicates in threads; bound numpy/BLAS threads externally for CPU benchmarks",
                **_code_provenance()}
    dump_json(manifest, root / "manifest.json")
    total_replicates = sum(len(graph.replicate_ids) for graph, *_ in prepared)
    total_fits = sum(len(graph.replicate_ids) * len(graph.sample_sizes) * len(arms) for graph, arms, *_ in prepared)
    print(f"Efficiency experiment: graphs={len(graphs)}, sample draws={total_replicates}, planned arm records={total_fits}, workers={config.execution.workers}", flush=True)
    summary = summarize([], reference_arm=config.execution.reference_arm, min_replicates=config.execution.min_replicates)
    overall_statuses = Counter()
    started = time.monotonic()
    output_stream = (root / "replicates.jsonl").open("w", encoding="utf-8") if config.output.per_replicate else None
    try:
        for graph, arms, roles, factors, prediction in prepared:
            directory = root / f"graph_{graph.graph_id:03d}"
            directory.mkdir()
            dump_json(graph.metadata, directory / "design.json")
            dump_json({"query_roles": roles, "arms": [arm.to_dict() for arm in arms],
                       "factors": factors, "prediction": prediction}, directory / "arms.json")
            if len(graph.replicate_ids) < config.execution.min_replicates:
                manifest["warnings"].append(f"图 {graph.graph_id} 仅 {len(graph.replicate_ids)} 次重复，低于配置的 {config.execution.min_replicates} 次精度提醒线；仅作流程检查。")
            if any(arm.admissible is False for arm in arms):
                manifest["warnings"].append(f"图 {graph.graph_id} 包含不合法调整对，已{'按诊断拟合' if config.execution.allow_non_admissible else '跳过拟合'}；不能作为合法方案效率优势。")
            graph_row = {"graph_id": graph.graph_id, "directory": directory.name, "sample_sizes": list(graph.sample_sizes),
                         "n_replicates": len(graph.replicate_ids), "factors": factors, "prediction": prediction,
                         "provenance": graph.provenance, "arms": [arm.to_dict() for arm in arms]}
            manifest["graphs"].append(graph_row)
            graph_rows = []
            task = lambda rep: _fit_replicate(graph, arms, roles, factors, prediction, config, rep)
            for fit_rows in _ordered_parallel(task, graph.replicate_ids, config.execution.workers):
                graph_rows.extend(fit_rows)
                overall_statuses.update(row["status"] for row in fit_rows)
                if output_stream is not None:
                    for row in fit_rows:
                        output_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    output_stream.flush()
            graph_summary = summarize(graph_rows, reference_arm=config.execution.reference_arm,
                                      min_replicates=config.execution.min_replicates)
            summary["groups"].extend(graph_summary["groups"])
            summary["n_records"] += graph_summary["n_records"]
            graph_row["statuses"] = dict(Counter(row["status"] for row in graph_rows))
            print(f"graph {graph.graph_id}: n={list(graph.sample_sizes)}, repeats={len(graph.replicate_ids)}, statuses={graph_row['statuses']}", flush=True)
            dump_json(manifest, root / "manifest.json")
            dump_json(summary, root / "summary.json")
            del graph_rows
        endpoints = evaluate_endpoints(summary, thresholds=config.endpoints.to_dict())
        dump_json(summary, root / "summary.json")
        dump_json(endpoints, root / "endpoints.json")
        _write_summary_csv(summary, root / "summary.csv")
        _write_comparisons_csv(summary, root / "comparisons.csv")
        if overall_statuses.get("error"):
            manifest["warnings"].append(f"共有 {overall_statuses['error']} 条拟合/输入记录失败，失败率已单独列出，方差比只使用配对成功重复。")
        if not overall_statuses.get("ok"):
            manifest["warnings"].append("本次运行没有成功拟合记录，不产生可用的效率比较结论。")
        manifest.update(status="complete", n_records=summary["n_records"], statuses=dict(overall_statuses),
                        files={"summary": "summary.json", "summary_csv": "summary.csv", "comparisons_csv": "comparisons.csv",
                               "endpoints": "endpoints.json", "report": "REPORT.md"})
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if output_stream is not None:
            output_stream.close()
        manifest["seconds"] = round(time.monotonic() - started, 6)
        dump_json(manifest, root / "manifest.json")
    _report(manifest, summary, endpoints, root / "REPORT.md")
    return root / "manifest.json"


def run(config_path, *, output=None):
    path = Path(config_path).resolve()
    return run_experiment(load_config(path), config_directory=path.parent, output=output, config_path=path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, help="Override output.directory, relative to cwd; must be new or empty")
    args = parser.parse_args(argv)
    print(run(args.config, output=args.out))


if __name__ == "__main__":
    main()
