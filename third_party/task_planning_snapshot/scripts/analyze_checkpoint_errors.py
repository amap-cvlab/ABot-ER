#!/usr/bin/env python
"""Generate reusable checkpoint error-analysis JSON and Chinese HTML reports."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from task_planning_config import job_output_dir, job_scratch_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def model_basename(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def as_number(value: Any) -> float | None:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def numeric_average(paths: list[Path]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for path in paths:
        try:
            data = load_json(path)
        except Exception:
            continue
        for key, value in data.items():
            number = as_number(value)
            if number is None:
                continue
            sums[key] = sums.get(key, 0.0) + number
            counts[key] = counts.get(key, 0) + 1
    return {key: round(sums[key] / counts[key], 6) for key in sorted(sums)}


def short_path(path: Path, root: Path | None = None) -> str:
    try:
        return str(path.relative_to(root or PROJECT_ROOT))
    except ValueError:
        return str(path)


def instruction_keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z_-]{2,}", text.lower())
    stop = {
        "the",
        "and",
        "you",
        "can",
        "please",
        "move",
        "bring",
        "find",
        "locate",
        "task",
        "with",
        "from",
        "into",
        "onto",
        "left",
        "right",
    }
    return [word for word in words if word not in stop]


def load_final_summary(root: Path) -> dict[str, Any]:
    path = root / "full" / "final_summary.json"
    if path.exists():
        return load_json(path)
    return {}


def load_et_strict_summary(root: Path) -> dict[str, Any]:
    path = root / "strict_rescore" / "et_strict_summary.json"
    if path.exists():
        return load_json(path)
    return {}


def summary_with_strict_et(final_summary: dict[str, Any], et_strict: dict[str, Any]) -> dict[str, Any]:
    if not final_summary or not et_strict:
        return final_summary
    sanitized = json.loads(json.dumps(final_summary, ensure_ascii=False))
    scores = sanitized.setdefault("scores", {})
    scores["ET-Plan_Bench"] = {
        "task_count": et_strict.get("task_count", 0),
        "summary_files": et_strict.get("summary_files", 0),
        "success_rate": et_strict.get("task_success_rate", et_strict.get("success_rate", 0)),
        "task_success_rate": et_strict.get("task_success_rate", 0),
        "format_success_rate": et_strict.get("format_success_rate", 0),
        "action_execution_success_rate": et_strict.get("action_execution_success_rate", 0),
        "average_steps": et_strict.get("average_steps", 0),
        "average_lcs": et_strict.get("average_lcs", 0),
        "average_lcs_ratio": et_strict.get("average_lcs_ratio", 0),
    }
    return sanitized


def analyze_et(root: Path, max_cases: int) -> dict[str, Any]:
    base = root / "strict_rescore" / "ET-Plan_Bench"
    result: dict[str, Any] = {}
    if not base.exists():
        return result

    for bench_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        rows: list[dict[str, Any]] = []
        for summary in sorted(bench_dir.glob("*/summary.json")):
            try:
                data = load_json(summary)
            except Exception:
                continue
            for row in data.get("results", []):
                item = dict(row)
                item["_summary"] = short_path(summary, root)
                rows.append(item)

        error_files: list[dict[str, Any]] = []
        error_types: Counter[str] = Counter()
        for error_path in sorted(bench_dir.glob("*/task_*_error.json")):
            try:
                payload = load_json(error_path)
            except Exception:
                payload = {}
            error_type = str(payload.get("error_type", "unknown"))
            error_types[error_type] += 1
            error_files.append(
                {
                    "error_file": short_path(error_path, root),
                    "task_id": payload.get("task_id"),
                    "error_type": error_type,
                    "returncode": payload.get("returncode"),
                }
            )

        bad_cases: list[dict[str, Any]] = []
        keyword_counter: Counter[str] = Counter()
        for row in rows:
            success = float(row.get("success_rate", 0) or 0)
            has_error = bool(row.get("error"))
            if has_error or success <= 0:
                task = str(row.get("task") or "")
                keyword_counter.update(instruction_keywords(task))
                bad_cases.append(
                    {
                        "task_id": row.get("task_id"),
                        "task": task,
                        "success_rate": row.get("success_rate"),
                        "average_steps": row.get("average_steps"),
                        "error": row.get("error"),
                        "prediction_file": row.get("prediction_file"),
                        "score_summary_file": row.get("score_summary_file"),
                        "summary_file": row.get("_summary"),
                    }
                )

        task_count = len(rows)
        success_sum = sum(float(row.get("success_rate", 0) or 0) for row in rows)
        result[bench_dir.name] = {
            "recorded_tasks": task_count,
            "success_tasks": sum(1 for row in rows if not row.get("error") and float(row.get("success_rate", 0) or 0) > 0),
            "failed_or_error_tasks": len(bad_cases),
            "success_rate_over_recorded_tasks": round(success_sum / task_count, 6) if task_count else None,
            "avg_steps_over_recorded_tasks": round(
                sum(float(row.get("average_steps", 0) or 0) for row in rows) / task_count,
                6,
            )
            if task_count
            else None,
            "task_error_files": len(error_files),
            "task_error_types": dict(error_types),
            "top_bad_case_keywords": keyword_counter.most_common(12),
            "sample_bad_cases": bad_cases[:max_cases],
            "sample_error_files": error_files[:max_cases],
        }
    return result


def eb_eval_set(path: Path) -> str:
    for part in path.parts:
        marker = "_full_alfred_"
        if marker in part:
            return part.split(marker, 1)[1].rsplit("_shard", 1)[0]
        marker = "_full_habitat_"
        if marker in part:
            return part.split(marker, 1)[1].rsplit("_shard", 1)[0]
    return "unknown"


def eb_case(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": short_path(path),
        "eval_set": eb_eval_set(path),
        "instruction": data.get("instruction", ""),
        "task_success": as_number(data.get("task_success", data.get("success"))),
        "task_progress": as_number(data.get("task_progress")),
        "planner_output_error": as_number(data.get("planner_output_error")),
        "num_invalid_actions": as_number(data.get("num_invalid_actions")),
        "num_invalid_action_ratio": as_number(data.get("num_invalid_action_ratio")),
        "num_steps": as_number(data.get("num_steps", data.get("path_length"))),
        "empty_plan": as_number(data.get("empty_plan")),
        "episode_elapsed_seconds": as_number(data.get("episode_elapsed_seconds")),
    }


def analyze_eb_dataset(paths: list[Path], max_cases: int) -> dict[str, Any]:
    by_eval_set: dict[str, dict[str, Any]] = defaultdict(lambda: {"paths": [], "bad_cases": []})
    keyword_counter: Counter[str] = Counter()
    all_bad_cases: list[dict[str, Any]] = []

    for path in sorted(paths):
        try:
            data = load_json(path)
        except Exception:
            continue
        eval_set = eb_eval_set(path)
        by_eval_set[eval_set]["paths"].append(path)
        case = eb_case(path, data)
        is_bad = (
            (case["task_success"] is not None and case["task_success"] <= 0)
            or (case["planner_output_error"] is not None and case["planner_output_error"] > 0)
            or (case["num_invalid_actions"] is not None and case["num_invalid_actions"] >= 10)
        )
        if is_bad:
            keyword_counter.update(instruction_keywords(str(case.get("instruction") or "")))
            by_eval_set[eval_set]["bad_cases"].append(case)
            all_bad_cases.append(case)

    eval_sets: dict[str, Any] = {}
    for eval_set, data in sorted(by_eval_set.items()):
        paths_for_set = data["paths"]
        bad_cases = data["bad_cases"]
        eval_sets[eval_set] = {
            "episode_count": len(paths_for_set),
            "metrics": numeric_average(paths_for_set),
            "bad_case_count": len(bad_cases),
            "bad_case_rate": round(len(bad_cases) / len(paths_for_set), 6) if paths_for_set else None,
            "sample_bad_cases": bad_cases[:max_cases],
        }

    return {
        "episode_count": len(paths),
        "metrics": numeric_average(paths),
        "eval_sets": eval_sets,
        "top_bad_case_keywords": keyword_counter.most_common(20),
        "sample_bad_cases": all_bad_cases[:max_cases],
    }


def mirrored_or_running_paths(root: Path, dataset: str, run_id: str, model: str) -> list[Path]:
    mirrored = sorted((root / "full" / dataset / "episode_results").glob("*/episode_*_final_res.json"))
    if mirrored:
        return mirrored
    model_name = model_basename(model)
    runtime_root = job_scratch_dir(run_id) / "embodiedbench" / "running"
    if dataset == "EB-ALFRED":
        base = runtime_root / "eb_alfred"
        pattern = f"{model_name}_{run_id}_full_alfred_*/*/results/episode_*_final_res.json"
        fallback = f"*_{run_id}_full_alfred_*/*/results/episode_*_final_res.json"
    else:
        base = runtime_root / "eb_habitat"
        pattern = f"{model_name}_{run_id}_full_habitat_*/*/results/episode_*_final_res.json"
        fallback = f"*_{run_id}_full_habitat_*/*/results/episode_*_final_res.json"
    paths = sorted(base.glob(pattern))
    return paths if paths else sorted(base.glob(fallback))


def analyze_eb(root: Path, run_id: str, model: str, max_cases: int) -> dict[str, Any]:
    alfred_paths = mirrored_or_running_paths(root, "EB-ALFRED", run_id, model)
    habitat_paths = mirrored_or_running_paths(root, "EB-Habitat", run_id, model)
    return {
        "EB-ALFRED": analyze_eb_dataset(alfred_paths, max_cases),
        "EB-Habitat": analyze_eb_dataset(habitat_paths, max_cases),
    }


def esc(value: Any) -> str:
    if isinstance(value, str):
        return html.escape(value)
    return html.escape(json.dumps(value, ensure_ascii=False))


def metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return "-"


def render_score_cards(report: dict[str, Any]) -> str:
    scores = report.get("final_summary", {}).get("scores", {})
    et = scores.get("ET-Plan_Bench", {})
    alfred = scores.get("EB-ALFRED", {})
    habitat = scores.get("EB-Habitat", {})
    cards = [
        (
            "ET-Plan_Bench",
            f"{et.get('task_count', 0)} tasks",
            f"task_success {metric(et, 'task_success_rate')}",
        ),
        ("EB-ALFRED", f"{alfred.get('episode_count', 0)} episodes", f"task_success {metric(alfred, 'task_success')}"),
        ("EB-Habitat", f"{habitat.get('episode_count', 0)} episodes", f"task_success {metric(habitat, 'task_success')}"),
    ]
    return "".join(
        f"<article class='metric-card'><b>{esc(name)}</b><span>{esc(count)}</span><strong>{esc(value)}</strong></article>"
        for name, count, value in cards
    )


def render_et_table(report: dict[str, Any]) -> str:
    rows = []
    for name, data in report["ET-Plan_Bench"].items():
        rows.append(
            "<tr>"
            f"<td><code>{esc(name)}</code></td>"
            f"<td>{data['recorded_tasks']}</td>"
            f"<td>{data['failed_or_error_tasks']}</td>"
            f"<td>{data['success_rate_over_recorded_tasks']}</td>"
            f"<td>{esc(data['task_error_types'])}</td>"
            f"<td>{esc(data['top_bad_case_keywords'][:5])}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_et_strict_block(report: dict[str, Any]) -> str:
    strict = report.get("ET-Plan_Bench_strict_rescore", {})
    if not strict:
        return (
            "<p class=\"warn\">尚未发现 <code>strict_rescore/et_strict_summary.json</code>。"
            "当前报告不展示 ET 分数；请先运行 "
            "<code>python scripts/experimental/rescore_et_predictions.py --source-run outputs/&lt;JOB_NAME&gt;</code> "
            "或等价命令得到严格重算结果。</p>"
        )
    return (
        "<table><thead><tr>"
        "<th>指标</th><th>值</th><th>说明</th>"
        "</tr></thead><tbody>"
        f"<tr><td><code>task_count</code></td><td>{esc(strict.get('task_count'))}</td><td>复用已有 prediction 重算的 ET 任务数。</td></tr>"
        f"<tr><td><code>task_success_rate</code></td><td>{metric(strict, 'task_success_rate')}</td><td>严格任务完成率，调用 ET-Plan-Bench 任务条件检查。</td></tr>"
        f"<tr><td><code>format_success_rate</code></td><td>{metric(strict, 'format_success_rate')}</td><td>预测动作是否为标准 VirtualHome script 格式。</td></tr>"
        f"<tr><td><code>action_execution_success_rate</code></td><td>{metric(strict, 'action_execution_success_rate')}</td><td>格式合法后，在 VirtualHome 中逐步执行是否成功。</td></tr>"
        "</tbody></table>"
    )


def render_eb_tables(report: dict[str, Any]) -> str:
    blocks = []
    for dataset, data in report["EmbodiedBench"].items():
        rows = []
        for eval_set, item in data["eval_sets"].items():
            metrics = item["metrics"]
            rows.append(
                "<tr>"
                f"<td><code>{esc(eval_set)}</code></td>"
                f"<td>{item['episode_count']}</td>"
                f"<td>{item['bad_case_count']}</td>"
                f"<td>{item['bad_case_rate']}</td>"
                f"<td>{metric(metrics, 'task_success')}</td>"
                f"<td>{metric(metrics, 'task_progress')}</td>"
                f"<td>{metric(metrics, 'planner_output_error')}</td>"
                f"<td>{metric(metrics, 'num_invalid_actions')}</td>"
                "</tr>"
            )
        blocks.append(
            f"<h3>{esc(dataset)}</h3>"
            "<table><thead><tr><th>eval_set</th><th>episodes</th><th>bad cases</th><th>bad rate</th>"
            "<th>task_success</th><th>task_progress</th><th>planner_output_error</th><th>invalid_actions</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
    return "".join(blocks)


def render_bad_cases(report: dict[str, Any]) -> str:
    items = []
    for bench, data in report["ET-Plan_Bench"].items():
        for case in data["sample_error_files"][:8]:
            items.append(
                f"<li><b>ET / {esc(bench)} / task {esc(case.get('task_id'))}</b>"
                f"<p><code>{esc(case.get('error_file'))}</code></p>"
                f"<p>error_type: <code>{esc(case.get('error_type'))}</code>, returncode: {esc(case.get('returncode'))}</p></li>"
            )
    for dataset, data in report["EmbodiedBench"].items():
        for case in data["sample_bad_cases"][:8]:
            items.append(
                f"<li><b>{esc(dataset)} / {esc(case.get('eval_set'))}</b>"
                f"<p>{esc(case.get('instruction') or '')}</p>"
                f"<p>success={esc(case.get('task_success'))}, progress={esc(case.get('task_progress'))}, "
                f"planner_error={esc(case.get('planner_output_error'))}, invalid_actions={esc(case.get('num_invalid_actions'))}</p>"
                f"<p><code>{esc(case.get('file'))}</code></p></li>"
            )
    return "".join(items[:30]) or "<li>没有收集到 bad case。</li>"


def render_html(report: dict[str, Any]) -> str:
    residual_errors = report.get("final_summary", {}).get("residual_errors", {}).get("et_task_errors", [])
    residual_text = f"ET 残余 task-level error 为 {len(residual_errors)} 个；这些样本经过并发重试和单 worker 重试后仍未恢复，优先按稳定失败分析。"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>当前 Checkpoint 错误分析报告</title>
  <style>
    :root {{
      --ink: #17202a; --muted: #5b677a; --line: #d9e2ef; --bg: #f6f8fb;
      --blue: #2563eb; --green: #047857; --orange: #c2410c; --purple: #7c3aed;
    }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); line-height: 1.65; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 34px 24px 72px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    h2 {{ margin-top: 34px; padding-top: 6px; }}
    h3 {{ margin-top: 22px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: #eef6ff; color: #1e3a8a; font-weight: 700; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    code {{ background: #eef2f7; border-radius: 5px; padding: 1px 5px; }}
    .lead {{ color: var(--muted); margin-top: 0; }}
    .note {{ border-left: 4px solid var(--blue); background: #eff6ff; padding: 12px 14px; border-radius: 6px; }}
    .warn {{ border-left: 4px solid var(--orange); background: #fff7ed; padding: 12px 14px; border-radius: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 22px 0; }}
    .metric-card {{ background: #fff; border: 1px solid var(--line); border-top: 4px solid var(--blue); border-radius: 8px; padding: 14px 16px; }}
    .metric-card:nth-child(2) {{ border-top-color: var(--orange); }}
    .metric-card:nth-child(3) {{ border-top-color: var(--green); }}
    .metric-card span {{ display: block; color: var(--muted); margin-top: 4px; }}
    .metric-card strong {{ display: block; font-size: 18px; margin-top: 8px; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .panel {{ background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    li {{ margin: 10px 0; }}
    @media (max-width: 860px) {{ .grid, .two-col {{ grid-template-columns: 1fr; }} table {{ font-size: 13px; }} }}
  </style>
</head>
<body>
<main>
  <h1>当前 Checkpoint 错误分析报告</h1>
  <p class="lead">run_id: <code>{esc(report['run_id'])}</code>；生成时间：<code>{esc(report['generated_at'])}</code>。</p>
  <p class="note">{esc(residual_text)}本报告用于区分模型能力短板、输出格式问题、模拟器执行问题和工程稳定性问题，后续 checkpoint 可以复用同一脚本生成。</p>

  <section class="grid">{render_score_cards(report)}</section>

  <section class="two-col">
    <div class="panel">
      <h2>能力结论</h2>
      <ul>
        <li>请依据上方本次评测的实际分数、错误类型统计和 bad case 判断模型能力；报告不预填特定 checkpoint 的结论。</li>
        <li>重点区分模型输出格式、无效动作、模拟器异常、API 异常和超时，避免把工程错误直接归因于模型能力。</li>
      </ul>
    </div>
    <div class="panel">
      <h2>优化方向</h2>
      <ul>
        <li>训练侧优化方向应由本次高频错误和失败样例决定，并在不同 checkpoint 间做同比验证。</li>
        <li>评测侧继续保留 task-level retry、单 worker 复核和原始模型输出留存。</li>
        <li>实验性 direct-plan 对照可使用 <code>scripts/experimental/et_plan_strict_score.py</code>，正式结果仍使用官方输出。</li>
      </ul>
    </div>
  </section>

  <h2>ET-Plan_Bench 严格重算</h2>
  <p class="note">本节使用 <code>outputs/{esc(report['run_id'])}/strict_rescore/et_strict_summary.json</code>，复用模型 prediction 文件重算，不重新调用模型 API。严格分数要求预测动作为标准 VirtualHome script，并在执行后满足 ET-Plan-Bench 的任务完成条件。</p>
  {render_et_strict_block(report)}

  <h2>ET-Plan_Bench 分项</h2>
  <table><thead><tr><th>benchmark</th><th>任务数</th><th>失败/错误</th><th>strict task_success_rate</th><th>错误类型</th><th>高频关键词</th></tr></thead><tbody>{render_et_table(report)}</tbody></table>

  <h2>EmbodiedBench 分项</h2>
  {render_eb_tables(report)}

  <h2>Bad Case 样例</h2>
  <ol>{render_bad_cases(report)}</ol>

  <h2>复用方法</h2>
  <p>后续 checkpoint 完成评测后运行：</p>
  <p><code>python scripts/finalize_full_eval.py --job-name &lt;JOB_NAME&gt; --model &lt;MODEL&gt; --skip-full-eval-status --skip-runtime-mirror</code></p>
  <p><code>python scripts/analyze_checkpoint_errors.py --job-name &lt;JOB_NAME&gt; --model &lt;MODEL&gt;</code></p>
</main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", dest="run_id", required=True)
    parser.add_argument("--model", default="qwen35-vl")
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--html", action="store_true", help="Also render an optional HTML report.")
    parser.add_argument("--docs-copy", action="store_true", help="Also write docs/当前checkpoint错误分析报告.html")
    args = parser.parse_args()

    run_id = args.run_id
    root = job_output_dir(run_id)
    et_strict = load_et_strict_summary(root)
    final_summary = summary_with_strict_et(load_final_summary(root), et_strict)
    report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "final_summary": final_summary,
        "ET-Plan_Bench_strict_rescore": et_strict,
        "ET-Plan_Bench": analyze_et(root, args.max_cases),
        "EmbodiedBench": analyze_eb(root, run_id, args.model, args.max_cases),
    }
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "checkpoint_error_analysis.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {"json": str(json_path)}
    if args.html:
        html_path = out_dir / "当前checkpoint错误分析报告.html"
        html_path.write_text(render_html(report), encoding="utf-8")
        payload["html"] = str(html_path)
    if args.docs_copy:
        docs_path = PROJECT_ROOT / "docs" / "当前checkpoint错误分析报告.html"
        docs_path.write_text(render_html(report), encoding="utf-8")
        payload["docs_html"] = str(docs_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
