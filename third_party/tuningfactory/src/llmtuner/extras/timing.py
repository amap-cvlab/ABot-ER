"""
Phase-level wall-clock timing for training/inference runs.

把训练/推理过程切分成命名 phase（load_template / get_dataset / load_model /
trainer_init / train / save_model / ...），每个 phase 记录起止时间、耗时与
（可选）出错状态。运行结束时把完整 timeline 落到 `{output_dir}/phase_timings.json`，
并在控制台打一份分阶段耗时表，方便后期分析 model load vs data load 等瓶颈。

约定：
- 仅 rank 0 记录（`RANK` 环境变量），避免多卡重复。
- 异常安全：phase 失败也记录 duration（带 `error=<class>`），run_exp 末尾用 try/finally 兜底 dump。
- 零第三方依赖：只用 stdlib（time / json / os / contextlib）。

典型用法：
    from llmtuner.extras.timing import phase, mark_run_start, dump, print_summary

    mark_run_start()
    with phase("load_template"):
        template = load_template(...)
    with phase("get_dataset"):
        dataset_module = get_dataset(...)
    ...
    dump(training_args.output_dir)
    print_summary()
"""

import json
import os
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

from .logging import get_logger


logger = get_logger(__name__)


_TIMELINE: List[Dict] = []
_RUN_START_TS: Optional[float] = None


def _is_rank_zero() -> bool:
    """RANK 未设置（本机 python 直跑）也视为 rank 0。"""
    return int(os.environ.get("RANK", "0")) == 0


def mark_run_start() -> None:
    """在 run_exp 最开头调用一次，作为后续 phase 的相对时间零点。可多次调用幂等。"""
    global _RUN_START_TS
    if _RUN_START_TS is None:
        _RUN_START_TS = time.time()


def reset() -> None:
    """清空全局 timeline（仅测试 / 多 run_exp 复用场景需要）。"""
    global _TIMELINE, _RUN_START_TS
    _TIMELINE = []
    _RUN_START_TS = None


def _record(name: str, start_ts: float, end_ts: float, error: Optional[str] = None) -> None:
    if not _is_rank_zero():
        return
    if _RUN_START_TS is None:
        mark_run_start()
    entry = {
        "phase": name,
        "rel_start_sec": round(start_ts - _RUN_START_TS, 3),
        "duration_sec": round(end_ts - start_ts, 3),
    }
    if error is not None:
        entry["error"] = error
    _TIMELINE.append(entry)


@contextmanager
def phase(name: str):
    """
    上下文管理器：包住一段代码，记录其耗时到全局 timeline。
    异常仍然向上抛出，但 phase 会以 `error=<class>` 标记记录。
    """
    start = time.time()
    if _is_rank_zero():
        logger.info(f"[phase:{name}] start")
    error_name: Optional[str] = None
    try:
        yield
    except BaseException as e:
        error_name = type(e).__name__
        raise
    finally:
        end = time.time()
        _record(name, start, end, error=error_name)
        if _is_rank_zero():
            tag = f" (error: {error_name})" if error_name else ""
            logger.info(f"[phase:{name}] done in {end - start:.2f}s{tag}")


def dump(output_dir: str, file_name: str = "phase_timings.json") -> None:
    """把累积的 timeline + run 元信息写到 {output_dir}/{file_name}。仅 rank 0 执行。"""
    if not _is_rank_zero():
        return
    if not _TIMELINE:
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, file_name)
        payload = {
            "run_start_ts": _RUN_START_TS,
            "run_start_iso": (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_RUN_START_TS))
                if _RUN_START_TS is not None
                else None
            ),
            "total_sec": round(time.time() - _RUN_START_TS, 3) if _RUN_START_TS is not None else None,
            "phases": _TIMELINE,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"[phase] timeline dumped to {path}")
    except Exception:
        logger.exception("[phase] failed to dump timeline; continuing")


def print_summary() -> None:
    """打印分阶段耗时表（rank 0）。"""
    if not _is_rank_zero() or not _TIMELINE:
        return
    total_recorded = sum(p["duration_sec"] for p in _TIMELINE)
    wall = time.time() - _RUN_START_TS if _RUN_START_TS is not None else total_recorded
    lines = ["", "=" * 64, "[phase] run timing summary"]
    lines.append(f"  {'phase':<24}{'duration':>14}{'% of recorded':>20}")
    lines.append("  " + "-" * 58)
    for p in _TIMELINE:
        pct = (p["duration_sec"] / total_recorded * 100) if total_recorded else 0
        marker = f"  ! {p['error']}" if "error" in p else ""
        lines.append(f"  {p['phase']:<24}{p['duration_sec']:>12.2f}s{pct:>18.1f}%{marker}")
    lines.append("  " + "-" * 58)
    lines.append(f"  {'recorded total':<24}{total_recorded:>12.2f}s")
    lines.append(f"  {'wall-clock total':<24}{wall:>12.2f}s")
    lines.append("=" * 64)
    logger.info("\n".join(lines))
