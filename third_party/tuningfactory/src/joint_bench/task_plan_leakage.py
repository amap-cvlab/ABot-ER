"""Task Planning evaluation-anchor leakage detection.

Scene overlap is deliberately not treated as leakage. EB-ALFRED is evaluated
on ``valid_seen`` scenes, so a scene-only guard would discard legitimate
training data. The hard guard instead covers source lineage, official episode
keys, trial/task identifiers, exact normalized instructions, and trajectory
paths derived from official evaluation outputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator


ALFRED_EVAL_SETS = {
    "base",
    "common_sense",
    "complex_instruction",
    "long_horizon",
    "spatial",
    "visual_appearance",
}
TRIAL_RE = re.compile(r"trial_T\d+_\d+_\d+", re.IGNORECASE)
EPISODE_PATH_RE = re.compile(
    r"(?:^|[/\\])"
    r"(base|common_sense|complex_instruction|long_horizon|spatial|visual_appearance)"
    r"[/\\]episode_(\d+)(?:[/\\]|_)",
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    """Normalize text for exact benchmark-instruction matching."""

    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def normalized_hash(value: Any) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def dotted_get(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def record_source_id(record: dict[str, Any]) -> str:
    for field in ("source_id", "metadata.source", "source"):
        value = dotted_get(record, field)
        if value:
            return str(value)
    return ""


def conversation_texts(record: dict[str, Any]) -> Iterator[str]:
    for message in record.get("conversations") or []:
        if not isinstance(message, dict):
            continue
        yield str(message.get("value", message.get("content", "")))


def record_text(record: dict[str, Any]) -> str:
    fields = [
        record.get("id"),
        record.get("parent_id"),
        record.get("group_id"),
        record.get("source_id"),
        dotted_get(record, "metadata.source"),
        dotted_get(record, "metadata.episode_id"),
        dotted_get(record, "metadata.task_id"),
        dotted_get(record, "metadata.instruction"),
    ]
    fields.extend(conversation_texts(record))
    return "\n".join(str(value) for value in fields if value)


@dataclass(frozen=True)
class TaskPlanEvalAnchor:
    benchmark: str
    eval_set: str
    episode_id: int
    instruction: str
    trial_id: str | None = None
    task_key: str | None = None

    @property
    def eval_key(self) -> tuple[str, int]:
        return self.eval_set, self.episode_id

    @property
    def instruction_normalized(self) -> str:
        return normalize_text(self.instruction)


class TaskPlanLeakageGuard:
    """Fail-closed matcher for Task Planning formal evaluation contamination."""

    def __init__(
        self,
        anchors: Iterable[TaskPlanEvalAnchor],
        *,
        forbidden_source_ids: Iterable[str] = (),
    ) -> None:
        self.anchors = tuple(anchors)
        if not self.anchors:
            raise ValueError("at least one Task Planning evaluation anchor is required")
        self.forbidden_source_ids = frozenset(str(value) for value in forbidden_source_ids)
        self.instructions = {
            anchor.instruction_normalized: anchor.eval_key
            for anchor in self.anchors
            if anchor.instruction_normalized
        }
        self.trial_ids = {
            str(anchor.trial_id).casefold(): anchor.eval_key
            for anchor in self.anchors
            if anchor.trial_id
        }
        self.task_keys = {
            normalize_text(anchor.task_key): anchor.eval_key
            for anchor in self.anchors
            if anchor.task_key
        }
        self.eval_keys = frozenset(anchor.eval_key for anchor in self.anchors)
        alternatives = sorted(self.instructions, key=len, reverse=True)
        self._instruction_pattern = (
            re.compile(
                r"(?<![a-z0-9])(?:"
                + "|".join(re.escape(value) for value in alternatives)
                + r")(?![a-z0-9])"
            )
            if alternatives
            else None
        )
        task_alternatives = sorted(self.task_keys, key=len, reverse=True)
        self._task_key_pattern = (
            re.compile(
                r"(?<![a-z0-9])(?:"
                + "|".join(re.escape(value) for value in task_alternatives)
                + r")(?![a-z0-9])"
            )
            if task_alternatives
            else None
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        forbidden_source_ids: Iterable[str] = (),
    ) -> "TaskPlanLeakageGuard":
        anchors: list[TaskPlanEvalAnchor] = []
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    anchors.append(
                        TaskPlanEvalAnchor(
                            benchmark=str(row["benchmark"]),
                            eval_set=str(row["eval_set"]),
                            episode_id=int(row["episode_id"]),
                            instruction=str(row["instruction"]),
                            trial_id=str(row["trial_id"]) if row.get("trial_id") else None,
                            task_key=str(row["task_key"]) if row.get("task_key") else None,
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid anchor at {path}:{line_number}") from exc
        return cls(anchors, forbidden_source_ids=forbidden_source_ids)

    def match(self, record: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        source_id = record_source_id(record)
        if source_id in self.forbidden_source_ids:
            return [f"forbidden_source:{source_id}"]

        text = record_text(record)
        text_casefold = text.casefold()
        for trial_id, eval_key in self.trial_ids.items():
            if trial_id in text_casefold:
                reasons.append(f"eval_trial:{eval_key[0]}:{eval_key[1]}:{trial_id}")

        normalized = normalize_text(text)
        if self._instruction_pattern is not None:
            for match in self._instruction_pattern.finditer(normalized):
                eval_key = self.instructions[match.group(0)]
                reasons.append(f"eval_instruction:{eval_key[0]}:{eval_key[1]}")

        for match in EPISODE_PATH_RE.finditer(text):
            eval_key = (match.group(1).casefold(), int(match.group(2)))
            if eval_key in self.eval_keys:
                reasons.append(f"eval_episode_path:{eval_key[0]}:{eval_key[1]}")

        if self._task_key_pattern is not None:
            for match in self._task_key_pattern.finditer(normalized):
                eval_key = self.task_keys[match.group(0)]
                reasons.append(f"eval_task_key:{eval_key[0]}:{eval_key[1]}")

        return list(dict.fromkeys(reasons))


def audit_records(
    records: Iterable[dict[str, Any]],
    guard: TaskPlanLeakageGuard,
    *,
    sample_limit: int = 100,
) -> dict[str, Any]:
    rows = 0
    contaminated = 0
    reasons: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for record in records:
        rows += 1
        matched = guard.match(record)
        if not matched:
            continue
        contaminated += 1
        source_id = record_source_id(record)
        sources[source_id] += 1
        reasons.update(reason.split(":", 1)[0] for reason in matched)
        if len(samples) < sample_limit:
            samples.append(
                {
                    "id": record.get("id"),
                    "source_id": source_id,
                    "r3_component": record.get("r3_component"),
                    "reasons": matched,
                }
            )
    return {
        "status": "pass" if contaminated == 0 else "fail",
        "records": rows,
        "contaminated_records": contaminated,
        "reason_counts": dict(sorted(reasons.items())),
        "source_counts": dict(sorted(sources.items())),
        "samples": samples,
    }
