"""Validate and aggregate ADHOC metrics from durable lifecycle events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable
from uuid import UUID

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode

_CONTEXT_FIELDS = (
    "execution_case_id",
    "case_attempt_id",
    "case_attempt_no",
    "step_id",
)


@dataclass(frozen=True, slots=True)
class AdhocEventAggregate:
    outcome: str
    action_count: int
    vision_tokens: int
    judge_tokens: int
    duration_ms: int
    started_at: str
    finished_at: str

    @property
    def summary(self) -> dict[str, object]:
        return {
            "execution_item_outcome": self.outcome,
            "action_count": self.action_count,
            "vision_tokens": self.vision_tokens,
            "judge_tokens": self.judge_tokens,
            "duration_ms": self.duration_ms,
        }


def aggregate_adhoc_events(
    events: Iterable[dict[str, Any]],
    *,
    task_run_id: str,
    execution_item_id: str,
) -> AdhocEventAggregate:
    """Aggregate one ADHOC item, treating exact composite-key replay as one event."""
    seen_composites: dict[tuple[str, str, int], str] = {}
    action_ids: set[str] = set()
    started: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    next_step_sequence = 1
    vision_tokens = 0
    judge_tokens = 0

    for event in events:
        _validate_envelope(event, task_run_id, execution_item_id)
        key = _composite_key(event)
        canonical = json.dumps(
            event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        previous = seen_composites.get(key)
        if previous is not None:
            if previous != canonical:
                _fail("same producer sequence was reused with different event content")
            continue
        seen_composites[key] = canonical

        event_type = event.get("type")
        if event_type == "RUN_STARTED":
            started.append(event)
        elif event_type == "RUN_FINISHED":
            finished.append(event)
        elif event_type == "ACTION_RECORDED":
            action_id = event["data"].get("action_id")
            try:
                normalized_action_id = str(UUID(str(action_id)))
            except (TypeError, ValueError, AttributeError):
                _fail("ACTION_RECORDED requires a valid action_id")
            if normalized_action_id in action_ids:
                _fail("ACTION_RECORDED action_id must be unique")
            action_ids.add(normalized_action_id)
        elif event_type == "STEP_FINISHED":
            data = event["data"]
            sequence = _non_negative_int(data.get("step_sequence"), "step_sequence")
            if sequence != next_step_sequence:
                _fail(
                    f"STEP_FINISHED step_sequence must be {next_step_sequence}, got {sequence}"
                )
            next_step_sequence += 1
            _non_negative_int(data.get("duration_ms"), "duration_ms")
            vision_tokens += _non_negative_int(
                data.get("vision_tokens"), "vision_tokens"
            )
            judge_tokens += _non_negative_int(data.get("judge_tokens"), "judge_tokens")

    if len(started) != 1:
        _fail("ADHOC requires exactly one RUN_STARTED event")
    if len(finished) != 1:
        _fail("ADHOC requires exactly one RUN_FINISHED event")

    started_at = started[0].get("occurred_at")
    finished_at = finished[0].get("occurred_at")
    start_time = _parse_timestamp(started_at, "RUN_STARTED.occurred_at")
    finish_time = _parse_timestamp(finished_at, "RUN_FINISHED.occurred_at")
    if finish_time < start_time:
        _fail("RUN_FINISHED.occurred_at precedes RUN_STARTED.occurred_at")
    outcome = finished[0]["data"].get("outcome")
    if not isinstance(outcome, str) or not outcome:
        _fail("RUN_FINISHED requires outcome")

    return AdhocEventAggregate(
        outcome=outcome,
        action_count=len(action_ids),
        vision_tokens=vision_tokens,
        judge_tokens=judge_tokens,
        duration_ms=int((finish_time - start_time).total_seconds() * 1000),
        started_at=str(started_at),
        finished_at=str(finished_at),
    )


def validate_adhoc_completion(
    aggregate: AdhocEventAggregate,
    *,
    outcome: str,
    started_at: str,
    finished_at: str,
    summary: dict[str, object],
) -> None:
    if outcome != aggregate.outcome:
        _fail("Task complete outcome does not match RUN_FINISHED")
    if started_at != aggregate.started_at or finished_at != aggregate.finished_at:
        _fail("Task complete timestamps do not match ADHOC boundary events")
    if summary != aggregate.summary:
        _fail("Task complete summary does not match ADHOC event aggregation")


def _validate_envelope(
    event: dict[str, Any], task_run_id: str, execution_item_id: str
) -> None:
    if event.get("task_run_id") != task_run_id:
        _fail("ADHOC event task_run_id does not match the completed Run")
    missing_context = [name for name in _CONTEXT_FIELDS if name not in event]
    if missing_context or any(event.get(name) is not None for name in _CONTEXT_FIELDS):
        _fail("ADHOC events must have null Case, Attempt, and step_id context")
    data = event.get("data")
    if not isinstance(data, dict) or data.get("execution_item_id") != execution_item_id:
        _fail("ADHOC event execution_item_id does not match the Plan")


def _composite_key(event: dict[str, Any]) -> tuple[str, str, int]:
    producer_id = event.get("producer_id")
    sequence = event.get("producer_seq")
    if not isinstance(producer_id, str) or not producer_id:
        _fail("ADHOC event requires producer_id")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        _fail("ADHOC event requires a positive producer_seq")
    return str(event.get("task_run_id")), producer_id, sequence


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"STEP_FINISHED {name} must be a non-negative integer")
    return value


def _parse_timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        _fail(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(f"{name} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        _fail(f"{name} must include timezone")
    return parsed


def _fail(message: str):
    raise ExecutionError(ExecutionErrorCode.EXECUTION_ERROR, message)
