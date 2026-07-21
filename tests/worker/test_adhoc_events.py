from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest

from phone_agent.execution.errors import ExecutionError
from phone_agent.worker.adhoc_events import (
    aggregate_adhoc_events,
    validate_adhoc_completion,
)


def _event(
    run_id, producer_id, item_id, sequence, event_type, data=None, occurred_at=None
):
    return {
        "schema_version": "autoglm.event.v1",
        "event_id": str(uuid4()),
        "idempotency_key": f"{run_id}:{producer_id}:{sequence}",
        "task_run_id": run_id,
        "producer_id": producer_id,
        "producer_seq": sequence,
        "fencing_token": 7,
        "type": event_type,
        "occurred_at": occurred_at or f"2026-07-14T10:00:0{sequence}Z",
        "execution_case_id": None,
        "case_attempt_id": None,
        "case_attempt_no": None,
        "step_id": None,
        "data": {"execution_item_id": item_id, **(data or {})},
    }


@pytest.fixture
def adhoc_events():
    run_id = str(uuid4())
    producer_id = str(uuid4())
    item_id = str(uuid4())
    action_id = str(uuid4())
    events = [
        _event(
            run_id,
            producer_id,
            item_id,
            1,
            "RUN_STARTED",
            occurred_at="2026-07-14T10:00:00Z",
        ),
        _event(
            run_id, producer_id, item_id, 2, "ACTION_RECORDED", {"action_id": action_id}
        ),
        _event(
            run_id,
            producer_id,
            item_id,
            3,
            "STEP_FINISHED",
            {
                "step_sequence": 1,
                "duration_ms": 120,
                "vision_tokens": 11,
                "judge_tokens": 2,
            },
        ),
        _event(
            run_id,
            producer_id,
            item_id,
            4,
            "RUN_FINISHED",
            {"outcome": "PASS"},
            occurred_at="2026-07-14T10:00:01.250Z",
        ),
    ]
    return run_id, producer_id, item_id, events


def _aggregate(values):
    run_id, _producer_id, item_id, events = values
    return aggregate_adhoc_events(events, task_run_id=run_id, execution_item_id=item_id)


def test_adhoc_events_aggregate_structured_metrics(adhoc_events):
    aggregate = _aggregate(adhoc_events)

    assert aggregate.started_at == "2026-07-14T10:00:00Z"
    assert aggregate.finished_at == "2026-07-14T10:00:01.250Z"
    assert aggregate.summary == {
        "execution_item_outcome": "PASS",
        "action_count": 1,
        "vision_tokens": 11,
        "judge_tokens": 2,
        "duration_ms": 1250,
    }


def test_missing_boundary_event_fails(adhoc_events):
    adhoc_events[3].pop()

    with pytest.raises(ExecutionError, match="exactly one RUN_FINISHED"):
        _aggregate(adhoc_events)


def test_duplicate_action_id_in_distinct_events_fails(adhoc_events):
    run_id, producer_id, item_id, events = adhoc_events
    duplicate = _event(
        run_id,
        producer_id,
        item_id,
        5,
        "ACTION_RECORDED",
        {"action_id": events[1]["data"]["action_id"]},
    )
    events.insert(-1, duplicate)

    with pytest.raises(ExecutionError, match="action_id must be unique"):
        _aggregate(adhoc_events)


def test_step_sequence_gap_fails(adhoc_events):
    adhoc_events[3][2]["data"]["step_sequence"] = 2

    with pytest.raises(ExecutionError, match="step_sequence must be 1"):
        _aggregate(adhoc_events)


@pytest.mark.parametrize("field", ["duration_ms", "vision_tokens", "judge_tokens"])
def test_negative_step_metric_fails(adhoc_events, field):
    adhoc_events[3][2]["data"][field] = -1

    with pytest.raises(ExecutionError, match="non-negative integer"):
        _aggregate(adhoc_events)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_case_id", str(uuid4())),
        ("case_attempt_id", str(uuid4())),
        ("case_attempt_no", 1),
        ("step_id", str(uuid4())),
    ],
)
def test_adhoc_event_cannot_forge_case_attempt_or_step(adhoc_events, field, value):
    adhoc_events[3][1][field] = value

    with pytest.raises(ExecutionError, match="must have null Case"):
        _aggregate(adhoc_events)


def test_exact_replay_does_not_increment_metrics(adhoc_events):
    replay = deepcopy(adhoc_events[3][1])
    adhoc_events[3].insert(2, replay)

    aggregate = _aggregate(adhoc_events)

    assert aggregate.action_count == 1


def test_same_producer_sequence_with_different_content_fails(adhoc_events):
    replay = deepcopy(adhoc_events[3][1])
    replay["data"]["action_id"] = str(uuid4())
    adhoc_events[3].insert(2, replay)

    with pytest.raises(ExecutionError, match="different event content"):
        _aggregate(adhoc_events)


def test_inconsistent_complete_summary_and_boundaries_fail(adhoc_events):
    aggregate = _aggregate(adhoc_events)

    with pytest.raises(ExecutionError, match="summary does not match"):
        validate_adhoc_completion(
            aggregate,
            outcome=aggregate.outcome,
            started_at=aggregate.started_at,
            finished_at=aggregate.finished_at,
            summary={**aggregate.summary, "action_count": 99},
        )
    with pytest.raises(ExecutionError, match="outcome does not match"):
        validate_adhoc_completion(
            aggregate,
            outcome="FAIL",
            started_at=aggregate.started_at,
            finished_at=aggregate.finished_at,
            summary=aggregate.summary,
        )
    with pytest.raises(ExecutionError, match="timestamps do not match"):
        validate_adhoc_completion(
            aggregate,
            outcome=aggregate.outcome,
            started_at="2026-07-14T09:59:59Z",
            finished_at=aggregate.finished_at,
            summary=aggregate.summary,
        )
