from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from phone_agent.execution.models import CaseOutcome
from phone_agent.execution.result import (
    AttemptResult,
    CaseExecutionResult,
    TaskExecutionResult,
    TaskOutcome,
)
from phone_agent.worker.child_process import (
    ChildProcessPlanExecutor,
    _ChildLifecycleSink,
    _decode_result,
    _encode_result,
)


def test_child_result_protocol_round_trip_preserves_duplicate_display_cases():
    task_id = uuid4()
    now = datetime.now(timezone.utc)
    cases = tuple(
        CaseExecutionResult(
            execution_case_id=uuid4(),
            ordinal=index,
            outcome=CaseOutcome.PASS,
            flaky=False,
            attempts=(AttemptResult(CaseOutcome.PASS, f"case {index}"),),
        )
        for index in range(1, 201)
    )
    original = TaskExecutionResult(
        task_id=task_id,
        outcome=TaskOutcome.PASS,
        started_at=now,
        finished_at=now,
        cases=cases,
    )

    decoded = _decode_result(_encode_result(original))

    assert decoded.task_id == task_id
    assert len(decoded.cases) == 200
    assert decoded.cases[-1].ordinal == 200
    assert len({case.execution_case_id for case in decoded.cases}) == 200


class FakeDurableSink:
    def __init__(self):
        self.events = []

    def emit(self, event_type, data):
        self.events.append((event_type, data))


def test_adhoc_position_uses_platform_names_and_allowed_states(monkeypatch):
    protocol = []
    monkeypatch.setattr(
        "phone_agent.worker.child_process._write_protocol",
        lambda _fd, message: protocol.append(message),
    )
    sink = _ChildLifecycleSink(FakeDurableSink(), 1, str(uuid4()), is_adhoc=True)

    sink.emit("RUN_STARTED", {})
    sink.emit("RUN_FINISHED", {"outcome": "PASS"})

    assert [message["payload"]["adhoc_item_state"] for message in protocol] == [
        "RUNNING",
        "FINALIZING",
    ]
    assert all(
        message["payload"]["current_execution_case_id"] is None for message in protocol
    )
    assert all(
        set(message["payload"])
        == {"current_execution_case_id", "adhoc_item_state"}
        for message in protocol
    )
    assert "FINISHED" not in str(protocol)
    assert "adhoc_state" not in str(protocol)


def test_test_run_position_contains_only_platform_heartbeat_fields(monkeypatch):
    protocol = []
    monkeypatch.setattr(
        "phone_agent.worker.child_process._write_protocol",
        lambda _fd, message: protocol.append(message),
    )
    sink = _ChildLifecycleSink(FakeDurableSink(), 1, str(uuid4()), is_adhoc=False)
    execution_case_id = str(uuid4())

    sink.emit(
        "STEP_STARTED",
        {"execution_case_id": execution_case_id, "step_id": str(uuid4())},
    )

    assert protocol == [
        {
            "type": "POSITION",
            "payload": {
                "current_execution_case_id": execution_case_id,
                "adhoc_item_state": None,
            },
        }
    ]


def test_parent_position_initializes_with_platform_field_names(tmp_path):
    executor = ChildProcessPlanExecutor(
        python_executable="python3.10",
        model_profiles_path=tmp_path / "models.yaml",
        report_root=tmp_path / "reports",
        outbox_db_path=tmp_path / "worker.db",
        sealing_key_path=tmp_path / "seal",
        platform_base_url="http://platform.internal",
        runtime_environment="dev",
    )

    assert executor.position() == {
        "current_execution_case_id": None,
        "adhoc_item_state": None,
    }
