from __future__ import annotations

from uuid import uuid4

import pytest

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.worker.outbox import DurableOutbox, LocalSealer
from phone_agent.worker.platform_events import (
    PLATFORM_EVENT_TYPES,
    OutboxLifecycleSink,
    OutboxPump,
)


class FakeApi:
    def __init__(self):
        self.batches = []

    def events_batch(self, task_run_id, payload):
        self.batches.append((task_run_id, payload))
        return {"accepted": True}


def test_event_is_persisted_with_monotonic_sequence_before_send(tmp_path):
    outbox = DurableOutbox(tmp_path / "worker.db")
    sealer = LocalSealer(tmp_path / "seal-key")
    outbox.save_credential("lease", "secret-lease", sealer)
    task_run_id = uuid4()
    producer_id = uuid4()
    sink = OutboxLifecycleSink(
        outbox=outbox,
        task_run_id=task_run_id,
        producer_id=producer_id,
        lease_credential_ref="lease",
        fencing_token=4,
    )
    original = {"execution_case_id": str(uuid4()), "value": 1}
    sink.emit("STEP_STARTED", original)
    sink.emit("STEP_FINISHED", {"value": 2})

    pending = outbox.due()
    assert [item.producer_seq for item in pending] == [1, 2]
    assert original["execution_case_id"] is not None
    assert all("lease_token" not in item.payload for item in pending)

    api = FakeApi()
    assert OutboxPump(outbox, api, sealer).flush_once() == 2
    assert outbox.pending_count() == 0
    assert api.batches[0][1]["events"][0]["lease_token"] == "secret-lease"
    assert set(api.batches[0][1]) == {"events"}


def test_internal_telemetry_is_filtered_before_sequence_allocation(tmp_path):
    outbox = DurableOutbox(tmp_path / "worker.db")
    producer_id = uuid4()
    sink = OutboxLifecycleSink(
        outbox=outbox,
        task_run_id=uuid4(),
        producer_id=producer_id,
        lease_credential_ref="lease",
        fencing_token=4,
    )

    sink.emit("DEVICE_PROBED", {"adb_serial": "SERIAL"})
    sink.emit("MODEL_REQUEST_STARTED", {"agent_step": 1})
    sink.emit("ACTION_STARTED", {"action_name": "Tap"})
    sink.emit("CASE_STARTED", {"execution_case_id": str(uuid4())})

    pending = outbox.due()
    assert len(pending) == 1
    assert pending[0].payload["type"] == "CASE_STARTED"
    assert pending[0].producer_seq == 1
    assert outbox.producer_positions() == {str(producer_id): 1}


def test_test_run_events_are_contiguous_and_all_acked_with_local_telemetry(
    tmp_path,
):
    outbox = DurableOutbox(tmp_path / "worker.db")
    sealer = LocalSealer(tmp_path / "seal-key")
    outbox.save_credential("lease", "secret-lease", sealer)
    producer_id = uuid4()
    sink = OutboxLifecycleSink(
        outbox=outbox,
        task_run_id=uuid4(),
        producer_id=producer_id,
        lease_credential_ref="lease",
        fencing_token=4,
    )
    telemetry = (
        "DEVICE_PROBED",
        "AGENT_STEP_STARTED",
        "MODEL_REQUEST_STARTED",
        "MODEL_REQUEST_FINISHED",
        "ACTION_STARTED",
        "ACTION_FINISHED",
    )
    expected = list(PLATFORM_EVENT_TYPES)
    for index, event_type in enumerate(expected):
        sink.emit(telemetry[index % len(telemetry)], {"local": True})
        sink.emit(event_type, {"index": index})

    pending = outbox.due(limit=100)
    assert [item.producer_seq for item in pending] == list(range(1, len(expected) + 1))
    assert [item.payload["type"] for item in pending] == expected

    class StrictSequenceApi:
        def __init__(self):
            self.next_sequence = 1
            self.acked = []

        def events_batch(self, _task_run_id, payload):
            event = payload["events"][0]
            assert event["type"] in PLATFORM_EVENT_TYPES
            assert event["producer_seq"] == self.next_sequence
            self.next_sequence += 1
            self.acked.append(event)
            return {"accepted": True}

    api = StrictSequenceApi()
    assert OutboxPump(outbox, api, sealer).flush_once(limit=100) == len(expected)
    assert [event["producer_seq"] for event in api.acked] == list(
        range(1, len(expected) + 1)
    )
    assert outbox.unacknowledged_event_count_for_run(str(pending[0].task_run_id)) == 0


def test_event_payload_redacts_reasoning_credentials_and_typed_text(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AUTOGLM_GITLAB_TOKEN", "known-gitlab-secret")
    outbox = DurableOutbox(tmp_path / "worker.db")
    sink = OutboxLifecycleSink(
        outbox=outbox,
        task_run_id=uuid4(),
        producer_id=uuid4(),
        lease_credential_ref="lease",
        fencing_token=4,
    )

    sink.emit(
        "ACTION_RECORDED",
        {
            "thinking": "private model chain",
            "action": {
                "action": "Type",
                "text": "user-password",
                "cookie": "session-cookie",
            },
            "message": "GitLab Token=known-gitlab-secret",
        },
    )

    data = outbox.due()[0].payload["data"]
    assert data["thinking"] == "[REDACTED]"
    assert data["action"]["text"] == "[REDACTED]"
    assert data["action"]["cookie"] == "[REDACTED]"
    assert "known-gitlab-secret" not in str(data)
    assert "private model chain" not in str(data)


def test_protocol_409_failure_does_not_orphan_the_run_outbox(tmp_path):
    outbox = DurableOutbox(tmp_path / "worker.db")
    sealer = LocalSealer(tmp_path / "seal-key")
    outbox.save_credential("lease", "secret-lease", sealer)
    run_id = uuid4()
    sink = OutboxLifecycleSink(
        outbox=outbox,
        task_run_id=run_id,
        producer_id=uuid4(),
        lease_credential_ref="lease",
        fencing_token=4,
    )
    sink.emit("RUN_STARTED", {"value": 1})
    sink.emit("RUN_FINISHED", {"value": 2})

    class ProtocolErrorApi:
        def events_batch(self, _task_run_id, _payload):
            raise ExecutionError(
                ExecutionErrorCode.EXECUTION_ERROR,
                "HTTP 409 INVALID_EVENT: invalid event",
                retryable=False,
            )

    assert OutboxPump(outbox, ProtocolErrorApi(), sealer).flush_once() == 0
    states = [
        row["state"]
        for row in outbox._connection.execute(
            "SELECT state FROM outbox ORDER BY id"
        ).fetchall()
    ]
    assert states == ["FAILED", "FAILED"]


def test_adhoc_sink_injects_item_and_forbids_case_context(tmp_path):
    outbox = DurableOutbox(tmp_path / "worker.db")
    item_id = uuid4()
    sink = OutboxLifecycleSink(
        outbox=outbox,
        task_run_id=uuid4(),
        producer_id=uuid4(),
        lease_credential_ref="lease",
        fencing_token=4,
        adhoc_execution_item_id=item_id,
    )

    sink.emit("RUN_STARTED", {})
    event = outbox.due()[0].payload

    assert event["data"]["execution_item_id"] == str(item_id)
    assert event["execution_case_id"] is None
    assert event["case_attempt_id"] is None
    assert event["case_attempt_no"] is None
    assert event["step_id"] is None
    with pytest.raises(ValueError, match="must not reference"):
        sink.emit("STEP_FINISHED", {"execution_case_id": str(uuid4())})
