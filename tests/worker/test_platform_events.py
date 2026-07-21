from __future__ import annotations

from uuid import uuid4

import pytest

from phone_agent.worker.outbox import DurableOutbox, LocalSealer
from phone_agent.worker.platform_events import OutboxLifecycleSink, OutboxPump


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
