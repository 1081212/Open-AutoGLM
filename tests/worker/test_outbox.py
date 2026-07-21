from __future__ import annotations

from phone_agent.worker.outbox import DurableOutbox, LocalSealer


def test_enqueue_is_idempotent_and_acknowledge(tmp_path):
    outbox = DurableOutbox(tmp_path / "worker.db")
    first = outbox.enqueue(idempotency_key="run:producer:1", kind="EVENT", payload={"x": 1})
    second = outbox.enqueue(idempotency_key="run:producer:1", kind="EVENT", payload={"x": 2})
    assert first == second
    assert len(outbox.due()) == 1
    assert outbox.due()[0].payload == {"x": 1}
    outbox.acknowledge(first)
    assert outbox.pending_count() == 0


def test_credentials_are_sealed_and_recoverable(tmp_path):
    outbox = DurableOutbox(tmp_path / "worker.db")
    sealer = LocalSealer(tmp_path / "sealing-key")
    outbox.save_credential("lease:run-1", "super-secret-token", sealer)
    assert outbox.load_credential("lease:run-1", sealer) == "super-secret-token"
    assert b"super-secret-token" not in (tmp_path / "worker.db").read_bytes()


def test_fence_rejection_orphans_run_writes_but_not_artifact_upload(tmp_path):
    outbox = DurableOutbox(tmp_path / "worker.db")
    outbox.enqueue(idempotency_key="event", kind="EVENT", payload={}, task_run_id="run-1")
    outbox.enqueue(
        idempotency_key="artifact",
        kind="ARTIFACT_COMPLETE",
        payload={},
        task_run_id="run-1",
    )
    assert outbox.mark_orphaned_for_run("run-1", "fence rejected") == 1
    assert [item.kind for item in outbox.due()] == ["ARTIFACT_COMPLETE"]


def test_producer_positions_are_persistent(tmp_path):
    outbox = DurableOutbox(tmp_path / "worker.db")
    assert outbox.next_producer_sequence("producer-a") == 1
    assert outbox.next_producer_sequence("producer-a") == 2
    assert outbox.next_producer_sequence("producer-b") == 1

    assert outbox.producer_positions() == {"producer-a": 2, "producer-b": 1}
