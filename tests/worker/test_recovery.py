from __future__ import annotations

from uuid import uuid4

import pytest

from phone_agent.execution.errors import ExecutionError
from phone_agent.worker.outbox import DurableOutbox, LocalSealer
from phone_agent.worker.recovery import StartupRecovery


class FakeApi:
    def __init__(self, responses):
        self.responses = list(responses)
        self.heartbeat_payloads = []
        self.completed = []
        self.calls = []

    def heartbeat(self, payload):
        self.calls.append(
            ("heartbeat", payload["startup_context"]["previous_task_run_id"])
        )
        self.heartbeat_payloads.append(payload)
        return self.responses.pop(0)

    def complete_run(self, task_run_id, payload):
        self.calls.append(("complete", task_run_id))
        self.completed.append((task_run_id, payload))
        return {}


def setup_active_run(tmp_path, *, outbox=None, sealer=None):
    outbox = outbox or DurableOutbox(tmp_path / "worker.db")
    sealer = sealer or LocalSealer(tmp_path / "seal")
    task_run_id = str(uuid4())
    lease_ref = f"lease:{task_run_id}"
    outbox.save_credential(lease_ref, f"lease-secret:{task_run_id}", sealer)
    claim = {
        "task_id": str(uuid4()),
        "task_run_id": task_run_id,
        "instance_id": str(uuid4()),
        "device_uid": str(uuid4()),
        "fencing_token": 8,
        "lease_expires_at": "2026-07-14T10:00:45Z",
        "lease_credential_ref": lease_ref,
    }
    outbox.save_task_run_state(task_run_id, state="ACTIVE", claim=claim)
    return outbox, sealer, task_run_id, claim


def decision(task_run_id, **overrides):
    value = {
        "task_run_id": task_run_id,
        "phase": "RUNNING",
        "outcome": None,
        "lease_state": "ACTIVE",
        "fence_state": "OWNED",
        "startup_termination_allowed": False,
    }
    value.update(overrides)
    return {"blocking_task_run": value}


def make_recovery(tmp_path, api, outbox, sealer):
    return StartupRecovery(
        api=api,
        outbox=outbox,
        sealer=sealer,
        spool_root=tmp_path,
        worker_id=uuid4(),
        instance_id=uuid4(),
        runtime_environment="prod",
    )


def test_startup_blocks_without_platform_decision(tmp_path):
    outbox, sealer, task_run_id, _ = setup_active_run(tmp_path)
    api = FakeApi([{}])

    with pytest.raises(
        ExecutionError, match=f"omitted startup decision for {task_run_id}"
    ):
        make_recovery(tmp_path, api, outbox, sealer).recover()

    assert outbox.active_task_runs()


def test_startup_blocks_before_heartbeat_when_local_summary_is_incomplete(tmp_path):
    outbox, sealer, task_run_id, claim = setup_active_run(tmp_path)
    claim.pop("lease_expires_at")
    outbox.save_task_run_state(task_run_id, state="ACTIVE", claim=claim)
    api = FakeApi([])

    with pytest.raises(ExecutionError, match="previous_lease_expires_at"):
        make_recovery(tmp_path, api, outbox, sealer).recover()

    assert api.heartbeat_payloads == []
    assert outbox.active_task_runs()


def test_startup_heartbeat_reports_one_run_summary_without_bearer_secret(tmp_path):
    outbox, sealer, task_run_id, claim = setup_active_run(tmp_path)
    api = FakeApi([decision(task_run_id, phase="COMPLETED", outcome="WORKER_LOST")])

    make_recovery(tmp_path, api, outbox, sealer).recover()

    payload = api.heartbeat_payloads[0]
    assert payload["startup_context"] == {
        "previous_task_run_id": task_run_id,
        "previous_instance_id": claim["instance_id"],
        "previous_fencing_token": 8,
        "previous_lease_expires_at": "2026-07-14T10:00:45Z",
    }
    assert "lease-secret" not in str(payload)
    assert payload["runtime_environment"] == "prod"
    assert payload["worker_version"] == "0.3.0"
    assert payload["heartbeat_at"]
    assert payload["availability"] == "ONLINE"
    assert payload["activity"] == "STARTING"
    assert payload["max_concurrency"] == 1
    assert payload["active_task_count"] == 0
    assert payload["current_task_run_id"] is None
    assert payload["supported_execution_versions"] == ["autoglm.execution.v1"]
    assert payload["devices"] == []
    assert payload["outbox_pending"] == 0
    assert payload["spool_free_bytes"] > 0
    assert payload["last_error_code"] is None


def test_startup_termination_never_resumes_and_completes_worker_lost(tmp_path):
    outbox, sealer, task_run_id, claim = setup_active_run(tmp_path)
    api = FakeApi([decision(task_run_id, startup_termination_allowed=True)])
    recovery = make_recovery(tmp_path, api, outbox, sealer)

    recovery.recover()

    assert outbox.active_task_runs() == ()
    complete = api.completed[0][1]
    assert complete["outcome"] == "WORKER_LOST"
    assert complete["completion_mode"] == "STARTUP_TERMINATION"
    assert complete["previous_instance_id"] == claim["instance_id"]
    assert complete["current_instance_id"] == str(recovery.instance_id)
    assert complete["fencing_token"] == 8
    assert complete["lease_token"] == f"lease-secret:{task_run_id}"
    assert complete["error"]["code"] == "WORKER_RESTARTED"
    assert "lease-secret" not in str(api.heartbeat_payloads)


@pytest.mark.parametrize(
    ("lease_state", "fence_state"),
    [("EXPIRED", "OWNED"), ("ACTIVE", "LOST")],
)
def test_invalid_lease_or_fence_only_marks_run_orphaned(
    tmp_path, lease_state, fence_state
):
    outbox, sealer, task_run_id, _ = setup_active_run(tmp_path)
    api = FakeApi(
        [decision(task_run_id, lease_state=lease_state, fence_state=fence_state)]
    )

    make_recovery(tmp_path, api, outbox, sealer).recover()

    assert outbox.active_task_runs() == ()
    assert api.completed == []


def test_multiple_interrupted_runs_are_decided_serially(tmp_path):
    outbox, sealer, first_id, _ = setup_active_run(tmp_path)
    outbox, sealer, second_id, _ = setup_active_run(
        tmp_path, outbox=outbox, sealer=sealer
    )
    api = FakeApi(
        [
            decision(first_id, startup_termination_allowed=True),
            decision(second_id, phase="COMPLETED", outcome="WORKER_LOST"),
        ]
    )

    make_recovery(tmp_path, api, outbox, sealer).recover()

    assert [
        payload["startup_context"]["previous_task_run_id"]
        for payload in api.heartbeat_payloads
    ] == [
        first_id,
        second_id,
    ]
    assert api.calls == [
        ("heartbeat", first_id),
        ("complete", first_id),
        ("heartbeat", second_id),
    ]


def test_blocking_first_run_prevents_deciding_next_run(tmp_path):
    outbox, sealer, first_id, _ = setup_active_run(tmp_path)
    outbox, sealer, _second_id, _ = setup_active_run(
        tmp_path, outbox=outbox, sealer=sealer
    )
    api = FakeApi([decision(first_id)])

    with pytest.raises(ExecutionError, match="is still blocking startup"):
        make_recovery(tmp_path, api, outbox, sealer).recover()

    assert len(api.heartbeat_payloads) == 1
    assert api.completed == []


def test_missing_lease_and_fence_decision_remains_blocked(tmp_path):
    outbox, sealer, task_run_id, _ = setup_active_run(tmp_path)
    response = decision(task_run_id)
    del response["blocking_task_run"]["lease_state"]
    del response["blocking_task_run"]["fence_state"]
    api = FakeApi([response])

    with pytest.raises(ExecutionError, match="is still blocking startup"):
        make_recovery(tmp_path, api, outbox, sealer).recover()

    assert outbox.active_task_runs()
    assert api.completed == []


def test_mismatched_blocking_run_is_rejected(tmp_path):
    outbox, sealer, _task_run_id, _ = setup_active_run(tmp_path)
    api = FakeApi([decision(str(uuid4()), phase="COMPLETED")])

    with pytest.raises(ExecutionError, match="does not match interrupted TaskRun"):
        make_recovery(tmp_path, api, outbox, sealer).recover()

    assert outbox.active_task_runs()
    assert api.completed == []
