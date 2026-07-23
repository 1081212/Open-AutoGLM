from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from phone_agent.execution.cancellation import CancellationToken
from phone_agent.worker.heartbeat import (
    ControlHeartbeatLoop,
    RunLeaseLoop,
    WorkerRuntimeState,
    _parse_time,
)
from phone_agent.worker.models import RunHeartbeatResponse, WorkerActivity


class FakeApi:
    def __init__(self):
        self.control_payload = None
        self.run_payload = None

    def heartbeat(self, payload):
        self.control_payload = payload
        return {"ok": True}

    def run_heartbeat(self, task_run_id, payload):
        self.run_payload = (task_run_id, payload)
        return RunHeartbeatResponse(
            lease_expires_at=(
                datetime.now(timezone.utc) + timedelta(seconds=45)
            ).isoformat(),
            cancel_requested=False,
            fence_owned=True,
        )


def test_control_heartbeat_uses_snapshot_callback_only(tmp_path):
    api = FakeApi()
    calls = []

    def snapshot(busy):
        calls.append(busy)
        return ({"device_uid": "cached"},)

    state = WorkerRuntimeState(activity=WorkerActivity.IDLE)
    loop = ControlHeartbeatLoop(
        api=api,
        worker_id=uuid4(),
        instance_id=uuid4(),
        runtime_environment="dev",
        state=state,
        device_snapshot=snapshot,
        outbox_pending=lambda: 3,
        spool_root=tmp_path,
    )
    loop.send_once()
    assert calls == [False]
    assert api.control_payload["active_task_count"] == 0
    assert api.control_payload["outbox_pending"] == 3
    assert api.control_payload["runtime_environment"] == "dev"
    assert api.control_payload["supported_execution_versions"] == [
        "autoglm.execution.v1",
        "autoglm.execution.v2",
    ]


def test_run_heartbeat_carries_binding_and_sequences():
    api = FakeApi()
    token = CancellationToken()
    task_run_id = str(uuid4())
    loop = RunLeaseLoop(
        api=api,
        task_run_id=task_run_id,
        worker_id=uuid4(),
        instance_id=uuid4(),
        device_uid=uuid4(),
        lease_token="lease",
        fencing_token=9,
        lease_expires_at=(
            datetime.now(timezone.utc) + timedelta(seconds=45)
        ).isoformat(),
        cancellation_token=token,
        position_provider=lambda: {
            "current_execution_case_id": "case",
            "adhoc_item_state": None,
        },
        producer_sequences=lambda: {"producer": 7},
    )
    loop.send_once()
    _, payload = api.run_payload
    assert payload["fencing_token"] == 9
    assert payload["last_sent_producer_seq"] == {"producer": 7}
    assert payload["current_execution_case_id"] == "case"
    assert payload["adhoc_item_state"] is None


def test_run_heartbeat_accepts_platform_five_digit_fractional_seconds():
    parsed = _parse_time("2026-07-23T03:07:50.17854+00:00")

    assert parsed == datetime(2026, 7, 23, 3, 7, 50, 178540, tzinfo=timezone.utc)
