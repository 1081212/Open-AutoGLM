from __future__ import annotations

import gzip
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from phone_agent.execution.models import CaseOutcome, ExecutionPlan
from phone_agent.execution.result import (
    AttemptResult,
    CaseExecutionResult,
    TaskExecutionResult,
    TaskOutcome,
)
from phone_agent.worker.heartbeat import WorkerRuntimeState
from phone_agent.worker.models import (
    ClaimResponse,
    DispatchNotification,
    RedisMessage,
    RunHeartbeatResponse,
    WorkerActivity,
)
from phone_agent.worker.outbox import DurableOutbox, LocalSealer
from phone_agent.worker.platform_events import OutboxLifecycleSink
from phone_agent.worker.spool import PlanSpool
from phone_agent.worker.supervisor import ClaimedRun, WorkerSupervisor


class FakeNotifier:
    def __init__(self, message):
        self.message = message
        self.acked = []

    def read_one(self, **kwargs):
        del kwargs
        message, self.message = self.message, None
        return message

    def acknowledge(self, message):
        self.acked.append(message.redis_id)


class FakeDiscovery:
    def __init__(self, device_uid):
        self.device_uid = str(device_uid)

    def resolve_serial(self, device_uid):
        return "SERIAL" if str(device_uid) == self.device_uid else None


class FakeApi:
    def __init__(self, claim, compressed):
        self._claim = claim
        self.compressed = compressed
        self.plan_accepted_payload = None
        self.completed_payload = None
        self.heartbeats = 0
        self.event_batches = []

    def claim(self, dispatch_id, payload):
        del dispatch_id, payload
        return self._claim

    @contextmanager
    def plan_chunks(self, download_url):
        del download_url
        yield iter((self.compressed,))

    def plan_accepted(self, task_run_id, payload):
        del task_run_id
        self.plan_accepted_payload = payload
        return {"accepted": True}

    def run_heartbeat(self, task_run_id, payload):
        del task_run_id, payload
        self.heartbeats += 1
        return RunHeartbeatResponse(
            lease_expires_at=(
                datetime.now(timezone.utc) + timedelta(seconds=45)
            ).isoformat(),
            cancel_requested=False,
            fence_owned=True,
        )

    def complete_run(self, task_run_id, payload):
        del task_run_id
        self.completed_payload = payload
        return {"completed": True}

    def events_batch(self, task_run_id, payload):
        self.event_batches.append((task_run_id, payload))
        return {"accepted": True}


def test_claim_download_accept_ack_execute_complete(tmp_path, test_run_plan):
    canonical = json.dumps(
        test_run_plan.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    compressed = gzip.compress(canonical, mtime=0)
    worker_id = uuid4()
    instance_id = uuid4()
    device_uid = uuid4()
    task_run_id = uuid4()
    notification = DispatchNotification(
        schema_version="autoglm.dispatch.v1",
        dispatch_id=uuid4(),
        task_id=test_run_plan.task_id,
        task_type="TEST_RUN",
        plan_id=test_run_plan.plan_id,
        plan_canonical_sha256="sha256:" + hashlib.sha256(canonical).hexdigest(),
        worker_id=worker_id,
        device_uid=device_uid,
    )
    message = RedisMessage(redis_id="1-0", notification=notification, raw_fields={})
    claim = ClaimResponse.model_validate(
        {
            "claimed": True,
            "task_id": str(test_run_plan.task_id),
            "worker_id": str(worker_id),
            "instance_id": str(instance_id),
            "device_uid": str(device_uid),
            "task_run_id": str(task_run_id),
            "lease_token": "lease-secret",
            "fencing_token": 3,
            "lease_expires_at": (
                datetime.now(timezone.utc) + timedelta(seconds=45)
            ).isoformat(),
            "renew_after_seconds": 10,
            "plan": {
                "schema_version": "autoglm.execution.v1",
                "plan_id": str(test_run_plan.plan_id),
                "download_url": f"/worker/v1/task-runs/{task_run_id}/plan",
                "wire_media_type": "application/vnd.autoglm.execution-plan+gzip",
                "inner_media_type": "application/vnd.autoglm.execution-plan+json",
                "wire_format": "gzip",
                "compressed_sha256": "sha256:" + hashlib.sha256(compressed).hexdigest(),
                "compressed_size": len(compressed),
                "canonical_sha256": "sha256:" + hashlib.sha256(canonical).hexdigest(),
                "canonical_size": len(canonical),
                "item_count": len(test_run_plan.test_run.cases),
                "case_count": len(test_run_plan.test_run.cases),
            },
        }
    )
    api = FakeApi(claim, compressed)
    notifier = FakeNotifier(message)
    state = WorkerRuntimeState(activity=WorkerActivity.IDLE)

    def execute(stored, claimed, cancellation):
        assert stored.plan.plan_id == test_run_plan.plan_id
        assert claimed.adb_serial == "SERIAL"
        assert cancellation.is_cancelled is False
        now = datetime.now(timezone.utc)
        return TaskExecutionResult(
            task_id=test_run_plan.task_id,
            outcome=TaskOutcome.PASS,
            started_at=now,
            finished_at=now,
        )

    supervisor = WorkerSupervisor(
        worker_id=worker_id,
        instance_id=instance_id,
        api=api,
        notifier=notifier,
        discovery=FakeDiscovery(device_uid),
        spool=PlanSpool(tmp_path / "spool"),
        outbox=DurableOutbox(tmp_path / "spool" / "worker.db"),
        sealer=LocalSealer(tmp_path / "spool" / "seal-key"),
        state=state,
        device_lock_dir=tmp_path / "spool" / "devices",
        execute_plan=execute,
    )

    assert supervisor.process_one() is True
    assert notifier.acked == ["1-0"]
    assert api.plan_accepted_payload is not None
    assert api.completed_payload["outcome"] == "PASS"
    persisted_claim = json.loads(
        (tmp_path / "spool" / "task-runs" / str(task_run_id) / "claim.json").read_text()
    )
    assert persisted_claim["lease_expires_at"] == claim.lease_expires_at
    assert state.snapshot()[0] is WorkerActivity.IDLE


def test_adhoc_complete_uses_durable_event_aggregation(tmp_path, test_run_plan):
    plan_data = test_run_plan.model_dump(mode="json")
    item_id = uuid4()
    plan_data.update(
        {
            "task_type": "ADHOC",
            "normalizer": None,
            "test_run": None,
            "adhoc": {"execution_item_id": str(item_id), "prompt": "open settings"},
        }
    )
    plan = ExecutionPlan.model_validate(plan_data)
    task_run_id = uuid4()
    worker_id = uuid4()
    device_uid = uuid4()
    outbox = DurableOutbox(tmp_path / "worker.db")
    sealer = LocalSealer(tmp_path / "seal")
    outbox.save_credential("lease", "lease-secret", sealer)
    sink = OutboxLifecycleSink(
        outbox=outbox,
        task_run_id=task_run_id,
        producer_id=uuid4(),
        lease_credential_ref="lease",
        fencing_token=3,
        adhoc_execution_item_id=item_id,
    )
    sink.emit("RUN_STARTED", {})
    sink.emit("ACTION_RECORDED", {"action_id": str(uuid4())})
    sink.emit(
        "STEP_FINISHED",
        {
            "step_sequence": 1,
            "duration_ms": 50,
            "vision_tokens": 13,
            "judge_tokens": 0,
        },
    )
    sink.emit("RUN_FINISHED", {"outcome": "PASS"})
    events = outbox.events_for_run(str(task_run_id))
    api = FakeApi(None, b"")
    supervisor = WorkerSupervisor(
        worker_id=worker_id,
        instance_id=uuid4(),
        api=api,
        notifier=None,
        discovery=None,
        spool=None,
        outbox=outbox,
        sealer=sealer,
        state=WorkerRuntimeState(),
        device_lock_dir=tmp_path,
        execute_plan=lambda *_args: None,
    )
    claimed = ClaimedRun(
        task_id=plan.task_id,
        task_run_id=task_run_id,
        plan_id=plan.plan_id,
        worker_id=worker_id,
        instance_id=supervisor.instance_id,
        device_uid=device_uid,
        adb_serial="SERIAL",
        lease_token="lease-secret",
        fencing_token=3,
        lease_expires_at="2026-07-14T10:00:45Z",
        renew_after_seconds=10,
    )
    unrelated_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    result = TaskExecutionResult(
        task_id=plan.task_id,
        outcome=TaskOutcome.PASS,
        started_at=unrelated_time,
        finished_at=unrelated_time,
        adhoc_result=AttemptResult(CaseOutcome.PASS, "done"),
    )

    assert supervisor._complete_run(claimed, result, "lease", plan) is True

    assert api.completed_payload["started_at"] == events[0]["occurred_at"]
    assert api.completed_payload["finished_at"] == events[-1]["occurred_at"]
    assert api.completed_payload["summary"] == {
        "execution_item_outcome": "PASS",
        "action_count": 1,
        "vision_tokens": 13,
        "judge_tokens": 0,
        "duration_ms": int(
            (
                datetime.fromisoformat(events[-1]["occurred_at"])
                - datetime.fromisoformat(events[0]["occurred_at"])
            ).total_seconds()
            * 1000
        ),
    }
    assert "case_total" not in api.completed_payload["summary"]
    assert len(api.event_batches) == 4
    assert outbox.unacknowledged_event_count_for_run(str(task_run_id)) == 0


def test_test_run_complete_summary_and_failed_artifact_are_contract_aligned(
    tmp_path, test_run_plan
):
    case = test_run_plan.test_run.cases[0]
    task_run_id = uuid4()
    worker_id = uuid4()
    outbox = DurableOutbox(tmp_path / "worker.db")
    sink = OutboxLifecycleSink(
        outbox=outbox,
        task_run_id=task_run_id,
        producer_id=uuid4(),
        lease_credential_ref="lease",
        fencing_token=3,
    )
    sink.emit(
        "STEP_FINISHED",
        {
            "execution_case_id": str(case.execution_case_id),
            "case_attempt_id": str(uuid4()),
            "case_attempt_no": 1,
            "step_id": str(case.steps[0].step_id),
            "duration_ms": 250,
            "vision_tokens": 21,
            "judge_tokens": 4,
        },
    )
    failed_artifact = outbox.enqueue(
        idempotency_key="artifact:upload",
        kind="ARTIFACT_UPLOAD",
        payload={},
        task_run_id=str(task_run_id),
    )
    outbox.mark_failed(failed_artifact, "COS unavailable")
    api = FakeApi(None, b"")
    supervisor = WorkerSupervisor(
        worker_id=worker_id,
        instance_id=uuid4(),
        api=api,
        notifier=None,
        discovery=None,
        spool=None,
        outbox=outbox,
        sealer=LocalSealer(tmp_path / "seal"),
        state=WorkerRuntimeState(),
        device_lock_dir=tmp_path,
        execute_plan=lambda *_args: None,
    )
    claimed = ClaimedRun(
        task_id=test_run_plan.task_id,
        task_run_id=task_run_id,
        plan_id=test_run_plan.plan_id,
        worker_id=worker_id,
        instance_id=supervisor.instance_id,
        device_uid=uuid4(),
        adb_serial="SERIAL",
        lease_token="lease-secret",
        fencing_token=3,
        lease_expires_at="2026-07-14T10:00:45Z",
        renew_after_seconds=10,
    )
    started_at = datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc)
    attempt = AttemptResult(CaseOutcome.PASS, "passed")
    case_result = CaseExecutionResult(
        execution_case_id=case.execution_case_id,
        ordinal=case.ordinal,
        outcome=CaseOutcome.PASS,
        flaky=False,
        attempts=(attempt,),
    )
    result = TaskExecutionResult(
        task_id=test_run_plan.task_id,
        outcome=TaskOutcome.PASS,
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=1),
        cases=(case_result,),
    )

    assert supervisor._complete_run(claimed, result, "lease", test_run_plan) is True

    assert api.completed_payload["summary"] == {
        "case_total": 1,
        "passed": 1,
        "failed": 0,
        "blocked": 0,
        "review": 0,
        "skipped": 0,
        "infra_error": 0,
        "flaky": 0,
        "not_run": 0,
        "vision_tokens": 21,
        "judge_tokens": 4,
        "duration_ms": 1000,
    }
    assert api.completed_payload["outcome"] == "PASS"
    assert api.completed_payload["result_completeness"] == "PARTIAL"
    assert "pass" not in api.completed_payload["summary"]
    assert "fail" not in api.completed_payload["summary"]
