from __future__ import annotations

from uuid import uuid4

from phone_agent.execution.models import CaseOutcome
from phone_agent.execution.result import AttemptResult, CaseExecutionResult
from phone_agent.worker.case_coordinator import PlatformCaseCoordinator
from phone_agent.worker.outbox import DurableOutbox, LocalSealer
from phone_agent.worker.platform_events import OutboxLifecycleSink


class FakeApi:
    def __init__(self, attempt_id):
        self.attempt_id = attempt_id
        self.begin_payload = None
        self.complete_payload = None
        self.checkpoint_payload = None
        self.events = []

    def begin_attempt(self, _task_run_id, _execution_case_id, payload):
        self.begin_payload = payload
        return {"case_attempt_id": str(self.attempt_id)}

    def complete_attempt(self, _attempt_id, payload):
        self.complete_payload = payload
        return {}

    def events_batch(self, _task_run_id, payload):
        self.events.extend(payload["events"])
        return {"accepted": True}

    def checkpoint(self, _task_run_id, _execution_case_id, payload):
        self.checkpoint_payload = payload
        return {"accepted": True}


def test_attempt_and_checkpoint_match_platform_contract(tmp_path, test_run_plan):
    case = test_run_plan.test_run.cases[0]
    worker_id = uuid4()
    instance_id = uuid4()
    device_uid = uuid4()
    task_run_id = uuid4()
    attempt_id = uuid4()
    producer_id = uuid4()
    outbox = DurableOutbox(tmp_path / "worker.db")
    sealer = LocalSealer(tmp_path / "seal")
    outbox.save_credential("lease-ref", "lease-secret", sealer)
    api = FakeApi(attempt_id)
    coordinator = PlatformCaseCoordinator(
        api=api,
        outbox=outbox,
        sealer=sealer,
        project_id=test_run_plan.project_id,
        task_id=test_run_plan.task_id,
        task_run_id=task_run_id,
        worker_id=worker_id,
        instance_id=instance_id,
        device_uid=device_uid,
        lease_token="lease-secret",
        fencing_token=9,
        encrypted_root=tmp_path / "encrypted",
        runtime_environment="dev",
    )

    assert coordinator.begin_attempt(case, 1) == attempt_id
    sink = OutboxLifecycleSink(
        outbox=outbox,
        task_run_id=task_run_id,
        producer_id=producer_id,
        lease_credential_ref="lease-ref",
        fencing_token=9,
    )
    sink.emit(
        "STEP_FINISHED",
        {
            "execution_case_id": str(case.execution_case_id),
            "case_attempt_id": str(attempt_id),
            "case_attempt_no": 1,
            "step_id": str(case.steps[0].step_id),
            "duration_ms": 120,
            "vision_tokens": 13,
            "judge_tokens": 2,
        },
    )
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    attempt = AttemptResult(CaseOutcome.PASS, "passed")
    coordinator.finish_attempt(case, 1, attempt, attempt_dir)
    case_result = CaseExecutionResult(
        execution_case_id=case.execution_case_id,
        ordinal=case.ordinal,
        outcome=CaseOutcome.PASS,
        flaky=False,
        attempts=(attempt,),
    )
    coordinator.checkpoint(case, case_result)

    assert set(api.begin_payload) == {
        "worker_id",
        "instance_id",
        "device_uid",
        "lease_token",
        "fencing_token",
        "idempotency_key",
        "attempt_no",
        "attempt_kind",
        "started_at",
    }
    assert api.begin_payload["attempt_kind"] == "INITIAL"
    assert set(api.complete_payload) == {
        "worker_id",
        "instance_id",
        "device_uid",
        "lease_token",
        "fencing_token",
        "idempotency_key",
        "outcome",
        "result_message",
        "metrics",
        "finished_at",
    }
    assert api.complete_payload["metrics"] == {
        "duration_ms": 120,
        "vision_tokens": 13,
        "judge_tokens": 2,
    }
    assert api.checkpoint_payload["producer_positions"] == {str(producer_id): 1}
    assert api.checkpoint_payload["artifacts"] == []
    assert len(api.events) == 1


def test_second_attempt_is_case_retry(tmp_path, test_run_plan):
    case = test_run_plan.test_run.cases[0]
    api = FakeApi(uuid4())
    coordinator = PlatformCaseCoordinator(
        api=api,
        outbox=DurableOutbox(tmp_path / "worker.db"),
        sealer=LocalSealer(tmp_path / "seal"),
        project_id=test_run_plan.project_id,
        task_id=test_run_plan.task_id,
        task_run_id=uuid4(),
        worker_id=uuid4(),
        instance_id=uuid4(),
        device_uid=uuid4(),
        lease_token="lease",
        fencing_token=1,
        encrypted_root=tmp_path / "encrypted",
        runtime_environment="dev",
    )

    coordinator.begin_attempt(case, 2)

    assert api.begin_payload["attempt_kind"] == "CASE_RETRY"
