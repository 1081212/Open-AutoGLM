from __future__ import annotations

import json
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


def test_case_artifact_upload_is_released_only_after_checkpoint(
    tmp_path, test_run_plan
):
    case = test_run_plan.test_run.cases[0]
    task_run_id = uuid4()
    attempt_id = uuid4()
    artifact_id = uuid4()
    outbox = DurableOutbox(tmp_path / "worker.db")
    coordinator = PlatformCaseCoordinator(
        api=FakeApi(attempt_id),
        outbox=outbox,
        sealer=LocalSealer(tmp_path / "seal"),
        project_id=test_run_plan.project_id,
        task_id=test_run_plan.task_id,
        task_run_id=task_run_id,
        worker_id=uuid4(),
        instance_id=uuid4(),
        device_uid=uuid4(),
        lease_token="lease",
        fencing_token=1,
        encrypted_root=tmp_path / "encrypted",
        runtime_environment="dev",
    )
    outbox.enqueue(
        idempotency_key=f"{artifact_id}:upload-complete",
        kind="ARTIFACT_UPLOAD",
        payload={"artifact_id": str(artifact_id)},
        task_run_id=str(task_run_id),
        defer_seconds=3600,
    )
    coordinator._attempt_ids[(case.execution_case_id, 1)] = attempt_id
    coordinator._artifacts[(case.execution_case_id, 1)] = [
        {
            "artifact_id": str(artifact_id),
            "upload_state": "PENDING",
            "encrypted_local_sha256": "sha256:" + "a" * 64,
            "type": "SCREENSHOT",
            "relative_name": "screen.png",
        }
    ]
    result = CaseExecutionResult(
        execution_case_id=case.execution_case_id,
        ordinal=case.ordinal,
        outcome=CaseOutcome.PASS,
        flaky=False,
        attempts=(AttemptResult(CaseOutcome.PASS, "passed"),),
    )

    assert outbox.due() == ()
    coordinator.checkpoint(case, result)

    released = outbox.due()
    assert len(released) == 1
    assert released[0].payload["artifact_id"] == str(artifact_id)


def test_screenshot_and_ui_xml_bind_to_frozen_plan_step_but_log_does_not(
    tmp_path, test_run_plan
):
    case = test_run_plan.test_run.cases[0]
    task_run_id = uuid4()
    attempt_id = uuid4()
    outbox = DurableOutbox(tmp_path / "worker.db")
    coordinator = PlatformCaseCoordinator(
        api=FakeApi(attempt_id),
        outbox=outbox,
        sealer=LocalSealer(tmp_path / "seal"),
        project_id=test_run_plan.project_id,
        task_id=test_run_plan.task_id,
        task_run_id=task_run_id,
        worker_id=uuid4(),
        instance_id=uuid4(),
        device_uid=uuid4(),
        lease_token="lease",
        fencing_token=1,
        encrypted_root=tmp_path / "encrypted",
        runtime_environment="dev",
    )
    case_dir = tmp_path / "case"
    attempt_dir = case_dir / "attempt_01"
    screenshot = attempt_dir / "screenshots" / "step_001.png"
    ui_xml = attempt_dir / "ui" / "step_001.xml"
    logcat = attempt_dir / "logcat.txt"
    screenshot.parent.mkdir(parents=True)
    ui_xml.parent.mkdir(parents=True)
    screenshot.write_bytes(b"png")
    ui_xml.write_text("<hierarchy/>", encoding="utf-8")
    logcat.write_text("log", encoding="utf-8")
    (case_dir / "report.json").write_text(
        json.dumps(
            {
                "execution_case_id": str(case.execution_case_id),
                "attempts": [
                    {
                        "attempt": 1,
                        "steps": [
                            {
                                "test_step_index": case.steps[0].index,
                                "screenshot": "attempt_01/screenshots/step_001.png",
                                "ui_xml": "attempt_01/ui/step_001.xml",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class RecordingPreparer:
        def __init__(self):
            self.sources = []

        def prepare(self, source, **_kwargs):
            self.sources.append(source)
            artifact_id = uuid4()
            encrypted = tmp_path / f"{artifact_id}.enc"
            encrypted.write_bytes(b"ciphertext")
            outbox.enqueue(
                idempotency_key=f"{artifact_id}:upload-complete",
                kind="ARTIFACT_UPLOAD",
                payload={
                    "artifact_id": str(artifact_id),
                    "ciphertext_sha256": "sha256:" + "b" * 64,
                },
                task_run_id=str(task_run_id),
                local_path=str(encrypted),
                defer_seconds=3600,
            )
            return artifact_id

    preparer = RecordingPreparer()
    coordinator.preparer = preparer

    coordinator._stage_artifacts(case, 1, attempt_id, attempt_dir)

    sources = {source.path.name: source for source in preparer.sources}
    expected_step_id = case.steps[0].step_id
    assert sources["step_001.png"].execution_case_id == case.execution_case_id
    assert sources["step_001.png"].case_attempt_id == attempt_id
    assert sources["step_001.png"].case_attempt_no == 1
    assert sources["step_001.png"].step_id == expected_step_id
    assert sources["step_001.xml"].step_id == expected_step_id
    assert sources["logcat.txt"].step_id is None
