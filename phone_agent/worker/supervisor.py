"""Single-capacity Worker claim/download/execute/finalize orchestration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable
from uuid import UUID

from phone_agent.execution.cancellation import CancellationToken
from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.execution.models import ExecutionPlan, TaskType
from phone_agent.execution.result import TaskExecutionResult
from phone_agent.worker.adhoc_events import (
    aggregate_adhoc_events,
    validate_adhoc_completion,
)
from phone_agent.worker.api_client import WorkerApiClient
from phone_agent.worker.device_discovery import DeviceDiscoveryCache
from phone_agent.worker.device_lock import ActiveDeviceLock
from phone_agent.worker.heartbeat import RunLeaseLoop, WorkerRuntimeState
from phone_agent.worker.models import ClaimResponse, RedisMessage, WorkerActivity
from phone_agent.worker.outbox import DurableOutbox, LocalSealer
from phone_agent.worker.platform_events import OutboxPump
from phone_agent.worker.redis_notifier import RedisDispatchNotifier
from phone_agent.worker.spool import PlanDescriptor, PlanSpool, SpoolPlan


@dataclass(frozen=True, slots=True)
class ClaimedRun:
    task_id: UUID
    task_run_id: UUID
    plan_id: UUID
    worker_id: UUID
    instance_id: UUID
    device_uid: UUID
    adb_serial: str
    lease_token: str
    fencing_token: int
    lease_expires_at: str
    renew_after_seconds: int


class WorkerSupervisor:
    def __init__(
        self,
        *,
        worker_id: UUID,
        instance_id: UUID,
        api: WorkerApiClient,
        notifier: RedisDispatchNotifier,
        discovery: DeviceDiscoveryCache,
        spool: PlanSpool,
        outbox: DurableOutbox,
        sealer: LocalSealer,
        state: WorkerRuntimeState,
        device_lock_dir,
        execute_plan: Callable[
            [SpoolPlan, ClaimedRun, CancellationToken], TaskExecutionResult
        ],
    ) -> None:
        self.worker_id = worker_id
        self.instance_id = instance_id
        self.api = api
        self.notifier = notifier
        self.discovery = discovery
        self.spool = spool
        self.outbox = outbox
        self.sealer = sealer
        self.state = state
        self.device_lock_dir = device_lock_dir
        self.execute_plan = execute_plan

    def process_one(self, *, block_ms: int = 1000) -> bool:
        activity, current_run, _ = self.state.snapshot()
        if activity is not WorkerActivity.IDLE or current_run is not None:
            return False
        message = self.notifier.read_one(worker_busy=False, block_ms=block_ms)
        if message is None:
            return False
        self._process_message(message)
        return True

    def _process_message(self, message: RedisMessage) -> None:
        notification = message.notification
        if notification.worker_id != self.worker_id:
            raise ExecutionError(
                ExecutionErrorCode.CLAIM_REJECTED, "dispatch worker binding mismatch"
            )
        adb_serial = self.discovery.resolve_serial(notification.device_uid)
        if adb_serial is None:
            raise ExecutionError(
                ExecutionErrorCode.DEVICE_NOT_FOUND, "dispatch device is not online"
            )
        device_lock = ActiveDeviceLock(self.device_lock_dir, notification.device_uid)
        if not device_lock.acquire(
            worker_id=self.worker_id,
            instance_id=self.instance_id,
            adb_serial=adb_serial,
        ):
            raise ExecutionError(
                ExecutionErrorCode.CLAIM_REJECTED, "active device lock is busy"
            )

        lease_loop: RunLeaseLoop | None = None
        claimed: ClaimedRun | None = None
        cancellation = CancellationToken()
        try:
            self.state.set_activity(WorkerActivity.CLAIMING)
            claim = self.api.claim(
                str(notification.dispatch_id),
                {
                    "worker_id": str(self.worker_id),
                    "instance_id": str(self.instance_id),
                    "device_uid": str(notification.device_uid),
                    "supported_execution_versions": ["autoglm.execution.v1"],
                    "supported_event_versions": ["autoglm.event.v1"],
                },
            )
            if not claim.claimed:
                if claim.ack_disposition == "ACK":
                    self.notifier.acknowledge(message)
                return
            claimed = self._validate_claim(notification, claim, adb_serial)
            self.state.set_activity(
                WorkerActivity.DOWNLOADING_PLAN, task_run_id=str(claimed.task_run_id)
            )
            lease_ref = f"lease:{claimed.task_run_id}"
            self.outbox.save_credential(lease_ref, claimed.lease_token, self.sealer)
            claim_metadata = {
                "task_id": str(claimed.task_id),
                "task_run_id": str(claimed.task_run_id),
                "plan_id": str(claimed.plan_id),
                "worker_id": str(claimed.worker_id),
                "instance_id": str(claimed.instance_id),
                "device_uid": str(claimed.device_uid),
                "adb_serial": claimed.adb_serial,
                "fencing_token": claimed.fencing_token,
                "lease_expires_at": claimed.lease_expires_at,
                "lease_credential_ref": lease_ref,
            }
            self.outbox.save_task_run_state(
                str(claimed.task_run_id), state="ACTIVE", claim=claim_metadata
            )
            self.spool.write_claim_metadata(str(claimed.task_run_id), claim_metadata)
            lease_loop = RunLeaseLoop(
                api=self.api,
                task_run_id=str(claimed.task_run_id),
                worker_id=self.worker_id,
                instance_id=self.instance_id,
                device_uid=claimed.device_uid,
                lease_token=claimed.lease_token,
                fencing_token=claimed.fencing_token,
                lease_expires_at=claimed.lease_expires_at,
                cancellation_token=cancellation,
                position_provider=getattr(
                    self.execute_plan,
                    "position",
                    lambda: {
                        "current_execution_case_id": None,
                        "adhoc_item_state": None,
                    },
                ),
                producer_sequences=lambda: self.outbox.producer_positions_for_run(
                    str(claimed.task_run_id)
                ),
                renew_after_seconds=claimed.renew_after_seconds,
            )
            lease_loop.start()
            descriptor = _descriptor(claim)
            with self.api.plan_chunks(claim.plan.download_url) as chunks:
                stored = self.spool.store(str(claimed.task_run_id), chunks, descriptor)
            self._validate_plan_binding(stored, notification, claimed)
            self.outbox.save_task_run_state(
                str(claimed.task_run_id),
                state="ACTIVE",
                claim=claim_metadata,
                plan_ready=True,
            )
            self.api.plan_accepted(
                str(claimed.task_run_id),
                {
                    "worker_id": str(self.worker_id),
                    "instance_id": str(self.instance_id),
                    "lease_token": claimed.lease_token,
                    "fencing_token": claimed.fencing_token,
                    "plan_id": str(claimed.plan_id),
                    "compressed_sha256": descriptor.compressed_sha256,
                    "compressed_size": descriptor.compressed_size,
                    "canonical_sha256": descriptor.canonical_sha256,
                    "canonical_size": descriptor.canonical_size,
                },
            )
            self.outbox.save_task_run_state(
                str(claimed.task_run_id),
                state="ACTIVE",
                claim=claim_metadata,
                plan_ready=True,
                plan_accepted=True,
            )
            self.notifier.acknowledge(message)
            self.state.set_activity(
                WorkerActivity.BUSY, task_run_id=str(claimed.task_run_id)
            )
            result = self.execute_plan(stored, claimed, cancellation)
            self.state.set_activity(
                WorkerActivity.FINALIZING, task_run_id=str(claimed.task_run_id)
            )
            self.outbox.save_task_run_state(
                str(claimed.task_run_id),
                state="FINALIZING",
                claim=claim_metadata,
                plan_ready=True,
                plan_accepted=True,
            )
            if not self._complete_run(claimed, result, lease_ref, stored.plan):
                raise ExecutionError(
                    ExecutionErrorCode.RETRYABLE_ERROR,
                    "Run completion is durably queued; Worker remains blocked until recovery",
                    retryable=True,
                )
            self.outbox.save_task_run_state(
                str(claimed.task_run_id),
                state="COMPLETED",
                claim=claim_metadata,
                plan_ready=True,
                plan_accepted=True,
            )
        except ExecutionError as error:
            self.state.set_activity(
                WorkerActivity.DEGRADED,
                task_run_id=str(claimed.task_run_id) if claimed else None,
                last_error_code=error.code.value,
            )
            raise
        finally:
            if lease_loop:
                lease_loop.stop()
            device_lock.release()
            activity, _, error_code = self.state.snapshot()
            if activity is not WorkerActivity.DEGRADED:
                self.state.set_activity(WorkerActivity.IDLE)
            elif claimed is None:
                self.state.set_activity(WorkerActivity.IDLE, last_error_code=error_code)

    def _validate_claim(
        self, notification, claim: ClaimResponse, adb_serial: str
    ) -> ClaimedRun:
        assert claim.plan is not None
        expected = {
            "task_id": (notification.task_id, claim.task_id),
            "worker_id": (self.worker_id, claim.worker_id),
            "instance_id": (self.instance_id, claim.instance_id),
            "device_uid": (notification.device_uid, claim.device_uid),
            "plan_id": (notification.plan_id, claim.plan.plan_id),
            "canonical_sha256": (
                notification.plan_canonical_sha256.lower(),
                claim.plan.canonical_sha256.lower(),
            ),
        }
        mismatched = [name for name, (left, right) in expected.items() if left != right]
        if mismatched:
            raise ExecutionError(
                ExecutionErrorCode.PLAN_INVALID,
                "dispatch/claim binding mismatch: " + ", ".join(mismatched),
            )
        assert (
            claim.task_id
            and claim.task_run_id
            and claim.worker_id
            and claim.instance_id
        )
        assert (
            claim.device_uid and claim.lease_token and claim.fencing_token is not None
        )
        assert claim.lease_expires_at and claim.renew_after_seconds
        return ClaimedRun(
            task_id=claim.task_id,
            task_run_id=claim.task_run_id,
            plan_id=claim.plan.plan_id,
            worker_id=claim.worker_id,
            instance_id=claim.instance_id,
            device_uid=claim.device_uid,
            adb_serial=adb_serial,
            lease_token=claim.lease_token,
            fencing_token=claim.fencing_token,
            lease_expires_at=claim.lease_expires_at,
            renew_after_seconds=claim.renew_after_seconds,
        )

    @staticmethod
    def _validate_plan_binding(
        stored: SpoolPlan, notification, claimed: ClaimedRun
    ) -> None:
        if (
            stored.plan.task_id != claimed.task_id
            or stored.plan.plan_id != claimed.plan_id
        ):
            raise ExecutionError(
                ExecutionErrorCode.PLAN_INVALID, "downloaded Plan binding mismatch"
            )
        digest = (
            "sha256:" + hashlib.sha256(stored.canonical_path.read_bytes()).hexdigest()
        )
        if digest.lower() != notification.plan_canonical_sha256.lower():
            raise ExecutionError(
                ExecutionErrorCode.PLAN_HASH_MISMATCH,
                "dispatch canonical hash mismatch",
            )

    def _complete_run(
        self,
        claimed: ClaimedRun,
        result: TaskExecutionResult,
        lease_ref: str,
        plan: ExecutionPlan,
    ) -> bool:
        outcome = result.outcome.value
        started_at = result.started_at.isoformat()
        finished_at = result.finished_at.isoformat()
        summary: dict[str, object] = self._test_run_summary(
            result, str(claimed.task_run_id)
        )
        if plan.task_type is TaskType.ADHOC:
            assert plan.adhoc is not None
            aggregate = aggregate_adhoc_events(
                self.outbox.events_for_run(str(claimed.task_run_id)),
                task_run_id=str(claimed.task_run_id),
                execution_item_id=str(plan.adhoc.execution_item_id),
            )
            if result.outcome.value != aggregate.outcome:
                raise ExecutionError(
                    ExecutionErrorCode.EXECUTION_ERROR,
                    "TaskExecutionResult outcome does not match ADHOC RUN_FINISHED",
                )
            outcome = aggregate.outcome
            started_at = aggregate.started_at
            finished_at = aggregate.finished_at
            summary = aggregate.summary
            validate_adhoc_completion(
                aggregate,
                outcome=outcome,
                started_at=started_at,
                finished_at=finished_at,
                summary=summary,
            )
            # The platform aggregates ADHOC completion from events. Flush the
            # durable stream synchronously so RUN_FINISHED cannot race complete.
            OutboxPump(self.outbox, self.api, self.sealer).flush_once(limit=10_000)
            if self.outbox.unacknowledged_event_count_for_run(str(claimed.task_run_id)):
                raise ExecutionError(
                    ExecutionErrorCode.RETRYABLE_ERROR,
                    "ADHOC events are not acknowledged; Task complete is blocked",
                    retryable=True,
                )
        payload = {
            "schema_version": "autoglm.result.v1",
            "idempotency_key": f"{claimed.task_run_id}:final",
            "task_id": str(claimed.task_id),
            "task_run_id": str(claimed.task_run_id),
            "lease_token": claimed.lease_token,
            "fencing_token": claimed.fencing_token,
            "worker_id": str(self.worker_id),
            "device_uid": str(claimed.device_uid),
            "phase": "COMPLETED",
            "outcome": outcome,
            "result_completeness": (
                "PARTIAL"
                if self.outbox.unacknowledged_count_for_run(
                    str(claimed.task_run_id), kind="ARTIFACT_UPLOAD"
                )
                else "COMPLETE"
            ),
            "started_at": started_at,
            "finished_at": finished_at,
            "summary": summary,
            "error": result.error.as_dict() if result.error else None,
        }
        try:
            self.api.complete_run(str(claimed.task_run_id), payload)
            return True
        except ExecutionError as error:
            if not error.retryable:
                raise
            payload_without_secret = {
                key: value for key, value in payload.items() if key != "lease_token"
            }
            self.outbox.enqueue(
                idempotency_key=payload["idempotency_key"],
                kind="RUN_COMPLETE",
                payload=payload_without_secret,
                task_run_id=str(claimed.task_run_id),
                lease_credential_ref=lease_ref,
                fencing_token=claimed.fencing_token,
            )
            return False

    def _test_run_summary(
        self, result: TaskExecutionResult, task_run_id: str
    ) -> dict[str, int]:
        counts = {
            "passed": 0,
            "failed": 0,
            "blocked": 0,
            "review": 0,
            "skipped": 0,
            "infra_error": 0,
        }
        mapping = {
            "PASS": "passed",
            "FAIL": "failed",
            "BLOCKED": "blocked",
            "REVIEW": "review",
            "SKIPPED": "skipped",
        }
        for case in result.cases:
            key = mapping.get(case.outcome.value, "infra_error")
            counts[key] += 1
        vision_tokens = 0
        judge_tokens = 0
        for event in self.outbox.events_for_run(task_run_id):
            if event.get("type") != "STEP_FINISHED":
                continue
            data = event.get("data") or {}
            vision_tokens += _event_metric(data, "vision_tokens")
            judge_tokens += _event_metric(data, "judge_tokens")
        return {
            "case_total": len(result.cases) + len(result.not_run_execution_case_ids),
            **counts,
            "flaky": sum(1 for case in result.cases if case.flaky),
            "not_run": len(result.not_run_execution_case_ids),
            "vision_tokens": vision_tokens,
            "judge_tokens": judge_tokens,
            "duration_ms": max(
                0, int((result.finished_at - result.started_at).total_seconds() * 1000)
            ),
        }


def _event_metric(data: dict, name: str) -> int:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExecutionError(
            ExecutionErrorCode.EXECUTION_ERROR,
            f"STEP_FINISHED {name} must be a non-negative integer",
        )
    return value


def _descriptor(claim: ClaimResponse) -> PlanDescriptor:
    assert claim.plan is not None
    return PlanDescriptor(
        plan_id=str(claim.plan.plan_id),
        compressed_sha256=claim.plan.compressed_sha256,
        compressed_size=claim.plan.compressed_size,
        canonical_sha256=claim.plan.canonical_sha256,
        canonical_size=claim.plan.canonical_size,
        item_count=claim.plan.item_count,
        case_count=claim.plan.case_count,
    )
