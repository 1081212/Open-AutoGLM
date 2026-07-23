"""Control heartbeat and run lease loops; neither loop performs ADB calls."""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from uuid import UUID

from phone_agent.execution.cancellation import CancellationToken
from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.execution.models import SUPPORTED_EXECUTION_VERSIONS
from phone_agent.worker.api_client import WorkerApiClient
from phone_agent.worker.models import WorkerActivity
from phone_agent.worker.config import RuntimeEnvironment, parse_runtime_environment
from phone_agent.worker.time_utils import parse_aware_iso8601

WORKER_VERSION = "0.3.0"
logger = logging.getLogger(__name__)


@dataclass
class WorkerRuntimeState:
    activity: WorkerActivity = WorkerActivity.STARTING
    current_task_run_id: str | None = None
    last_error_code: str | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def set_activity(
        self,
        activity: WorkerActivity,
        *,
        task_run_id: str | None = None,
        last_error_code: str | None = None,
    ) -> None:
        with self._lock:
            self.activity = activity
            self.current_task_run_id = task_run_id
            self.last_error_code = last_error_code

    def snapshot(self) -> tuple[WorkerActivity, str | None, str | None]:
        with self._lock:
            return self.activity, self.current_task_run_id, self.last_error_code


class ControlHeartbeatLoop:
    """Periodic heartbeat that consumes cached device state only."""

    def __init__(
        self,
        *,
        api: WorkerApiClient,
        worker_id: UUID,
        instance_id: UUID,
        runtime_environment: RuntimeEnvironment,
        state: WorkerRuntimeState,
        device_snapshot: Callable[[bool], tuple[dict[str, object], ...]],
        outbox_pending: Callable[[], int],
        spool_root: Path,
        interval_seconds: int = 10,
        worker_version: str = WORKER_VERSION,
    ) -> None:
        self.api = api
        self.worker_id = worker_id
        self.instance_id = instance_id
        self.runtime_environment = parse_runtime_environment(runtime_environment)
        self.state = state
        self.device_snapshot = device_snapshot
        self.outbox_pending = outbox_pending
        self.spool_root = spool_root
        self.interval_seconds = interval_seconds
        self.worker_version = worker_version
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="worker-control-heartbeat", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 2)

    def send_once(self) -> dict[str, object]:
        activity, task_run_id, error_code = self.state.snapshot()
        busy = activity in {
            WorkerActivity.CLAIMING,
            WorkerActivity.DOWNLOADING_PLAN,
            WorkerActivity.BUSY,
            WorkerActivity.FINALIZING,
        }
        payload = {
            "schema_version": "autoglm.worker-heartbeat.v1",
            "worker_id": str(self.worker_id),
            "instance_id": str(self.instance_id),
            "runtime_environment": self.runtime_environment,
            "worker_version": self.worker_version,
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "availability": "ONLINE",
            "activity": activity.value,
            "max_concurrency": 1,
            "active_task_count": 1 if task_run_id else 0,
            "current_task_run_id": task_run_id,
            "supported_execution_versions": list(SUPPORTED_EXECUTION_VERSIONS),
            "devices": list(self.device_snapshot(busy)),
            "outbox_pending": self.outbox_pending(),
            "spool_free_bytes": shutil.disk_usage(self.spool_root).free,
            "last_error_code": error_code,
        }
        return self.api.heartbeat(payload)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.send_once()
            except Exception:
                # Presence expiry is handled by the platform. Never call ADB here.
                pass
            self._stop.wait(self.interval_seconds)


class RunLeaseLoop:
    """Renew a claimed Run immediately and every renew interval."""

    def __init__(
        self,
        *,
        api: WorkerApiClient,
        task_run_id: str,
        worker_id: UUID,
        instance_id: UUID,
        device_uid: UUID,
        lease_token: str,
        fencing_token: int,
        lease_expires_at: str,
        cancellation_token: CancellationToken,
        position_provider: Callable[[], dict[str, object]],
        producer_sequences: Callable[[], dict[str, int]],
        renew_after_seconds: int = 10,
        safety_seconds: int = 5,
    ) -> None:
        self.api = api
        self.task_run_id = task_run_id
        self.worker_id = worker_id
        self.instance_id = instance_id
        self.device_uid = device_uid
        self.lease_token = lease_token
        self.fencing_token = fencing_token
        self.lease_expires_at = _parse_time(lease_expires_at)
        self.cancellation_token = cancellation_token
        self.position_provider = position_provider
        self.producer_sequences = producer_sequences
        self.renew_after_seconds = renew_after_seconds
        self.safety = timedelta(seconds=safety_seconds)
        self.lost_error: ExecutionError | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="worker-run-heartbeat", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.renew_after_seconds + 2)

    def send_once(self) -> None:
        payload = {
            "worker_id": str(self.worker_id),
            "instance_id": str(self.instance_id),
            "device_uid": str(self.device_uid),
            "lease_token": self.lease_token,
            "fencing_token": self.fencing_token,
            **self.position_provider(),
            "last_sent_producer_seq": self.producer_sequences(),
        }
        response = self.api.run_heartbeat(self.task_run_id, payload)
        self.lease_expires_at = _parse_time(response.lease_expires_at)
        if response.cancel_requested:
            self.cancellation_token.cancel("Platform requested cancellation")
        if not response.fence_owned:
            self._lose_lease(
                "Platform reports that this Worker no longer owns the fence"
            )

    def _run(self) -> None:
        while not self._stop.is_set() and not self.cancellation_token.is_cancelled:
            try:
                self.send_once()
            except ExecutionError as error:
                if error.code is ExecutionErrorCode.LEASE_LOST:
                    self._lose_lease(str(error))
                    return
            except ValueError:
                self._lose_lease(
                    "Platform returned an invalid lease_expires_at timestamp"
                )
                return
            if datetime.now(timezone.utc) >= self.lease_expires_at - self.safety:
                self._lose_lease("Lease could not be renewed before the safety window")
                return
            self._stop.wait(self.renew_after_seconds)

    def _lose_lease(self, message: str) -> None:
        logger.error(
            "Run lease lost task_run_id=%s reason=%s", self.task_run_id, message
        )
        self.lost_error = ExecutionError(ExecutionErrorCode.LEASE_LOST, message)
        self.cancellation_token.cancel(message)


def _parse_time(value: str) -> datetime:
    return parse_aware_iso8601(value, "lease_expires_at").astimezone(timezone.utc)
