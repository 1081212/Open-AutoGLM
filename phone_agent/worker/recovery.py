"""Conservative startup recovery: never resume non-idempotent phone actions."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.worker.config import RuntimeEnvironment, parse_runtime_environment
from phone_agent.worker.heartbeat import WORKER_VERSION
from phone_agent.worker.outbox import DurableOutbox, LocalSealer


class StartupRecovery:
    def __init__(
        self,
        *,
        api,
        outbox: DurableOutbox,
        sealer: LocalSealer,
        spool_root: Path,
        worker_id: UUID,
        instance_id: UUID,
        runtime_environment: RuntimeEnvironment,
    ) -> None:
        self.api = api
        self.outbox = outbox
        self.sealer = sealer
        self.spool_root = spool_root
        self.worker_id = worker_id
        self.instance_id = instance_id
        self.runtime_environment = parse_runtime_environment(runtime_environment)

    def recover(self) -> None:
        active = self.outbox.active_task_runs()
        for run in active:
            self._stop_old_child(run["task_run_id"])
            response = self.api.heartbeat(self._startup_heartbeat_payload(run))
            decision = response.get("blocking_task_run")
            if not isinstance(decision, dict):
                raise ExecutionError(
                    ExecutionErrorCode.WORKER_RESTARTED,
                    f"Platform omitted startup decision for {run['task_run_id']}",
                    retryable=True,
                )
            self._apply_platform_decision(run, decision)

    def _startup_heartbeat_payload(self, run: dict) -> dict[str, object]:
        claim = run["claim"]
        required_summary = {
            "previous_instance_id": claim.get("instance_id"),
            "previous_fencing_token": claim.get("fencing_token"),
            "previous_lease_expires_at": claim.get("lease_expires_at"),
        }
        missing = [name for name, value in required_summary.items() if value is None]
        if missing:
            raise ExecutionError(
                ExecutionErrorCode.WORKER_RESTARTED,
                "Interrupted TaskRun is missing startup recovery metadata: "
                + ", ".join(missing),
                retryable=True,
            )
        return {
            "schema_version": "autoglm.worker-heartbeat.v1",
            "worker_id": str(self.worker_id),
            "instance_id": str(self.instance_id),
            "runtime_environment": self.runtime_environment,
            "worker_version": WORKER_VERSION,
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "availability": "ONLINE",
            "activity": "STARTING",
            "max_concurrency": 1,
            "active_task_count": 0,
            "current_task_run_id": None,
            "supported_execution_versions": ["autoglm.execution.v1"],
            "devices": [],
            "outbox_pending": self.outbox.pending_count(),
            "spool_free_bytes": shutil.disk_usage(self.spool_root).free,
            "last_error_code": None,
            "startup_context": {
                "previous_task_run_id": run["task_run_id"],
                **required_summary,
            },
        }

    def _apply_platform_decision(self, run: dict, decision: dict) -> None:
        task_run_id = run["task_run_id"]
        if str(decision.get("task_run_id")) != task_run_id:
            raise ExecutionError(
                ExecutionErrorCode.WORKER_RESTARTED,
                f"Platform startup decision does not match interrupted TaskRun {task_run_id}",
                retryable=True,
            )
        if decision.get("phase") == "COMPLETED":
            self.outbox.save_task_run_state(
                task_run_id,
                state="COMPLETED",
                claim=run["claim"],
                plan_ready=run["plan_ready"],
                plan_accepted=run["plan_accepted"],
            )
        elif decision.get("startup_termination_allowed") is True:
            self._complete_as_worker_lost(run)
        elif (
            decision.get("lease_state") is not None
            and decision.get("lease_state") != "ACTIVE"
        ) or (
            decision.get("fence_state") is not None
            and decision.get("fence_state") != "OWNED"
        ):
            self.outbox.mark_orphaned_for_run(
                task_run_id, "startup lease or fence is no longer valid"
            )
            self.outbox.save_task_run_state(
                task_run_id,
                state="ORPHANED",
                claim=run["claim"],
                plan_ready=run["plan_ready"],
                plan_accepted=run["plan_accepted"],
            )
        else:
            raise ExecutionError(
                ExecutionErrorCode.WORKER_RESTARTED,
                f"Interrupted TaskRun {task_run_id} is still blocking startup",
                retryable=True,
            )

    def _complete_as_worker_lost(self, run: dict) -> None:
        claim = run["claim"]
        task_run_id = run["task_run_id"]
        lease_ref = claim["lease_credential_ref"]
        lease = self.outbox.load_credential(lease_ref, self.sealer)
        self.api.complete_run(
            task_run_id,
            {
                "schema_version": "autoglm.result.v1",
                "idempotency_key": f"{task_run_id}:startup-termination",
                "task_id": claim["task_id"],
                "task_run_id": task_run_id,
                "lease_token": lease,
                "fencing_token": claim["fencing_token"],
                "worker_id": str(self.worker_id),
                "device_uid": claim["device_uid"],
                "phase": "COMPLETED",
                "outcome": "WORKER_LOST",
                "result_completeness": "PARTIAL",
                "completion_mode": "STARTUP_TERMINATION",
                "previous_instance_id": claim["instance_id"],
                "current_instance_id": str(self.instance_id),
                "error": {
                    "code": "WORKER_RESTARTED",
                    "category": "WORKER",
                    "message": "Worker parent restarted; phone execution was not resumed",
                    "retryable": False,
                },
            },
        )
        self.outbox.mark_orphaned_for_run(task_run_id, "startup termination")
        self.outbox.save_task_run_state(
            task_run_id,
            state="COMPLETED",
            claim=claim,
            plan_ready=run["plan_ready"],
            plan_accepted=run["plan_accepted"],
        )

    def _stop_old_child(self, task_run_id: str) -> None:
        marker = self.spool_root / "task-runs" / task_run_id / "child.json"
        if not marker.exists():
            return
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            pgid = int(metadata["pgid"])
            if pgid > 1:
                exists = _process_group_exists(pgid)
                if exists and not _is_worker_child(pgid):
                    raise ExecutionError(
                        ExecutionErrorCode.WORKER_RESTARTED,
                        f"Refusing to signal unverified stale process group {pgid}",
                    )
                if exists:
                    os.killpg(pgid, signal.SIGTERM)
                    deadline = time.monotonic() + 5
                    while _process_group_exists(pgid) and time.monotonic() < deadline:
                        time.sleep(0.1)
                    if _process_group_exists(pgid):
                        os.killpg(pgid, signal.SIGKILL)
                        deadline = time.monotonic() + 2
                        while (
                            _process_group_exists(pgid) and time.monotonic() < deadline
                        ):
                            time.sleep(0.1)
                    if _process_group_exists(pgid):
                        raise ExecutionError(
                            ExecutionErrorCode.WORKER_RESTARTED,
                            f"Old task child process group {pgid} did not exit",
                        )
        except ExecutionError:
            raise
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ExecutionError(
                ExecutionErrorCode.WORKER_RESTARTED,
                f"Cannot safely verify old child marker for {task_run_id}: {exc}",
            ) from exc
        marker.unlink(missing_ok=True)


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _is_worker_child(pid: int) -> bool:
    try:
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return (
        completed.returncode == 0
        and "phone_agent.worker.child_process" in completed.stdout
    )
