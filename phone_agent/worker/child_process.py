"""Task child process and parent-side process-group supervisor."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from phone_agent.execution.cancellation import CancellationToken
from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.execution.models import CaseOutcome, ExecutionPlan
from phone_agent.execution.result import (
    AttemptResult,
    CaseExecutionResult,
    TaskExecutionResult,
    TaskOutcome,
)
from phone_agent.worker.identity import uuid7
from phone_agent.worker.model_profiles import ModelProfileStore
from phone_agent.worker.outbox import DurableOutbox, LocalSealer
from phone_agent.worker.platform_events import OutboxLifecycleSink
from phone_agent.worker.spool import SpoolPlan
from phone_agent.worker.structured_runner import StructuredPlanRunner
from phone_agent.worker.api_client import WorkerApiClient
from phone_agent.worker.case_coordinator import PlatformCaseCoordinator
from phone_agent.worker.report_bundle import create_local_report_bundle
from phone_agent.worker.config import RuntimeEnvironment, parse_runtime_environment


class ChildProcessPlanExecutor:
    """Launch one isolated child for one claimed Run."""

    def __init__(
        self,
        *,
        python_executable: str,
        model_profiles_path: Path,
        report_root: Path,
        outbox_db_path: Path,
        sealing_key_path: Path,
        platform_base_url: str,
        runtime_environment: RuntimeEnvironment,
        active_probe_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_grace_seconds: int = 30,
    ) -> None:
        self.python_executable = python_executable
        self.model_profiles_path = model_profiles_path
        self.report_root = report_root
        self.outbox_db_path = outbox_db_path
        self.sealing_key_path = sealing_key_path
        self.platform_base_url = platform_base_url
        self.runtime_environment = parse_runtime_environment(runtime_environment)
        self.active_probe_callback = active_probe_callback
        self._position_lock = threading.RLock()
        self._position: dict[str, object] = {
            "current_execution_case_id": None,
            "adhoc_item_state": None,
        }
        self.cancel_grace_seconds = cancel_grace_seconds

    def position(self) -> dict[str, object]:
        with self._position_lock:
            return dict(self._position)

    def _handle_protocol_message(self, message: dict[str, Any]) -> None:
        if message.get("type") == "DEVICE_PROBED" and self.active_probe_callback:
            self.active_probe_callback(message["payload"])
        elif message.get("type") == "POSITION":
            with self._position_lock:
                self._position = dict(message["payload"])

    def __call__(
        self, stored: SpoolPlan, claimed, cancellation: CancellationToken
    ) -> TaskExecutionResult:
        with self._position_lock:
            self._position = {
                "current_execution_case_id": None,
                "adhoc_item_state": "PENDING" if stored.plan.adhoc else None,
            }
        event_read, event_write = os.pipe()
        messages: list[dict[str, Any]] = []
        run_root = stored.root
        log_root = run_root / "logs"
        log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        request = {
            "plan_path": str(stored.canonical_path),
            "task_run_id": str(claimed.task_run_id),
            "worker_id": str(claimed.worker_id),
            "instance_id": str(claimed.instance_id),
            "adb_serial": claimed.adb_serial,
            "model_profiles_path": str(self.model_profiles_path),
            "report_root": str(self.report_root),
            "outbox_db_path": str(self.outbox_db_path),
            "sealing_key_path": str(self.sealing_key_path),
            "platform_base_url": self.platform_base_url,
            "runtime_environment": self.runtime_environment,
            "lease_credential_ref": f"lease:{claimed.task_run_id}",
            "fencing_token": claimed.fencing_token,
            "producer_id": str(uuid7()),
            "device_uid": str(claimed.device_uid),
        }
        with (
            (log_root / "stdout.log").open("ab", buffering=0) as stdout_log,
            (log_root / "stderr.log").open("ab", buffering=0) as stderr_log,
        ):
            process = subprocess.Popen(
                [
                    self.python_executable,
                    "-m",
                    "phone_agent.worker.child_process",
                    "--event-fd",
                    str(event_write),
                ],
                stdin=subprocess.PIPE,
                stdout=stdout_log,
                stderr=stderr_log,
                pass_fds=(event_write,),
                start_new_session=True,
                text=True,
            )
            os.close(event_write)
            reader = threading.Thread(
                target=_read_protocol,
                args=(event_read, messages, self._handle_protocol_message),
                name="worker-child-protocol",
                daemon=True,
            )
            reader.start()
            _atomic_json(
                run_root / "child.json", {"pid": process.pid, "pgid": process.pid}
            )
            assert process.stdin is not None
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.close()
            cancelled_at: float | None = None
            while process.poll() is None:
                if cancellation.is_cancelled and cancelled_at is None:
                    cancelled_at = time.monotonic()
                    _signal_group(process.pid, signal.SIGTERM)
                elif (
                    cancelled_at is not None
                    and time.monotonic() - cancelled_at >= self.cancel_grace_seconds
                ):
                    _signal_group(process.pid, signal.SIGKILL)
                time.sleep(0.1)
            reader.join(timeout=5)
        (run_root / "child.json").unlink(missing_ok=True)
        result_messages = [
            message for message in messages if message.get("type") == "RESULT"
        ]
        if result_messages:
            return _decode_result(result_messages[-1]["payload"])
        code = (
            ExecutionErrorCode.CANCELLED
            if cancellation.is_cancelled
            else ExecutionErrorCode.WORKER_LOST
        )
        error = ExecutionError(
            code, f"task child exited with code {process.returncode}"
        )
        return TaskExecutionResult.now(
            task_id=stored.plan.task_id,
            outcome=TaskOutcome.CANCELLED
            if cancellation.is_cancelled
            else TaskOutcome.WORKER_LOST,
            started_at=datetime.now(timezone.utc),
            error=error,
        )


def child_main(event_fd: int) -> int:
    request = json.loads(sys.stdin.readline())
    os.environ["AUTOGLM_WORKER_CHILD"] = "1"
    cancellation = CancellationToken()
    _install_child_guards(cancellation)
    outbox = DurableOutbox(request["outbox_db_path"])
    sealer = LocalSealer(request["sealing_key_path"])
    plan = ExecutionPlan.model_validate_json(Path(request["plan_path"]).read_bytes())
    durable_sink = OutboxLifecycleSink(
        outbox=outbox,
        task_run_id=UUID(request["task_run_id"]),
        producer_id=UUID(request["producer_id"]),
        lease_credential_ref=request["lease_credential_ref"],
        fencing_token=int(request["fencing_token"]),
        adhoc_execution_item_id=(
            plan.adhoc.execution_item_id if plan.adhoc is not None else None
        ),
    )
    sink = _ChildLifecycleSink(
        durable_sink,
        event_fd,
        request["device_uid"],
        is_adhoc=plan.adhoc is not None,
    )
    timed_out = threading.Event()

    def cancel_for_timeout() -> None:
        timed_out.set()
        cancellation.cancel("Task execution timeout")

    timer = threading.Timer(
        plan.execution_options.task_timeout_seconds,
        cancel_for_timeout,
    )
    timer.daemon = True
    timer.start()
    try:
        worker_credential = os.getenv("AUTOGLM_WORKER_CREDENTIAL", "")
        if not worker_credential:
            raise ExecutionError(
                ExecutionErrorCode.CLAIM_REJECTED,
                "AUTOGLM_WORKER_CREDENTIAL is unavailable in task child",
            )
        api = WorkerApiClient(
            request["platform_base_url"],
            worker_credential,
            parse_runtime_environment(request["runtime_environment"]),
        )
        lease_token = outbox.load_credential(request["lease_credential_ref"], sealer)
        coordinator = PlatformCaseCoordinator(
            api=api,
            outbox=outbox,
            sealer=sealer,
            project_id=plan.project_id,
            task_id=plan.task_id,
            task_run_id=UUID(request["task_run_id"]),
            worker_id=UUID(request["worker_id"]),
            instance_id=UUID(request["instance_id"]),
            device_uid=UUID(request["device_uid"]),
            lease_token=lease_token,
            fencing_token=int(request["fencing_token"]),
            encrypted_root=Path(request["plan_path"]).parent / "encrypted",
            runtime_environment=parse_runtime_environment(
                request["runtime_environment"]
            ),
        )
        runner = StructuredPlanRunner(
            profiles=ModelProfileStore.load(request["model_profiles_path"]),
            adb_serial=request["adb_serial"],
            report_root=Path(request["report_root"]),
            cancellation_token=cancellation,
            lifecycle_sink=sink,
            case_coordinator=coordinator,
        )
        result = runner.execute(plan, task_run_id=request["task_run_id"])
        if timed_out.is_set() and result.outcome is TaskOutcome.CANCELLED:
            timeout_error = ExecutionError(
                ExecutionErrorCode.TASK_TIMEOUT,
                "Task execution exceeded its configured timeout",
            )
            result = replace(result, outcome=TaskOutcome.TIMED_OUT, error=timeout_error)
        if plan.execution_options.local_report_compatibility:
            bundle = create_local_report_bundle(
                Path(request["report_root"]) / request["task_run_id"],
                Path(request["plan_path"]).parent / "local-report-bundle.zip",
            )
            coordinator.stage_run_artifact(
                bundle,
                "LOCAL_REPORT_BUNDLE",
                "application/zip",
            )
        _write_protocol(event_fd, {"type": "RESULT", "payload": _encode_result(result)})
        return 0
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, ExecutionError)
            else ExecutionError(
                ExecutionErrorCode.EXECUTION_ERROR, f"{type(exc).__name__}: {exc}"
            )
        )
        result = TaskExecutionResult.now(
            task_id=plan.task_id,
            outcome=TaskOutcome.INFRA_ERROR,
            started_at=datetime.now(timezone.utc),
            error=error,
        )
        _write_protocol(event_fd, {"type": "RESULT", "payload": _encode_result(result)})
        return 1
    finally:
        timer.cancel()
        outbox.close()
        os.close(event_fd)


def _install_child_guards(cancellation: CancellationToken) -> None:
    parent_pid = os.getppid()

    def handle_signal(_signum, _frame):
        cancellation.cancel("Task child received termination signal")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    def watch_parent() -> None:
        while not cancellation.wait(1):
            if os.getppid() != parent_pid or os.getppid() == 1:
                cancellation.cancel("Worker parent process exited")
                # The parent can no longer enforce cancel_grace_seconds. Give
                # cooperative cleanup a short window, then kill this entire
                # task process group so an ADB/player subprocess cannot survive.
                time.sleep(5)
                try:
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return

    threading.Thread(
        target=watch_parent, name="worker-parent-watch", daemon=True
    ).start()


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _write_protocol(fd: int, message: dict[str, Any]) -> None:
    data = (
        json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _read_protocol(
    fd: int,
    messages: list[dict[str, Any]],
    message_callback: Callable[[dict[str, Any]], None] | None,
) -> None:
    with os.fdopen(fd, "r", encoding="utf-8") as events:
        for line in events:
            if line.strip():
                message = json.loads(line)
                messages.append(message)
                if message_callback:
                    message_callback(message)


class _ChildLifecycleSink:
    def __init__(
        self, durable_sink, event_fd: int, device_uid: str, *, is_adhoc: bool
    ) -> None:
        self.durable_sink = durable_sink
        self.event_fd = event_fd
        self.device_uid = device_uid
        self.is_adhoc = is_adhoc

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        self.durable_sink.emit(event_type, data)
        if event_type == "DEVICE_PROBED":
            _write_protocol(
                self.event_fd,
                {
                    "type": "DEVICE_PROBED",
                    "payload": {**data, "device_uid": self.device_uid},
                },
            )
        elif event_type in {
            "RUN_STARTED",
            "CASE_STARTED",
            "STEP_STARTED",
            "RUN_FINISHED",
        }:
            adhoc_item_state = None
            if self.is_adhoc:
                adhoc_item_state = (
                    "FINALIZING" if event_type == "RUN_FINISHED" else "RUNNING"
                )
            _write_protocol(
                self.event_fd,
                {
                    "type": "POSITION",
                    "payload": {
                        "current_execution_case_id": (
                            None
                            if event_type == "RUN_FINISHED"
                            else data.get("execution_case_id")
                        ),
                        "adhoc_item_state": adhoc_item_state,
                    },
                },
            )


def _encode_result(result: TaskExecutionResult) -> dict[str, Any]:
    return {
        "task_id": str(result.task_id),
        "outcome": result.outcome.value,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
        "cases": [
            {
                "execution_case_id": str(case.execution_case_id),
                "ordinal": case.ordinal,
                "outcome": case.outcome.value,
                "flaky": case.flaky,
                "attempts": [
                    {
                        "outcome": attempt.outcome.value,
                        "message": attempt.message,
                        "error": attempt.error.as_dict() if attempt.error else None,
                    }
                    for attempt in case.attempts
                ],
            }
            for case in result.cases
        ],
        "adhoc_result": (
            {
                "outcome": result.adhoc_result.outcome.value,
                "message": result.adhoc_result.message,
                "error": result.adhoc_result.error.as_dict()
                if result.adhoc_result.error
                else None,
            }
            if result.adhoc_result
            else None
        ),
        "error": result.error.as_dict() if result.error else None,
        "not_run_execution_case_ids": [
            str(value) for value in result.not_run_execution_case_ids
        ],
    }


def _decode_result(payload: dict[str, Any]) -> TaskExecutionResult:
    def decode_error(value):
        if not value:
            return None
        return ExecutionError(
            ExecutionErrorCode(value["code"]),
            value["message"],
            bool(value.get("retryable")),
            value.get("details_artifact_id"),
        )

    def decode_attempt(value):
        return AttemptResult(
            CaseOutcome(value["outcome"]),
            value["message"],
            decode_error(value.get("error")),
        )

    adhoc = payload.get("adhoc_result")
    return TaskExecutionResult(
        task_id=UUID(payload["task_id"]),
        outcome=TaskOutcome(payload["outcome"]),
        started_at=datetime.fromisoformat(payload["started_at"]),
        finished_at=datetime.fromisoformat(payload["finished_at"]),
        cases=tuple(
            CaseExecutionResult(
                execution_case_id=UUID(case["execution_case_id"]),
                ordinal=case["ordinal"],
                outcome=CaseOutcome(case["outcome"]),
                flaky=case["flaky"],
                attempts=tuple(decode_attempt(attempt) for attempt in case["attempts"]),
            )
            for case in payload["cases"]
        ),
        adhoc_result=decode_attempt(adhoc) if adhoc else None,
        error=decode_error(payload.get("error")),
        not_run_execution_case_ids=tuple(
            UUID(value) for value in payload["not_run_execution_case_ids"]
        ),
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    os.replace(part, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-fd", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(child_main(_parse_args().event_fd))
