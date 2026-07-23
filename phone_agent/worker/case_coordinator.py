"""Platform Attempt, encrypted Artifact, and Case checkpoint boundary."""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.execution.models import ExecutionCase
from phone_agent.execution.result import AttemptResult, CaseExecutionResult
from phone_agent.worker.artifact_uploader import ArtifactPreparer, ArtifactSource
from phone_agent.worker.config import RuntimeEnvironment, parse_runtime_environment
from phone_agent.worker.platform_events import OutboxPump

MAX_ARTIFACT_PLAINTEXT_SIZE = 64 * 1024 * 1024
logger = logging.getLogger(__name__)


class PlatformCaseCoordinator:
    def __init__(
        self,
        *,
        api,
        outbox,
        sealer,
        project_id: UUID,
        task_id: UUID,
        task_run_id: UUID,
        worker_id: UUID,
        instance_id: UUID,
        device_uid: UUID,
        lease_token: str,
        fencing_token: int,
        encrypted_root: Path,
        runtime_environment: RuntimeEnvironment,
    ) -> None:
        self.api = api
        self.outbox = outbox
        self.sealer = sealer
        self.project_id = project_id
        self.task_id = task_id
        self.task_run_id = task_run_id
        self.worker_id = worker_id
        self.instance_id = instance_id
        self.device_uid = device_uid
        self.lease_token = lease_token
        self.fencing_token = fencing_token
        self.runtime_environment = parse_runtime_environment(runtime_environment)
        self.preparer = ArtifactPreparer(
            api,
            outbox,
            sealer,
            encrypted_root,
            self.runtime_environment,
            worker_id=worker_id,
            instance_id=instance_id,
            device_uid=device_uid,
        )
        self._attempt_ids: dict[tuple[UUID, int], UUID] = {}
        self._artifacts: dict[tuple[UUID, int], list[dict[str, object]]] = {}

    def stage_run_artifact(
        self, path: Path, artifact_type: str, content_type: str
    ) -> UUID:
        identity = hashlib.sha256(path.name.encode()).hexdigest()[:16]
        artifact_id = self.preparer.prepare(
            ArtifactSource(
                path=path,
                artifact_type=artifact_type,
                content_type=content_type,
                project_id=self.project_id,
                task_id=self.task_id,
                task_run_id=self.task_run_id,
            ),
            initiate_idempotency_key=f"{self.task_run_id}:{artifact_type}:{identity}",
            lease_token=self.lease_token,
            fencing_token=self.fencing_token,
        )
        return artifact_id

    def begin_attempt(self, case: ExecutionCase, attempt_no: int) -> UUID:
        logger.info(
            "Creating platform Case Attempt task_run_id=%s case=%s "
            "execution_case_id=%s attempt_no=%d",
            self.task_run_id,
            case.display_id,
            case.execution_case_id,
            attempt_no,
        )
        response = self.api.begin_attempt(
            str(self.task_run_id),
            str(case.execution_case_id),
            {
                "worker_id": str(self.worker_id),
                "instance_id": str(self.instance_id),
                "device_uid": str(self.device_uid),
                "idempotency_key": f"{self.task_run_id}:{case.execution_case_id}:attempt:{attempt_no}",
                "lease_token": self.lease_token,
                "fencing_token": self.fencing_token,
                "attempt_no": attempt_no,
                "attempt_kind": "INITIAL" if attempt_no == 1 else "CASE_RETRY",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        try:
            attempt_id = UUID(str(response["case_attempt_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutionError(
                ExecutionErrorCode.EXECUTION_ERROR,
                "attempts:begin returned no valid case_attempt_id",
            ) from exc
        self._attempt_ids[(case.execution_case_id, attempt_no)] = attempt_id
        logger.info(
            "Platform Case Attempt created task_run_id=%s "
            "execution_case_id=%s attempt_no=%d case_attempt_id=%s",
            self.task_run_id,
            case.execution_case_id,
            attempt_no,
            attempt_id,
        )
        return attempt_id

    def finish_attempt(
        self,
        case: ExecutionCase,
        attempt_no: int,
        result: AttemptResult,
        attempt_dir: Path,
    ) -> None:
        key = (case.execution_case_id, attempt_no)
        attempt_id = self._attempt_ids[key]
        artifacts = self._stage_artifacts(case, attempt_no, attempt_id, attempt_dir)
        self._artifacts[key] = artifacts
        logger.info(
            "Completing platform Case Attempt task_run_id=%s "
            "case_attempt_id=%s outcome=%s artifact_count=%d",
            self.task_run_id,
            attempt_id,
            result.outcome.value,
            len(artifacts),
        )
        self.api.complete_attempt(
            str(attempt_id),
            {
                "worker_id": str(self.worker_id),
                "instance_id": str(self.instance_id),
                "device_uid": str(self.device_uid),
                "idempotency_key": f"{attempt_id}:complete",
                "lease_token": self.lease_token,
                "fencing_token": self.fencing_token,
                "outcome": result.outcome.value,
                "result_message": result.message,
                "metrics": self._attempt_metrics(case.execution_case_id, attempt_no),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info(
            "Platform Case Attempt completed task_run_id=%s case_attempt_id=%s",
            self.task_run_id,
            attempt_id,
        )

    def checkpoint(self, case: ExecutionCase, result: CaseExecutionResult) -> None:
        final_attempt_no = len(result.attempts)
        final_attempt_id = self._attempt_ids[(case.execution_case_id, final_attempt_no)]
        artifacts = [
            artifact
            for attempt_no in range(1, final_attempt_no + 1)
            for artifact in self._artifacts.get(
                (case.execution_case_id, attempt_no), []
            )
        ]
        artifact_ids = [str(item["artifact_id"]) for item in artifacts]
        try:
            OutboxPump(self.outbox, self.api, self.sealer).flush_once(limit=10_000)
            if self.outbox.unacknowledged_event_count_for_run(str(self.task_run_id)):
                raise ExecutionError(
                    ExecutionErrorCode.RETRYABLE_ERROR,
                    "Case events are not acknowledged; checkpoint is blocked",
                    retryable=True,
                )
            logger.info(
                "Submitting Case checkpoint task_run_id=%s execution_case_id=%s "
                "outcome=%s flaky=%s attempt_count=%d artifact_count=%d",
                self.task_run_id,
                case.execution_case_id,
                result.outcome.value,
                result.flaky,
                len(result.attempts),
                len(artifacts),
            )
            self.api.checkpoint(
                str(self.task_run_id),
                str(case.execution_case_id),
                {
                    "schema_version": "autoglm.checkpoint.v1",
                    "task_run_id": str(self.task_run_id),
                    "lease_token": self.lease_token,
                    "fencing_token": self.fencing_token,
                    "execution_case_id": str(case.execution_case_id),
                    "ordinal": case.ordinal,
                    "case_outcome": result.outcome.value,
                    "flaky": result.flaky,
                    "final_case_attempt_id": str(final_attempt_id),
                    "final_case_attempt_no": final_attempt_no,
                    "attempt_count": len(result.attempts),
                    "producer_positions": self.outbox.producer_positions_for_run(
                        str(self.task_run_id)
                    ),
                    "artifacts": artifacts,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except ExecutionError as error:
            logger.error(
                "Case checkpoint rejected task_run_id=%s execution_case_id=%s "
                "error_code=%s retryable=%s message=%s",
                self.task_run_id,
                case.execution_case_id,
                error.code.value,
                error.retryable,
                error.message,
            )
            raise
        finally:
            released = self.outbox.release_artifact_uploads(artifact_ids)
            logger.info(
                "Released checkpoint-gated Artifact uploads task_run_id=%s "
                "execution_case_id=%s artifact_count=%d",
                self.task_run_id,
                case.execution_case_id,
                released,
            )
        logger.info(
            "Case checkpoint accepted task_run_id=%s execution_case_id=%s",
            self.task_run_id,
            case.execution_case_id,
        )

    def _attempt_metrics(
        self, execution_case_id: UUID, attempt_no: int
    ) -> dict[str, int]:
        duration_ms = 0
        vision_tokens = 0
        judge_tokens = 0
        for event in self.outbox.events_for_run(str(self.task_run_id)):
            if (
                event.get("type") != "STEP_FINISHED"
                or event.get("execution_case_id") != str(execution_case_id)
                or event.get("case_attempt_no") != attempt_no
            ):
                continue
            data = event.get("data") or {}
            duration_ms += _non_negative_metric(data, "duration_ms")
            vision_tokens += _non_negative_metric(data, "vision_tokens")
            judge_tokens += _non_negative_metric(data, "judge_tokens")
        return {
            "duration_ms": duration_ms,
            "vision_tokens": vision_tokens,
            "judge_tokens": judge_tokens,
        }

    def _stage_artifacts(
        self,
        case: ExecutionCase,
        attempt_no: int,
        attempt_id: UUID,
        attempt_dir: Path,
    ) -> list[dict[str, object]]:
        staged: list[dict[str, object]] = []
        root = attempt_dir.resolve()
        if not root.is_dir():
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                f"Attempt artifact directory is missing: {attempt_dir}",
            )
        paths = sorted(attempt_dir.rglob("*"))
        requires_step_bindings = any(
            path.is_file() and _artifact_identity(path)[0] in {"SCREENSHOT", "UI_XML"}
            for path in paths
        )
        step_bindings = (
            self._load_step_artifact_bindings(case, attempt_no, attempt_dir)
            if requires_step_bindings
            else {}
        )
        for path in paths:
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve()
            if root not in resolved.parents:
                raise ExecutionError(
                    ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                    "Artifact path escapes the attempt directory",
                )
            size = path.stat().st_size
            if size > MAX_ARTIFACT_PLAINTEXT_SIZE:
                raise ExecutionError(
                    ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                    f"Artifact exceeds 64 MiB rolling limit: {path.name}",
                )
            relative = path.relative_to(attempt_dir).as_posix()
            artifact_type, content_type = _artifact_identity(path)
            step_id = (
                step_bindings.get(resolved)
                if artifact_type in {"SCREENSHOT", "UI_XML"}
                else None
            )
            if artifact_type in {"SCREENSHOT", "UI_XML"} and step_id is None:
                raise ExecutionError(
                    ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                    f"{artifact_type} is missing a test_step_index binding in "
                    f"report.json: {relative}",
                )
            identity = hashlib.sha256(relative.encode()).hexdigest()[:16]
            upload_key = f"{attempt_id}:{identity}"
            artifact_id = self.preparer.prepare(
                ArtifactSource(
                    path=path,
                    artifact_type=artifact_type,
                    content_type=content_type,
                    project_id=self.project_id,
                    task_id=self.task_id,
                    task_run_id=self.task_run_id,
                    execution_case_id=case.execution_case_id,
                    case_attempt_no=attempt_no,
                    case_attempt_id=attempt_id,
                    step_id=step_id,
                ),
                initiate_idempotency_key=upload_key,
                lease_token=self.lease_token,
                fencing_token=self.fencing_token,
                defer_upload_until_checkpoint=True,
            )
            item = self.outbox.find_by_idempotency_key(f"{artifact_id}:upload-complete")
            if item is None or item.kind != "ARTIFACT_UPLOAD":
                raise ExecutionError(
                    ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                    "Prepared Artifact has no durable upload item",
                )
            # Upload is best effort. Encrypted durable staging is the Case boundary.
            staged.append(
                {
                    "artifact_id": str(artifact_id),
                    "upload_state": "PENDING",
                    "encrypted_local_sha256": item.payload["ciphertext_sha256"],
                    "type": artifact_type,
                    "relative_name": relative,
                }
            )
        return staged

    @staticmethod
    def _load_step_artifact_bindings(
        case: ExecutionCase,
        attempt_no: int,
        attempt_dir: Path,
    ) -> dict[Path, UUID]:
        report_path = attempt_dir.parent / "report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                "Case report.json is missing or invalid; screenshot and UI XML "
                "cannot be bound to a frozen Plan step",
            ) from exc
        if not isinstance(report, dict) or report.get("execution_case_id") != str(
            case.execution_case_id
        ):
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                "Case report.json execution_case_id does not match the Plan",
            )
        attempts = report.get("attempts")
        if not isinstance(attempts, list):
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                "Case report.json has no Attempt metadata",
            )
        attempt = next(
            (
                item
                for item in attempts
                if isinstance(item, dict) and item.get("attempt") == attempt_no
            ),
            None,
        )
        if attempt is None or not isinstance(attempt.get("steps"), list):
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                f"Case report.json has no metadata for Attempt {attempt_no}",
            )
        plan_steps = {step.index: step.step_id for step in case.steps}
        case_root = attempt_dir.parent.resolve()
        attempt_root = attempt_dir.resolve()
        bindings: dict[Path, UUID] = {}
        for artifact in attempt["steps"]:
            if not isinstance(artifact, dict):
                continue
            test_step_index = artifact.get("test_step_index")
            if (
                isinstance(test_step_index, bool)
                or not isinstance(test_step_index, int)
                or test_step_index not in plan_steps
            ):
                raise ExecutionError(
                    ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                    "Case report.json contains an invalid test_step_index",
                )
            for field in ("screenshot", "ui_xml"):
                relative = artifact.get(field)
                if relative is None:
                    continue
                if not isinstance(relative, str) or not relative:
                    raise ExecutionError(
                        ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                        f"Case report.json contains an invalid {field} path",
                    )
                resolved = (case_root / relative).resolve()
                if attempt_root not in resolved.parents:
                    raise ExecutionError(
                        ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                        f"Case report.json {field} path escapes the current Attempt",
                    )
                previous = bindings.setdefault(resolved, plan_steps[test_step_index])
                if previous != plan_steps[test_step_index]:
                    raise ExecutionError(
                        ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                        f"Case report.json binds {field} to conflicting Plan steps",
                    )
        return bindings


def _artifact_identity(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        artifact_type = "SCREENSHOT"
    elif suffix == ".xml":
        artifact_type = "UI_XML"
    elif suffix in {".log", ".txt"}:
        artifact_type = "LOG"
    else:
        artifact_type = "ATTEMPT_ATTACHMENT"
    return (
        artifact_type,
        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )


def _non_negative_metric(data: dict, name: str) -> int:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExecutionError(
            ExecutionErrorCode.EXECUTION_ERROR,
            f"STEP_FINISHED {name} must be a non-negative integer",
        )
    return value
