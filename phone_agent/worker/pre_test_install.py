"""Platform-frozen GitLab CI APK installation for execution v2 plans."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlparse

import requests

from phone_agent.adb.command import AdbCommandAdapter
from phone_agent.execution.cancellation import CancellationToken
from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.execution.models import ExecutionPlan, PreTestInstall, TaskType
from phone_agent.gitlab_install import (
    FrozenArtifactError,
    create_session,
    download_frozen_job_artifact,
    extract_single_apk_secure,
)
from phone_agent.worker.identity import uuid7
from phone_agent.worker.outbox import DurableOutbox
from phone_agent.worker.platform_events import OutboxLifecycleSink
from phone_agent.worker.spool import SpoolPlan

INSTALLATION_FACT_SCHEMA = "autoglm.installation-fact.v1"
logger = logging.getLogger(__name__)


class ClaimedRunLike(Protocol):
    task_run_id: object
    device_uid: object
    adb_serial: str
    fencing_token: int


@dataclass(frozen=True, slots=True)
class WorkerGitLabConfig:
    base_url: str | None
    token: str | None
    verify_ssl: bool = True
    use_env_proxy: bool = False
    timeout_seconds: int = 300
    max_artifact_bytes: int = 2 * 1024 * 1024 * 1024
    max_apk_bytes: int = 1024 * 1024 * 1024
    min_free_bytes: int = 512 * 1024 * 1024
    apk_metadata_tool: str | None = None


@dataclass(frozen=True, slots=True)
class ApkMetadata:
    package_name: str
    version_name: str
    version_code: str


@dataclass(frozen=True, slots=True)
class InstallationFact:
    schema_version: str
    idempotency_key: str
    task_run_id: str
    ci_build_id: str
    pipeline_id: int
    job_id: int | None
    apk_sha256: str | None
    apk_size: int | None
    package_name: str | None
    version_name: str | None
    version_code: str | None
    device_uid: str
    started_at: str
    finished_at: str
    install_result: str
    error_code: str | None = None

    def as_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "InstallationFact":
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class InstallationAttempt:
    fact: InstallationFact
    error: ExecutionError | None = None


class ApkMetadataReader:
    """Read immutable APK manifest facts without installing the APK first."""

    def __init__(self, configured_tool: str | None = None) -> None:
        self.configured_tool = configured_tool

    def read(self, apk_path: Path) -> ApkMetadata:
        tool = self._resolve_tool()
        if Path(tool).name.startswith("apkanalyzer"):
            return ApkMetadata(
                package_name=self._run(tool, "application-id", apk_path),
                version_name=self._run(tool, "version-name", apk_path),
                version_code=self._run(tool, "version-code", apk_path),
            )
        completed = self._run_process(
            [tool, "dump", "badging", str(apk_path)], timeout=60
        )
        first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
        package = _badging_value(first_line, "name")
        version_name = _badging_value(first_line, "versionName")
        version_code = _badging_value(first_line, "versionCode")
        if not package or version_name is None or version_code is None:
            raise ExecutionError(
                ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                "APK manifest metadata is incomplete",
            )
        return ApkMetadata(package, version_name, version_code)

    def _resolve_tool(self) -> str:
        if self.configured_tool:
            return self.configured_tool
        for name in ("apkanalyzer", "aapt2", "aapt"):
            found = shutil.which(name)
            if found:
                return found
        raise ExecutionError(
            ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
            "No APK metadata tool is available; configure AUTOGLM_ANDROID_APK_METADATA_TOOL",
        )

    def _run(self, tool: str, operation: str, apk_path: Path) -> str:
        completed = self._run_process(
            [tool, "manifest", operation, str(apk_path)], timeout=60
        )
        value = completed.stdout.strip()
        if not value:
            raise ExecutionError(
                ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                f"APK manifest {operation} is empty",
            )
        return value

    @staticmethod
    def _run_process(
        argv: list[str], *, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExecutionError(
                ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                "APK metadata inspection could not be executed",
            ) from exc
        if completed.returncode != 0:
            raise ExecutionError(
                ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                "APK metadata inspection failed",
            )
        return completed


class FrozenGitLabApkInstaller:
    """Download candidates in Plan order and install exactly one verified APK."""

    def __init__(
        self,
        *,
        config: WorkerGitLabConfig,
        outbox: DurableOutbox,
        adb: AdbCommandAdapter | None = None,
        metadata_reader: ApkMetadataReader | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.outbox = outbox
        self.adb = adb or AdbCommandAdapter()
        self.metadata_reader = metadata_reader or ApkMetadataReader(
            config.apk_metadata_tool
        )
        self.session = session

    def install(
        self,
        stored: SpoolPlan,
        claimed: ClaimedRunLike,
        cancellation: CancellationToken,
        *,
        lease_credential_ref: str,
        guard: Callable[[], None],
    ) -> InstallationAttempt | None:
        plan = stored.plan
        instruction = plan.pre_test_install
        if instruction is None:
            return None
        started_at = _now()
        logger.info(
            "Frozen APK install preparing task_run_id=%s ci_build_id=%s "
            "pipeline_id=%s candidate_count=%d adb_serial=%s",
            claimed.task_run_id,
            instruction.ci_build_id,
            instruction.pipeline_id,
            len(instruction.artifact_candidates),
            claimed.adb_serial,
        )
        try:
            self._validate_local_config(instruction)
        except ExecutionError as error:
            return self._failed(
                plan,
                claimed,
                lease_credential_ref,
                started_at,
                None,
                None,
                None,
                None,
                error.message,
            )
        session = self.session or create_session(
            verify_ssl=self.config.verify_ssl,
            tls_version="auto",
            use_env_proxy=self.config.use_env_proxy,
        )
        install_root = stored.root / "pre-test-install"
        install_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        selected_job_id: int | None = None
        apk_path: Path | None = None
        apk_sha256: str | None = None
        apk_size: int | None = None
        metadata: ApkMetadata | None = None
        try:
            for candidate in instruction.artifact_candidates:
                guard()
                cancellation.raise_if_cancelled()
                zip_path: Path | None = None
                try:
                    logger.info(
                        "Downloading frozen GitLab artifact task_run_id=%s "
                        "job_id=%s declared_size=%s",
                        claimed.task_run_id,
                        candidate.job_id,
                        candidate.artifact_size,
                    )
                    zip_path = download_frozen_job_artifact(
                        session=session,
                        gitlab_base_url=self.config.base_url or "",
                        project_path=instruction.gitlab_project_path,
                        job_id=candidate.job_id,
                        token=self.config.token or "",
                        output_dir=install_root,
                        declared_size=candidate.artifact_size,
                        max_artifact_bytes=self.config.max_artifact_bytes,
                        min_free_bytes=self.config.min_free_bytes,
                        timeout_seconds=self.config.timeout_seconds,
                        verify_ssl=self.config.verify_ssl,
                        cancellation_check=lambda: (
                            guard(),
                            cancellation.raise_if_cancelled(),
                        ),
                    )
                    apk_path = extract_single_apk_secure(
                        zip_path,
                        install_root,
                        output_filename=f"job-{candidate.job_id}.apk",
                        max_apk_bytes=self.config.max_apk_bytes,
                        min_free_bytes=self.config.min_free_bytes,
                        cancellation_check=lambda: (
                            guard(),
                            cancellation.raise_if_cancelled(),
                        ),
                    )
                    selected_job_id = candidate.job_id
                    logger.info(
                        "Frozen artifact selected task_run_id=%s job_id=%s",
                        claimed.task_run_id,
                        selected_job_id,
                    )
                    break
                except FrozenArtifactError as error:
                    logger.warning(
                        "Frozen artifact candidate failed task_run_id=%s "
                        "job_id=%s recoverable=%s retryable=%s message=%s",
                        claimed.task_run_id,
                        candidate.job_id,
                        error.candidate_recoverable,
                        error.retryable,
                        str(error),
                    )
                    if error.candidate_recoverable:
                        continue
                    return self._failed(
                        plan,
                        claimed,
                        lease_credential_ref,
                        started_at,
                        selected_job_id or candidate.job_id,
                        apk_sha256,
                        apk_size,
                        metadata,
                        str(error),
                        retryable=error.retryable,
                    )
                finally:
                    if zip_path is not None:
                        zip_path.unlink(missing_ok=True)
            if apk_path is None or selected_job_id is None:
                return self._failed(
                    plan,
                    claimed,
                    lease_credential_ref,
                    started_at,
                    None,
                    None,
                    None,
                    None,
                    "No frozen artifact candidate contained exactly one APK",
                )

            try:
                apk_size = apk_path.stat().st_size
                apk_sha256 = "sha256:" + _sha256(apk_path)
            except OSError:
                return self._failed(
                    plan,
                    claimed,
                    lease_credential_ref,
                    started_at,
                    selected_job_id,
                    None,
                    None,
                    None,
                    "Downloaded APK could not be read from the Worker spool",
                )
            logger.info(
                "APK verified task_run_id=%s job_id=%s apk_size=%d apk_sha256=%s",
                claimed.task_run_id,
                selected_job_id,
                apk_size,
                apk_sha256,
            )
            fact_key = _success_fact_key(
                str(claimed.task_run_id),
                str(instruction.ci_build_id),
                selected_job_id,
                apk_sha256,
                str(claimed.device_uid),
            )
            durable_success_fact: InstallationFact | None = None
            existing = self.outbox.find_by_idempotency_key(fact_key)
            if existing is not None:
                fact = InstallationFact.from_payload(existing.payload)
                if (
                    fact.install_result != "SUCCESS"
                    or fact.task_run_id != str(claimed.task_run_id)
                    or fact.ci_build_id != str(instruction.ci_build_id)
                    or fact.job_id != selected_job_id
                    or fact.apk_sha256 != apk_sha256
                    or fact.device_uid != str(claimed.device_uid)
                ):
                    raise ExecutionError(
                        ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                        "Installation idempotency key is bound to a conflicting fact",
                    )
                try:
                    self._verify_installed_package(
                        claimed.adb_serial,
                        plan.target_requirements.app_package,
                    )
                except ExecutionError:
                    # The prior external fact remains durable, but it cannot
                    # prove that the package is still present after a restart,
                    # manual uninstall, or device reset. Reinstall the already
                    # downloaded and verified APK before allowing any Case.
                    logger.warning(
                        "Durable install fact found but target package is absent; "
                        "reinstalling task_run_id=%s adb_serial=%s job_id=%s",
                        claimed.task_run_id,
                        claimed.adb_serial,
                        selected_job_id,
                    )
                    durable_success_fact = fact
                else:
                    self._emit_once(
                        plan, claimed, lease_credential_ref, fact, "RUN_STARTED"
                    )
                    logger.info(
                        "Reusing durable successful install fact task_run_id=%s "
                        "job_id=%s",
                        claimed.task_run_id,
                        selected_job_id,
                    )
                    return InstallationAttempt(fact)

            # Manifest inspection is temporarily disabled. The downloaded file
            # has already passed the frozen Job, ZIP safety, single-APK, size,
            # and SHA-256 checks; proceed directly to the locked-device install.
            guard()
            cancellation.raise_if_cancelled()
            try:
                android_user = self._current_android_user(claimed.adb_serial)
                logger.info(
                    "ADB APK install starting task_run_id=%s adb_serial=%s "
                    "android_user=%s job_id=%s",
                    claimed.task_run_id,
                    claimed.adb_serial,
                    android_user,
                    selected_job_id,
                )
                install_result = self.adb.run(
                    claimed.adb_serial,
                    ["install", "-r", "-d", str(apk_path)],
                    timeout=300,
                )
                install_output = _adb_output(
                    install_result.stdout, install_result.stderr
                )
                if not any(
                    line.strip() == "Success" for line in install_output.splitlines()
                ):
                    raise ExecutionError(
                        ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                        "ADB install returned without an explicit Success result",
                    )
                logger.info(
                    "ADB APK install command completed task_run_id=%s "
                    "adb_serial=%s android_user=%s job_id=%s "
                    "duration_ms=%s result=Success",
                    claimed.task_run_id,
                    claimed.adb_serial,
                    android_user,
                    selected_job_id,
                    getattr(install_result, "duration_ms", "-"),
                )
                self._ensure_installed_for_user(
                    claimed.adb_serial,
                    plan.target_requirements.app_package,
                    android_user,
                )
            except ExecutionError as error:
                return self._failed(
                    plan,
                    claimed,
                    lease_credential_ref,
                    started_at,
                    selected_job_id,
                    apk_sha256,
                    apk_size,
                    metadata,
                    f"ADB APK installation verification failed: {_safe_error(error)}",
                )
            if durable_success_fact is not None:
                self._emit_once(
                    plan,
                    claimed,
                    lease_credential_ref,
                    durable_success_fact,
                    "RUN_STARTED",
                )
                logger.info(
                    "ADB APK reinstall succeeded task_run_id=%s adb_serial=%s "
                    "job_id=%s",
                    claimed.task_run_id,
                    claimed.adb_serial,
                    selected_job_id,
                )
                guard()
                cancellation.raise_if_cancelled()
                return InstallationAttempt(durable_success_fact)
            fact = InstallationFact(
                schema_version=INSTALLATION_FACT_SCHEMA,
                idempotency_key=fact_key,
                task_run_id=str(claimed.task_run_id),
                ci_build_id=str(instruction.ci_build_id),
                pipeline_id=instruction.pipeline_id,
                job_id=selected_job_id,
                apk_sha256=apk_sha256,
                apk_size=apk_size,
                package_name=plan.target_requirements.app_package,
                version_name=None,
                version_code=None,
                device_uid=str(claimed.device_uid),
                started_at=started_at,
                finished_at=_now(),
                install_result="SUCCESS",
            )
            self._record_fact(fact)
            self._emit_once(plan, claimed, lease_credential_ref, fact, "RUN_STARTED")
            logger.info(
                "ADB APK install succeeded task_run_id=%s adb_serial=%s " "job_id=%s",
                claimed.task_run_id,
                claimed.adb_serial,
                selected_job_id,
            )
            # Installation may have completed while cancellation or fence loss was
            # being delivered. Record the external fact first, then prohibit Cases.
            guard()
            cancellation.raise_if_cancelled()
            return InstallationAttempt(fact)
        finally:
            if apk_path is not None:
                apk_path.unlink(missing_ok=True)
            for part in install_root.glob("*.part"):
                part.unlink(missing_ok=True)

    def _current_android_user(self, adb_serial: str) -> str:
        result = self.adb.run(
            adb_serial,
            ["shell", "am", "get-current-user"],
            timeout=30,
        )
        android_user = result.stdout.strip()
        if not android_user.isdigit():
            raise ExecutionError(
                ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                "ADB did not return a valid current Android user",
            )
        return android_user

    def _verify_installed_package(
        self,
        adb_serial: str,
        expected_package: str | None,
        *,
        android_user: str | None = None,
    ) -> None:
        if not expected_package:
            raise ExecutionError(
                ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                "execution v2 Plan target_requirements.app_package is missing",
            )
        user = android_user or self._current_android_user(adb_serial)
        result = self.adb.run(
            adb_serial,
            ["shell", "pm", "path", "--user", user, expected_package],
            timeout=30,
            check=False,
        )
        paths = [
            line.removeprefix("package:").strip()
            for line in result.stdout.splitlines()
            if line.startswith("package:") and line.removeprefix("package:").strip()
        ]
        if not paths:
            raise ExecutionError(
                ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                f"target package {expected_package} is absent for Android "
                f"user {user} after adb install",
            )
        logger.info(
            "Installed package verified adb_serial=%s android_user=%s "
            "package_name=%s apk_path_count=%d",
            adb_serial,
            user,
            expected_package,
            len(paths),
        )

    def _ensure_installed_for_user(
        self,
        adb_serial: str,
        expected_package: str | None,
        android_user: str,
    ) -> None:
        try:
            self._verify_installed_package(
                adb_serial,
                expected_package,
                android_user=android_user,
            )
            return
        except ExecutionError:
            if not expected_package:
                raise
        # On multi-user devices, `adb install -r` can update an existing base
        # package that is enabled only for another user. The command reports
        # Success, but the current foreground user still cannot see the app.
        logger.warning(
            "APK base package updated but is not enabled for current Android "
            "user; enabling existing package adb_serial=%s android_user=%s "
            "package_name=%s",
            adb_serial,
            android_user,
            expected_package,
        )
        try:
            self.adb.run(
                adb_serial,
                [
                    "shell",
                    "cmd",
                    "package",
                    "install-existing",
                    "--user",
                    android_user,
                    "--wait",
                    expected_package,
                ],
                timeout=60,
            )
        except ExecutionError as error:
            raise ExecutionError(
                ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                f"target package {expected_package} could not be enabled for "
                f"Android user {android_user}: {_safe_error(error)}",
            ) from error
        self._verify_installed_package(
            adb_serial,
            expected_package,
            android_user=android_user,
        )

    def emit_failed_run_finished(
        self,
        plan: ExecutionPlan,
        claimed: ClaimedRunLike,
        lease_credential_ref: str,
        fact: InstallationFact,
    ) -> None:
        self._emit_once(
            plan,
            claimed,
            lease_credential_ref,
            fact,
            "RUN_FINISHED",
            outcome="INFRA_ERROR",
        )

    def _failed(
        self,
        plan: ExecutionPlan,
        claimed: ClaimedRunLike,
        lease_credential_ref: str,
        started_at: str,
        job_id: int | None,
        apk_sha256: str | None,
        apk_size: int | None,
        metadata: ApkMetadata | None,
        message: str,
        *,
        retryable: bool = False,
    ) -> InstallationAttempt:
        instruction = plan.pre_test_install
        assert instruction is not None
        failure_key = (
            f"install-failure:{claimed.task_run_id}:{instruction.ci_build_id}:"
            f"{job_id or 'none'}:{claimed.device_uid}"
        )
        fact = InstallationFact(
            schema_version=INSTALLATION_FACT_SCHEMA,
            idempotency_key=failure_key,
            task_run_id=str(claimed.task_run_id),
            ci_build_id=str(instruction.ci_build_id),
            pipeline_id=instruction.pipeline_id,
            job_id=job_id,
            apk_sha256=apk_sha256,
            apk_size=apk_size,
            package_name=metadata.package_name if metadata else None,
            version_name=metadata.version_name if metadata else None,
            version_code=metadata.version_code if metadata else None,
            device_uid=str(claimed.device_uid),
            started_at=started_at,
            finished_at=_now(),
            install_result="FAILED",
            error_code=ExecutionErrorCode.PRE_TEST_INSTALL_FAILED.value,
        )
        self._record_fact(fact)
        self._emit_once(plan, claimed, lease_credential_ref, fact, "RUN_STARTED")
        self._emit_once(plan, claimed, lease_credential_ref, fact, "RUN_ERROR")
        logger.error(
            "Frozen APK install failed task_run_id=%s job_id=%s "
            "retryable=%s message=%s",
            claimed.task_run_id,
            job_id,
            retryable,
            message,
        )
        return InstallationAttempt(
            fact,
            ExecutionError(
                ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                message,
                retryable=retryable,
            ),
        )

    def _record_fact(self, fact: InstallationFact) -> None:
        item_id = self.outbox.enqueue(
            idempotency_key=fact.idempotency_key,
            kind="INSTALLATION_FACT",
            payload=fact.as_payload(),
            task_run_id=fact.task_run_id,
        )
        # The fact is intentionally retained in the outbox table, but it is a
        # local migration source rather than a separately sendable API item.
        self.outbox.acknowledge(item_id)

    def _emit_once(
        self,
        plan: ExecutionPlan,
        claimed: ClaimedRunLike,
        lease_credential_ref: str,
        fact: InstallationFact,
        event_type: str,
        *,
        outcome: str | None = None,
    ) -> None:
        if plan.task_type is not TaskType.TEST_RUN or plan.test_run is None:
            # Current autoglm.event.v1 makes ADHOC event data closed and has no
            # installation event type. The durable fact remains available for a
            # future dedicated platform API without corrupting ADHOC metrics.
            return
        marker = f"{fact.idempotency_key}:{event_type}"
        for event in self.outbox.events_for_run(str(claimed.task_run_id)):
            if (event.get("data") or {}).get("installation_event_key") == marker:
                return
        first_case = plan.test_run.cases[0]
        sink = OutboxLifecycleSink(
            outbox=self.outbox,
            task_run_id=claimed.task_run_id,  # type: ignore[arg-type]
            producer_id=uuid7(),
            lease_credential_ref=lease_credential_ref,
            fencing_token=claimed.fencing_token,
        )
        data: dict[str, object] = {
            "execution_case_id": str(first_case.execution_case_id),
            "installation_event_key": marker,
            "pre_test_install": fact.as_payload(),
        }
        if outcome is not None:
            data["outcome"] = outcome
        sink.emit(event_type, data)

    def _validate_local_config(self, instruction: PreTestInstall) -> None:
        if not self.config.base_url or not self.config.token:
            raise ExecutionError(
                ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                "GitLab download Secret configuration is missing",
            )
        base = urlparse(self.config.base_url)
        repository = urlparse(instruction.repository_url)
        if (
            base.scheme != "https"
            or not base.netloc
            or (base.scheme.lower(), base.netloc.lower())
            != (repository.scheme.lower(), repository.netloc.lower())
        ):
            raise ExecutionError(
                ExecutionErrorCode.PRE_TEST_INSTALL_FAILED,
                "Frozen repository origin does not match the configured GitLab origin",
            )


def _success_fact_key(
    task_run_id: str,
    ci_build_id: str,
    job_id: int,
    apk_sha256: str,
    device_uid: str,
) -> str:
    return (
        f"install:{task_run_id}:{ci_build_id}:{job_id}:"
        f"{apk_sha256.removeprefix('sha256:')}:{device_uid}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _badging_value(line: str, name: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(name)}='([^']*)'", line)
    return match.group(1) if match else None


def _adb_output(stdout: str, stderr: str) -> str:
    return "\n".join(part.strip() for part in (stdout, stderr) if part.strip())


def _safe_error(error: ExecutionError) -> str:
    # Preserve useful ADB diagnostics without allowing multiline/unbounded logs.
    message = " ".join(error.message.split())
    return message[:500] if message else error.code.value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
