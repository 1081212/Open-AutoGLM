"""Initiate, encrypt, upload and complete Artifact records without plaintext COS writes."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import requests
from pydantic import BaseModel, ConfigDict

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.worker.encryption import ArtifactAadV1, encrypt_artifact_file
from phone_agent.worker.outbox import DurableOutbox, LocalSealer, OutboxItem
from phone_agent.worker.config import RuntimeEnvironment, parse_runtime_environment


class ArtifactInitiateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: UUID
    object_key: str
    artifact_type: str
    content_type: str
    plaintext_size: int
    plaintext_sha256: str
    encryption_algorithm: str
    key_wrap_algorithm: str
    kek_version: str
    wrapped_dek: str
    nonce: str
    aad: ArtifactAadV1
    aad_sha256: str
    upload_deadline: str
    upload_url: str
    upload_url_expires_at: str
    artifact_upload_token: str
    plaintext_dek: str | None
    key_disclosed: bool


@dataclass(frozen=True, slots=True)
class ArtifactSource:
    path: Path
    artifact_type: str
    content_type: str
    project_id: UUID
    task_id: UUID
    task_run_id: UUID
    execution_case_id: UUID | None = None
    case_attempt_no: int | None = None
    case_attempt_id: UUID | None = None
    step_id: UUID | None = None
    sensitive: bool = True


class ArtifactPreparer:
    def __init__(
        self,
        api,
        outbox: DurableOutbox,
        sealer: LocalSealer,
        encrypted_root: Path,
        runtime_environment: RuntimeEnvironment,
        *,
        worker_id: UUID,
        instance_id: UUID,
        device_uid: UUID,
    ) -> None:
        self.api = api
        self.outbox = outbox
        self.sealer = sealer
        self.encrypted_root = encrypted_root
        self.runtime_environment = parse_runtime_environment(runtime_environment)
        self.worker_id = worker_id
        self.instance_id = instance_id
        self.device_uid = device_uid

    def prepare(
        self,
        source: ArtifactSource,
        *,
        initiate_idempotency_key: str,
        lease_token: str,
        fencing_token: int,
    ) -> UUID:
        plaintext_size, plaintext_sha256 = _file_identity(source.path)
        response = ArtifactInitiateResponse.model_validate(
            self.api.initiate_artifact(
                str(source.task_run_id),
                {
                    "worker_id": str(self.worker_id),
                    "instance_id": str(self.instance_id),
                    "device_uid": str(self.device_uid),
                    "lease_token": lease_token,
                    "fencing_token": fencing_token,
                    "initiate_idempotency_key": initiate_idempotency_key,
                    "artifact_type": source.artifact_type,
                    "content_type": source.content_type,
                    "plaintext_size": plaintext_size,
                    "plaintext_sha256": plaintext_sha256,
                    "execution_case_id": str(source.execution_case_id)
                    if source.execution_case_id
                    else None,
                    "case_attempt_id": str(source.case_attempt_id)
                    if source.case_attempt_id
                    else None,
                    "case_attempt_no": source.case_attempt_no,
                    "step_id": str(source.step_id) if source.step_id else None,
                    "sensitive": source.sensitive,
                },
            )
        )
        expected_object_prefix = f"mobile-automation-hub/{self.runtime_environment}/"
        if not response.object_key.startswith(expected_object_prefix):
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                "Platform returned an Artifact object key for another runtime environment",
            )
        if (
            response.artifact_type != source.artifact_type
            or response.content_type != source.content_type
            or response.plaintext_size != plaintext_size
            or response.plaintext_sha256 != plaintext_sha256
        ):
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                "Platform returned mismatched Artifact plaintext identity",
            )
        if not response.key_disclosed or response.plaintext_dek is None:
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                "Artifact initiate replay did not disclose a DEK; use a new idempotency key",
            )
        expected_aad = ArtifactAadV1(
            project_id=source.project_id,
            task_id=source.task_id,
            task_run_id=source.task_run_id,
            execution_case_id=source.execution_case_id,
            case_attempt_no=source.case_attempt_no,
            artifact_id=response.artifact_id,
            type=source.artifact_type,
            content_type=source.content_type,
            aad_version=1,
        )
        if response.aad != expected_aad:
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                "Platform returned mismatched Artifact AAD",
            )
        expected_aad_sha256 = (
            "sha256:" + hashlib.sha256(expected_aad.canonical_bytes()).hexdigest()
        )
        if response.aad_sha256 != expected_aad_sha256:
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                "Platform returned mismatched Artifact AAD hash",
            )
        try:
            dek = base64.b64decode(response.plaintext_dek, validate=True)
            nonce = base64.b64decode(response.nonce, validate=True)
        except ValueError as exc:
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                "Invalid Artifact key material encoding",
            ) from exc

        token_ref = f"artifact-token:{response.artifact_id}"
        url_ref = f"artifact-url:{response.artifact_id}"
        self.outbox.save_credential(
            token_ref, response.artifact_upload_token, self.sealer
        )
        self.outbox.save_credential(url_ref, response.upload_url, self.sealer)
        encrypted_path = self.encrypted_root / f"{response.artifact_id}.enc"
        try:
            try:
                encrypted = encrypt_artifact_file(
                    source.path,
                    encrypted_path,
                    dek=dek,
                    nonce=nonce,
                    aad=expected_aad,
                )
            except (OSError, ValueError) as exc:
                raise ExecutionError(
                    ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                    f"Artifact encryption failed: {type(exc).__name__}",
                ) from exc
        finally:
            del dek
        self.outbox.enqueue(
            idempotency_key=f"{response.artifact_id}:upload-complete",
            kind="ARTIFACT_UPLOAD",
            payload={
                "artifact_id": str(response.artifact_id),
                "url_credential_ref": url_ref,
                "ciphertext_size": encrypted.ciphertext_size,
                "ciphertext_sha256": encrypted.ciphertext_sha256,
                "plaintext_size": encrypted.plaintext_size,
                "plaintext_sha256": encrypted.plaintext_sha256,
                "auth_tag": base64.b64encode(encrypted.tag).decode("ascii"),
                "upload_deadline": response.upload_deadline,
            },
            task_run_id=str(source.task_run_id),
            artifact_upload_credential_ref=token_ref,
            fencing_token=fencing_token,
            local_path=str(encrypted.path),
        )
        return response.artifact_id


class ArtifactOutboxPump:
    def __init__(
        self, api, outbox: DurableOutbox, sealer: LocalSealer, session=None
    ) -> None:
        self.api = api
        self.outbox = outbox
        self.sealer = sealer
        self.session = session or requests.Session()

    def flush_item(self, item: OutboxItem) -> bool:
        if item.kind != "ARTIFACT_UPLOAD":
            return False
        try:
            self._upload_and_complete(item)
            self.outbox.acknowledge(item.id)
            return True
        except Exception as error:
            if getattr(error, "retryable", False) or isinstance(
                error, requests.RequestException
            ):
                self.outbox.retry(item.id, str(error))
            else:
                self.outbox.mark_failed(item.id, str(error))
            return False

    def _upload_and_complete(self, item: OutboxItem) -> None:
        if not item.local_path or not item.artifact_upload_credential_ref:
            raise ValueError(
                "Artifact outbox item is missing local path or token reference"
            )
        payload = item.payload
        url_ref = payload["url_credential_ref"]
        upload_url = self.outbox.load_credential(url_ref, self.sealer)
        token = self.outbox.load_credential(
            item.artifact_upload_credential_ref, self.sealer
        )
        path = Path(item.local_path)
        with path.open("rb") as handle:
            response = self.session.put(
                upload_url,
                data=handle,
                headers={"Content-Type": "application/octet-stream"},
                timeout=(5, 300),
            )
        if response.status_code in {401, 403}:
            refreshed = self.api.refresh_artifact_upload(payload["artifact_id"], token)
            new_url = refreshed.get("upload_url")
            if not isinstance(new_url, str) or not new_url:
                raise ExecutionError(
                    ExecutionErrorCode.ARTIFACT_UPLOAD_FAILED,
                    "Artifact upload URL refresh did not return a URL",
                )
            self.outbox.save_credential(url_ref, new_url, self.sealer)
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_UPLOAD_FAILED,
                "Artifact upload URL refreshed; retry queued",
                retryable=True,
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_UPLOAD_FAILED,
                f"COS temporary status {response.status_code}",
                retryable=True,
            )
        if not 200 <= response.status_code < 300:
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_UPLOAD_FAILED,
                f"COS rejected upload with status {response.status_code}",
            )
        self.api.complete_artifact(
            payload["artifact_id"],
            {
                "ciphertext_size": payload["ciphertext_size"],
                "ciphertext_sha256": payload["ciphertext_sha256"],
                "auth_tag": payload["auth_tag"],
            },
            token,
        )


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, "sha256:" + digest.hexdigest()
