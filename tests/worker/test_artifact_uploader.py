from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from uuid import uuid4

import pytest

from phone_agent.execution.errors import ExecutionError
from phone_agent.worker.artifact_uploader import (
    ArtifactOutboxPump,
    ArtifactPreparer,
    ArtifactSource,
)
from phone_agent.worker.encryption import ArtifactAadV1
from phone_agent.worker.outbox import DurableOutbox, LocalSealer


class FakeApi:
    def __init__(
        self,
        source: ArtifactSource,
        *,
        disclose_dek: bool = True,
        bad_aad: bool = False,
    ):
        self.source = source
        self.artifact_id = uuid4()
        self.disclose_dek = disclose_dek
        self.bad_aad = bad_aad
        self.initiate_payload = None
        self.refreshed = []
        self.completed = []

    def initiate_artifact(self, task_run_id, payload):
        self.initiate_payload = payload
        aad = ArtifactAadV1(
            project_id=uuid4() if self.bad_aad else self.source.project_id,
            task_id=self.source.task_id,
            task_run_id=self.source.task_run_id,
            execution_case_id=self.source.execution_case_id,
            case_attempt_no=self.source.case_attempt_no,
            artifact_id=self.artifact_id,
            type=self.source.artifact_type,
            content_type=self.source.content_type,
        )
        aad_sha256 = "sha256:" + hashlib.sha256(aad.canonical_bytes()).hexdigest()
        return {
            "artifact_id": str(self.artifact_id),
            "object_key": f"mobile-automation-hub/dev/artifacts/{self.artifact_id}",
            "artifact_type": self.source.artifact_type,
            "content_type": self.source.content_type,
            "plaintext_size": self.source.path.stat().st_size,
            "plaintext_sha256": "sha256:"
            + hashlib.sha256(self.source.path.read_bytes()).hexdigest(),
            "encryption_algorithm": "AES-256-GCM",
            "key_wrap_algorithm": "AES-KW",
            "kek_version": "local-v1",
            "wrapped_dek": base64.b64encode(b"wrapped").decode(),
            "nonce": base64.b64encode(b"n" * 12).decode(),
            "aad": aad.model_dump(mode="json"),
            "aad_sha256": aad_sha256,
            "upload_deadline": "2030-01-01T00:00:00Z",
            "upload_url": "https://cos.example/first",
            "upload_url_expires_at": "2030-01-01T00:00:00Z",
            "artifact_upload_token": "artifact-secret-token",
            "plaintext_dek": (
                base64.b64encode(b"d" * 32).decode() if self.disclose_dek else None
            ),
            "key_disclosed": self.disclose_dek,
        }

    def refresh_artifact_upload(self, artifact_id, token):
        self.refreshed.append((artifact_id, token))
        return {"upload_url": "https://cos.example/refreshed"}

    def complete_artifact(self, artifact_id, payload, token):
        self.completed.append((artifact_id, payload, token))
        return {}


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeSession:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.puts = []

    def put(self, url, *, data, headers, timeout):
        body = data.read()
        self.puts.append((url, body, headers, timeout))
        return FakeResponse(next(self.statuses))


def make_source(tmp_path):
    path = tmp_path / "sensitive.png"
    path.write_bytes(b"private-screenshot-contents")
    return ArtifactSource(
        path=path,
        artifact_type="SCREENSHOT",
        content_type="image/png",
        project_id=uuid4(),
        task_id=uuid4(),
        task_run_id=uuid4(),
        execution_case_id=uuid4(),
        case_attempt_no=1,
        case_attempt_id=uuid4(),
        step_id=uuid4(),
    )


def prepare(tmp_path, *, disclose_dek=True, bad_aad=False):
    source = make_source(tmp_path)
    api = FakeApi(source, disclose_dek=disclose_dek, bad_aad=bad_aad)
    outbox = DurableOutbox(tmp_path / "worker.db")
    sealer = LocalSealer(tmp_path / "sealing-key")
    preparer = ArtifactPreparer(
        api,
        outbox,
        sealer,
        tmp_path / "encrypted",
        "dev",
        worker_id=uuid4(),
        instance_id=uuid4(),
        device_uid=uuid4(),
    )
    artifact_id = preparer.prepare(
        source,
        initiate_idempotency_key="attempt:step:screenshot",
        lease_token="lease-secret-token",
        fencing_token=7,
    )
    return source, api, outbox, sealer, artifact_id


def test_prepare_encrypts_and_keeps_credentials_out_of_payload(tmp_path):
    source, api, outbox, _sealer, artifact_id = prepare(tmp_path)
    item = outbox.due()[0]

    assert artifact_id == api.artifact_id
    assert item.kind == "ARTIFACT_UPLOAD"
    assert item.local_path
    ciphertext = Path(item.local_path).read_bytes()
    assert ciphertext != source.path.read_bytes()
    assert b"private-screenshot-contents" not in ciphertext
    serialized_payload = str(item.payload)
    assert "artifact-secret-token" not in serialized_payload
    assert "lease-secret-token" not in serialized_payload
    assert "https://cos.example" not in serialized_payload
    assert item.artifact_upload_credential_ref
    assert api.initiate_payload["initiate_idempotency_key"] == "attempt:step:screenshot"
    assert api.initiate_payload["execution_case_id"] == str(source.execution_case_id)
    assert api.initiate_payload["case_attempt_id"] == str(source.case_attempt_id)
    assert api.initiate_payload["case_attempt_no"] == source.case_attempt_no
    assert api.initiate_payload["step_id"] == str(source.step_id)
    assert set(api.initiate_payload) == {
        "worker_id",
        "instance_id",
        "device_uid",
        "lease_token",
        "fencing_token",
        "initiate_idempotency_key",
        "artifact_type",
        "content_type",
        "plaintext_size",
        "plaintext_sha256",
        "execution_case_id",
        "case_attempt_id",
        "case_attempt_no",
        "step_id",
        "sensitive",
    }


def test_prepare_rejects_platform_aad_mismatch(tmp_path):
    with pytest.raises(ExecutionError, match="mismatched Artifact AAD"):
        prepare(tmp_path, bad_aad=True)


def test_prepare_rejects_cross_environment_object_key(tmp_path):
    source = make_source(tmp_path)
    api = FakeApi(source)
    original = api.initiate_artifact

    def wrong_environment(task_run_id, payload):
        response = original(task_run_id, payload)
        response["object_key"] = (
            f"mobile-automation-hub/prod/artifacts/{api.artifact_id}"
        )
        return response

    api.initiate_artifact = wrong_environment
    outbox = DurableOutbox(tmp_path / "worker.db")
    sealer = LocalSealer(tmp_path / "sealing-key")
    preparer = ArtifactPreparer(
        api,
        outbox,
        sealer,
        tmp_path / "encrypted",
        "dev",
        worker_id=uuid4(),
        instance_id=uuid4(),
        device_uid=uuid4(),
    )

    with pytest.raises(ExecutionError, match="another runtime environment"):
        preparer.prepare(
            source,
            initiate_idempotency_key="cross-env",
            lease_token="lease",
            fencing_token=1,
        )


def test_initiate_replay_without_dek_requires_new_idempotency_key(tmp_path):
    with pytest.raises(ExecutionError, match="new idempotency key"):
        prepare(tmp_path, disclose_dek=False)


def test_upload_then_complete_uses_ciphertext_and_scoped_token(tmp_path):
    source, api, outbox, sealer, _ = prepare(tmp_path)
    session = FakeSession([200])
    pump = ArtifactOutboxPump(api, outbox, sealer, session=session)
    ciphertext_path = Path(outbox.due()[0].local_path)

    assert pump.flush_item(outbox.due()[0]) is True
    assert outbox.pending_count() == 0
    assert not ciphertext_path.exists()
    assert session.puts[0][0] == "https://cos.example/first"
    assert session.puts[0][1] != source.path.read_bytes()
    assert api.completed[0][2] == "artifact-secret-token"
    assert set(api.completed[0][1]) == {
        "ciphertext_size",
        "ciphertext_sha256",
        "auth_tag",
    }


def test_403_refreshes_url_and_queues_retry(tmp_path):
    _source, api, outbox, sealer, _ = prepare(tmp_path)
    first_session = FakeSession([403])
    pump = ArtifactOutboxPump(api, outbox, sealer, session=first_session)
    item = outbox.due()[0]

    assert pump.flush_item(item) is False
    assert api.refreshed == [(str(api.artifact_id), "artifact-secret-token")]
    assert outbox.load_credential(item.payload["url_credential_ref"], sealer) == (
        "https://cos.example/refreshed"
    )
