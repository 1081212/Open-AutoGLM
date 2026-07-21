"""AES-256-GCM artifact encryption for public-readable COS buckets."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from pydantic import BaseModel, ConfigDict, Field


class ArtifactAadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: UUID
    task_id: UUID
    task_run_id: UUID
    execution_case_id: UUID | None
    case_attempt_no: int | None = Field(default=None, ge=1)
    artifact_id: UUID
    type: str
    content_type: str
    aad_version: int = Field(default=1, ge=1, le=1)

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class EncryptedArtifact:
    path: Path
    plaintext_size: int
    ciphertext_size: int
    plaintext_sha256: str
    ciphertext_sha256: str
    tag: bytes


def encrypt_artifact_file(
    plaintext_path: str | os.PathLike[str],
    encrypted_path: str | os.PathLike[str],
    *,
    dek: bytes,
    nonce: bytes,
    aad: ArtifactAadV1,
    chunk_size: int = 1024 * 1024,
) -> EncryptedArtifact:
    if len(dek) != 32:
        raise ValueError("Artifact DEK must be exactly 256 bits")
    if len(nonce) != 12:
        raise ValueError("Artifact GCM nonce must be exactly 96 bits")
    source_path = Path(plaintext_path)
    target_path = Path(encrypted_path)
    target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    part_path = target_path.with_name(target_path.name + ".part")
    plaintext_hash = hashlib.sha256()
    ciphertext_hash = hashlib.sha256()
    plaintext_size = 0
    ciphertext_size = 0
    encryptor = Cipher(algorithms.AES(dek), modes.GCM(nonce)).encryptor()
    encryptor.authenticate_additional_data(aad.canonical_bytes())
    try:
        with source_path.open("rb") as source, part_path.open("wb") as target:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                plaintext_hash.update(chunk)
                plaintext_size += len(chunk)
                encrypted = encryptor.update(chunk)
                ciphertext_hash.update(encrypted)
                ciphertext_size += len(encrypted)
                target.write(encrypted)
            tail = encryptor.finalize()
            if tail:
                ciphertext_hash.update(tail)
                ciphertext_size += len(tail)
                target.write(tail)
            target.flush()
            os.fsync(target.fileno())
        os.replace(part_path, target_path)
    finally:
        part_path.unlink(missing_ok=True)
    return EncryptedArtifact(
        path=target_path,
        plaintext_size=plaintext_size,
        ciphertext_size=ciphertext_size,
        plaintext_sha256="sha256:" + plaintext_hash.hexdigest(),
        ciphertext_sha256="sha256:" + ciphertext_hash.hexdigest(),
        tag=encryptor.tag,
    )
