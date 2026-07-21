from __future__ import annotations

import os
from uuid import uuid4

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from phone_agent.worker.encryption import ArtifactAadV1, encrypt_artifact_file


def _aad():
    return ArtifactAadV1(
        project_id=uuid4(),
        task_id=uuid4(),
        task_run_id=uuid4(),
        execution_case_id=uuid4(),
        case_attempt_no=1,
        artifact_id=uuid4(),
        type="SCREENSHOT",
        content_type="image/png",
        aad_version=1,
    )


def _decrypt(ciphertext, key, nonce, tag, aad):
    decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
    decryptor.authenticate_additional_data(aad.canonical_bytes())
    return decryptor.update(ciphertext) + decryptor.finalize()


def test_encrypt_artifact_round_trip(tmp_path):
    plaintext = b"sensitive screenshot bytes" * 100
    source = tmp_path / "screen.png"
    target = tmp_path / "encrypted" / "artifact.enc"
    source.write_bytes(plaintext)
    key = os.urandom(32)
    nonce = os.urandom(12)
    aad = _aad()

    result = encrypt_artifact_file(source, target, dek=key, nonce=nonce, aad=aad)

    assert target.read_bytes() != plaintext
    assert _decrypt(target.read_bytes(), key, nonce, result.tag, aad) == plaintext
    assert result.plaintext_size == len(plaintext)
    assert result.ciphertext_size == len(plaintext)


def test_aad_tampering_fails_authentication(tmp_path):
    source = tmp_path / "data"
    target = tmp_path / "data.enc"
    source.write_bytes(b"secret")
    key = os.urandom(32)
    nonce = os.urandom(12)
    aad = _aad()
    result = encrypt_artifact_file(source, target, dek=key, nonce=nonce, aad=aad)
    tampered = aad.model_copy(update={"type": "LOG"})

    with pytest.raises(InvalidTag):
        _decrypt(target.read_bytes(), key, nonce, result.tag, tampered)
