from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from phone_agent.worker.identity import (
    derive_device_uid,
    load_or_create_worker_id,
    uuid7,
)


def test_device_uid_golden_vector():
    actual = derive_device_uid(
        "018f47a0-7b5c-7a05-8c00-000000000005",
        "SJT0220804008129",
    )
    assert str(actual) == "66a655c2-01c6-5d89-bc0d-0f337fb139ea"


def test_device_uid_preserves_serial_case():
    worker_id = UUID("018f47a0-7b5c-7a05-8c00-000000000005")
    assert derive_device_uid(worker_id, "Serial") != derive_device_uid(
        worker_id, "serial"
    )


def test_device_uid_rejects_nul():
    with pytest.raises(ValueError, match="NUL"):
        derive_device_uid(uuid7(), "bad\x00serial")


def test_worker_id_is_stable_and_uuid7(tmp_path):
    path = tmp_path / "identity" / "worker-id"
    first = load_or_create_worker_id(path)
    second = load_or_create_worker_id(path)
    assert first == second
    assert first.version == 7
    assert path.stat().st_mode & 0o777 == 0o600


def test_configured_worker_id_is_persisted_and_mismatch_is_rejected(tmp_path):
    path = tmp_path / "identity" / "worker-id"
    configured = uuid4()

    assert load_or_create_worker_id(path, configured) == configured
    with pytest.raises(ValueError, match="does not match"):
        load_or_create_worker_id(path, uuid4())
