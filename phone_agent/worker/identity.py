"""Stable worker identity and cross-language device UID derivation."""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from uuid import UUID, uuid5


AUTOGLM_DEVICE_NAMESPACE = UUID("7f0f6d4e-7d56-5c63-9b6e-4ee1d91d0a33")


def uuid7(timestamp_ms: int | None = None) -> UUID:
    """Generate an RFC 9562 UUIDv7 on Python 3.10."""
    milliseconds = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    if not 0 <= milliseconds < 1 << 48:
        raise ValueError("timestamp_ms is outside UUIDv7 range")
    random_a = secrets.randbits(12)
    random_b = secrets.randbits(62)
    value = (
        (milliseconds << 80) | (0x7 << 76) | (random_a << 64) | (0b10 << 62) | random_b
    )
    return UUID(int=value)


def derive_device_uid(worker_id: UUID | str, adb_serial: str) -> UUID:
    serial = _normalize_serial(adb_serial)
    worker = str(UUID(str(worker_id)))
    return uuid5(AUTOGLM_DEVICE_NAMESPACE, f"{worker}\x00{serial}")


def load_or_create_worker_id(
    path: str | os.PathLike[str], expected_worker_id: UUID | None = None
) -> UUID:
    identity_path = Path(path)
    identity_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(identity_path.parent, 0o700)
    except OSError:
        pass
    if identity_path.exists():
        persisted = UUID(identity_path.read_text(encoding="ascii").strip())
        if expected_worker_id is not None and persisted != expected_worker_id:
            raise ValueError(
                "AUTOGLM_WORKER_ID does not match the persisted environment Worker identity"
            )
        return persisted
    worker_id = expected_worker_id or uuid7()
    temp = identity_path.with_name(f".{identity_path.name}.{os.getpid()}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(str(worker_id) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, identity_path)
        except FileExistsError:
            persisted = UUID(identity_path.read_text(encoding="ascii").strip())
            if expected_worker_id is not None and persisted != expected_worker_id:
                raise ValueError(
                    "AUTOGLM_WORKER_ID does not match the persisted environment Worker identity"
                )
            return persisted
    finally:
        temp.unlink(missing_ok=True)
    return worker_id


def _normalize_serial(serial: str) -> str:
    if "\x00" in serial:
        raise ValueError("adb_serial must not contain NUL")
    normalized = serial.strip(" \t\r\n\f\v")
    if not normalized:
        raise ValueError("adb_serial must not be empty")
    return normalized
