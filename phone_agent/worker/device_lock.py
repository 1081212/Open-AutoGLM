"""Cross-process exclusive lock for the currently active Android device."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from filelock import FileLock, Timeout


class ActiveDeviceLock:
    def __init__(self, lock_dir: Path, device_uid: UUID | str) -> None:
        lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = lock_dir / f"{device_uid}.lock"
        self._lock = FileLock(str(self.path) + ".guard")
        self._held = False
        self._payload: dict[str, object] | None = None

    def acquire(
        self,
        *,
        worker_id: UUID,
        instance_id: UUID,
        adb_serial: str,
        task_run_id: UUID | None = None,
        timeout: float = 0,
    ) -> bool:
        try:
            self._lock.acquire(timeout=timeout)
        except Timeout:
            return False
        payload = {
            "worker_id": str(worker_id),
            "instance_id": str(instance_id),
            "device_uid": self.path.stem,
            "adb_serial": adb_serial,
            "task_run_id": str(task_run_id) if task_run_id else None,
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_payload(payload)
        self._payload = payload
        self._held = True
        return True

    def bind_task_run(self, task_run_id: UUID | str) -> None:
        if not self._held or self._payload is None:
            raise RuntimeError("device lock is not held")
        payload = {**self._payload, "task_run_id": str(task_run_id)}
        self._write_payload(payload)
        self._payload = payload

    def assert_held(
        self,
        *,
        worker_id: UUID | str,
        instance_id: UUID | str,
        device_uid: UUID | str,
        adb_serial: str,
        task_run_id: UUID | str,
    ) -> None:
        if not self._held or self._payload is None or not self.path.is_file():
            raise RuntimeError("active device lock is no longer held")
        try:
            persisted = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("active device lock metadata is unreadable") from exc
        expected = {
            "worker_id": str(worker_id),
            "instance_id": str(instance_id),
            "device_uid": str(device_uid),
            "adb_serial": adb_serial,
            "task_run_id": str(task_run_id),
        }
        if any(persisted.get(name) != value for name, value in expected.items()):
            raise RuntimeError("active device lock binding changed")

    def _write_payload(self, payload: dict[str, object]) -> None:
        part = self.path.with_suffix(self.path.suffix + ".part")
        try:
            with part.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(part, 0o600)
            os.replace(part, self.path)
        finally:
            part.unlink(missing_ok=True)

    def release(self) -> None:
        if not self._held:
            return
        self.path.unlink(missing_ok=True)
        self._lock.release()
        self._held = False
        self._payload = None

    def __enter__(self) -> "ActiveDeviceLock":
        if not self._held:
            raise RuntimeError("device lock must be acquired explicitly")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
