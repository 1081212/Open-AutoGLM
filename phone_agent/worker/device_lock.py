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
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.chmod(self.path, 0o600)
        self._held = True
        return True

    def release(self) -> None:
        if not self._held:
            return
        self.path.unlink(missing_ok=True)
        self._lock.release()
        self._held = False

    def __enter__(self) -> "ActiveDeviceLock":
        if not self._held:
            raise RuntimeError("device lock must be acquired explicitly")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
