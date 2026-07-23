"""Bounded, terminal-state-aware cleanup for the environment Worker spool."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from phone_agent.worker.outbox import DurableOutbox, TerminalTaskRun
from phone_agent.worker.time_utils import parse_aware_iso8601

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SpoolGcResult:
    deleted_runs: tuple[str, ...]
    spool_bytes: int
    free_bytes: int
    can_claim: bool


class SpoolGarbageCollector:
    def __init__(
        self,
        *,
        spool_root: Path,
        outbox: DurableOutbox,
        retention_days: int,
        max_bytes: int,
        min_free_bytes: int,
    ) -> None:
        self.spool_root = spool_root.absolute()
        self.outbox = outbox
        self.retention = timedelta(days=retention_days)
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.trash_root = self.spool_root / "gc-trash"

    def collect(self, *, now: datetime | None = None) -> SpoolGcResult:
        """Apply age retention first, then oldest-first disk-pressure cleanup."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("GC current time must include a timezone")
        self._clean_abandoned_trash()
        terminal_runs = list(self.outbox.terminal_task_runs())
        deleted: list[str] = []

        for run in tuple(terminal_runs):
            updated_at = parse_aware_iso8601(run.updated_at, "task_run.updated_at")
            if now - updated_at >= self.retention and self._purge_run(run):
                deleted.append(run.task_run_id)
                terminal_runs.remove(run)

        spool_bytes = self._tree_size(self.spool_root)
        free_bytes = shutil.disk_usage(self.spool_root).free
        while (
            spool_bytes > self.max_bytes or free_bytes < self.min_free_bytes
        ) and terminal_runs:
            run = terminal_runs.pop(0)
            if self._purge_run(run):
                deleted.append(run.task_run_id)
            spool_bytes = self._tree_size(self.spool_root)
            free_bytes = shutil.disk_usage(self.spool_root).free

        if deleted:
            self.outbox.compact()
            spool_bytes = self._tree_size(self.spool_root)
            free_bytes = shutil.disk_usage(self.spool_root).free
        can_claim = spool_bytes <= self.max_bytes and free_bytes >= self.min_free_bytes
        if deleted or not can_claim:
            logger.info(
                "Worker spool GC completed deleted_runs=%d spool_bytes=%d "
                "free_bytes=%d can_claim=%s",
                len(deleted),
                spool_bytes,
                free_bytes,
                can_claim,
            )
        return SpoolGcResult(tuple(deleted), spool_bytes, free_bytes, can_claim)

    def _purge_run(self, run: TerminalTaskRun) -> bool:
        staging = self.trash_root / f"{run.task_run_id}-{uuid4().hex}.tmp"
        moved_any = False
        try:
            for parent_name in ("task-runs", "local-reports"):
                source = self.spool_root / parent_name / run.task_run_id
                if not source.exists() and not source.is_symlink():
                    continue
                self._validate_deletion_target(source, parent_name, run.task_run_id)
                staging.mkdir(parents=True, exist_ok=True, mode=0o700)
                target_parent = staging / parent_name
                target_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.replace(source, target_parent / run.task_run_id)
                moved_any = True
            if not self.outbox.purge_terminal_task_run(run.task_run_id):
                logger.warning(
                    "Spool GC skipped non-terminal Run after filesystem staging "
                    "task_run_id=%s",
                    run.task_run_id,
                )
                self._restore_staged(staging, run.task_run_id)
                return False
            if moved_any:
                shutil.rmtree(staging)
            logger.info(
                "Worker terminal Run removed from spool task_run_id=%s state=%s",
                run.task_run_id,
                run.state,
            )
            return True
        except Exception as error:
            logger.warning(
                "Worker spool GC failed task_run_id=%s error_type=%s",
                run.task_run_id,
                type(error).__name__,
            )
            self._restore_staged(staging, run.task_run_id)
            return False

    def _validate_deletion_target(
        self, path: Path, parent_name: str, task_run_id: str
    ) -> None:
        expected = self.spool_root / parent_name / task_run_id
        if path != expected or path.is_symlink():
            raise OSError("unsafe GC deletion target")
        if path.parent.resolve() != (self.spool_root / parent_name).resolve():
            raise OSError("GC deletion target escapes spool root")

    def _restore_staged(self, staging: Path, task_run_id: str) -> None:
        if not staging.exists() or staging.is_symlink():
            return
        for parent_name in ("task-runs", "local-reports"):
            staged = staging / parent_name / task_run_id
            destination = self.spool_root / parent_name / task_run_id
            if staged.exists() and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.replace(staged, destination)
        try:
            shutil.rmtree(staging)
        except OSError:
            pass

    def _clean_abandoned_trash(self) -> None:
        if not self.trash_root.exists():
            return
        if self.trash_root.is_symlink():
            logger.warning("Refusing to clean symlinked Worker GC trash directory")
            return
        for path in self.trash_root.iterdir():
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    @staticmethod
    def _tree_size(root: Path) -> int:
        total = 0
        for directory, _names, filenames in os.walk(root, followlinks=False):
            for filename in filenames:
                path = Path(directory) / filename
                try:
                    if not path.is_symlink():
                        total += path.stat().st_size
                except FileNotFoundError:
                    continue
        return total
