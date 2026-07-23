"""Crash-safe immutable plan spool with compressed and canonical verification."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.execution.models import ExecutionPlan, TaskType

MAX_COMPRESSED_SIZE = 16 * 1024 * 1024
MAX_CANONICAL_SIZE = 64 * 1024 * 1024
MAX_CASE_COUNT = 1000


@dataclass(frozen=True, slots=True)
class PlanDescriptor:
    plan_id: str
    compressed_sha256: str
    compressed_size: int
    canonical_sha256: str
    canonical_size: int
    item_count: int
    case_count: int


@dataclass(frozen=True, slots=True)
class SpoolPlan:
    root: Path
    compressed_path: Path
    canonical_path: Path
    plan: ExecutionPlan


class PlanSpool:
    def __init__(self, spool_root: str | os.PathLike[str]) -> None:
        self.spool_root = Path(spool_root)
        self.spool_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.spool_root, 0o700)
        except OSError:
            pass

    def store(
        self,
        task_run_id: str,
        compressed_chunks,
        descriptor: PlanDescriptor,
    ) -> SpoolPlan:
        self._validate_descriptor(descriptor)
        run_root = self.spool_root / "task-runs" / task_run_id
        run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        compressed_part = run_root / "plan.json.gz.part"
        compressed_path = run_root / "plan.json.gz"
        canonical_part = run_root / "plan.json.part"
        canonical_path = run_root / "plan.json"

        compressed_hash = hashlib.sha256()
        compressed_size = 0
        try:
            with compressed_part.open("wb") as handle:
                for chunk in compressed_chunks:
                    if not chunk:
                        continue
                    compressed_size += len(chunk)
                    if (
                        compressed_size > descriptor.compressed_size
                        or compressed_size > MAX_COMPRESSED_SIZE
                    ):
                        raise self._invalid(
                            "compressed plan exceeds declared or configured size"
                        )
                    compressed_hash.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            _verify_size_hash(
                "compressed",
                compressed_size,
                compressed_hash.hexdigest(),
                descriptor.compressed_size,
                descriptor.compressed_sha256,
            )
            os.replace(compressed_part, compressed_path)

            canonical_hash = hashlib.sha256()
            canonical_size = 0
            with gzip.open(compressed_path, "rb") as source, canonical_part.open(
                "wb"
            ) as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    canonical_size += len(chunk)
                    if (
                        canonical_size > descriptor.canonical_size
                        or canonical_size > MAX_CANONICAL_SIZE
                    ):
                        raise self._invalid(
                            "canonical plan exceeds declared or configured size"
                        )
                    canonical_hash.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            _verify_size_hash(
                "canonical",
                canonical_size,
                canonical_hash.hexdigest(),
                descriptor.canonical_size,
                descriptor.canonical_sha256,
            )
            try:
                raw = json.loads(canonical_part.read_bytes())
                plan = ExecutionPlan.model_validate(raw)
            except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
                raise self._invalid(f"plan schema validation failed: {exc}") from exc
            self._validate_counts(plan, descriptor)
            if plan.target_requirements.device_type != "adb":
                raise self._invalid("platform Worker supports adb plans only")
            if str(plan.plan_id) != descriptor.plan_id:
                raise self._invalid("plan_id does not match claim descriptor")
            os.replace(canonical_part, canonical_path)
            return SpoolPlan(run_root, compressed_path, canonical_path, plan)
        except ExecutionError:
            raise
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise self._invalid(f"plan download/decompression failed: {exc}") from exc
        finally:
            compressed_part.unlink(missing_ok=True)
            canonical_part.unlink(missing_ok=True)

    def write_claim_metadata(
        self, task_run_id: str, metadata: dict[str, object]
    ) -> Path:
        forbidden = {
            "lease_token",
            "artifact_upload_token",
            "authorization",
            "plaintext_dek",
        }
        if forbidden.intersection(key.lower() for key in metadata):
            raise ValueError("claim.json must not contain bearer credentials")
        path = self.spool_root / "task-runs" / task_run_id / "claim.json"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _atomic_write(path, json.dumps(metadata, ensure_ascii=False, indent=2).encode())
        return path

    @staticmethod
    def _validate_descriptor(descriptor: PlanDescriptor) -> None:
        if (
            descriptor.compressed_size < 0
            or descriptor.compressed_size > MAX_COMPRESSED_SIZE
        ):
            raise ExecutionError(
                ExecutionErrorCode.PLAN_INVALID,
                "compressed_size is outside Worker limit",
            )
        if (
            descriptor.canonical_size < 0
            or descriptor.canonical_size > MAX_CANONICAL_SIZE
        ):
            raise ExecutionError(
                ExecutionErrorCode.PLAN_INVALID,
                "canonical_size is outside Worker limit",
            )
        if not 0 <= descriptor.case_count <= MAX_CASE_COUNT:
            raise ExecutionError(
                ExecutionErrorCode.PLAN_INVALID, "case_count is outside Worker limit"
            )

    @staticmethod
    def _validate_counts(plan: ExecutionPlan, descriptor: PlanDescriptor) -> None:
        if plan.task_type is TaskType.TEST_RUN:
            assert plan.test_run is not None
            actual_cases = len(plan.test_run.cases)
            actual_items = actual_cases
        else:
            actual_cases = 0
            actual_items = 1
        if (actual_cases, actual_items) != (
            descriptor.case_count,
            descriptor.item_count,
        ):
            raise ExecutionError(
                ExecutionErrorCode.PLAN_INVALID, "plan item/case count mismatch"
            )

    @staticmethod
    def _invalid(message: str) -> ExecutionError:
        return ExecutionError(ExecutionErrorCode.PLAN_INVALID, message, retryable=False)


def _verify_size_hash(
    label: str,
    actual_size: int,
    actual_hash: str,
    expected_size: int,
    expected_hash: str,
) -> None:
    expected = expected_hash.removeprefix("sha256:").lower()
    if actual_size != expected_size:
        raise ExecutionError(
            ExecutionErrorCode.PLAN_HASH_MISMATCH, f"{label} size mismatch"
        )
    if actual_hash.lower() != expected:
        raise ExecutionError(
            ExecutionErrorCode.PLAN_HASH_MISMATCH, f"{label} SHA-256 mismatch"
        )


def _atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
