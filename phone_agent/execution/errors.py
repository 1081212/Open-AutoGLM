"""Explicit execution errors; stdout text is never used as a status channel."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class ErrorCategory(str, Enum):
    PLAN = "PLAN"
    CLAIM = "CLAIM"
    LEASE = "LEASE"
    WORKER = "WORKER"
    DEVICE = "DEVICE"
    MODEL = "MODEL"
    JUDGE = "JUDGE"
    ACTION = "ACTION"
    ARTIFACT = "ARTIFACT"
    OUTBOX = "OUTBOX"
    EXECUTION = "EXECUTION"
    CANCELLATION = "CANCELLATION"


class ExecutionErrorCode(str, Enum):
    PLAN_VERSION_UNSUPPORTED = "PLAN_VERSION_UNSUPPORTED"
    PLAN_HASH_MISMATCH = "PLAN_HASH_MISMATCH"
    PLAN_INVALID = "PLAN_INVALID"
    CLAIM_REJECTED = "CLAIM_REJECTED"
    LEASE_LOST = "LEASE_LOST"
    WORKER_RESTARTED = "WORKER_RESTARTED"
    WORKER_LOST = "WORKER_LOST"
    DEVICE_LOST = "DEVICE_LOST"
    DEVICE_NOT_FOUND = "DEVICE_NOT_FOUND"
    DEVICE_DISCONNECTED = "DEVICE_DISCONNECTED"
    DEVICE_COMMAND_TIMEOUT = "DEVICE_COMMAND_TIMEOUT"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_ERROR = "MODEL_ERROR"
    JUDGE_ERROR = "JUDGE_ERROR"
    ACTION_ERROR = "ACTION_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    RETRYABLE_ERROR = "RETRYABLE_ERROR"
    CASE_ERROR = "CASE_ERROR"
    ARTIFACT_ENCRYPT_FAILED = "ARTIFACT_ENCRYPT_FAILED"
    ARTIFACT_UPLOAD_FAILED = "ARTIFACT_UPLOAD_FAILED"
    PRE_TEST_INSTALL_FAILED = "PRE_TEST_INSTALL_FAILED"
    OUTBOX_FULL = "OUTBOX_FULL"
    CANCELLED = "CANCELLED"
    TASK_TIMEOUT = "TASK_TIMEOUT"


_CATEGORY_BY_CODE = {
    ExecutionErrorCode.PLAN_VERSION_UNSUPPORTED: ErrorCategory.PLAN,
    ExecutionErrorCode.PLAN_HASH_MISMATCH: ErrorCategory.PLAN,
    ExecutionErrorCode.PLAN_INVALID: ErrorCategory.PLAN,
    ExecutionErrorCode.CLAIM_REJECTED: ErrorCategory.CLAIM,
    ExecutionErrorCode.LEASE_LOST: ErrorCategory.LEASE,
    ExecutionErrorCode.WORKER_RESTARTED: ErrorCategory.WORKER,
    ExecutionErrorCode.WORKER_LOST: ErrorCategory.WORKER,
    ExecutionErrorCode.DEVICE_LOST: ErrorCategory.DEVICE,
    ExecutionErrorCode.DEVICE_NOT_FOUND: ErrorCategory.DEVICE,
    ExecutionErrorCode.DEVICE_DISCONNECTED: ErrorCategory.DEVICE,
    ExecutionErrorCode.DEVICE_COMMAND_TIMEOUT: ErrorCategory.DEVICE,
    ExecutionErrorCode.MODEL_TIMEOUT: ErrorCategory.MODEL,
    ExecutionErrorCode.MODEL_ERROR: ErrorCategory.MODEL,
    ExecutionErrorCode.JUDGE_ERROR: ErrorCategory.JUDGE,
    ExecutionErrorCode.ACTION_ERROR: ErrorCategory.ACTION,
    ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED: ErrorCategory.ARTIFACT,
    ExecutionErrorCode.ARTIFACT_UPLOAD_FAILED: ErrorCategory.ARTIFACT,
    ExecutionErrorCode.PRE_TEST_INSTALL_FAILED: ErrorCategory.EXECUTION,
    ExecutionErrorCode.OUTBOX_FULL: ErrorCategory.OUTBOX,
    ExecutionErrorCode.CANCELLED: ErrorCategory.CANCELLATION,
    ExecutionErrorCode.TASK_TIMEOUT: ErrorCategory.EXECUTION,
}


@dataclass(slots=True)
class ExecutionError(Exception):
    code: ExecutionErrorCode
    message: str
    retryable: bool = False
    details_artifact_id: str | None = None

    @property
    def category(self) -> ErrorCategory:
        return _CATEGORY_BY_CODE.get(self.code, ErrorCategory.EXECUTION)

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "category": self.category.value,
            "message": self.message,
            "retryable": self.retryable,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "details_artifact_id": self.details_artifact_id,
        }

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"
