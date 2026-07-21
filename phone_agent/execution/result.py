"""Structured results returned by TaskExecutor."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID

from phone_agent.execution.errors import ExecutionError
from phone_agent.execution.models import CaseOutcome


class TaskOutcome(str, Enum):
    PASS = "PASS"
    PASS_WITH_FLAKY = "PASS_WITH_FLAKY"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    REVIEW = "REVIEW"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    WORKER_LOST = "WORKER_LOST"
    DEVICE_LOST = "DEVICE_LOST"
    INFRA_ERROR = "INFRA_ERROR"


@dataclass(frozen=True, slots=True)
class AttemptResult:
    outcome: CaseOutcome
    message: str
    error: ExecutionError | None = None


@dataclass(frozen=True, slots=True)
class CaseExecutionResult:
    execution_case_id: UUID
    ordinal: int
    outcome: CaseOutcome
    flaky: bool
    attempts: tuple[AttemptResult, ...]


@dataclass(frozen=True, slots=True)
class TaskExecutionResult:
    task_id: UUID
    outcome: TaskOutcome
    started_at: datetime
    finished_at: datetime
    cases: tuple[CaseExecutionResult, ...] = ()
    adhoc_result: AttemptResult | None = None
    error: ExecutionError | None = None
    not_run_execution_case_ids: tuple[UUID, ...] = field(default_factory=tuple)

    @classmethod
    def now(cls, **kwargs):
        return cls(finished_at=datetime.now(timezone.utc), **kwargs)
