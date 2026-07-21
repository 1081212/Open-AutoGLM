"""Structured execution primitives shared by the CLI and platform worker."""

from phone_agent.execution.cancellation import CancellationToken
from phone_agent.execution.errors import ErrorCategory, ExecutionError, ExecutionErrorCode
from phone_agent.execution.models import ExecutionPlan, TaskType
from phone_agent.execution.result import AttemptResult, CaseExecutionResult, TaskExecutionResult
from phone_agent.execution.task_executor import TaskExecutor

__all__ = [
    "AttemptResult",
    "CancellationToken",
    "CaseExecutionResult",
    "ErrorCategory",
    "ExecutionError",
    "ExecutionErrorCode",
    "ExecutionPlan",
    "TaskExecutionResult",
    "TaskExecutor",
    "TaskType",
]
