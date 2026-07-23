"""Sequential, retry-safe execution orchestration independent of CLI parsing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Protocol

from phone_agent.execution.cancellation import CancellationToken
from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.execution.lifecycle import LifecycleSink, NullLifecycleSink
from phone_agent.execution.models import (
    CaseOutcome,
    ExecutionCase,
    ExecutionPlan,
    TaskType,
)
from phone_agent.execution.result import (
    AttemptResult,
    CaseExecutionResult,
    TaskExecutionResult,
    TaskOutcome,
)


class AttemptRunner(Protocol):
    def __call__(self, case: ExecutionCase, attempt_no: int) -> AttemptResult: ...


class AdhocRunner(Protocol):
    def __call__(self, prompt: str) -> AttemptResult: ...


_TASK_FATAL = {
    CaseOutcome.CANCELLED,
    CaseOutcome.WORKER_LOST,
    CaseOutcome.DEVICE_LOST,
    CaseOutcome.TIMED_OUT,
}


class TaskExecutor:
    """Execute a validated immutable plan without parsing Markdown."""

    def __init__(
        self,
        *,
        attempt_runner: AttemptRunner | None = None,
        adhoc_runner: AdhocRunner | None = None,
        cancellation_token: CancellationToken | None = None,
        lifecycle_sink: LifecycleSink | None = None,
        before_case: Callable[[ExecutionCase], None] | None = None,
        after_case: Callable[[ExecutionCase], None] | None = None,
        on_case_finished: (
            Callable[[ExecutionCase, CaseExecutionResult], None] | None
        ) = None,
        emit_run_started: bool = True,
        bind_test_run_boundaries: bool = False,
    ) -> None:
        self.attempt_runner = attempt_runner
        self.adhoc_runner = adhoc_runner
        self.cancellation_token = cancellation_token or CancellationToken()
        self.lifecycle_sink = lifecycle_sink or NullLifecycleSink()
        self.before_case = before_case
        self.after_case = after_case
        self.on_case_finished = on_case_finished
        self.emit_run_started = emit_run_started
        self.bind_test_run_boundaries = bind_test_run_boundaries

    def execute(self, plan: ExecutionPlan) -> TaskExecutionResult:
        started_at = datetime.now(timezone.utc)
        execution_item_id = (
            str(plan.adhoc.execution_item_id)
            if plan.task_type is TaskType.ADHOC and plan.adhoc
            else None
        )
        boundary_execution_case_id = None
        if (
            self.bind_test_run_boundaries
            and plan.task_type is TaskType.TEST_RUN
            and plan.test_run
        ):
            boundary_execution_case_id = str(plan.test_run.cases[0].execution_case_id)
        if self.emit_run_started:
            self.lifecycle_sink.emit(
                "RUN_STARTED",
                {
                    "task_id": str(plan.task_id),
                    "plan_id": str(plan.plan_id),
                    "task_type": plan.task_type.value,
                    **(
                        {"execution_case_id": boundary_execution_case_id}
                        if boundary_execution_case_id
                        else {}
                    ),
                    **(
                        {"execution_item_id": execution_item_id}
                        if execution_item_id
                        else {}
                    ),
                },
            )
        try:
            if plan.task_type is TaskType.ADHOC:
                result = self._execute_adhoc(plan, started_at)
            else:
                result = self._execute_test_run(plan, started_at)
        except ExecutionError as error:
            outcome = _task_outcome_from_error(error)
            result = TaskExecutionResult.now(
                task_id=plan.task_id,
                outcome=outcome,
                started_at=started_at,
                error=error,
            )
        if result.error is not None:
            self.lifecycle_sink.emit(
                "RUN_ERROR",
                {
                    "task_id": str(plan.task_id),
                    "error_code": result.error.code.value,
                    "error_category": result.error.category.value,
                    "message": result.error.message,
                    "retryable": result.error.retryable,
                    **(
                        {"execution_case_id": boundary_execution_case_id}
                        if boundary_execution_case_id
                        else {}
                    ),
                    **(
                        {"execution_item_id": execution_item_id}
                        if execution_item_id
                        else {}
                    ),
                },
            )
        self.lifecycle_sink.emit(
            "RUN_FINISHED",
            {
                "task_id": str(plan.task_id),
                "outcome": result.outcome.value,
                **(
                    {"execution_case_id": boundary_execution_case_id}
                    if boundary_execution_case_id
                    else {}
                ),
                **(
                    {"execution_item_id": execution_item_id}
                    if execution_item_id
                    else {}
                ),
            },
        )
        return result

    def _execute_adhoc(
        self, plan: ExecutionPlan, started_at: datetime
    ) -> TaskExecutionResult:
        if self.adhoc_runner is None or plan.adhoc is None:
            raise ExecutionError(
                ExecutionErrorCode.PLAN_INVALID, "ADHOC runner is not configured"
            )
        self.cancellation_token.raise_if_cancelled()
        result = self.adhoc_runner(plan.adhoc.prompt)
        return TaskExecutionResult.now(
            task_id=plan.task_id,
            outcome=_task_outcome_from_case(result.outcome, flaky=False),
            started_at=started_at,
            adhoc_result=result,
            error=result.error,
        )

    def _execute_test_run(
        self, plan: ExecutionPlan, started_at: datetime
    ) -> TaskExecutionResult:
        if self.attempt_runner is None or plan.test_run is None:
            raise ExecutionError(
                ExecutionErrorCode.PLAN_INVALID, "TEST_RUN runner is not configured"
            )
        case_results: list[CaseExecutionResult] = []
        not_run = []
        fatal_error: ExecutionError | None = None

        for position, case in enumerate(plan.test_run.cases):
            try:
                self.cancellation_token.raise_if_cancelled()
                if self.before_case:
                    self.before_case(case)
                case_result = self._execute_case(case, plan)
            except ExecutionError as error:
                fatal_error = error
                not_run.extend(
                    item.execution_case_id for item in plan.test_run.cases[position:]
                )
                break
            case_results.append(case_result)
            try:
                if self.on_case_finished:
                    self.on_case_finished(case, case_result)
            except ExecutionError as error:
                fatal_error = error
                # The platform has not accepted this Case checkpoint. For an
                # infrastructure termination it will close the still-RUNNING
                # Case as INFRA_ERROR, so the local aggregate must describe the
                # same fact instead of retaining the pre-checkpoint outcome.
                if _task_outcome_from_error(error) is TaskOutcome.INFRA_ERROR:
                    case_results[-1] = CaseExecutionResult(
                        execution_case_id=case_result.execution_case_id,
                        ordinal=case_result.ordinal,
                        outcome=CaseOutcome.INFRA_ERROR,
                        flaky=False,
                        attempts=case_result.attempts,
                    )
                not_run.extend(
                    item.execution_case_id
                    for item in plan.test_run.cases[position + 1 :]
                )
                break
            if case_result.outcome in _TASK_FATAL:
                fatal_error = case_result.attempts[
                    -1
                ].error or _error_for_fatal_outcome(case_result.outcome)
                not_run.extend(
                    item.execution_case_id
                    for item in plan.test_run.cases[position + 1 :]
                )
                break
            try:
                if self.after_case:
                    self.after_case(case)
            except ExecutionError as error:
                fatal_error = error
                not_run.extend(
                    item.execution_case_id
                    for item in plan.test_run.cases[position + 1 :]
                )
                break

        outcome = _aggregate_task_outcome(case_results, fatal_error)
        return TaskExecutionResult.now(
            task_id=plan.task_id,
            outcome=outcome,
            started_at=started_at,
            cases=tuple(case_results),
            error=fatal_error,
            not_run_execution_case_ids=tuple(not_run),
        )

    def _execute_case(
        self, case: ExecutionCase, plan: ExecutionPlan
    ) -> CaseExecutionResult:
        assert plan.test_run is not None
        retry = plan.test_run.case_retry
        attempts: list[AttemptResult] = []
        self.lifecycle_sink.emit(
            "CASE_STARTED",
            {"execution_case_id": str(case.execution_case_id), "ordinal": case.ordinal},
        )
        max_attempts = retry.max_retries + 1
        for attempt_no in range(1, max_attempts + 1):
            self.cancellation_token.raise_if_cancelled()
            result = self.attempt_runner(case, attempt_no)
            attempts.append(result)
            if (
                result.outcome in _TASK_FATAL
                or result.outcome not in retry.eligible_outcomes
            ):
                break

        final = attempts[-1]
        flaky = (
            len(attempts) == 2
            and attempts[0].outcome != CaseOutcome.PASS
            and final.outcome == CaseOutcome.PASS
        )
        outcome = (
            CaseOutcome.INFRA_ERROR
            if len(attempts) == 2 and final.outcome == CaseOutcome.RETRYABLE_ERROR
            else final.outcome
        )
        case_result = CaseExecutionResult(
            execution_case_id=case.execution_case_id,
            ordinal=case.ordinal,
            outcome=outcome,
            flaky=flaky,
            attempts=tuple(attempts),
        )
        self.lifecycle_sink.emit(
            "CASE_FINISHED",
            {
                "execution_case_id": str(case.execution_case_id),
                "outcome": outcome.value,
                "flaky": flaky,
            },
        )
        return case_result


def _aggregate_task_outcome(
    cases: list[CaseExecutionResult], fatal_error: ExecutionError | None
) -> TaskOutcome:
    if fatal_error:
        return _task_outcome_from_error(fatal_error)
    outcomes = {case.outcome for case in cases}
    if (
        CaseOutcome.INFRA_ERROR in outcomes
        or CaseOutcome.CASE_ERROR in outcomes
        or CaseOutcome.RETRYABLE_ERROR in outcomes
    ):
        return TaskOutcome.INFRA_ERROR
    if CaseOutcome.FAIL in outcomes:
        return TaskOutcome.FAIL
    if CaseOutcome.BLOCKED in outcomes:
        return TaskOutcome.BLOCKED
    if CaseOutcome.REVIEW in outcomes:
        return TaskOutcome.REVIEW
    if any(case.flaky for case in cases):
        return TaskOutcome.PASS_WITH_FLAKY
    return TaskOutcome.PASS


def _task_outcome_from_case(outcome: CaseOutcome, flaky: bool) -> TaskOutcome:
    if flaky and outcome is CaseOutcome.PASS:
        return TaskOutcome.PASS_WITH_FLAKY
    mapping = {
        CaseOutcome.PASS: TaskOutcome.PASS,
        CaseOutcome.FAIL: TaskOutcome.FAIL,
        CaseOutcome.BLOCKED: TaskOutcome.BLOCKED,
        CaseOutcome.REVIEW: TaskOutcome.REVIEW,
        CaseOutcome.CANCELLED: TaskOutcome.CANCELLED,
        CaseOutcome.TIMED_OUT: TaskOutcome.TIMED_OUT,
        CaseOutcome.WORKER_LOST: TaskOutcome.WORKER_LOST,
        CaseOutcome.DEVICE_LOST: TaskOutcome.DEVICE_LOST,
    }
    return mapping.get(outcome, TaskOutcome.INFRA_ERROR)


def _task_outcome_from_error(error: ExecutionError) -> TaskOutcome:
    mapping = {
        ExecutionErrorCode.CANCELLED: TaskOutcome.CANCELLED,
        ExecutionErrorCode.TASK_TIMEOUT: TaskOutcome.TIMED_OUT,
        ExecutionErrorCode.WORKER_LOST: TaskOutcome.WORKER_LOST,
        ExecutionErrorCode.WORKER_RESTARTED: TaskOutcome.WORKER_LOST,
        ExecutionErrorCode.DEVICE_LOST: TaskOutcome.DEVICE_LOST,
    }
    return mapping.get(error.code, TaskOutcome.INFRA_ERROR)


def _error_for_fatal_outcome(outcome: CaseOutcome) -> ExecutionError:
    mapping = {
        CaseOutcome.CANCELLED: ExecutionErrorCode.CANCELLED,
        CaseOutcome.TIMED_OUT: ExecutionErrorCode.TASK_TIMEOUT,
        CaseOutcome.WORKER_LOST: ExecutionErrorCode.WORKER_LOST,
        CaseOutcome.DEVICE_LOST: ExecutionErrorCode.DEVICE_LOST,
    }
    return ExecutionError(mapping[outcome], outcome.value, retryable=False)
