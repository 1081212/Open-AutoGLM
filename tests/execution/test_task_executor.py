from __future__ import annotations

from uuid import uuid4

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.execution.models import CaseOutcome, ExecutionPlan
from phone_agent.execution.result import AttemptResult, TaskOutcome
from phone_agent.execution.task_executor import TaskExecutor


class RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event_type, data):
        self.events.append((event_type, data))


def test_retry_pass_marks_flaky_and_continues(test_run_plan):
    calls: dict[int, int] = {}

    def runner(case, attempt_no):
        calls[case.ordinal] = calls.get(case.ordinal, 0) + 1
        if case.ordinal == 1 and attempt_no == 1:
            return AttemptResult(CaseOutcome.FAIL, "failed")
        return AttemptResult(CaseOutcome.PASS, "passed")

    result = TaskExecutor(attempt_runner=runner).execute(test_run_plan)

    assert result.outcome is TaskOutcome.PASS_WITH_FLAKY
    assert [case.ordinal for case in result.cases] == [1, 2]
    assert result.cases[0].flaky is True
    assert len(result.cases[0].attempts) == 2
    assert calls == {1: 2, 2: 1}


def test_review_does_not_retry_and_next_case_runs(test_run_plan):
    calls: list[tuple[int, int]] = []

    def runner(case, attempt_no):
        calls.append((case.ordinal, attempt_no))
        outcome = CaseOutcome.REVIEW if case.ordinal == 1 else CaseOutcome.PASS
        return AttemptResult(outcome, outcome.value)

    result = TaskExecutor(attempt_runner=runner).execute(test_run_plan)

    assert result.outcome is TaskOutcome.REVIEW
    assert calls == [(1, 1), (2, 1)]


def test_device_lost_stops_remaining_cases(test_run_plan):
    error = ExecutionError(ExecutionErrorCode.DEVICE_LOST, "device disconnected")
    sink = RecordingSink()

    def runner(case, attempt_no):
        del attempt_no
        return AttemptResult(CaseOutcome.DEVICE_LOST, "lost", error=error)

    result = TaskExecutor(
        attempt_runner=runner,
        lifecycle_sink=sink,
        bind_test_run_boundaries=True,
    ).execute(test_run_plan)

    assert result.outcome is TaskOutcome.DEVICE_LOST
    assert len(result.cases) == 1
    assert result.not_run_execution_case_ids == (
        test_run_plan.test_run.cases[1].execution_case_id,
    )
    assert [event_type for event_type, _ in sink.events[-3:]] == [
        "CASE_FINISHED",
        "RUN_ERROR",
        "RUN_FINISHED",
    ]
    run_error = sink.events[-2][1]
    assert run_error["error_code"] == "DEVICE_LOST"
    assert run_error["message"] == "device disconnected"
    assert run_error["execution_case_id"] == str(
        test_run_plan.test_run.cases[0].execution_case_id
    )


def test_device_lost_after_case_keeps_completed_case(test_run_plan):
    def runner(case, attempt_no):
        del case, attempt_no
        return AttemptResult(CaseOutcome.PASS, "passed")

    def after_case(case):
        if case.ordinal == 1:
            raise ExecutionError(ExecutionErrorCode.DEVICE_LOST, "lost after case")

    result = TaskExecutor(attempt_runner=runner, after_case=after_case).execute(
        test_run_plan
    )

    assert result.outcome is TaskOutcome.DEVICE_LOST
    assert len(result.cases) == 1
    assert result.cases[0].outcome is CaseOutcome.PASS
    assert result.not_run_execution_case_ids == (
        test_run_plan.test_run.cases[1].execution_case_id,
    )


def test_checkpoint_failure_stops_before_next_case(test_run_plan):
    executed = []

    def runner(case, attempt_no):
        executed.append((case.ordinal, attempt_no))
        return AttemptResult(CaseOutcome.PASS, "passed")

    def checkpoint(case, result):
        del result
        if case.ordinal == 1:
            raise ExecutionError(
                ExecutionErrorCode.RETRYABLE_ERROR,
                "checkpoint unavailable",
                retryable=True,
            )

    result = TaskExecutor(
        attempt_runner=runner,
        on_case_finished=checkpoint,
    ).execute(test_run_plan)

    assert executed == [(1, 1)]
    assert result.outcome is TaskOutcome.INFRA_ERROR
    assert len(result.cases) == 1
    assert result.cases[0].outcome is CaseOutcome.INFRA_ERROR
    assert result.not_run_execution_case_ids == (
        test_run_plan.test_run.cases[1].execution_case_id,
    )


def test_adhoc_run_boundaries_reference_the_plan_item(test_run_plan):
    plan_data = test_run_plan.model_dump(mode="json")
    item_id = uuid4()
    plan_data.update(
        {
            "task_type": "ADHOC",
            "normalizer": None,
            "test_run": None,
            "adhoc": {"execution_item_id": str(item_id), "prompt": "open settings"},
        }
    )
    plan = ExecutionPlan.model_validate(plan_data)
    sink = RecordingSink()

    result = TaskExecutor(
        adhoc_runner=lambda _prompt: AttemptResult(CaseOutcome.PASS, "done"),
        lifecycle_sink=sink,
    ).execute(plan)

    assert result.outcome is TaskOutcome.PASS
    assert sink.events[0] == (
        "RUN_STARTED",
        {
            "task_id": str(plan.task_id),
            "plan_id": str(plan.plan_id),
            "task_type": "ADHOC",
            "execution_item_id": str(item_id),
        },
    )
    assert sink.events[-1] == (
        "RUN_FINISHED",
        {
            "task_id": str(plan.task_id),
            "outcome": "PASS",
            "execution_item_id": str(item_id),
        },
    )
