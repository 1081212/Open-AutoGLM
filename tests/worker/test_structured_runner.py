from __future__ import annotations

from types import SimpleNamespace

from phone_agent.execution.cancellation import CancellationToken
from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.execution.models import ResetPolicy, TargetRequirements
from phone_agent.worker.structured_runner import StructuredPlanRunner


class RecordingAdb:
    def __init__(self):
        self.calls = []

    def run(self, serial, args, **kwargs):
        self.calls.append((serial, list(args), kwargs))


class RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event_type, data):
        self.events.append((event_type, data))


class FailingAgent:
    def __init__(self, error):
        self.error = error
        self.run_vision_tokens = 7
        self.agent_config = SimpleNamespace(run_context={})

    def run(self, _prompt):
        raise self.error


class MinimalReporter:
    def __init__(self):
        self.current_case = True

    def start_case(self, *_args, **_kwargs):
        return SimpleNamespace(attempts=[])

    def set_test_steps(self, _steps):
        pass

    def begin_test_step(self, _step_index):
        pass

    def finish_case(self, _message):
        self.current_case = False
        return "REVIEW"


def _runner(tmp_path, adb):
    return StructuredPlanRunner(
        profiles=None,
        adb_serial="LOCKED-SERIAL",
        report_root=tmp_path,
        cancellation_token=CancellationToken(),
        adb=adb,
    )


def test_android_activity_reset_matches_legacy_main_launch_command(
    tmp_path, test_run_plan
):
    adb = RecordingAdb()
    runner = _runner(tmp_path, adb)
    target = TargetRequirements(
        device_type="adb",
        app_package="com.wakeup.howear",
        reset_policy=ResetPolicy(
            type="ANDROID_ACTIVITY",
            component=(
                "com.wakeup.howear/" "com.wakeup.howear.view.app.SplashActivity"
            ),
            wait_seconds=0,
        ),
    )
    plan = test_run_plan.model_copy(update={"target_requirements": target})

    runner._apply_reset(plan)

    assert adb.calls == [
        (
            "LOCKED-SERIAL",
            [
                "shell",
                "am",
                "start",
                "-W",
                "-a",
                "android.intent.action.MAIN",
                "-c",
                "android.intent.category.LAUNCHER",
                "-n",
                ("com.wakeup.howear/" "com.wakeup.howear.view.app.SplashActivity"),
                "-f",
                "0x10008000",
            ],
            {"timeout": 30},
        )
    ]


def test_none_reset_policy_does_not_guess_or_launch_an_app(tmp_path, test_run_plan):
    adb = RecordingAdb()
    runner = _runner(tmp_path, adb)

    runner._apply_reset(test_run_plan)

    assert adb.calls == []


def test_fatal_step_error_emits_bound_terminal_step_context(tmp_path, test_run_plan):
    sink = RecordingSink()
    runner = StructuredPlanRunner(
        profiles=None,
        adb_serial="LOCKED-SERIAL",
        report_root=tmp_path,
        cancellation_token=CancellationToken(),
        lifecycle_sink=sink,
        adb=RecordingAdb(),
    )
    runner._agent = SimpleNamespace(run_vision_tokens=19)
    case = test_run_plan.test_run.cases[0]
    step = case.steps[0]
    error = ExecutionError(
        ExecutionErrorCode.DEVICE_DISCONNECTED,
        "ADB screenshot returned no data",
    )

    runner._emit_failed_step_finished(
        case=case,
        attempt_no=2,
        attempt_id="attempt-uuid",
        step=step,
        started_at=None,
        error=error,
    )

    assert sink.events == [
        (
            "STEP_FINISHED",
            {
                "execution_case_id": str(case.execution_case_id),
                "case_attempt_id": "attempt-uuid",
                "case_attempt_no": 2,
                "step_id": str(step.step_id),
                "outcome": "DEVICE_LOST",
                "message": "ADB screenshot returned no data",
                "duration_ms": 0,
                "vision_tokens": 19,
                "judge_tokens": 0,
            },
        )
    ]


def test_case_attempt_closes_step_before_attempt_after_execution_error(
    tmp_path, test_run_plan
):
    sink = RecordingSink()
    runner = StructuredPlanRunner(
        profiles=None,
        adb_serial="LOCKED-SERIAL",
        report_root=tmp_path,
        cancellation_token=CancellationToken(),
        lifecycle_sink=sink,
        adb=RecordingAdb(),
    )
    error = ExecutionError(
        ExecutionErrorCode.MODEL_ERROR,
        "model request failed",
        retryable=True,
    )
    runner._agent = FailingAgent(error)
    runner._reporter = MinimalReporter()

    result = runner._run_case_attempt(
        test_run_plan,
        test_run_plan.test_run.cases[0],
        1,
    )

    assert result.error is error
    assert [event_type for event_type, _ in sink.events] == [
        "CASE_ATTEMPT_STARTED",
        "STEP_STARTED",
        "STEP_FINISHED",
        "CASE_ATTEMPT_FINISHED",
    ]
    step_finished = sink.events[2][1]
    assert step_finished["outcome"] == "RETRYABLE_ERROR"
    assert step_finished["message"] == "model request failed"
    assert step_finished["vision_tokens"] == 7
    assert step_finished["judge_tokens"] == 0
    assert step_finished["execution_case_id"] == str(
        test_run_plan.test_run.cases[0].execution_case_id
    )
    assert step_finished["step_id"] == str(
        test_run_plan.test_run.cases[0].steps[0].step_id
    )
