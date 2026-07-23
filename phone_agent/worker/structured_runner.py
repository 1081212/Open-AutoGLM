"""Execute an immutable platform Plan without parsing source Markdown."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from phone_agent import PhoneAgent
from phone_agent.adb.command import AdbCommandAdapter
from phone_agent.agent import AgentConfig
from phone_agent.execution.cancellation import CancellationToken
from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.execution.lifecycle import LifecycleSink, NullLifecycleSink
from phone_agent.execution.models import (
    CaseOutcome,
    ExecutionCase,
    ExecutionPlan,
    ExecutionStep,
)
from phone_agent.execution.result import AttemptResult, TaskExecutionResult
from phone_agent.execution.task_executor import TaskExecutor
from phone_agent.reporting import TestRunReporter
from phone_agent.worker.model_profiles import ModelProfileStore
from phone_agent.worker.status_judge import StructuredStatusJudge

_STATUS = re.compile(
    r"^\s*(?:STATUS|状态)\s*[:：]\s*(PASS|SKIPPED|BLOCKED|FAIL|REVIEW)\b", re.I | re.M
)
logger = logging.getLogger(__name__)


class StructuredPlanRunner:
    def __init__(
        self,
        *,
        profiles: ModelProfileStore,
        adb_serial: str,
        report_root: Path,
        cancellation_token: CancellationToken,
        lifecycle_sink: LifecycleSink | None = None,
        adb: AdbCommandAdapter | None = None,
        case_coordinator=None,
        run_started_already: bool = False,
    ) -> None:
        self.profiles = profiles
        self.adb_serial = adb_serial
        self.report_root = report_root
        self.cancellation_token = cancellation_token
        self.lifecycle_sink = lifecycle_sink or NullLifecycleSink()
        self.adb = adb or AdbCommandAdapter()
        self.case_coordinator = case_coordinator
        self.run_started_already = run_started_already
        self._agent: PhoneAgent | None = None
        self._reporter: TestRunReporter | None = None
        self._judge: StructuredStatusJudge | None = None

    def execute(self, plan: ExecutionPlan, *, task_run_id: str) -> TaskExecutionResult:
        options = plan.execution_options
        logger.info(
            "Structured execution starting task_run_id=%s task_type=%s "
            "case_count=%d vision_profile=%s judge_enabled=%s reset_policy=%s",
            task_run_id,
            plan.task_type.value,
            len(plan.test_run.cases) if plan.test_run else 0,
            plan.model_profiles.vision_profile,
            options.status_judge_enabled,
            plan.target_requirements.reset_policy.type,
        )
        model_config = self.profiles.resolve(
            plan.model_profiles.vision_profile,
            lang=options.language,
            timeout_seconds=options.model_call_timeout_seconds,
        )
        reporter = TestRunReporter(
            artifact_name=task_run_id,
            base_dir=self.report_root,
            device_type="adb",
            device_id=self.adb_serial,
            model_name=model_config.model_name,
            base_url=model_config.base_url,
        )
        agent_config = AgentConfig(
            max_steps=options.max_steps_per_agent_call,
            device_id=self.adb_serial,
            lang=options.language,
            reporter=reporter if plan.test_run else None,
            auto_manage_report_case=False,
            require_structured_finish_status=not options.status_judge_enabled,
            cancellation_token=self.cancellation_token,
            lifecycle_sink=self.lifecycle_sink,
            model_call_timeout=options.model_call_timeout_seconds,
            run_context={
                "task_run_id": task_run_id,
                "plan_id": str(plan.plan_id),
                **(
                    {"execution_item_id": str(plan.adhoc.execution_item_id)}
                    if plan.adhoc
                    else {}
                ),
            },
        )
        if options.status_judge_enabled:
            agent_config.system_prompt = (agent_config.system_prompt or "") + (
                "\n\n当前运行启用了独立文本判定模型。你只负责手机操作和描述可观察证据，"
                "不要自行判定 PASS/FAIL；完成或无法继续时直接 finish 并说明事实。"
            )
        agent = PhoneAgent(
            model_config=model_config,
            agent_config=agent_config,
            confirmation_callback=lambda _message: False,
            takeover_callback=lambda _message: None,
        )
        self._agent = agent
        self._reporter = reporter
        self._judge = None
        if options.status_judge_enabled:
            assert plan.model_profiles.judge_profile
            self._judge = StructuredStatusJudge(
                self.profiles.resolve(
                    plan.model_profiles.judge_profile,
                    lang=options.language,
                    timeout_seconds=options.model_call_timeout_seconds,
                )
            )
        try:
            executor = TaskExecutor(
                attempt_runner=lambda case, attempt: self._run_case_attempt(
                    plan, case, attempt
                ),
                adhoc_runner=lambda prompt: self._run_adhoc(prompt, task_run_id),
                cancellation_token=self.cancellation_token,
                lifecycle_sink=self.lifecycle_sink,
                before_case=lambda _case: self._probe_device(),
                after_case=lambda _case: self._probe_device(),
                on_case_finished=(
                    self.case_coordinator.checkpoint if self.case_coordinator else None
                ),
                emit_run_started=not self.run_started_already,
                bind_test_run_boundaries=self.run_started_already,
            )
            result = executor.execute(plan)
            logger.info(
                "Structured execution finished task_run_id=%s outcome=%s "
                "completed_cases=%d not_run_cases=%d",
                task_run_id,
                result.outcome.value,
                len(result.cases),
                len(result.not_run_execution_case_ids),
            )
            return result
        finally:
            reporter.finish_run()
            self._agent = None
            self._reporter = None
            self._judge = None

    def _run_case_attempt(
        self, plan: ExecutionPlan, case: ExecutionCase, attempt_no: int
    ) -> AttemptResult:
        assert self._agent is not None and self._reporter is not None
        logger.info(
            "Case attempt starting case=%s execution_case_id=%s ordinal=%d "
            "attempt_no=%d",
            case.display_id,
            case.execution_case_id,
            case.ordinal,
            attempt_no,
        )
        self._apply_reset(plan)
        attempt_id = None
        if self.case_coordinator:
            attempt_id = self.case_coordinator.begin_attempt(case, attempt_no)
        self.lifecycle_sink.emit(
            "CASE_ATTEMPT_STARTED",
            {
                "execution_case_id": str(case.execution_case_id),
                "case_attempt_id": str(attempt_id) if attempt_id else None,
                "case_attempt_no": attempt_no,
            },
        )
        reporter = self._reporter
        case_report = reporter.start_case(
            self._case_summary(case),
            case.ordinal,
            attempt=attempt_no,
            execution_case_id=str(case.execution_case_id),
            ordinal=case.ordinal,
        )
        reporter.set_test_steps(
            [
                {
                    "index": step.index,
                    "text": step.instruction,
                    "target_state": step.target_state,
                    "activity": step.expected_activity,
                }
                for step in case.steps
            ]
        )
        last_status = "REVIEW"
        messages: list[str] = []
        active_step: ExecutionStep | None = None
        active_step_started: float | None = None
        active_step_finished = False
        try:
            for step in case.steps:
                step_started = time.monotonic()
                logger.info(
                    "Step starting case=%s attempt_no=%d step_index=%d step_id=%s",
                    case.display_id,
                    attempt_no,
                    step.index,
                    step.step_id,
                )
                self.cancellation_token.raise_if_cancelled()
                self.lifecycle_sink.emit(
                    "STEP_STARTED",
                    {
                        "execution_case_id": str(case.execution_case_id),
                        "case_attempt_no": attempt_no,
                        "case_attempt_id": str(attempt_id) if attempt_id else None,
                        "step_id": str(step.step_id),
                        "step_index": step.index,
                    },
                )
                active_step = step
                active_step_started = step_started
                active_step_finished = False
                reporter.begin_test_step(step.index)
                self._agent.agent_config.run_context.update(
                    {
                        "execution_case_id": str(case.execution_case_id),
                        "case_attempt_no": attempt_no,
                        "case_attempt_id": str(attempt_id) if attempt_id else None,
                        "step_id": str(step.step_id),
                    }
                )
                message = self._agent.run(self._step_prompt(case, step))
                vision_total_tokens = self._agent.run_vision_tokens
                if message.strip() == "Max steps reached":
                    message = self._agent.request_finish_status_only(
                        step.instruction, step.target_state
                    )
                judge_raw = None
                judge_prompt_tokens = 0
                judge_completion_tokens = 0
                judge_total_tokens = 0
                if self._judge:
                    try:
                        judged = self._judge.judge(
                            case=case,
                            step=step,
                            execution_message=message,
                        )
                        status = judged.status
                        message = judged.message
                        judge_raw = judged.raw
                        judge_prompt_tokens = judged.prompt_tokens
                        judge_completion_tokens = judged.completion_tokens
                        judge_total_tokens = judged.total_tokens
                    except Exception as exc:
                        status = "REVIEW"
                        judge_raw = f"{type(exc).__name__}: judge request failed"
                        message = "独立判定模型调用失败，需要人工复核"
                        self.lifecycle_sink.emit(
                            "RUN_ERROR",
                            {
                                **self._agent.agent_config.run_context,
                                "error_code": "JUDGE_ERROR",
                                "error_type": type(exc).__name__,
                            },
                        )
                else:
                    status = _parse_status(message)
                reporter.finish_test_step(
                    step.index,
                    status,
                    message,
                    judge_raw=judge_raw,
                    judge_prompt_tokens=judge_prompt_tokens,
                    judge_completion_tokens=judge_completion_tokens,
                    judge_total_tokens=judge_total_tokens,
                )
                duration_ms = max(0, round((time.monotonic() - step_started) * 1000))
                self.lifecycle_sink.emit(
                    "STEP_FINISHED",
                    {
                        "execution_case_id": str(case.execution_case_id),
                        "case_attempt_no": attempt_no,
                        "case_attempt_id": str(attempt_id) if attempt_id else None,
                        "step_id": str(step.step_id),
                        "outcome": status,
                        "message": message,
                        "duration_ms": duration_ms,
                        "vision_tokens": vision_total_tokens,
                        "judge_tokens": judge_total_tokens,
                    },
                )
                active_step_finished = True
                logger.info(
                    "Step finished case=%s attempt_no=%d step_index=%d "
                    "outcome=%s duration_ms=%d vision_tokens=%d judge_tokens=%d",
                    case.display_id,
                    attempt_no,
                    step.index,
                    status,
                    duration_ms,
                    vision_total_tokens,
                    judge_total_tokens,
                )
                messages.append(f"{step.index}. {status}: {message}")
                last_status = status
                self._agent.reset()
                active_step = None
                active_step_started = None
                if status in {"FAIL", "BLOCKED"}:
                    break
            final_message = (
                "\n".join(messages) or "STATUS: REVIEW\nREASON: 用例没有可执行步骤"
            )
            reported_status = reporter.finish_case(final_message)
            result = AttemptResult(
                _case_outcome(reported_status or last_status), final_message
            )
            if self.case_coordinator:
                self.case_coordinator.finish_attempt(
                    case,
                    attempt_no,
                    result,
                    Path(case_report.attempts[-1].artifacts_dir),
                )
            self._emit_attempt_finished(case, attempt_no, attempt_id, result)
            logger.info(
                "Case attempt finished case=%s execution_case_id=%s "
                "attempt_no=%d outcome=%s",
                case.display_id,
                case.execution_case_id,
                attempt_no,
                result.outcome.value,
            )
            return result
        except ExecutionError as error:
            if active_step is not None and not active_step_finished:
                self._emit_failed_step_finished(
                    case=case,
                    attempt_no=attempt_no,
                    attempt_id=attempt_id,
                    step=active_step,
                    started_at=active_step_started,
                    error=error,
                )
            logger.error(
                "Case attempt failed case=%s execution_case_id=%s "
                "attempt_no=%d error_code=%s message=%s",
                case.display_id,
                case.execution_case_id,
                attempt_no,
                error.code.value,
                error.message,
            )
            if reporter.current_case:
                reporter.finish_case(f"STATUS: REVIEW\nREASON: {error}")
            result = AttemptResult(_error_outcome(error), str(error), error)
            if self.case_coordinator:
                self.case_coordinator.finish_attempt(
                    case,
                    attempt_no,
                    result,
                    Path(case_report.attempts[-1].artifacts_dir),
                )
            self._emit_attempt_finished(case, attempt_no, attempt_id, result)
            return result
        except Exception as exc:
            logger.exception(
                "Case attempt crashed case=%s execution_case_id=%s attempt_no=%d",
                case.display_id,
                case.execution_case_id,
                attempt_no,
            )
            error = ExecutionError(
                ExecutionErrorCode.CASE_ERROR,
                f"{type(exc).__name__}: {exc}",
            )
            if active_step is not None and not active_step_finished:
                self._emit_failed_step_finished(
                    case=case,
                    attempt_no=attempt_no,
                    attempt_id=attempt_id,
                    step=active_step,
                    started_at=active_step_started,
                    error=error,
                )
            if reporter.current_case:
                reporter.finish_case(f"STATUS: REVIEW\nREASON: {error}")
            result = AttemptResult(CaseOutcome.CASE_ERROR, str(error), error)
            if self.case_coordinator:
                self.case_coordinator.finish_attempt(
                    case,
                    attempt_no,
                    result,
                    Path(case_report.attempts[-1].artifacts_dir),
                )
            self._emit_attempt_finished(case, attempt_no, attempt_id, result)
            return result

    def _emit_failed_step_finished(
        self,
        *,
        case: ExecutionCase,
        attempt_no: int,
        attempt_id,
        step: ExecutionStep,
        started_at: float | None,
        error: ExecutionError,
    ) -> None:
        """Close an opened platform Step before its Attempt is terminated."""
        duration_ms = (
            max(0, round((time.monotonic() - started_at) * 1000))
            if started_at is not None
            else 0
        )
        vision_tokens = (
            max(0, self._agent.run_vision_tokens) if self._agent is not None else 0
        )
        outcome = _error_outcome(error).value
        self.lifecycle_sink.emit(
            "STEP_FINISHED",
            {
                "execution_case_id": str(case.execution_case_id),
                "case_attempt_id": str(attempt_id) if attempt_id else None,
                "case_attempt_no": attempt_no,
                "step_id": str(step.step_id),
                "outcome": outcome,
                "message": error.message,
                "duration_ms": duration_ms,
                "vision_tokens": vision_tokens,
                # A failed judge call does not expose reliable usage. Never guess.
                "judge_tokens": 0,
            },
        )
        logger.info(
            "Failed step terminal event recorded case=%s attempt_no=%d "
            "step_index=%d outcome=%s duration_ms=%d vision_tokens=%d",
            case.display_id,
            attempt_no,
            step.index,
            outcome,
            duration_ms,
            vision_tokens,
        )

    def _emit_attempt_finished(
        self,
        case: ExecutionCase,
        attempt_no: int,
        attempt_id,
        result: AttemptResult,
    ) -> None:
        self.lifecycle_sink.emit(
            "CASE_ATTEMPT_FINISHED",
            {
                "execution_case_id": str(case.execution_case_id),
                "case_attempt_id": str(attempt_id) if attempt_id else None,
                "case_attempt_no": attempt_no,
                "outcome": result.outcome.value,
            },
        )

    def _run_adhoc(self, prompt: str, task_run_id: str) -> AttemptResult:
        assert self._agent is not None
        try:
            message = self._agent.run(prompt)
            outcome = _case_outcome(_parse_status(message))
            self._write_adhoc_result(task_run_id, prompt, message, outcome)
            return AttemptResult(outcome, message)
        except ExecutionError as error:
            return AttemptResult(_error_outcome(error), str(error), error)

    def _probe_device(self) -> None:
        try:
            self.adb.get_state(self.adb_serial)
        except ExecutionError as error:
            raise ExecutionError(
                ExecutionErrorCode.DEVICE_LOST,
                error.message,
                retryable=False,
            ) from error
        self.lifecycle_sink.emit(
            "DEVICE_PROBED",
            {
                **(self._agent.agent_config.run_context if self._agent else {}),
                "adb_serial": self.adb_serial,
            },
        )

    def _apply_reset(self, plan: ExecutionPlan) -> None:
        reset = plan.target_requirements.reset_policy
        if reset.type == "NONE":
            logger.warning(
                "Case app reset skipped because Plan reset_policy=NONE "
                "app_package=%s",
                plan.target_requirements.app_package,
            )
            return
        assert reset.component
        logger.info(
            "Resetting Android app adb_serial=%s component=%s " "wait_seconds=%d",
            self.adb_serial,
            reset.component,
            reset.wait_seconds,
        )
        # Keep the original main.py Case-boundary behavior, while taking the
        # component from the frozen Plan instead of hard-coding Wearfit.
        self.adb.run(
            self.adb_serial,
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
                reset.component,
                "-f",
                "0x10008000",
            ],
            timeout=30,
        )
        deadline = time.monotonic() + reset.wait_seconds
        while time.monotonic() < deadline:
            self.cancellation_token.raise_if_cancelled()
            time.sleep(min(0.25, deadline - time.monotonic()))
        logger.info(
            "Android app reset completed adb_serial=%s component=%s",
            self.adb_serial,
            reset.component,
        )

    @staticmethod
    def _case_summary(case: ExecutionCase) -> str:
        return f"{case.display_id}：{case.title}\n测试目标：{case.goal}\n预期结果：{case.expected_result}"

    @staticmethod
    def _step_prompt(case: ExecutionCase, step: ExecutionStep) -> str:
        preconditions = "；".join(case.preconditions) or "无额外前置条件"
        failures = "；".join(case.failure_conditions) or "以页面可观察结果为准"
        return (
            f"执行结构化测试用例 {case.display_id}：{case.title}\n"
            f"测试目标：{case.goal}\n前置条件：{preconditions}\n"
            f"当前步骤 {step.index}：{step.instruction}\n"
            f"目标状态：{step.target_state or case.expected_result}\n"
            f"预期 Activity：{step.expected_activity or '未指定'}\n"
            f"失败条件：{failures}\n"
            "完成本步骤后必须用 finish 返回 STATUS: PASS、SKIPPED、BLOCKED、FAIL 或 REVIEW。"
        )

    def _write_adhoc_result(
        self, task_run_id: str, prompt: str, message: str, outcome: CaseOutcome
    ) -> None:
        target = self.report_root / task_run_id / "adhoc.json"
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        part = target.with_suffix(".json.part")
        part.write_text(
            json.dumps(
                {
                    "schema_version": "autoglm.local-adhoc.v1",
                    "prompt": prompt,
                    "outcome": outcome.value,
                    "message": message,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with part.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(part, target)


def _parse_status(message: str) -> str:
    match = _STATUS.search(message or "")
    return match.group(1).upper() if match else "REVIEW"


def _case_outcome(status: str) -> CaseOutcome:
    return {
        "PASS": CaseOutcome.PASS,
        "SKIPPED": CaseOutcome.SKIPPED,
        "BLOCKED": CaseOutcome.BLOCKED,
        "FAIL": CaseOutcome.FAIL,
        "REVIEW": CaseOutcome.REVIEW,
    }.get(status.upper(), CaseOutcome.REVIEW)


def _error_outcome(error: ExecutionError) -> CaseOutcome:
    if error.code is ExecutionErrorCode.CANCELLED:
        return CaseOutcome.CANCELLED
    if error.code in {
        ExecutionErrorCode.DEVICE_LOST,
        ExecutionErrorCode.DEVICE_DISCONNECTED,
    }:
        return CaseOutcome.DEVICE_LOST
    if error.code is ExecutionErrorCode.TASK_TIMEOUT:
        return CaseOutcome.TIMED_OUT
    return CaseOutcome.RETRYABLE_ERROR if error.retryable else CaseOutcome.CASE_ERROR
