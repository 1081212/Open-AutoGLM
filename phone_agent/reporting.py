"""Test artifact and report generation utilities."""

from __future__ import annotations

import base64
import html
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


FAIL_KEYWORDS = (
    "bug",
    "fail",
    "failed",
    "失败",
    "异常",
    "错误",
    "不符合",
    "未通过",
    "没生效",
    "无响应",
    "未进入",
    "未增加",
    "没有成功",
)

BLOCKED_KEYWORDS = (
    "blocked",
    "max steps reached",
    "model error",
    "user interaction required",
    "user intervention",
    "take_over",
    "接管",
    "人工",
    "登录",
    "验证码",
    "权限",
    "无法继续",
    "卡住",
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class StepArtifact:
    step: int
    screenshot: str | None
    ui_xml: str | None
    current_app: str
    current_activity: str | None
    test_step_index: int | None = None
    action: dict[str, Any] | None = None
    action_success: bool | None = None
    action_message: str | None = None
    vision_prompt_tokens: int = 0
    vision_completion_tokens: int = 0
    vision_cached_tokens: int = 0
    vision_total_tokens: int = 0
    sensitive_screenshot: bool = False
    ui_texts: list[str] = field(default_factory=list)


@dataclass
class TestStepArtifact:
    index: int
    text: str
    target_state: str | None = None
    activity: str | None = None
    status: str = "PENDING"
    result_message: str | None = None
    judge_raw: str | None = None
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_total_tokens: int = 0
    started_at: str | None = None
    finished_at: str | None = None


@dataclass
class CaseAttemptReport:
    attempt: int
    status: str = "RUNNING"
    result_message: str | None = None
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    artifacts_dir: str = ""
    test_steps: list[TestStepArtifact] = field(default_factory=list)
    steps: list[StepArtifact] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    logcat: str | None = None
    logcat_size: int = 0
    logcat_error: str | None = None


@dataclass
class CaseReport:
    case_id: str
    title: str
    task: str
    status: str = "RUNNING"
    result_message: str | None = None
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    artifacts_dir: str = ""
    test_steps: list[TestStepArtifact] = field(default_factory=list)
    steps: list[StepArtifact] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    attempts: list[CaseAttemptReport] = field(default_factory=list)


def extract_case_id(task: str, fallback_index: int) -> str:
    match = re.search(r"\bTC[-_][A-Za-z0-9_-]+\b", task)
    if match:
        return match.group(0).replace("_", "-")
    return f"CASE-{fallback_index:03d}"


def extract_package_name(task: str) -> str | None:
    patterns = (
        r"包名\s*[:：]?\s*([A-Za-z0-9_.]+)",
        r"package\s*[:：]?\s*([A-Za-z0-9_.]+)",
        r"\b([a-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+){2,})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, task)
        if match:
            return match.group(1)
    return None


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._-") or "run"


class TestRunReporter:
    """Collect screenshots, UI state and structured reports for a test run."""

    def __init__(
        self,
        artifact_name: str | None = None,
        base_dir: str | os.PathLike[str] = "test_artifacts",
        device_type: str = "adb",
        device_id: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        wda_url: str | None = None,
    ) -> None:
        run_name = artifact_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root_dir = Path(base_dir) / sanitize_name(run_name)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.device_type = device_type
        self.device_id = device_id
        self.model_name = model_name
        self.base_url = base_url
        self.wda_url = wda_url or os.getenv("PHONE_AGENT_WDA_URL", "http://localhost:8100")
        self.cases: list[CaseReport] = []
        self.current_case: CaseReport | None = None
        self.current_attempt: CaseAttemptReport | None = None
        self.current_test_step_index: int | None = None
        self._case_step_counter = 0
        self._logcat_process: subprocess.Popen[Any] | None = None
        self._logcat_file: Any | None = None
        self._logcat_capture_path: Path | None = None
        self.run_started_at = _now()
        self.environment: dict[str, Any] = self._collect_environment()

    def start_case(self, task: str, index: int, attempt: int = 1) -> CaseReport:
        case_id = extract_case_id(task, index)
        title = task.split("：", 1)[0].split(":", 1)[0][:120]
        case_dir = self.root_dir / case_id
        (case_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (case_dir / "ui").mkdir(parents=True, exist_ok=True)
        case = next((item for item in self.cases if item.case_id == case_id), None)
        if case is None:
            case = CaseReport(
                case_id=case_id,
                title=title,
                task=task,
                artifacts_dir=str(case_dir),
            )
            self.cases.append(case)
        attempt_dir = case_dir / f"attempt_{attempt:02d}"
        (attempt_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (attempt_dir / "ui").mkdir(parents=True, exist_ok=True)
        attempt_report = CaseAttemptReport(
            attempt=attempt,
            artifacts_dir=str(attempt_dir),
        )
        case.attempts.append(attempt_report)
        case.status = "RUNNING"
        case.result_message = None
        case.finished_at = None
        case.test_steps = attempt_report.test_steps
        case.steps = attempt_report.steps
        case.issues = attempt_report.issues
        self.current_case = case
        self.current_attempt = attempt_report
        self.current_test_step_index = None
        self._case_step_counter = 0
        package_name = extract_package_name(task)
        if package_name:
            self.environment.setdefault("apps", {})[package_name] = (
                self._collect_app_info(package_name)
            )
        self._write_run_metadata()
        self._start_logcat_capture(attempt_report)
        return case

    def set_test_steps(self, steps: list[dict[str, Any]]) -> None:
        """Register parsed Markdown test steps for the current case."""
        if not self.current_case:
            return
        parsed_steps = [
            TestStepArtifact(
                index=int(step["index"]),
                text=str(step["text"]),
                target_state=step.get("target_state"),
                activity=step.get("activity"),
            )
            for step in steps
        ]
        self.current_case.test_steps = parsed_steps
        if self.current_attempt:
            self.current_attempt.test_steps = parsed_steps
        self._write_case_json(self.current_case)

    def begin_test_step(self, index: int) -> None:
        """Mark the current Markdown test step before model actions are recorded."""
        self.current_test_step_index = index
        if not self.current_case:
            return
        for test_step in self.current_case.test_steps:
            if test_step.index == index:
                test_step.status = "RUNNING"
                test_step.started_at = test_step.started_at or _now()
                break
        self._write_case_json(self.current_case)

    def finish_test_step(
        self,
        index: int,
        status: str,
        result_message: str,
        judge_raw: str | None = None,
        judge_prompt_tokens: int = 0,
        judge_completion_tokens: int = 0,
        judge_total_tokens: int = 0,
    ) -> None:
        """Finish one Markdown test step while keeping the case open."""
        if not self.current_case:
            return
        for test_step in self.current_case.test_steps:
            if test_step.index == index:
                test_step.status = status
                test_step.result_message = result_message
                test_step.judge_raw = judge_raw
                test_step.judge_prompt_tokens = judge_prompt_tokens
                test_step.judge_completion_tokens = judge_completion_tokens
                test_step.judge_total_tokens = judge_total_tokens
                test_step.finished_at = _now()
                break
        self.current_test_step_index = None
        self._write_case_json(self.current_case)

    def save_step(
        self,
        step: int,
        screenshot_base64: str,
        width: int,
        height: int,
        current_app: str,
        sensitive_screenshot: bool = False,
    ) -> StepArtifact | None:
        if not self.current_case:
            return None

        case_dir = Path(self.current_case.artifacts_dir)
        attempt_dir = (
            Path(self.current_attempt.artifacts_dir)
            if self.current_attempt
            else case_dir
        )
        self._case_step_counter += 1
        artifact_step = self._case_step_counter
        attempt_rel = attempt_dir.relative_to(case_dir)
        screenshot_rel = str(attempt_rel / "screenshots" / f"step_{artifact_step:03d}.png")
        screenshot_path = attempt_dir / "screenshots" / f"step_{artifact_step:03d}.png"
        try:
            screenshot_path.write_bytes(base64.b64decode(screenshot_base64))
        except Exception:
            screenshot_rel = None

        ui_rel = None
        current_activity = self._get_current_activity()
        xml = self._dump_ui_xml()
        if xml:
            ui_rel = str(attempt_rel / "ui" / f"step_{artifact_step:03d}.xml")
            (attempt_dir / "ui" / f"step_{artifact_step:03d}.xml").write_text(
                xml, encoding="utf-8"
            )
        ui_texts = _extract_ui_texts(xml or "")

        artifact = StepArtifact(
            step=artifact_step,
            screenshot=screenshot_rel,
            ui_xml=ui_rel,
            current_app=current_app,
            current_activity=current_activity,
            test_step_index=self.current_test_step_index,
            sensitive_screenshot=sensitive_screenshot,
            ui_texts=ui_texts,
        )
        self.current_case.steps.append(artifact)
        self._write_case_json(self.current_case)
        return artifact

    def record_action(
        self,
        step: int,
        action: dict[str, Any] | None,
        success: bool | None,
        message: str | None,
        vision_prompt_tokens: int = 0,
        vision_completion_tokens: int = 0,
        vision_cached_tokens: int = 0,
        vision_total_tokens: int = 0,
    ) -> None:
        if not self.current_case or not self.current_case.steps:
            return
        artifact = self.current_case.steps[-1]
        artifact.action = action
        artifact.action_success = success
        artifact.action_message = message
        artifact.vision_prompt_tokens = vision_prompt_tokens
        artifact.vision_completion_tokens = vision_completion_tokens
        artifact.vision_cached_tokens = vision_cached_tokens
        artifact.vision_total_tokens = vision_total_tokens
        if success is False and message:
            self.current_case.issues.append(message)
        self._write_case_json(self.current_case)

    def finish_case(self, result_message: str, max_steps_reached: bool = False) -> str:
        if not self.current_case:
            return "UNKNOWN"
        case = self.current_case
        finished_at = _now()
        attempt_status = self._classify_status(case, result_message, max_steps_reached)
        if self.current_attempt:
            self.current_attempt.status = attempt_status
            self.current_attempt.result_message = result_message
            self.current_attempt.finished_at = finished_at
            self._stop_logcat_capture(
                keep=attempt_status != "PASS",
                attempt=self.current_attempt,
            )

        selected_attempt = self.current_attempt
        if case.attempts:
            pass_attempts = [attempt for attempt in case.attempts if attempt.status == "PASS"]
            selected_attempt = pass_attempts[-1] if pass_attempts else case.attempts[-1]

        case.result_message = selected_attempt.result_message if selected_attempt else result_message
        case.finished_at = finished_at
        case.status = selected_attempt.status if selected_attempt else attempt_status
        if selected_attempt:
            case.test_steps = selected_attempt.test_steps
            case.steps = selected_attempt.steps
            case.issues = selected_attempt.issues
        self._write_case_json(case)
        self._write_case_markdown(case)
        self._write_case_html(case)
        self._write_summary()
        self._write_summary_html()
        self.current_case = None
        self.current_attempt = None
        return case.status

    def finish_run(self) -> None:
        if self._logcat_process or self._logcat_file:
            self._stop_logcat_capture(
                keep=True,
                attempt=self.current_attempt,
            )
        self._write_run_metadata()
        self._write_summary()
        self._write_summary_html()

    def _classify_status(
        self, case: CaseReport, result_message: str, max_steps_reached: bool
    ) -> str:
        if max_steps_reached:
            return "BLOCKED"
        failed_actions = [s for s in case.steps if s.action_success is False]
        if failed_actions:
            return "FAIL"
        step_statuses = [s.status for s in case.test_steps]
        if any(status == "FAIL" for status in step_statuses):
            return "FAIL"

        executed_statuses = [
            status for status in step_statuses if status and status != "PENDING"
        ]
        if executed_statuses:
            last_status = executed_statuses[-1]
            if last_status in {"PASS", "SKIPPED"}:
                return "PASS"
            if last_status == "BLOCKED":
                return "BLOCKED"
            if last_status in {"UNKNOWN", "STEP_LIMIT", "REVIEW"}:
                return "REVIEW"

        if any(status in {"UNKNOWN", "STEP_LIMIT", "REVIEW"} for status in step_statuses):
            return "REVIEW"
        explicit = _parse_explicit_status(result_message or "")
        if explicit in {"FAIL", "BLOCKED", "REVIEW"}:
            return explicit
        return "PASS"

    def _collect_environment(self) -> dict[str, Any]:
        env = {
            "started_at": self.run_started_at,
            "device_type": self.device_type,
            "device_id": self.device_id,
            "model": self.model_name,
            "base_url": self.base_url,
            "apps": {},
        }
        if self.device_type == "adb":
            env.update(
                {
                    "device_model": self._run_device_shell("getprop ro.product.model"),
                    "device_brand": self._run_device_shell("getprop ro.product.brand"),
                    "android_version": self._run_device_shell(
                        "getprop ro.build.version.release"
                    ),
                    "sdk_version": self._run_device_shell("getprop ro.build.version.sdk"),
                }
            )
        return env

    def _collect_app_info(self, package_name: str) -> dict[str, str | None]:
        if self.device_type != "adb":
            return {"package": package_name}
        output = self._run_adb(["shell", "dumpsys", "package", package_name])
        version_name = _search_line_value(output, r"versionName=([^\s]+)")
        version_code = _search_line_value(output, r"versionCode=(\d+)")
        return {
            "package": package_name,
            "version_name": version_name,
            "version_code": version_code,
        }

    def _dump_ui_xml(self) -> str | None:
        if self.device_type == "ios":
            return self._dump_ios_source()
        if self.device_type != "adb":
            return None
        self._run_adb(["shell", "uiautomator", "dump", "/sdcard/window.xml"])
        xml = self._run_adb(["shell", "cat", "/sdcard/window.xml"])
        return xml if xml.strip().startswith("<?xml") else None

    def _get_current_activity(self) -> str | None:
        if self.device_type == "ios":
            return self._get_ios_current_app()
        if self.device_type != "adb":
            return None
        output = self._run_adb(["shell", "dumpsys", "window"])
        for line in output.splitlines():
            if "mCurrentFocus" in line or "mFocusedApp" in line:
                return line.strip()
        return None

    def _dump_ios_source(self) -> str | None:
        output = self._run_wda("source")
        if output.strip().startswith("<?xml") or "<XCUIElementType" in output:
            return output
        return None

    def _get_ios_current_app(self) -> str | None:
        output = self._run_wda("wda/activeAppInfo")
        if not output:
            return None
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return None
        value = data.get("value", {})
        bundle_id = value.get("bundleId")
        name = value.get("name")
        if bundle_id and name:
            return f"{bundle_id} ({name})"
        return bundle_id or name

    def _run_device_shell(self, command: str) -> str | None:
        output = self._run_adb(["shell", *command.split()])
        return output.strip() or None

    def _run_adb(self, args: list[str]) -> str:
        if self.device_type != "adb":
            return ""
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(args)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", timeout=15
            )
            return result.stdout
        except Exception:
            return ""

    def _adb_command(self, args: list[str]) -> list[str]:
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        return [*cmd, *args]

    def _start_logcat_capture(self, attempt: CaseAttemptReport) -> None:
        """Clear Android logs and start an isolated capture for one attempt."""
        if self.device_type != "adb":
            return
        if self._logcat_process or self._logcat_file:
            self._stop_logcat_capture(
                keep=True,
                attempt=self.current_attempt,
            )

        capture_path = Path(attempt.artifacts_dir) / "logcat.capture"
        try:
            cleared = subprocess.run(
                self._adb_command(["logcat", "-b", "all", "-c"]),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
            )
            if cleared.returncode != 0:
                detail = (cleared.stderr or cleared.stdout or "unknown error").strip()
                attempt.logcat_error = f"清空 logcat 失败: {detail}"
                return

            self._logcat_file = capture_path.open("wb")
            self._logcat_capture_path = capture_path
            self._logcat_process = subprocess.Popen(
                self._adb_command(["logcat", "-b", "all", "-v", "threadtime"]),
                stdout=self._logcat_file,
                stderr=subprocess.STDOUT,
            )
            print(f"Logcat capture started: {capture_path}")
        except Exception as exc:
            attempt.logcat_error = f"启动 logcat 采集失败: {exc}"
            self._close_logcat_file()
            try:
                capture_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _stop_logcat_capture(
        self,
        keep: bool,
        attempt: CaseAttemptReport | None,
    ) -> None:
        """Stop the active adb client and keep its output only for non-PASS attempts."""
        process = self._logcat_process
        capture_path = self._logcat_capture_path
        self._logcat_process = None
        self._logcat_capture_path = None
        if process:
            try:
                process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            except Exception as exc:
                if attempt and not attempt.logcat_error:
                    attempt.logcat_error = f"停止 logcat 采集失败: {exc}"
        self._close_logcat_file()

        if not capture_path:
            return
        if not keep:
            try:
                capture_path.unlink(missing_ok=True)
            except OSError as exc:
                if attempt:
                    attempt.logcat_error = f"删除 PASS 日志失败: {exc}"
            return

        final_path = capture_path.with_name("logcat.txt")
        try:
            capture_path.replace(final_path)
            if attempt:
                log_case_dir = capture_path.parent.parent
                attempt.logcat = str(final_path.relative_to(log_case_dir))
                attempt.logcat_size = final_path.stat().st_size
            print(f"Logcat retained: {final_path}")
        except OSError as exc:
            if attempt:
                attempt.logcat_error = f"保留 logcat 失败: {exc}"

    def _close_logcat_file(self) -> None:
        log_file = self._logcat_file
        self._logcat_file = None
        if log_file:
            try:
                log_file.close()
            except OSError:
                pass

    def _run_wda(self, endpoint: str) -> str:
        if self.device_type != "ios":
            return ""
        try:
            import requests

            base_url = self.wda_url.rstrip("/")
            response = requests.get(
                f"{base_url}/{endpoint.lstrip('/')}",
                timeout=10,
                verify=False,
            )
            if response.status_code != 200:
                return ""
            if endpoint == "source":
                data = response.json()
                return data.get("value", "")
            return response.text
        except Exception:
            return ""

    def _write_run_metadata(self) -> None:
        (self.root_dir / "run.json").write_text(
            json.dumps(
                {
                    "environment": self.environment,
                    "cases": [case.case_id for case in self.cases],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_case_json(self, case: CaseReport) -> None:
        path = Path(case.artifacts_dir) / "report.json"
        path.write_text(
            json.dumps(asdict(case), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_case_markdown(self, case: CaseReport) -> None:
        path = Path(case.artifacts_dir) / "report.md"
        last_step = case.steps[-1] if case.steps else None
        lines = [
            f"# {case.case_id}",
            "",
            f"- Status: {case.status}",
            f"- Attempts: {_attempt_count(case)}",
            f"- Started: {case.started_at}",
            f"- Finished: {case.finished_at}",
            f"- Result: {case.result_message or ''}",
            "",
            "## Task",
            "",
            case.task,
            "",
            "## Evidence",
            "",
        ]
        if last_step:
            lines.extend(
                [
                    f"- Last screenshot: {last_step.screenshot or ''}",
                    f"- Last UI XML: {last_step.ui_xml or ''}",
                    f"- Last activity: {last_step.current_activity or ''}",
                    f"- Last UI texts: {', '.join(last_step.ui_texts[:20])}",
                    "",
                ]
            )
        if case.issues:
            lines.extend(["## Issues", ""])
            lines.extend(f"- {issue}" for issue in case.issues)
            lines.append("")
        retained_logs = [attempt for attempt in _case_attempts(case) if attempt.logcat]
        log_errors = [
            attempt for attempt in _case_attempts(case) if attempt.logcat_error
        ]
        if retained_logs or log_errors:
            lines.extend(["## Android Logs", ""])
            for attempt in retained_logs:
                lines.append(
                    f"- Attempt {attempt.attempt}: [{attempt.logcat}]({attempt.logcat}) "
                    f"({_format_bytes(attempt.logcat_size)})"
                )
            for attempt in log_errors:
                lines.append(
                    f"- Attempt {attempt.attempt} collection error: "
                    f"{attempt.logcat_error}"
                )
            lines.append("")
        if case.test_steps:
            usage = _judge_usage_for_case(case)
            vision_usage = _vision_usage_for_case(case)
            lines.extend(
                [
                    "## Judge Token Usage",
                    "",
                    f"- Prompt tokens: {usage['prompt']}",
                    f"- Completion tokens: {usage['completion']}",
                    f"- Total tokens: {usage['total']}",
                    "",
                    "## Vision Token Usage",
                    "",
                    f"- Total tokens: {vision_usage['total']}",
                    "",
                ]
            )
        lines.extend(["## Steps", ""])
        for step in case.steps:
            action = step.action or {}
            action_name = action.get("action") or action.get("_metadata") or ""
            lines.append(
                f"- Step {step.step:03d}: {action_name} | "
                f"success={step.action_success} | "
                f"vision_tokens={step.vision_total_tokens} | "
                f"screenshot={step.screenshot}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_summary(self) -> None:
        summary_path = self.root_dir / "summary.md"
        lines = [
            "# Test Summary",
            "",
            f"- Started: {self.run_started_at}",
            f"- Artifact dir: {self.root_dir}",
            f"- Device: {self.environment.get('device_id')}",
            f"- Model: {self.model_name}",
            "",
            "## Environment",
            "",
        ]
        for key, value in self.environment.items():
            if key == "apps":
                continue
            lines.append(f"- {key}: {value}")
        if self.environment.get("apps"):
            lines.extend(["", "## Apps", ""])
            for package, info in self.environment["apps"].items():
                lines.append(
                    f"- {package}: {info.get('version_name')} "
                    f"({info.get('version_code')})"
                )
        usage = _judge_usage_for_run(self.cases)
        vision_usage = _vision_usage_for_run(self.cases)
        lines.extend(
            [
                "",
                "## Judge Token Usage",
                "",
                f"- Prompt tokens: {usage['prompt']}",
                f"- Completion tokens: {usage['completion']}",
                f"- Total tokens: {usage['total']}",
                "",
                "## Vision Token Usage",
                "",
                f"- Total tokens: {vision_usage['total']}",
            ]
        )
        lines.extend(["", "## Cases", ""])
        lines.append("| Case | Status | Attempts | Vision Tokens | Judge Tokens | Report | Last Screenshot |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- | --- |")
        for case in self.cases:
            last = case.steps[-1] if case.steps else None
            screenshot = f"{case.case_id}/{last.screenshot}" if last and last.screenshot else ""
            case_usage = _judge_usage_for_case(case)
            case_vision_usage = _vision_usage_for_case(case)
            lines.append(
                f"| {case.case_id} | {case.status} | {_attempt_count(case)} | {case_vision_usage['total']} | {case_usage['total']} | {case.case_id}/report.md | {screenshot} |"
            )
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_case_html(self, case: CaseReport) -> None:
        path = Path(case.artifacts_dir) / "report.html"
        meta = _case_metadata(case)
        judge_usage = _judge_usage_for_case(case)
        vision_usage = _vision_usage_for_case(case)
        lines = [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_h(case.case_id)} - Test Report</title>",
            f"<style>{_report_css()}</style>",
            "</head>",
            "<body>",
            '<header class="detail-bar">',
            '<a class="back-button" href="../index.html">← 汇总报告</a>',
            f'<span class="status {case.status.lower()}">{_h(case.status)}</span>',
            "</header>",
            '<main class="page">',
            '<section class="hero detail-hero">',
            f"<div><h1>{_h(case.case_id)}</h1><p>{_h(_display_title(case))}</p></div>",
            '<dl class="case-meta">',
            f"<dt>Type</dt><dd>{_h(meta.get('rule_type') or '-')}</dd>",
            f"<dt>Priority</dt><dd>{_h(meta.get('priority') or '-')}</dd>",
            f"<dt>Module</dt><dd>{_h(meta.get('module') or '-')}</dd>",
            f"<dt>Test Type</dt><dd>{_h(' / '.join(meta.get('test_types') or []) or '-')}</dd>",
            f"<dt>Duration</dt><dd>{_h(_case_duration(case))}</dd>",
            "</dl>",
            "</section>",
            '<section class="meta-grid">',
            _metric_card("开始时间", case.started_at),
            _metric_card("结束时间", case.finished_at or ""),
            _metric_card("执行时长", _case_duration(case)),
            _metric_card("测试步骤数", str(len(case.test_steps) or len(case.steps))),
            _metric_card("模型动作数", str(len(case.steps))),
            _metric_card("尝试次数", str(_attempt_count(case))),
            _metric_card("视觉模型 Tokens", str(vision_usage["total"])),
            _metric_card(
                "判定模型 Tokens",
                _format_token_usage(
                    judge_usage["prompt"],
                    judge_usage["completion"],
                    judge_usage["total"],
                ),
            ),
            "</section>",
            '<section class="panel task-panel">',
            "<h2>测试任务</h2>",
            '<div class="markdown-body">',
            _render_markdown(case.task),
            "</div>",
            "</section>",
            '<section class="panel result-panel">',
            '<div class="section-head"><div><h2>执行结果</h2><p class="subtle">模型最终输出或框架收尾信息，完整保留。</p></div></div>',
            f'<div class="result-message">{_render_result_message(case.result_message or "")}</div>',
            "</section>",
        ]
        if case.issues:
            lines.extend(
                [
                    '<section class="panel">',
                    "<h2>问题记录</h2>",
                    "<ul>",
                    *[f"<li>{_h(issue)}</li>" for issue in case.issues],
                    "</ul>",
                    "</section>",
                ]
            )

        lines.extend(['<section class="panel">', "<h2>步骤详情</h2>"])
        attempts = _case_attempts(case)
        if len(attempts) > 1:
            lines.append('<div class="attempt-tabs" role="tablist">')
            active_attempt = attempts[-1].attempt
            for attempt in attempts:
                active_class = " active" if attempt.attempt == active_attempt else ""
                lines.append(
                    f'<button class="attempt-tab{active_class}" type="button" '
                    f'data-attempt-tab="{attempt.attempt}">'
                    f'Attempt {attempt.attempt} · {_h(attempt.status)}</button>'
                )
            lines.append("</div>")
        for attempt in attempts:
            active_class = " active" if attempt == attempts[-1] else ""
            lines.append(_render_attempt_html(case, attempt, active_class))
        lines.extend(["</section>", "</main>", "</body>", "</html>"])
        lines.insert(-2, f"<script>{_report_js()}</script>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_summary_html(self) -> None:
        summary_path = self.root_dir / "index.html"
        summary_path.write_text(
            _build_summary_html(
                root_dir=self.root_dir,
                run_started_at=self.run_started_at,
                environment=self.environment,
                cases=self.cases,
            ),
            encoding="utf-8",
        )


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lower_text = text.lower()
    return any(keyword.lower() in lower_text for keyword in keywords)


def _parse_explicit_status(text: str) -> str | None:
    match = re.search(
        r"^\s*(?:STATUS|状态)\s*[:：]\s*(PASS|SKIPPED|BLOCKED|FAIL|REVIEW)\b",
        text or "",
        re.IGNORECASE | re.MULTILINE,
    )
    if match:
        return match.group(1).upper()
    match = re.search(r"^\s*(PASS|SKIPPED|BLOCKED|FAIL|REVIEW)\s*[:：]", text or "")
    return match.group(1).upper() if match else None


def _search_line_value(output: str, pattern: str) -> str | None:
    match = re.search(pattern, output)
    return match.group(1) if match else None


def _extract_ui_texts(xml: str) -> list[str]:
    if not xml.strip():
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return _extract_ui_texts_fallback(xml)

    values: list[str] = []
    root_bounds = _parse_bounds(root.attrib.get("bounds", ""))
    screen_width = root_bounds[2] if root_bounds else None
    screen_height = root_bounds[3] if root_bounds else None

    for node in root.iter():
        if not _node_is_visible(node, screen_width, screen_height):
            continue
        for attr in ("text", "content-desc", "name", "label", "value"):
            value = (node.attrib.get(attr) or "").strip()
            value = value.strip()
            if value and value not in values:
                values.append(value)
            if len(values) >= 80:
                return values
    return values


def _extract_ui_texts_fallback(xml: str) -> list[str]:
    values: list[str] = []
    for attr in ("text", "content-desc", "name", "label", "value"):
        for value in re.findall(fr'{attr}="([^"]+)"', xml):
            value = value.strip()
            if value and value not in values:
                values.append(value)
            if len(values) >= 80:
                return values
    return values


def _parse_bounds(value: str) -> tuple[int, int, int, int] | None:
    match = re.match(r"\[(\-?\d+),(\-?\d+)\]\[(\-?\d+),(\-?\d+)\]", value or "")
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _node_is_visible(
    node: ET.Element, screen_width: int | None, screen_height: int | None
) -> bool:
    bounds = _parse_bounds(node.attrib.get("bounds", ""))
    if not bounds:
        return True
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        return False
    if screen_width is not None and (right <= 0 or left >= screen_width):
        return False
    if screen_height is not None and (bottom <= 0 or top >= screen_height):
        return False
    return True


def _h(value: str) -> str:
    return html.escape(value, quote=True)


def _metric_card(label: str, value: str, tone: str = "") -> str:
    tone_class = f" metric-{tone}" if tone else ""
    return (
        f'<div class="metric{tone_class}">'
        f"<span>{_h(label)}</span>"
        f"<strong>{_h(value)}</strong>"
        "</div>"
    )


def _judge_usage_for_case(case: CaseReport) -> dict[str, int]:
    attempts = case.attempts or [
        CaseAttemptReport(
            attempt=1,
            test_steps=case.test_steps,
            steps=case.steps,
            issues=case.issues,
        )
    ]
    prompt = sum(
        step.judge_prompt_tokens for attempt in attempts for step in attempt.test_steps
    )
    completion = sum(
        step.judge_completion_tokens
        for attempt in attempts
        for step in attempt.test_steps
    )
    total = sum(
        step.judge_total_tokens for attempt in attempts for step in attempt.test_steps
    )
    return {"prompt": prompt, "completion": completion, "total": total}


def _judge_usage_for_run(cases: list[CaseReport]) -> dict[str, int]:
    prompt = 0
    completion = 0
    total = 0
    for case in cases:
        usage = _judge_usage_for_case(case)
        prompt += usage["prompt"]
        completion += usage["completion"]
        total += usage["total"]
    return {"prompt": prompt, "completion": completion, "total": total}


def _vision_usage_for_case(case: CaseReport) -> dict[str, int]:
    attempts = case.attempts or [
        CaseAttemptReport(
            attempt=1,
            test_steps=case.test_steps,
            steps=case.steps,
            issues=case.issues,
        )
    ]
    prompt = sum(
        step.vision_prompt_tokens for attempt in attempts for step in attempt.steps
    )
    completion = sum(
        step.vision_completion_tokens for attempt in attempts for step in attempt.steps
    )
    cached = sum(
        step.vision_cached_tokens for attempt in attempts for step in attempt.steps
    )
    total = sum(
        step.vision_total_tokens for attempt in attempts for step in attempt.steps
    )
    return {
        "prompt": prompt,
        "completion": completion,
        "cached": cached,
        "total": total,
    }


def _vision_usage_for_run(cases: list[CaseReport]) -> dict[str, int]:
    prompt = 0
    completion = 0
    cached = 0
    total = 0
    for case in cases:
        usage = _vision_usage_for_case(case)
        prompt += usage["prompt"]
        completion += usage["completion"]
        cached += usage["cached"]
        total += usage["total"]
    return {
        "prompt": prompt,
        "completion": completion,
        "cached": cached,
        "total": total,
    }


def _format_token_usage(prompt: int, completion: int, total: int) -> str:
    if not (prompt or completion or total):
        return "-"
    return f"{total} total / {prompt} prompt / {completion} completion"


def _build_summary_html(
    root_dir: Path,
    run_started_at: str,
    environment: dict[str, Any],
    cases: list[CaseReport],
) -> str:
    total = len(cases)
    pass_count = sum(1 for case in cases if case.status == "PASS")
    fail_count = sum(1 for case in cases if case.status == "FAIL")
    blocked_count = sum(1 for case in cases if case.status == "BLOCKED")
    other_count = total - pass_count - fail_count - blocked_count
    non_pass_cases = [case for case in cases if case.status != "PASS"]
    pass_rate = round((pass_count / total) * 100) if total else 0
    duration = _run_duration(cases)
    judge_usage = _judge_usage_for_run(cases)
    vision_usage = _vision_usage_for_run(cases)
    metadata = {case.case_id: _case_metadata(case) for case in cases}

    lines = [
        "<!doctype html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>AutoGLM Test Report</title>",
        f"<style>{_report_css()}</style>",
        "</head>",
        "<body>",
        '<main class="shell">',
        '<aside class="sidebar">',
        '<div class="brand"><span class="brand-mark"></span><div><strong>Test Report</strong><small>自动化测试报告</small></div></div>',
        '<nav class="side-nav"><a href="#overview">总览</a><a href="#module-stats">模块统计</a><a href="#tokens">Token 成本</a><a href="#cases">搜索与筛选</a><a href="#defects">非 PASS 用例</a><a href="#environment">环境信息</a></nav>',
        '<div class="side-foot">生成时间<br>' + _h(run_started_at) + "</div>",
        "</aside>",
        '<section class="content">',
        '<header class="topbar">',
        "<div>",
        "<h1>测试执行报告</h1>",
        f'<p class="subtle">{_h(str(root_dir))}</p>',
        "</div>",
        f'<div class="pass-ring" style="--value:{pass_rate}"><strong>{pass_rate}%</strong><span>PASS RATE</span></div>',
        "</header>",
        '<section id="overview" class="overview-grid">',
        _summary_tile("Total", str(total), "全部用例", "total"),
        _summary_tile("Passed", str(pass_count), "执行通过", "pass"),
        _summary_tile("Failed", str(fail_count), "需要排查", "fail"),
        _summary_tile("Blocked", str(blocked_count), "环境或流程阻塞", "blocked"),
        _summary_tile("Other", str(other_count), "未完成/未知", "other"),
        _summary_tile("Duration", duration, "首尾用例时间跨度", "duration"),
        _summary_tile("Vision Tokens", str(vision_usage["total"]), "视觉执行模型总消耗", "other"),
        _summary_tile("Judge Tokens", str(judge_usage["total"]), "文本判定模型总消耗", "other"),
        "</section>",
        _module_statistics_html(cases),
        _token_cost_panel(vision_usage, judge_usage),
        '<section class="panel split-panel">',
        '<div><h2>Status Overview</h2><p class="subtle">按最终状态聚合，便于先判断本轮执行质量。</p></div>',
        _status_bar(pass_count, fail_count, blocked_count, other_count),
        "</section>",
        '<section id="cases" class="panel">',
        '<div class="section-head"><div><h2>搜索与筛选</h2><p class="subtle">可组合筛选状态、模块、优先级、规则类型和测试类型。</p></div><span id="case-result-count" class="count-pill"></span></div>',
        _case_filter_html(cases),
        '<div id="case-filter-empty" class="empty-state filter-empty"><strong>没有匹配的测试用例</strong><span>请调整关键词或筛选条件。</span></div>',
        '<div class="case-list">',
    ]
    for case in cases:
        meta = metadata[case.case_id]
        report_link = f"{case.case_id}/report.html"
        issue_step = _find_issue_step(case)
        reason = _issue_reason(case, issue_step) if case.status != "PASS" else case.result_message or "Passed"
        case_usage = _judge_usage_for_case(case)
        case_vision_usage = _vision_usage_for_case(case)
        test_types = meta.get("test_types") or []
        tag_labels = [
            meta.get("module") or "未分模块",
            meta.get("priority") or "无优先级",
            meta.get("rule_type") or "无规则类型",
            *test_types,
        ]
        tags_html = "".join(
            f'<span class="case-tag">{_h(str(label))}</span>' for label in tag_labels
        )
        lines.append(
            '<a class="case-row" href="' + _h(report_link) + '" '
            f'data-search="{_h((case.case_id + " " + _display_title(case)).lower())}" '
            f'data-status="{_h(case.status)}" '
            f'data-module="{_h(str(meta.get("module") or ""))}" '
            f'data-priority="{_h(str(meta.get("priority") or ""))}" '
            f'data-rule-type="{_h(str(meta.get("rule_type") or ""))}" '
            f'data-test-types="{_h("|".join(test_types))}">'
            f'<span class="status-dot {case.status.lower()}"></span>'
            f'<span class="case-name"><strong>{_h(case.case_id)}</strong><em>{_h(_display_title(case))}</em><span class="case-tags">{tags_html}</span></span>'
            f'<span class="case-steps">{_case_step_count(case)} · Attempts {_attempt_count(case)} · {_h(_case_duration(case))} · Vision {case_vision_usage["total"]} · Judge {case_usage["total"]}</span>'
            f'<span class="case-reason">{_h(_shorten(reason, 110))}</span>'
            f'<span class="status {case.status.lower()}">{_h(case.status)}</span>'
            "</a>"
        )
    lines.extend(
        [
            "</div>",
            "</section>",
        '<section id="defects" class="panel">',
        '<div class="section-head"><div><h2>非 PASS 用例</h2><p class="subtle">优先查看失败截图、失败步骤和模型/动作输出原因。</p></div>',
        f'<span class="count-pill">{len(non_pass_cases)} items</span></div>',
        ]
    )

    if non_pass_cases:
        lines.append('<div class="defect-list">')
        for case in non_pass_cases:
            lines.append(_render_issue_case_html(case))
        lines.append("</div>")
    else:
        lines.append('<div class="empty-state"><strong>暂无失败或阻塞用例</strong><span>本轮所有用例均为 PASS。</span></div>')
    lines.append("</section>")

    lines.extend(
        [
            '<section id="environment" class="panel">',
            '<div class="section-head"><div><h2>环境信息</h2><p class="subtle">来自执行时采集的设备、模型和应用信息。</p></div></div>',
            '<dl class="env-list">',
        ]
    )
    for key, value in environment.items():
        if key == "apps":
            continue
        lines.append(f"<dt>{_h(str(key))}</dt><dd>{_h(str(value))}</dd>")
    lines.extend(
        [
            "</dl>",
            "</section>",
            "</section>",
            "</main>",
            f"<script>{_report_js()}</script>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(lines) + "\n"


def _summary_tile(label: str, value: str, note: str, tone: str) -> str:
    return (
        f'<article class="summary-tile tile-{tone}">'
        f"<span>{_h(label)}</span>"
        f"<strong>{_h(value)}</strong>"
        f"<em>{_h(note)}</em>"
        "</article>"
    )


def _case_filter_html(cases: list[CaseReport]) -> str:
    metadata = [_case_metadata(case) for case in cases]
    statuses = sorted({case.status for case in cases if case.status})
    modules = sorted({str(item["module"]) for item in metadata if item.get("module")})
    priorities = sorted(
        {str(item["priority"]) for item in metadata if item.get("priority")}
    )
    rule_types = sorted(
        {str(item["rule_type"]) for item in metadata if item.get("rule_type")}
    )
    test_types = sorted(
        {
            str(label)
            for item in metadata
            for label in (item.get("test_types") or [])
        }
    )
    return "\n".join(
        [
            '<div class="case-filters">',
            '<label class="search-field"><span>用例编号或标题</span><input id="case-search" type="search" placeholder="例如 TC-WEARFIT-WATCH-095"></label>',
            _filter_select("case-status-filter", "状态", statuses),
            _filter_select("case-module-filter", "关联模块", modules),
            _filter_select("case-priority-filter", "优先级", priorities),
            _filter_select("case-rule-filter", "规则类型", rule_types),
            _filter_select("case-test-type-filter", "测试类型", test_types),
            '<button id="case-filter-reset" class="filter-reset" type="button">重置筛选</button>',
            "</div>",
        ]
    )


def _filter_select(element_id: str, label: str, values: list[str]) -> str:
    options = ['<option value="">全部</option>']
    options.extend(f'<option value="{_h(value)}">{_h(value)}</option>' for value in values)
    return (
        f'<label class="filter-field"><span>{_h(label)}</span>'
        f'<select id="{_h(element_id)}">{"".join(options)}</select></label>'
    )


def _module_statistics_html(cases: list[CaseReport]) -> str:
    grouped: dict[str, list[CaseReport]] = {}
    for case in cases:
        module = str(_case_metadata(case).get("module") or "未分模块")
        grouped.setdefault(module, []).append(case)
    rows: list[str] = []
    for module, module_cases in sorted(grouped.items()):
        total = len(module_cases)
        counts = {
            status: sum(1 for case in module_cases if case.status == status)
            for status in ("PASS", "FAIL", "BLOCKED", "REVIEW")
        }
        other = total - sum(counts.values())
        priorities = _metadata_label_counts(module_cases, "priority")
        rules = _metadata_label_counts(module_cases, "rule_type")
        test_types = _metadata_label_counts(module_cases, "test_types")
        pass_rate = round(counts["PASS"] / total * 100) if total else 0
        rows.append(
            "<tr>"
            f"<td><strong>{_h(module)}</strong></td><td>{total}</td>"
            f'<td class="stat-pass">{counts["PASS"]}</td>'
            f'<td class="stat-fail">{counts["FAIL"]}</td>'
            f'<td class="stat-blocked">{counts["BLOCKED"]}</td>'
            f'<td class="stat-other">{counts["REVIEW"] + other}</td>'
            f"<td>{pass_rate}%</td>"
            f"<td>{_label_counts_html(priorities)}</td>"
            f"<td>{_label_counts_html(rules)}</td>"
            f"<td>{_label_counts_html(test_types)}</td>"
            "</tr>"
        )
    body = "".join(rows) or '<tr><td colspan="10">暂无用例</td></tr>'
    return "\n".join(
        [
            '<section id="module-stats" class="panel module-stats">',
            '<div class="section-head"><div><h2>模块统计</h2><p class="subtle">按关联模块汇总执行结论及标签分布。</p></div></div>',
            '<div class="stats-table-wrap"><table class="stats-table">',
            '<thead><tr><th>模块</th><th>总数</th><th>PASS</th><th>FAIL</th><th>BLOCKED</th><th>其他</th><th>通过率</th><th>优先级</th><th>规则类型</th><th>测试类型</th></tr></thead>',
            f"<tbody>{body}</tbody></table></div></section>",
        ]
    )


def _metadata_label_counts(cases: list[CaseReport], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        value = _case_metadata(case).get(key)
        values = value if isinstance(value, list) else [value]
        for label in values:
            if label:
                counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def _label_counts_html(counts: dict[str, int]) -> str:
    if not counts:
        return '<span class="muted">-</span>'
    return "".join(
        f'<span class="stat-tag">{_h(label)} {count}</span>'
        for label, count in sorted(counts.items())
    )


def _token_cost_panel(
    vision_usage: dict[str, int],
    judge_usage: dict[str, int],
) -> str:
    rows = [
        (
            "GLM 视觉执行模型",
            "执行点击、滑动、输入等手机操作",
            vision_usage,
            0.0,
            0.0,
        ),
        (
            "Judge 文本判定模型",
            "判定每个测试步骤 PASS / FAIL / BLOCKED / REVIEW",
            judge_usage,
            0.8,
            2.0,
        ),
    ]
    body = []
    for index, (name, note, usage, input_price, output_price) in enumerate(rows):
        cached = usage.get("cached", 0)
        body.append(
            '<tr class="token-row" '
            f'data-prompt="{usage["prompt"]}" '
            f'data-completion="{usage["completion"]}" '
            f'data-total="{usage["total"]}">'
            f'<td><strong>{_h(name)}</strong><span>{_h(note)}</span></td>'
            f'<td>{usage["prompt"]:,}</td>'
            f'<td>{usage["completion"]:,}</td>'
            f'<td>{cached:,}</td>'
            f'<td><strong>{usage["total"]:,}</strong></td>'
            '<td>'
            f'<input class="price-input token-input-price" type="number" min="0" step="0.01" value="{input_price:g}" aria-label="{_h(name)} 输入单价">'
            '</td>'
            '<td>'
            f'<input class="price-input token-output-price" type="number" min="0" step="0.01" value="{output_price:g}" aria-label="{_h(name)} 输出单价">'
            '</td>'
            f'<td><strong class="token-cost" data-cost-index="{index}">¥0.0000</strong></td>'
            "</tr>"
        )
    return "\n".join(
        [
            '<section id="tokens" class="panel token-panel">',
            '<div class="section-head">',
            "<div>",
            "<h2>Token 与成本</h2>",
            '<p class="subtle">单价单位为元 / 百万 token，可直接修改后自动重算。本地或未公开价格的模型默认按 0 计算。</p>',
            "</div>",
            '<div class="token-total-cost"><span>预估总成本</span><strong id="token-total-cost">¥0.0000</strong></div>',
            "</div>",
            '<div class="token-table-wrap">',
            '<table class="token-table">',
            "<thead><tr>"
            "<th>模型</th>"
            "<th>输入 Token</th>"
            "<th>输出 Token</th>"
            "<th>缓存 Token</th>"
            "<th>总 Token</th>"
            "<th>输入单价</th>"
            "<th>输出单价</th>"
            "<th>预估成本</th>"
            "</tr></thead>",
            f"<tbody>{''.join(body)}</tbody>",
            "</table>",
            "</div>",
            "</section>",
        ]
    )


def _status_bar(pass_count: int, fail_count: int, blocked_count: int, other_count: int) -> str:
    total = pass_count + fail_count + blocked_count + other_count
    parts = [
        ("pass", pass_count, "PASS"),
        ("fail", fail_count, "FAIL"),
        ("blocked", blocked_count, "BLOCKED"),
        ("other", other_count, "OTHER"),
    ]
    segments = []
    legend = []
    for tone, count, label in parts:
        width = (count / total * 100) if total else 0
        if count:
            segments.append(f'<span class="bar-{tone}" style="width:{width:.2f}%"></span>')
        legend.append(f'<span><i class="legend-{tone}"></i>{label} {count}</span>')
    return (
        '<div class="status-overview">'
        f'<div class="stacked-bar">{"".join(segments) or "<span></span>"}</div>'
        f'<div class="bar-legend">{"".join(legend)}</div>'
        "</div>"
    )


def _render_step_html(case: CaseReport, step: StepArtifact) -> str:
    screenshot = f"{step.screenshot}" if step.screenshot else ""
    action = json.dumps(step.action or {}, ensure_ascii=False, indent=2)
    ui_texts = "、".join(step.ui_texts[:30])
    screenshot_html = _screenshot_frame(screenshot, step.action, f"step {step.step} screenshot")
    return "\n".join(
        [
            '<article class="step-card">',
            '<div class="step-main">',
            f"<h3>Step {step.step:03d}</h3>",
            '<dl class="step-meta">',
            f"<dt>App</dt><dd>{_h(step.current_app)}</dd>",
            f"<dt>Activity</dt><dd>{_h(step.current_activity or '')}</dd>",
            f"<dt>动作成功</dt><dd>{_h(str(step.action_success))}</dd>",
            f"<dt>视觉 Tokens</dt><dd>{step.vision_total_tokens}</dd>",
            f"<dt>动作消息</dt><dd>{_h(step.action_message or '')}</dd>",
            f"<dt>UI 文本</dt><dd>{_h(ui_texts)}</dd>",
            "</dl>",
            f"<pre>{_h(action)}</pre>",
            "</div>",
            f'<div class="step-shot">{screenshot_html}</div>',
            "</article>",
        ]
    )


def _render_attempt_html(
    case: CaseReport, attempt: CaseAttemptReport, active_class: str = ""
) -> str:
    model_html: list[str] = []
    if attempt.test_steps:
        for test_step in attempt.test_steps:
            model_steps = [
                step for step in attempt.steps if step.test_step_index == test_step.index
            ]
            model_html.append(_render_test_step_html(case, test_step, model_steps))
    elif attempt.steps:
        for step in attempt.steps:
            model_html.append(_render_step_html(case, step))
    else:
        model_html.append('<div class="empty-state"><strong>无模型动作记录</strong></div>')

    vision_total = sum(step.vision_total_tokens for step in attempt.steps)
    judge_total = sum(step.judge_total_tokens for step in attempt.test_steps)
    logcat_html = _render_logcat_html(case, attempt)
    return "\n".join(
        [
            f'<div class="attempt-panel{active_class}" data-attempt-panel="{attempt.attempt}">',
            '<div class="attempt-summary">',
            f'<span class="status {attempt.status.lower()}">{_h(attempt.status)}</span>',
            f"<span>Attempt {attempt.attempt}</span>",
            f"<span>{_h(_attempt_duration(attempt))}</span>",
            f"<span>Vision {vision_total}</span>",
            f"<span>Judge {judge_total}</span>",
            "</div>",
            logcat_html,
            *model_html,
            "</div>",
        ]
    )


def _render_logcat_html(case: CaseReport, attempt: CaseAttemptReport) -> str:
    if not attempt.logcat and not attempt.logcat_error:
        return ""
    parts = ['<section class="logcat-panel">', '<div class="section-head">']
    parts.append('<div><h3>Android Logcat</h3><p class="subtle">仅非 PASS Attempt 保留。</p></div>')
    if attempt.logcat:
        parts.append(
            f'<a class="logcat-link" href="{_h(attempt.logcat)}">查看完整日志 · '
            f'{_h(_format_bytes(attempt.logcat_size))}</a>'
        )
    parts.append("</div>")
    if attempt.logcat_error:
        parts.append(f'<div class="logcat-error">{_h(attempt.logcat_error)}</div>')
    if attempt.logcat:
        log_path = Path(case.artifacts_dir) / attempt.logcat
        preview = _read_log_tail(log_path)
        if preview:
            parts.append('<details><summary>查看日志末尾（最多 400 行）</summary>')
            parts.append(f'<pre class="logcat-preview">{_h(preview)}</pre></details>')
    parts.append("</section>")
    return "\n".join(parts)


def _read_log_tail(path: Path, max_lines: int = 400, max_bytes: int = 256_000) -> str:
    try:
        with path.open("rb") as file:
            size = path.stat().st_size
            file.seek(max(0, size - max_bytes))
            data = file.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if size > max_bytes and lines:
            lines = lines[1:]
        return "\n".join(lines[-max_lines:])
    except OSError:
        return ""


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _render_test_step_html(
    case: CaseReport, test_step: TestStepArtifact, model_steps: list[StepArtifact]
) -> str:
    status_class = test_step.status.lower()
    token_usage = _format_token_usage(
        test_step.judge_prompt_tokens,
        test_step.judge_completion_tokens,
        test_step.judge_total_tokens,
    )
    vision_total_tokens = sum(step.vision_total_tokens for step in model_steps)
    model_html = []
    if model_steps:
        for step in model_steps:
            model_html.append(_render_step_html(case, step))
    else:
        model_html.append('<div class="empty-state"><strong>无模型动作记录</strong></div>')

    return "\n".join(
        [
            '<article class="test-step-card">',
            '<div class="test-step-head">',
            '<div>',
            f"<h3>测试步骤 {test_step.index}</h3>",
            f"<p>{_h(test_step.text)}</p>",
            "</div>",
            f'<span class="status {status_class}">{_h(test_step.status)}</span>',
            "</div>",
            '<dl class="step-meta test-step-meta">',
            f"<dt>目标状态</dt><dd>{_h(test_step.target_state or '')}</dd>",
            f"<dt>期望 Activity</dt><dd>{_h(test_step.activity or '')}</dd>",
            f"<dt>执行时长</dt><dd>{_h(_test_step_duration(test_step))}</dd>",
            f"<dt>视觉 Tokens</dt><dd>{vision_total_tokens}</dd>",
            f"<dt>判定 Tokens</dt><dd>{_h(token_usage)}</dd>",
            f"<dt>结果</dt><dd>{_h(test_step.result_message or '')}</dd>",
            f"<dt>判定原文</dt><dd>{_h(test_step.judge_raw or '')}</dd>",
            "</dl>",
            "<details open>",
            f"<summary>模型实现详情（{len(model_steps)} 个动作）</summary>",
            *model_html,
            "</details>",
            "</article>",
        ]
    )


def _render_issue_case_html(case: CaseReport) -> str:
    issue_step = _find_issue_step(case)
    screenshot = ""
    if issue_step and issue_step.screenshot:
        screenshot = f"{case.case_id}/{issue_step.screenshot}"
    reason = _issue_reason(case, issue_step)
    screenshot_html = _screenshot_frame(
        screenshot,
        issue_step.action if issue_step else None,
        f"{case.case_id} issue screenshot",
    )
    step_label = f"Step {issue_step.step:03d}" if issue_step else "无步骤"
    action = issue_step.action if issue_step else {}
    action_name = action.get("action") or action.get("_metadata") or "unknown"
    return "\n".join(
        [
            '<article class="defect-card">',
            '<div class="defect-shot">',
            screenshot_html,
            "</div>",
            '<div class="defect-body">',
            '<div class="defect-top">',
            f'<span class="status {case.status.lower()}">{_h(case.status)}</span>',
            f'<span class="step-chip">{_h(step_label)}</span>',
            "</div>",
            f"<h3>{_h(case.case_id)}</h3>",
            f'<p class="defect-title">{_h(_display_title(case))}</p>',
            '<div class="reason-box">',
            "<span>出错原因</span>",
            f"<strong>{_h(_shorten(reason, 220))}</strong>",
            "</div>",
            '<div class="action-strip">',
            f"<span>Action</span><code>{_h(action_name)}</code>",
            f"<span>Activity</span><code>{_h((issue_step.current_activity if issue_step else '') or 'unknown')}</code>",
            "</div>",
            f'<a class="detail-link" href="{_h(case.case_id)}/report.html">查看完整步骤</a>',
            "</div>",
            "</article>",
        ]
    )


def _find_issue_step(case: CaseReport) -> StepArtifact | None:
    for step in case.steps:
        if step.action_success is False:
            return step
    issue_test_step = _find_issue_test_step(case)
    if issue_test_step:
        for step in case.steps:
            if step.test_step_index == issue_test_step.index:
                return step
    return case.steps[-1] if case.steps else None


def _issue_reason(case: CaseReport, step: StepArtifact | None) -> str:
    issue_test_step = _find_issue_test_step(case)
    if issue_test_step and issue_test_step.result_message:
        return issue_test_step.result_message
    if step and step.action_message:
        return step.action_message
    if case.issues:
        return case.issues[-1]
    return case.result_message or ""


def _find_issue_test_step(case: CaseReport) -> TestStepArtifact | None:
    for status in ("FAIL", "BLOCKED", "UNKNOWN", "STEP_LIMIT", "REVIEW"):
        for step in case.test_steps:
            if step.status == status:
                return step
    return None


def _case_metadata(case: CaseReport) -> dict[str, Any]:
    match = re.search(r"^(TC-WEARFIT-(.+)-(\d{3})-(normal|audio))$", case.case_id)
    module = None
    rule_type = None
    if match:
        module = match.group(2)
        rule_type = match.group(4)
    priority = _task_list_field(case.task, "优先级")
    declared_module = _task_list_field(case.task, "关联模块")
    test_type = _task_list_field(case.task, "测试类型") or ""
    test_types = [
        value.strip()
        for value in re.split(r"\s*[/、|,，]\s*", test_type)
        if value.strip()
    ]
    return {
        "module": declared_module or module,
        "rule_type": rule_type,
        "priority": priority,
        "test_types": test_types,
    }


def _task_list_field(task: str, field_name: str) -> str | None:
    match = re.search(
        rf"^\s*-\s*{re.escape(field_name)}\s*[:：]\s*(.+?)\s*$",
        task,
        re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def _case_duration(case: CaseReport) -> str:
    if not case.started_at or not case.finished_at:
        return "-"
    try:
        start = datetime.fromisoformat(case.started_at)
        end = datetime.fromisoformat(case.finished_at)
        seconds = max(0, int((end - start).total_seconds()))
    except ValueError:
        return "-"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _attempt_duration(attempt: CaseAttemptReport) -> str:
    if not attempt.started_at or not attempt.finished_at:
        return "-"
    try:
        start = datetime.fromisoformat(attempt.started_at)
        end = datetime.fromisoformat(attempt.finished_at)
        seconds = max(0, int((end - start).total_seconds()))
    except ValueError:
        return "-"
    minutes, sec = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _test_step_duration(step: TestStepArtifact) -> str:
    if not step.started_at or not step.finished_at:
        return "-"
    try:
        start = datetime.fromisoformat(step.started_at)
        end = datetime.fromisoformat(step.finished_at)
        seconds = max(0, int((end - start).total_seconds()))
    except ValueError:
        return "-"
    minutes, sec = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _case_attempts(case: CaseReport) -> list[CaseAttemptReport]:
    if case.attempts:
        return case.attempts
    return [
        CaseAttemptReport(
            attempt=1,
            status=case.status,
            result_message=case.result_message,
            started_at=case.started_at,
            finished_at=case.finished_at,
            artifacts_dir=case.artifacts_dir,
            test_steps=case.test_steps,
            steps=case.steps,
            issues=case.issues,
        )
    ]


def _attempt_count(case: CaseReport) -> int:
    return len(case.attempts) if case.attempts else 1


def _case_step_count(case: CaseReport) -> str:
    if case.test_steps:
        return f"{len(case.test_steps)} test steps / {len(case.steps)} actions"
    return f"{len(case.steps)} steps"


def _render_result_message(message: str) -> str:
    if not message:
        return '<p class="muted">无结果信息。</p>'
    return _render_markdown(message)


def _screenshot_frame(screenshot: str, action: dict[str, Any] | None, alt: str) -> str:
    if not screenshot:
        return '<div class="empty-shot">无截图</div>'
    marker = _action_marker(action)
    marker_attrs = _action_marker_attrs(action)
    return (
        f'<a class="shot-frame" href="{_h(screenshot)}"{marker_attrs}>'
        f'<img src="{_h(screenshot)}" alt="{_h(alt)}">'
        '<span class="coord-overlay">'
        '<span class="origin-label">0,0</span>'
        '<span class="axis-label axis-x">x px</span><span class="axis-label axis-y">y px</span>'
        '<span class="tick-layer"></span>'
        "</span>"
        f"{marker}"
        "</a>"
        '<button class="coord-toggle" type="button">显示坐标</button>'
        '<div class="coord-caption">坐标：无点击坐标</div>'
    )


def _action_marker(action: dict[str, Any] | None) -> str:
    if not _action_marker_attrs(action):
        return ""
    return '<span class="tap-marker"><span></span></span>'


def _action_marker_attrs(action: dict[str, Any] | None) -> str:
    if not action:
        return ""
    point = action.get("element")
    if not (
        isinstance(point, list)
        and len(point) >= 2
        and isinstance(point[0], (int, float))
        and isinstance(point[1], (int, float))
    ):
        return ""
    x = max(0, min(1000, float(point[0])))
    y = max(0, min(1000, float(point[1])))
    return f' data-action-x="{x:.2f}" data-action-y="{y:.2f}"'


def _display_title(case: CaseReport) -> str:
    match = re.search(r"^##\s+TC-WEARFIT-\S+\s+(.+)$", case.task, re.MULTILINE)
    if match:
        return match.group(1).strip()
    title = str(case.title or "").strip()
    if title.startswith("## "):
        return title[3:].strip()
    return title


def _render_markdown(markdown: str) -> str:
    """Render a small Markdown subset used by generated test cases."""
    lines = markdown.splitlines()
    html_lines: list[str] = []
    in_list = False
    in_code = False
    code_lines: list[str] = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            html_lines.append("</ul>")
            in_list = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                html_lines.append(f"<pre><code>{_h(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code = False
            else:
                close_list()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not stripped:
            close_list()
            continue
        if stripped.startswith("### "):
            close_list()
            html_lines.append(f"<h3>{_h(stripped[4:])}</h3>")
            continue
        if stripped.startswith("## "):
            close_list()
            html_lines.append(f"<h2>{_h(stripped[3:])}</h2>")
            continue
        if stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{_inline_markdown(stripped[2:])}</li>")
            continue
        step_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if step_match:
            close_list()
            html_lines.append(
                f'<p class="md-step"><strong>{_h(step_match.group(1))}.</strong> '
                f"{_inline_markdown(step_match.group(2))}</p>"
            )
            continue
        html_lines.append(f"<p>{_inline_markdown(stripped)}</p>")

    close_list()
    if in_code:
        html_lines.append(f"<pre><code>{_h(chr(10).join(code_lines))}</code></pre>")
    return "\n".join(html_lines)


def _inline_markdown(text: str) -> str:
    escaped = _h(text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


def _shorten(value: str, max_len: int) -> str:
    value = " ".join(str(value).split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 1].rstrip() + "…"


def _run_duration(cases: list[CaseReport]) -> str:
    starts = [case.started_at for case in cases if case.started_at]
    ends = [case.finished_at for case in cases if case.finished_at]
    if not starts or not ends:
        return "-"
    try:
        start = datetime.fromisoformat(min(starts))
        end = datetime.fromisoformat(max(ends))
        seconds = max(0, int((end - start).total_seconds()))
    except ValueError:
        return "-"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _report_css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f4f6fb;
  --panel: #ffffff;
  --panel-2: #f8fafc;
  --text: #111827;
  --muted: #6b7280;
  --line: #e5e7eb;
  --pass: #16a34a;
  --fail: #dc2626;
  --blocked: #d97706;
  --review: #2563eb;
  --other: #64748b;
  --link: #2563eb;
  --shadow: 0 18px 45px rgba(15, 23, 42, .08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
.shell {
  display: grid;
  grid-template-columns: 248px minmax(0, 1fr);
  min-height: 100vh;
}
.sidebar {
  position: sticky;
  top: 0;
  height: 100vh;
  padding: 24px 18px;
  color: #dbeafe;
  background: linear-gradient(180deg, #0f172a 0%, #111827 55%, #1e1b4b 100%);
}
.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 30px;
}
.brand-mark {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: linear-gradient(135deg, #60a5fa, #22c55e);
  box-shadow: 0 10px 30px rgba(96, 165, 250, .35);
}
.brand strong { display: block; font-size: 22px; color: #fff; letter-spacing: 0; }
.brand small { display: block; color: #93c5fd; }
.side-nav {
  display: grid;
  gap: 8px;
}
.side-nav a {
  color: #cbd5e1;
  border-radius: 8px;
  padding: 10px 12px;
}
.side-nav a:hover {
  color: #fff;
  background: rgba(255,255,255,.08);
  text-decoration: none;
}
.side-foot {
  position: absolute;
  left: 18px;
  right: 18px;
  bottom: 22px;
  color: #94a3b8;
  font-size: 12px;
}
.content { padding: 28px 32px 42px; min-width: 0; }
.page { max-width: 1280px; margin: 0 auto; padding: 24px; }
.detail-bar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  height: 58px;
  padding: 0 24px;
  background: rgba(255,255,255,.92);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(14px);
}
.back-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #111827;
  background: #f1f5f9;
  border: 1px solid #dbe3ef;
  border-radius: 999px;
  padding: 8px 12px;
  font-weight: 700;
}
.back-button:hover {
  background: #e2e8f0;
  text-decoration: none;
}
.topbar,
.hero {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  margin-bottom: 18px;
}
h1, h2, h3 { margin: 0; line-height: 1.25; }
h1 { font-size: 30px; letter-spacing: 0; }
h2 { font-size: 18px; }
h3 { font-size: 15px; }
p { margin: 8px 0; }
.hero p, .muted, .subtle, .case-table span { color: var(--muted); }
.subtle { margin: 6px 0 0; }
.detail-hero {
  align-items: flex-start;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 18px;
  box-shadow: var(--shadow);
}
.case-meta {
  display: grid;
  grid-template-columns: auto auto;
  gap: 6px 14px;
  min-width: 280px;
  margin: 0;
  padding: 12px;
  border-radius: 12px;
  background: var(--panel-2);
}
.case-meta dt {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.case-meta dd {
  margin: 0;
  font-weight: 700;
  overflow-wrap: anywhere;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  margin-top: 16px;
  padding: 18px;
  box-shadow: var(--shadow);
}
.section-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 16px;
}
.count-pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  color: var(--muted);
  padding: 5px 10px;
  background: var(--panel-2);
}
.overview-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 16px;
}
.summary-tile {
  min-height: 118px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 16px;
  box-shadow: var(--shadow);
}
.summary-tile span,
.summary-tile em {
  display: block;
  color: var(--muted);
  font-style: normal;
}
.summary-tile strong {
  display: block;
  margin: 10px 0 8px;
  font-size: 30px;
  line-height: 1;
}
.tile-pass strong { color: var(--pass); }
.tile-fail strong { color: var(--fail); }
.tile-blocked strong { color: var(--blocked); }
.tile-other strong { color: var(--other); }
.token-panel {
  padding: 0;
  overflow: hidden;
}
.token-panel .section-head {
  margin: 0;
  padding: 18px;
  border-bottom: 1px solid var(--line);
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}
.token-total-cost {
  min-width: 180px;
  text-align: right;
}
.token-total-cost span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}
.token-total-cost strong {
  display: block;
  margin-top: 4px;
  color: #0f766e;
  font-size: 24px;
}
.token-table-wrap {
  overflow-x: auto;
}
.token-table {
  width: 100%;
  border-collapse: collapse;
  min-width: 900px;
}
.token-table th,
.token-table td {
  padding: 13px 14px;
  border-bottom: 1px solid var(--line);
  text-align: right;
  white-space: nowrap;
}
.token-table th:first-child,
.token-table td:first-child {
  text-align: left;
  white-space: normal;
  min-width: 260px;
}
.token-table th {
  color: #475569;
  background: #f8fafc;
  font-size: 12px;
  text-transform: uppercase;
}
.token-table td:first-child strong,
.token-table td:first-child span {
  display: block;
}
.token-table td:first-child span {
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
}
.token-cost {
  color: #0f766e;
}
.price-input {
  width: 96px;
  height: 34px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 0 8px;
  text-align: right;
  color: var(--text);
  background: #fff;
  font: inherit;
}
.price-input:focus {
  outline: 2px solid #bfdbfe;
  border-color: #60a5fa;
}
.pass-ring {
  width: 112px;
  height: 112px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  text-align: center;
  background:
    radial-gradient(circle closest-side, #fff 72%, transparent 73%),
    conic-gradient(var(--pass) calc(var(--value) * 1%), #e5e7eb 0);
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
}
.pass-ring strong { display: block; font-size: 24px; }
.pass-ring span { display: block; color: var(--muted); font-size: 10px; font-weight: 700; }
.split-panel {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  gap: 20px;
  align-items: center;
}
.status-overview { display: grid; gap: 10px; }
.stacked-bar {
  display: flex;
  height: 18px;
  overflow: hidden;
  border-radius: 999px;
  background: #e5e7eb;
}
.stacked-bar span { display: block; }
.bar-pass, .legend-pass { background: var(--pass); }
.bar-fail, .legend-fail { background: var(--fail); }
.bar-blocked, .legend-blocked { background: var(--blocked); }
.bar-other, .legend-other { background: var(--other); }
.bar-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  color: var(--muted);
  font-size: 12px;
}
.bar-legend i {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 999px;
  margin-right: 6px;
}
.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.metric {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px;
}
.metric span { display: block; color: var(--muted); margin-bottom: 6px; }
.metric strong { font-size: 22px; }
.metric-pass strong { color: var(--pass); }
.metric-fail strong { color: var(--fail); }
.metric-blocked strong { color: var(--blocked); }
.metric-other strong { color: var(--other); }
.status {
  display: inline-block;
  min-width: 74px;
  text-align: center;
  border-radius: 999px;
  padding: 4px 10px;
  color: #fff;
  font-weight: 700;
  font-size: 12px;
}
.status.pass { background: var(--pass); }
.status.skipped { background: var(--pass); }
.status.fail { background: var(--fail); }
.status.blocked { background: var(--blocked); }
.status.review,
.status.unknown,
.status.step_limit { background: var(--review); }
.status.running { background: var(--other); }
.case-filters {
  display: grid;
  grid-template-columns: minmax(240px, 2fr) repeat(5, minmax(120px, 1fr)) auto;
  gap: 10px;
  align-items: end;
  margin-bottom: 16px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel-2);
}
.search-field,
.filter-field { display: grid; gap: 5px; }
.search-field span,
.filter-field span {
  color: #475569;
  font-size: 12px;
  font-weight: 700;
}
.search-field input,
.filter-field select {
  width: 100%;
  height: 40px;
  border: 1px solid #cbd5e1;
  border-radius: 9px;
  padding: 0 10px;
  color: var(--text);
  background: #fff;
  font: inherit;
}
.search-field input:focus,
.filter-field select:focus {
  outline: 2px solid #bfdbfe;
  border-color: #60a5fa;
}
.filter-reset {
  height: 40px;
  border: 1px solid #cbd5e1;
  border-radius: 9px;
  padding: 0 14px;
  color: #334155;
  background: #fff;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
}
.filter-reset:hover { background: #eef2ff; border-color: #93c5fd; }
.filter-empty { display: none; margin-bottom: 12px; }
.filter-empty.visible { display: grid; }
.stats-table-wrap { overflow-x: auto; }
.stats-table { width: 100%; min-width: 1050px; border-collapse: collapse; }
.stats-table th,
.stats-table td { padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
.stats-table th { color: #475569; background: #f8fafc; font-size: 12px; white-space: nowrap; }
.stats-table .stat-pass { color: var(--pass); font-weight: 800; }
.stats-table .stat-fail { color: var(--fail); font-weight: 800; }
.stats-table .stat-blocked { color: var(--blocked); font-weight: 800; }
.stats-table .stat-other { color: var(--review); font-weight: 800; }
.stat-tag,
.case-tag {
  display: inline-flex;
  margin: 3px 4px 0 0;
  border: 1px solid #dbeafe;
  border-radius: 999px;
  padding: 2px 7px;
  color: #1e40af;
  background: #eff6ff;
  font-size: 11px;
  font-weight: 700;
  white-space: nowrap;
}
.case-tags { display: flex; flex-wrap: wrap; margin-top: 5px; }
.case-list { display: grid; gap: 8px; }
.case-row {
  display: grid;
  grid-template-columns: 12px minmax(260px, 1.2fr) 90px minmax(180px, 1fr) 92px;
  gap: 12px;
  align-items: center;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
  color: var(--text);
}
.case-row[hidden] { display: none; }
.case-row:hover {
  border-color: #93c5fd;
  box-shadow: 0 10px 26px rgba(37, 99, 235, .08);
  text-decoration: none;
}
.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
}
.status-dot.pass { background: var(--pass); }
.status-dot.fail { background: var(--fail); }
.status-dot.blocked { background: var(--blocked); }
.status-dot.review,
.status-dot.unknown,
.status-dot.step_limit { background: var(--review); }
.status-dot.running { background: var(--other); }
.case-name strong,
.case-name em {
  display: block;
  font-style: normal;
}
.case-name em,
.case-steps,
.case-reason { color: var(--muted); }
.case-reason { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.env-list, .step-meta {
  display: grid;
  grid-template-columns: 120px minmax(0, 1fr);
  gap: 6px 12px;
}
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  margin: 10px 0 0;
  max-height: 280px;
  overflow: auto;
}
.task-panel { padding-bottom: 0; }
.result-panel {
  border-left: 5px solid #2563eb;
}
.result-message {
  max-height: 360px;
  overflow: auto;
  padding: 14px;
  border: 1px solid #bfdbfe;
  border-radius: 12px;
  background: #eff6ff;
}
.result-message h2,
.result-message h3 {
  color: #1e3a8a;
}
.result-message p,
.result-message li {
  color: #172554;
}
.markdown-body {
  max-height: 520px;
  overflow: auto;
  padding: 4px 4px 16px;
  border-top: 1px solid var(--line);
}
.markdown-body h2 {
  margin: 16px 0 10px;
  font-size: 17px;
}
.markdown-body h3 {
  margin: 16px 0 8px;
  font-size: 14px;
  color: #334155;
}
.markdown-body p {
  margin: 7px 0;
}
.markdown-body ul {
  margin: 6px 0 12px 18px;
  padding: 0;
}
.markdown-body li {
  margin: 5px 0;
}
.markdown-body code {
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  color: #3730a3;
  border-radius: 6px;
  padding: 1px 5px;
}
.markdown-body .md-step {
  margin-top: 12px;
  padding: 9px 10px;
  border-radius: 10px;
  background: #f8fafc;
  border: 1px solid var(--line);
}
.defect-list { display: grid; gap: 14px; }
.defect-card {
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  gap: 16px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: linear-gradient(180deg, #fff 0%, #fbfcff 100%);
  padding: 12px;
}
.defect-shot .shot-frame,
.defect-shot .empty-shot {
  width: 100%;
  height: 240px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #eef2f7;
}
.defect-body { min-width: 0; }
.defect-top {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 10px;
}
.step-chip {
  color: #475569;
  background: #e2e8f0;
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 12px;
  font-weight: 700;
}
.defect-card h3 { font-size: 17px; }
.defect-title { color: var(--muted); margin: 4px 0 12px; }
.reason-box {
  border-left: 4px solid var(--fail);
  background: #fef2f2;
  border-radius: 10px;
  padding: 10px 12px;
  margin: 10px 0;
}
.reason-box span { display: block; color: #991b1b; font-size: 12px; font-weight: 700; }
.reason-box strong { display: block; margin-top: 4px; font-weight: 600; }
.action-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin: 12px 0;
}
.action-strip span {
  color: var(--muted);
  font-size: 12px;
}
.action-strip code {
  background: #eff6ff;
  color: #1d4ed8;
  border: 1px solid #bfdbfe;
  border-radius: 7px;
  padding: 3px 7px;
}
.detail-link {
  display: inline-flex;
  align-items: center;
  margin-top: 4px;
  font-weight: 700;
}
.test-step-card {
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  background: #f8fafc;
  padding: 16px;
  margin-bottom: 18px;
}
.test-step-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}
.test-step-head h3 {
  margin: 0 0 6px;
}
.test-step-head p {
  margin: 0;
  color: var(--text);
  line-height: 1.6;
}
.test-step-card details {
  margin-top: 12px;
}
.test-step-card summary {
  cursor: pointer;
  color: var(--muted);
  font-weight: 700;
  margin-bottom: 10px;
}
.attempt-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 14px 0;
  padding: 6px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel-2);
}
.attempt-tab {
  border: 1px solid transparent;
  border-radius: 9px;
  padding: 8px 12px;
  color: #475569;
  background: transparent;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
}
.attempt-tab.active {
  color: #0f172a;
  background: #fff;
  border-color: #cbd5e1;
  box-shadow: 0 6px 18px rgba(15, 23, 42, .08);
}
.attempt-panel {
  display: none;
}
.attempt-panel.active {
  display: block;
}
.attempt-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin: 12px 0;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
}
.attempt-summary span:not(.status) {
  color: #475569;
  background: #f1f5f9;
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 12px;
  font-weight: 700;
}
.logcat-panel {
  margin: 12px 0 16px;
  padding: 14px;
  border: 1px solid #fed7aa;
  border-radius: 12px;
  background: #fff7ed;
}
.logcat-panel .section-head { margin-bottom: 8px; }
.logcat-link { font-weight: 800; }
.logcat-error {
  margin: 8px 0;
  border-radius: 8px;
  padding: 9px 10px;
  color: #991b1b;
  background: #fef2f2;
}
.logcat-panel summary { cursor: pointer; color: #9a3412; font-weight: 700; }
.logcat-preview { max-height: 480px; color: #e2e8f0; background: #0f172a; }
.step-card {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #fff;
  padding: 12px;
}
.step-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 260px;
  gap: 14px;
  margin-bottom: 12px;
}
.step-shot .shot-frame,
.issue-shot .shot-frame {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #eef1f4;
}
.step-shot .shot-frame,
.issue-shot .shot-frame {
  height: auto;
}
.shot-frame {
  position: relative;
  display: block;
  overflow: hidden;
  background: #eef2f7;
}
.shot-frame img {
  display: block;
  width: 100%;
  height: auto;
  object-fit: contain;
}
.defect-shot .shot-frame img {
  height: 100%;
}
.coord-overlay,
.coord-overlay::before,
.coord-overlay::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  opacity: 0;
}
.coord-overlay {
  background:
    linear-gradient(to right, rgba(37, 99, 235, .18) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(37, 99, 235, .18) 1px, transparent 1px);
  background-size: 25% 25%;
}
.coord-overlay::before,
.coord-overlay::after {
  background: rgba(15, 23, 42, .28);
}
.coord-overlay::before {
  left: 0;
  right: 0;
  top: 0;
  height: 1px;
  bottom: auto;
  background: rgba(29, 78, 216, .75);
}
.coord-overlay::after {
  top: 0;
  bottom: 0;
  left: 0;
  width: 1px;
  right: auto;
  background: rgba(29, 78, 216, .75);
}
.show-coords .coord-overlay,
.show-coords .coord-overlay::before,
.show-coords .coord-overlay::after {
  opacity: 1;
}
.origin-label,
.axis-label,
.pixel-tick {
  position: absolute;
  z-index: 2;
  color: #1d4ed8;
  background: rgba(255,255,255,.86);
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  padding: 1px 6px;
  font-size: 11px;
  font-weight: 800;
  pointer-events: none;
  opacity: 0;
}
.show-coords .origin-label,
.show-coords .axis-label,
.show-coords .pixel-tick { opacity: 1; }
.origin-label { left: 6px; top: 6px; }
.axis-x { right: 8px; top: 6px; }
.axis-y { left: 6px; bottom: 8px; }
.pixel-tick::before {
  content: "";
  position: absolute;
  background: rgba(29, 78, 216, .7);
}
.tick-x {
  top: 6px;
  transform: translateX(-50%);
}
.tick-x::before {
  left: 50%;
  top: 20px;
  width: 1px;
  height: 12px;
}
.tick-y {
  left: 6px;
  transform: translateY(-50%);
}
.tick-y::before {
  left: 48px;
  top: 50%;
  width: 12px;
  height: 1px;
}
.tap-marker {
  position: absolute;
  z-index: 4;
  width: 12px;
  height: 12px;
  margin-left: -6px;
  margin-top: -6px;
  border-radius: 999px;
  background: #ef4444;
  border: 2px solid #fff;
  box-shadow: 0 0 0 2px rgba(239, 68, 68, .35), 0 8px 20px rgba(239, 68, 68, .35);
}
.tap-marker span {
  position: absolute;
  left: 14px;
  top: -9px;
  white-space: nowrap;
  color: #991b1b;
  background: #fff;
  border: 1px solid #fecaca;
  border-radius: 999px;
  padding: 2px 6px;
  font-size: 11px;
  font-weight: 800;
  opacity: 0;
}
.tap-marker:hover span,
.show-coords .tap-marker span { opacity: 1; }
.coord-toggle {
  margin-top: 8px;
  color: #1d4ed8;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
}
.coord-caption {
  margin-top: 6px;
  color: var(--muted);
  font-size: 12px;
  overflow-wrap: anywhere;
}
.empty-shot {
  min-height: 160px;
  display: grid;
  place-items: center;
  color: var(--muted);
  border: 1px dashed var(--line);
  border-radius: 10px;
  background: #f8fafc;
}
.empty-state {
  display: grid;
  place-items: center;
  gap: 4px;
  min-height: 150px;
  color: var(--muted);
  border: 1px dashed var(--line);
  border-radius: 14px;
  background: var(--panel-2);
}
.empty-state strong { color: var(--text); }
.empty-state span { display: block; }
.tile-duration strong { color: #2563eb; }
.tile-total strong { color: #111827; }
.back-link { color: var(--muted); }
@media (max-width: 760px) {
  .shell { grid-template-columns: 1fr; }
  .sidebar { position: static; height: auto; }
  .side-foot { position: static; margin-top: 28px; }
  .content { padding: 16px; }
  .page { padding: 14px; }
  .topbar, .hero, .section-head { flex-direction: column; align-items: flex-start; }
  .overview-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .split-panel { grid-template-columns: 1fr; }
  .case-row { grid-template-columns: 12px 1fr; }
  .case-steps, .case-reason, .case-row .status { grid-column: 2; }
  .defect-card { grid-template-columns: 1fr; }
  .step-card { grid-template-columns: 1fr; }
  .env-list, .step-meta { grid-template-columns: 1fr; }
  .case-filters { grid-template-columns: 1fr; }
}
@media (min-width: 761px) and (max-width: 1100px) {
  .overview-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .case-filters { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .search-field { grid-column: 1 / -1; }
}
"""


def _report_js() -> str:
    return r"""
(function () {
  function renderFrame(frame) {
    const img = frame.querySelector("img");
    const ticks = frame.querySelector(".tick-layer");
    const overlay = frame.querySelector(".coord-overlay");
    const marker = frame.querySelector(".tap-marker");
    if (!img || !ticks) return;

    const width = img.naturalWidth || img.clientWidth;
    const height = img.naturalHeight || img.clientHeight;
    if (!width || !height) return;

    const boxWidth = frame.clientWidth;
    const boxHeight = frame.clientHeight || img.clientHeight;
    const imageRatio = width / height;
    const boxRatio = boxWidth / boxHeight;
    let contentWidth = boxWidth;
    let contentHeight = boxHeight;
    let contentLeft = 0;
    let contentTop = 0;
    if (boxRatio > imageRatio) {
      contentHeight = boxHeight;
      contentWidth = contentHeight * imageRatio;
      contentLeft = (boxWidth - contentWidth) / 2;
    } else {
      contentWidth = boxWidth;
      contentHeight = contentWidth / imageRatio;
      contentTop = (boxHeight - contentHeight) / 2;
    }

    if (overlay) {
      overlay.style.left = contentLeft + "px";
      overlay.style.top = contentTop + "px";
      overlay.style.width = contentWidth + "px";
      overlay.style.height = contentHeight + "px";
    }

    ticks.innerHTML = "";
    const xCount = 4;
    const yCount = 4;
    for (let i = 1; i <= xCount; i += 1) {
      const pct = (i / xCount) * 100;
      const value = Math.round((i / xCount) * width);
      const tick = document.createElement("span");
      tick.className = "pixel-tick tick-x";
      tick.style.left = pct + "%";
      tick.textContent = value + "px";
      ticks.appendChild(tick);
    }
    for (let i = 1; i <= yCount; i += 1) {
      const pct = (i / yCount) * 100;
      const value = Math.round((i / yCount) * height);
      const tick = document.createElement("span");
      tick.className = "pixel-tick tick-y";
      tick.style.top = pct + "%";
      tick.textContent = value + "px";
      ticks.appendChild(tick);
    }

    if (marker && frame.dataset.actionX && frame.dataset.actionY) {
      const normalizedX = Number(frame.dataset.actionX);
      const normalizedY = Number(frame.dataset.actionY);
      const pixelX = Math.round((normalizedX / 1000) * width);
      const pixelY = Math.round((normalizedY / 1000) * height);
      marker.style.left = (contentLeft + (normalizedX / 1000) * contentWidth) + "px";
      marker.style.top = (contentTop + (normalizedY / 1000) * contentHeight) + "px";
      const label = marker.querySelector("span");
      if (label) {
        label.textContent = pixelX + "," + pixelY + "px";
      }
      const caption = frame.parentElement && frame.parentElement.querySelector(".coord-caption");
      if (caption) {
        caption.textContent = "点击坐标：model=[" + Math.round(normalizedX) + "," + Math.round(normalizedY) + "], pixel≈[" + pixelX + "," + pixelY + "]";
      }
    }
  }

  function init() {
    function updateCaseFilters() {
      const search = (document.getElementById("case-search") || {}).value || "";
      const status = (document.getElementById("case-status-filter") || {}).value || "";
      const module = (document.getElementById("case-module-filter") || {}).value || "";
      const priority = (document.getElementById("case-priority-filter") || {}).value || "";
      const ruleType = (document.getElementById("case-rule-filter") || {}).value || "";
      const testType = (document.getElementById("case-test-type-filter") || {}).value || "";
      const query = search.trim().toLocaleLowerCase();
      const rows = Array.from(document.querySelectorAll(".case-row"));
      let visible = 0;
      rows.forEach(function (row) {
        const testTypes = (row.dataset.testTypes || "").split("|");
        const matches = (!query || (row.dataset.search || "").includes(query))
          && (!status || row.dataset.status === status)
          && (!module || row.dataset.module === module)
          && (!priority || row.dataset.priority === priority)
          && (!ruleType || row.dataset.ruleType === ruleType)
          && (!testType || testTypes.includes(testType));
        row.hidden = !matches;
        if (matches) visible += 1;
      });
      const count = document.getElementById("case-result-count");
      if (count) count.textContent = "显示 " + visible + " / " + rows.length;
      const empty = document.getElementById("case-filter-empty");
      if (empty) empty.classList.toggle("visible", rows.length > 0 && visible === 0);
    }

    function updateTokenCosts() {
      let totalCost = 0;
      document.querySelectorAll(".token-row").forEach(function (row) {
        const prompt = Number(row.dataset.prompt || 0);
        const completion = Number(row.dataset.completion || 0);
        const inputPrice = Number((row.querySelector(".token-input-price") || {}).value || 0);
        const outputPrice = Number((row.querySelector(".token-output-price") || {}).value || 0);
        const cost = (prompt / 1000000) * inputPrice + (completion / 1000000) * outputPrice;
        totalCost += cost;
        const target = row.querySelector(".token-cost");
        if (target) {
          target.textContent = "¥" + cost.toFixed(4);
        }
      });
      const totalTarget = document.getElementById("token-total-cost");
      if (totalTarget) {
        totalTarget.textContent = "¥" + totalCost.toFixed(4);
      }
    }

    const frames = Array.from(document.querySelectorAll(".shot-frame"));
    frames.forEach(function (frame) {
      const img = frame.querySelector("img");
      if (img && !img.complete) {
        img.addEventListener("load", function () { renderFrame(frame); }, { once: true });
      } else {
        renderFrame(frame);
      }
    });
    window.addEventListener("resize", function () {
      frames.forEach(renderFrame);
    });
    document.querySelectorAll(".coord-toggle").forEach(function (button) {
      button.addEventListener("click", function (event) {
        event.preventDefault();
        const container = button.parentElement;
        const frame = container && container.querySelector(".shot-frame");
        if (!frame) return;
        frame.classList.toggle("show-coords");
        button.textContent = frame.classList.contains("show-coords") ? "隐藏坐标" : "显示坐标";
      });
    });
    document.querySelectorAll(".price-input").forEach(function (input) {
      input.addEventListener("input", updateTokenCosts);
    });
    document.querySelectorAll(".attempt-tab").forEach(function (button) {
      button.addEventListener("click", function () {
        const attempt = button.dataset.attemptTab;
        document.querySelectorAll(".attempt-tab").forEach(function (item) {
          item.classList.toggle("active", item === button);
        });
        document.querySelectorAll(".attempt-panel").forEach(function (panel) {
          panel.classList.toggle("active", panel.dataset.attemptPanel === attempt);
        });
        frames.forEach(renderFrame);
      });
    });
    [
      "case-search",
      "case-status-filter",
      "case-module-filter",
      "case-priority-filter",
      "case-rule-filter",
      "case-test-type-filter"
    ].forEach(function (id) {
      const control = document.getElementById(id);
      if (control) control.addEventListener("input", updateCaseFilters);
    });
    const reset = document.getElementById("case-filter-reset");
    if (reset) {
      reset.addEventListener("click", function () {
        [
          "case-search",
          "case-status-filter",
          "case-module-filter",
          "case-priority-filter",
          "case-rule-filter",
          "case-test-type-filter"
        ].forEach(function (id) {
          const control = document.getElementById(id);
          if (control) control.value = "";
        });
        updateCaseFilters();
      });
    }
    updateCaseFilters();
    updateTokenCosts();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
"""
