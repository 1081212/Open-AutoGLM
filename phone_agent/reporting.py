"""Test artifact and report generation utilities."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
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
    action: dict[str, Any] | None = None
    action_success: bool | None = None
    action_message: str | None = None
    sensitive_screenshot: bool = False
    ui_texts: list[str] = field(default_factory=list)


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
    steps: list[StepArtifact] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


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
    ) -> None:
        run_name = artifact_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root_dir = Path(base_dir) / sanitize_name(run_name)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.device_type = device_type
        self.device_id = device_id
        self.model_name = model_name
        self.base_url = base_url
        self.cases: list[CaseReport] = []
        self.current_case: CaseReport | None = None
        self.run_started_at = _now()
        self.environment: dict[str, Any] = self._collect_environment()

    def start_case(self, task: str, index: int) -> CaseReport:
        case_id = extract_case_id(task, index)
        title = task.split("：", 1)[0].split(":", 1)[0][:120]
        case_dir = self.root_dir / case_id
        (case_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (case_dir / "ui").mkdir(parents=True, exist_ok=True)
        case = CaseReport(
            case_id=case_id,
            title=title,
            task=task,
            artifacts_dir=str(case_dir),
        )
        self.cases.append(case)
        self.current_case = case
        package_name = extract_package_name(task)
        if package_name:
            self.environment.setdefault("apps", {})[package_name] = (
                self._collect_app_info(package_name)
            )
        self._write_run_metadata()
        return case

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
        screenshot_rel = f"screenshots/step_{step:03d}.png"
        screenshot_path = case_dir / screenshot_rel
        try:
            screenshot_path.write_bytes(base64.b64decode(screenshot_base64))
        except Exception:
            screenshot_rel = None

        ui_rel = None
        current_activity = self._get_current_activity()
        xml = self._dump_ui_xml()
        if xml:
            ui_rel = f"ui/step_{step:03d}.xml"
            (case_dir / ui_rel).write_text(xml, encoding="utf-8")
        ui_texts = _extract_ui_texts(xml or "")

        artifact = StepArtifact(
            step=step,
            screenshot=screenshot_rel,
            ui_xml=ui_rel,
            current_app=current_app,
            current_activity=current_activity,
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
    ) -> None:
        if not self.current_case or not self.current_case.steps:
            return
        artifact = self.current_case.steps[-1]
        if artifact.step == step:
            artifact.action = action
            artifact.action_success = success
            artifact.action_message = message
        if message and _contains_any(message, FAIL_KEYWORDS + BLOCKED_KEYWORDS):
            self.current_case.issues.append(message)
        self._write_case_json(self.current_case)

    def finish_case(self, result_message: str, max_steps_reached: bool = False) -> str:
        if not self.current_case:
            return "UNKNOWN"
        case = self.current_case
        case.result_message = result_message
        case.finished_at = _now()
        case.status = self._classify_status(case, result_message, max_steps_reached)
        self._write_case_json(case)
        self._write_case_markdown(case)
        self._write_summary()
        self.current_case = None
        return case.status

    def finish_run(self) -> None:
        self._write_run_metadata()
        self._write_summary()

    def _classify_status(
        self, case: CaseReport, result_message: str, max_steps_reached: bool
    ) -> str:
        text = " ".join([result_message or "", *case.issues]).lower()
        if max_steps_reached or _contains_any(text, BLOCKED_KEYWORDS):
            return "BLOCKED"
        if _contains_any(text, FAIL_KEYWORDS):
            return "FAIL"
        failed_actions = [s for s in case.steps if s.action_success is False]
        if failed_actions:
            return "FAIL"
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
        if self.device_type != "adb":
            return None
        self._run_adb(["shell", "uiautomator", "dump", "/sdcard/window.xml"])
        xml = self._run_adb(["shell", "cat", "/sdcard/window.xml"])
        return xml if xml.strip().startswith("<?xml") else None

    def _get_current_activity(self) -> str | None:
        if self.device_type != "adb":
            return None
        output = self._run_adb(["shell", "dumpsys", "window"])
        for line in output.splitlines():
            if "mCurrentFocus" in line or "mFocusedApp" in line:
                return line.strip()
        return None

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
        lines.extend(["## Steps", ""])
        for step in case.steps:
            action = step.action or {}
            action_name = action.get("action") or action.get("_metadata") or ""
            lines.append(
                f"- Step {step.step:03d}: {action_name} | "
                f"success={step.action_success} | screenshot={step.screenshot}"
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
        lines.extend(["", "## Cases", ""])
        lines.append("| Case | Status | Report | Last Screenshot |")
        lines.append("| --- | --- | --- | --- |")
        for case in self.cases:
            last = case.steps[-1] if case.steps else None
            screenshot = f"{case.case_id}/{last.screenshot}" if last and last.screenshot else ""
            lines.append(
                f"| {case.case_id} | {case.status} | {case.case_id}/report.md | {screenshot} |"
            )
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lower_text = text.lower()
    return any(keyword.lower() in lower_text for keyword in keywords)


def _search_line_value(output: str, pattern: str) -> str | None:
    match = re.search(pattern, output)
    return match.group(1) if match else None


def _extract_ui_texts(xml: str) -> list[str]:
    values: list[str] = []
    for attr in ("text", "content-desc", "resource-id"):
        for value in re.findall(fr'{attr}="([^"]+)"', xml):
            value = value.strip()
            if value and value not in values:
                values.append(value)
            if len(values) >= 80:
                return values
    return values
