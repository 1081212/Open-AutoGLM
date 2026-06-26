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
        self.current_test_step_index: int | None = None
        self._case_step_counter = 0
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
        self.current_test_step_index = None
        self._case_step_counter = 0
        package_name = extract_package_name(task)
        if package_name:
            self.environment.setdefault("apps", {})[package_name] = (
                self._collect_app_info(package_name)
            )
        self._write_run_metadata()
        return case

    def set_test_steps(self, steps: list[dict[str, Any]]) -> None:
        """Register parsed Markdown test steps for the current case."""
        if not self.current_case:
            return
        self.current_case.test_steps = [
            TestStepArtifact(
                index=int(step["index"]),
                text=str(step["text"]),
                target_state=step.get("target_state"),
                activity=step.get("activity"),
            )
            for step in steps
        ]
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
        self._case_step_counter += 1
        artifact_step = self._case_step_counter
        screenshot_rel = f"screenshots/step_{artifact_step:03d}.png"
        screenshot_path = case_dir / screenshot_rel
        try:
            screenshot_path.write_bytes(base64.b64decode(screenshot_base64))
        except Exception:
            screenshot_rel = None

        ui_rel = None
        current_activity = self._get_current_activity()
        xml = self._dump_ui_xml()
        if xml:
            ui_rel = f"ui/step_{artifact_step:03d}.xml"
            (case_dir / ui_rel).write_text(xml, encoding="utf-8")
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
    ) -> None:
        if not self.current_case or not self.current_case.steps:
            return
        artifact = self.current_case.steps[-1]
        artifact.action = action
        artifact.action_success = success
        artifact.action_message = message
        if success is False and message:
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
        self._write_case_html(case)
        self._write_summary()
        self._write_summary_html()
        self.current_case = None
        return case.status

    def finish_run(self) -> None:
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
        if case.test_steps:
            usage = _judge_usage_for_case(case)
            lines.extend(
                [
                    "## Judge Token Usage",
                    "",
                    f"- Prompt tokens: {usage['prompt']}",
                    f"- Completion tokens: {usage['completion']}",
                    f"- Total tokens: {usage['total']}",
                    "",
                ]
            )
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
        usage = _judge_usage_for_run(self.cases)
        lines.extend(
            [
                "",
                "## Judge Token Usage",
                "",
                f"- Prompt tokens: {usage['prompt']}",
                f"- Completion tokens: {usage['completion']}",
                f"- Total tokens: {usage['total']}",
            ]
        )
        lines.extend(["", "## Cases", ""])
        lines.append("| Case | Status | Judge Tokens | Report | Last Screenshot |")
        lines.append("| --- | --- | ---: | --- | --- |")
        for case in self.cases:
            last = case.steps[-1] if case.steps else None
            screenshot = f"{case.case_id}/{last.screenshot}" if last and last.screenshot else ""
            case_usage = _judge_usage_for_case(case)
            lines.append(
                f"| {case.case_id} | {case.status} | {case_usage['total']} | {case.case_id}/report.md | {screenshot} |"
            )
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_case_html(self, case: CaseReport) -> None:
        path = Path(case.artifacts_dir) / "report.html"
        meta = _case_metadata(case)
        judge_usage = _judge_usage_for_case(case)
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
            f"<dt>Duration</dt><dd>{_h(_case_duration(case))}</dd>",
            "</dl>",
            "</section>",
            '<section class="meta-grid">',
            _metric_card("开始时间", case.started_at),
            _metric_card("结束时间", case.finished_at or ""),
            _metric_card("执行时长", _case_duration(case)),
            _metric_card("测试步骤数", str(len(case.test_steps) or len(case.steps))),
            _metric_card("模型动作数", str(len(case.steps))),
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
        if case.test_steps:
            for test_step in case.test_steps:
                model_steps = [
                    step
                    for step in case.steps
                    if step.test_step_index == test_step.index
                ]
                lines.append(_render_test_step_html(case, test_step, model_steps))
        else:
            for step in case.steps:
                lines.append(_render_step_html(case, step))
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
    prompt = sum(step.judge_prompt_tokens for step in case.test_steps)
    completion = sum(step.judge_completion_tokens for step in case.test_steps)
    total = sum(step.judge_total_tokens for step in case.test_steps)
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
        '<nav class="side-nav"><a href="#overview">总览</a><a href="#cases">全部用例</a><a href="#defects">非 PASS 用例</a><a href="#environment">环境信息</a></nav>',
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
        _summary_tile("Judge Tokens", str(judge_usage["total"]), "文本判定模型总消耗", "other"),
        "</section>",
        '<section class="panel split-panel">',
        '<div><h2>Status Overview</h2><p class="subtle">按最终状态聚合，便于先判断本轮执行质量。</p></div>',
        _status_bar(pass_count, fail_count, blocked_count, other_count),
        "</section>",
        '<section id="cases" class="panel">',
        '<div class="section-head"><div><h2>全部用例</h2><p class="subtle">点击详情查看完整步骤、截图和动作记录。</p></div></div>',
        '<div class="case-list">',
    ]
    for case in cases:
        report_link = f"{case.case_id}/report.html"
        issue_step = _find_issue_step(case)
        reason = _issue_reason(case, issue_step) if case.status != "PASS" else case.result_message or "Passed"
        case_usage = _judge_usage_for_case(case)
        lines.append(
            '<a class="case-row" href="' + _h(report_link) + '">'
            f'<span class="status-dot {case.status.lower()}"></span>'
            f'<span class="case-name"><strong>{_h(case.case_id)}</strong><em>{_h(_display_title(case))}</em></span>'
            f'<span class="case-steps">{_case_step_count(case)} · {_h(_case_duration(case))} · Judge {case_usage["total"]}</span>'
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
            f"<dt>动作消息</dt><dd>{_h(step.action_message or '')}</dd>",
            f"<dt>UI 文本</dt><dd>{_h(ui_texts)}</dd>",
            "</dl>",
            f"<pre>{_h(action)}</pre>",
            "</div>",
            f'<div class="step-shot">{screenshot_html}</div>',
            "</article>",
        ]
    )


def _render_test_step_html(
    case: CaseReport, test_step: TestStepArtifact, model_steps: list[StepArtifact]
) -> str:
    status_class = test_step.status.lower()
    token_usage = _format_token_usage(
        test_step.judge_prompt_tokens,
        test_step.judge_completion_tokens,
        test_step.judge_total_tokens,
    )
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


def _case_metadata(case: CaseReport) -> dict[str, str | None]:
    match = re.search(r"^(TC-WEARFIT-(.+)-(\d{3})-(normal|audio))$", case.case_id)
    module = None
    rule_type = None
    if match:
        module = match.group(2)
        rule_type = match.group(4)
    priority_match = re.search(r"^\s*-\s*优先级\s*[:：]\s*(.+?)\s*$", case.task, re.MULTILINE)
    return {
        "module": module,
        "rule_type": rule_type,
        "priority": priority_match.group(1).strip() if priority_match else None,
    }


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
}
@media (max-width: 1100px) {
  .overview-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
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
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
"""
