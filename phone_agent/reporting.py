"""Test artifact and report generation utilities."""

from __future__ import annotations

import base64
import html
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

    def _write_case_html(self, case: CaseReport) -> None:
        path = Path(case.artifacts_dir) / "report.html"
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
            '<main class="page">',
            '<p><a class="back-link" href="../index.html">返回汇总报告</a></p>',
            '<section class="hero">',
            f"<div><h1>{_h(case.case_id)}</h1><p>{_h(case.title)}</p></div>",
            f'<span class="status {case.status.lower()}">{_h(case.status)}</span>',
            "</section>",
            '<section class="meta-grid">',
            _metric_card("开始时间", case.started_at),
            _metric_card("结束时间", case.finished_at or ""),
            _metric_card("步骤数", str(len(case.steps))),
            _metric_card("结果", case.result_message or ""),
            "</section>",
            '<section class="panel">',
            "<h2>测试任务</h2>",
            f"<pre>{_h(case.task)}</pre>",
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
        for step in case.steps:
            lines.append(_render_step_html(case, step))
        lines.extend(["</section>", "</main>", "</body>", "</html>"])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_summary_html(self) -> None:
        summary_path = self.root_dir / "index.html"
        total = len(self.cases)
        pass_count = sum(1 for case in self.cases if case.status == "PASS")
        fail_count = sum(1 for case in self.cases if case.status == "FAIL")
        blocked_count = sum(1 for case in self.cases if case.status == "BLOCKED")
        other_count = total - pass_count - fail_count - blocked_count
        non_pass_cases = [case for case in self.cases if case.status != "PASS"]

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
            '<main class="page">',
            '<section class="hero">',
            "<div><h1>AutoGLM 测试报告</h1>",
            f"<p>{_h(self.run_started_at)} · {_h(str(self.root_dir))}</p></div>",
            "</section>",
            '<section class="meta-grid">',
            _metric_card("用例总数", str(total)),
            _metric_card("成功", str(pass_count), "pass"),
            _metric_card("失败", str(fail_count), "fail"),
            _metric_card("阻塞", str(blocked_count), "blocked"),
            _metric_card("其他", str(other_count), "other"),
            "</section>",
            '<section class="panel">',
            "<h2>环境信息</h2>",
            '<dl class="env-list">',
        ]
        for key, value in self.environment.items():
            if key == "apps":
                continue
            lines.append(f"<dt>{_h(str(key))}</dt><dd>{_h(str(value))}</dd>")
        lines.extend(["</dl>", "</section>"])

        lines.extend(['<section class="panel">', "<h2>非 PASS 用例</h2>"])
        if non_pass_cases:
            lines.append('<div class="issue-grid">')
            for case in non_pass_cases:
                lines.append(_render_issue_case_html(case))
            lines.append("</div>")
        else:
            lines.append('<p class="muted">暂无非 PASS 用例。</p>')
        lines.append("</section>")

        lines.extend(
            [
                '<section class="panel">',
                "<h2>全部用例</h2>",
                '<table class="case-table">',
                "<thead><tr><th>用例</th><th>状态</th><th>步骤</th><th>结果</th><th>报告</th></tr></thead>",
                "<tbody>",
            ]
        )
        for case in self.cases:
            report_link = f"{case.case_id}/report.html"
            lines.append(
                "<tr>"
                f"<td><strong>{_h(case.case_id)}</strong><br><span>{_h(case.title)}</span></td>"
                f'<td><span class="status {case.status.lower()}">{_h(case.status)}</span></td>'
                f"<td>{len(case.steps)}</td>"
                f"<td>{_h(case.result_message or '')}</td>"
                f'<td><a href="{_h(report_link)}">查看详情</a></td>'
                "</tr>"
            )
        lines.extend(["</tbody>", "</table>", "</section>", "</main>", "</body>", "</html>"])
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


def _render_step_html(case: CaseReport, step: StepArtifact) -> str:
    screenshot = f"{step.screenshot}" if step.screenshot else ""
    action = json.dumps(step.action or {}, ensure_ascii=False, indent=2)
    ui_texts = "、".join(step.ui_texts[:30])
    screenshot_html = (
        f'<a href="{_h(screenshot)}"><img src="{_h(screenshot)}" alt="step {step.step} screenshot"></a>'
        if screenshot
        else '<div class="empty-shot">无截图</div>'
    )
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


def _render_issue_case_html(case: CaseReport) -> str:
    issue_step = _find_issue_step(case)
    screenshot = ""
    if issue_step and issue_step.screenshot:
        screenshot = f"{case.case_id}/{issue_step.screenshot}"
    reason = _issue_reason(case, issue_step)
    screenshot_html = (
        f'<a href="{_h(screenshot)}"><img src="{_h(screenshot)}" alt="{_h(case.case_id)} issue screenshot"></a>'
        if screenshot
        else '<div class="empty-shot">无截图</div>'
    )
    step_label = f"Step {issue_step.step:03d}" if issue_step else "无步骤"
    action = json.dumps(issue_step.action if issue_step else {}, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            '<article class="issue-card">',
            '<div class="issue-head">',
            f"<h3>{_h(case.case_id)}</h3>",
            f'<span class="status {case.status.lower()}">{_h(case.status)}</span>',
            "</div>",
            f"<p>{_h(case.title)}</p>",
            f"<p><strong>出错步骤：</strong>{_h(step_label)}</p>",
            f"<p><strong>出错原因：</strong>{_h(reason)}</p>",
            f"<pre>{_h(action)}</pre>",
            f'<div class="issue-shot">{screenshot_html}</div>',
            f'<p><a href="{_h(case.case_id)}/report.html">查看完整步骤</a></p>',
            "</article>",
        ]
    )


def _find_issue_step(case: CaseReport) -> StepArtifact | None:
    for step in case.steps:
        if step.action_success is False:
            return step
    for step in case.steps:
        if step.action_message and _contains_any(
            step.action_message, FAIL_KEYWORDS + BLOCKED_KEYWORDS
        ):
            return step
    return case.steps[-1] if case.steps else None


def _issue_reason(case: CaseReport, step: StepArtifact | None) -> str:
    if step and step.action_message:
        return step.action_message
    if case.issues:
        return case.issues[-1]
    return case.result_message or ""


def _report_css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #17202a;
  --muted: #667085;
  --line: #d8dde5;
  --pass: #1f8f4d;
  --fail: #c93535;
  --blocked: #a15c00;
  --other: #475467;
  --link: #1457b8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
.page { max-width: 1280px; margin: 0 auto; padding: 24px; }
.hero {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  padding: 20px 0 18px;
  border-bottom: 1px solid var(--line);
}
h1, h2, h3 { margin: 0; line-height: 1.25; }
h1 { font-size: 28px; }
h2 { font-size: 18px; margin-bottom: 14px; }
h3 { font-size: 15px; }
p { margin: 8px 0; }
.hero p, .muted, .case-table span { color: var(--muted); }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-top: 16px;
  padding: 16px;
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
  border-radius: 8px;
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
  padding: 3px 10px;
  color: #fff;
  font-weight: 700;
  font-size: 12px;
}
.status.pass { background: var(--pass); }
.status.fail { background: var(--fail); }
.status.blocked { background: var(--blocked); }
.status.running { background: var(--other); }
.case-table {
  width: 100%;
  border-collapse: collapse;
}
.case-table th, .case-table td {
  text-align: left;
  border-bottom: 1px solid var(--line);
  padding: 10px 8px;
  vertical-align: top;
}
.case-table th { color: var(--muted); font-weight: 600; }
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
  background: #f1f3f5;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
  margin: 10px 0 0;
  max-height: 280px;
  overflow: auto;
}
.issue-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 12px;
}
.issue-card, .step-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  padding: 12px;
}
.issue-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}
.step-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 260px;
  gap: 14px;
  margin-bottom: 12px;
}
.step-shot img, .issue-shot img {
  width: 100%;
  max-height: 520px;
  object-fit: contain;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #eef1f4;
}
.empty-shot {
  min-height: 160px;
  display: grid;
  place-items: center;
  color: var(--muted);
  border: 1px dashed var(--line);
  border-radius: 6px;
}
.back-link { color: var(--muted); }
@media (max-width: 760px) {
  .page { padding: 14px; }
  .hero, .issue-head { flex-direction: column; }
  .step-card { grid-template-columns: 1fr; }
  .env-list, .step-meta { grid-template-columns: 1fr; }
}
"""
