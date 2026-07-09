#!/usr/bin/env python3
"""
Phone Agent CLI - AI-powered phone automation.

Usage:
    python main.py [OPTIONS]

Environment Variables:
    PHONE_AGENT_BASE_URL: Model API base URL (default: http://localhost:8000/v1)
    PHONE_AGENT_MODEL: Model name (default: autoglm-phone-9b)
    PHONE_AGENT_API_KEY: API key for model authentication (default: EMPTY)
    PHONE_AGENT_MAX_STEPS: Maximum steps per task (default: 100)
    PHONE_AGENT_DEVICE_ID: ADB device ID for multi-device setups
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from openai import OpenAI

from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.agent_ios import IOSAgentConfig, IOSPhoneAgent
from phone_agent.config.apps import list_supported_apps
from phone_agent.config.apps_harmonyos import list_supported_apps as list_harmonyos_apps
from phone_agent.config.apps_ios import list_supported_apps as list_ios_apps
from phone_agent.device_factory import DeviceType, get_device_factory, set_device_type
from phone_agent.gitlab_install import (
    DEFAULT_GITLAB,
    DEFAULT_PROJECT,
    GitLabInstallConfig,
    GitLabInstallError,
    build_download_install,
)
from phone_agent.model import ModelConfig
from phone_agent.reporting import TestRunReporter
from phone_agent.status_judge import JudgeConfig, StepStatusJudge
from phone_agent.rule_loader import load_rules
from phone_agent.xctest import XCTestConnection
from phone_agent.xctest import list_devices as list_ios_devices


@dataclass
class ParsedTestCase:
    """从 Markdown 测试用例文件中解析出的单条用例。"""

    case_id: str | None
    title: str | None
    rule_type: str | None
    priority: str | None
    module: str | None
    task: str


@dataclass
class ParsedExecutionStep:
    """Markdown 测试用例中的单个固定执行步骤。"""

    index: int
    text: str
    target_state: str | None = None
    activity: str | None = None
    raw: str = ""


def parse_test_case_heading(task: str) -> tuple[str | None, str | None, str | None]:
    """从测试用例 Markdown 的二级标题中解析用例 ID、标题和规则类型。

    规范格式：
        ## TC-WEARFIT-[模块]-[序号]-[规则类型] 用例标题

    当前只做轻量解析，用于执行前打印和后续 rule 上下文扩展。
    """
    pattern = re.compile(r"^##\s+(TC-WEARFIT-[A-Za-z0-9]+-\d{3}-(normal|audio))\s+(.+)$")
    for line in task.splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1), match.group(3).strip(), match.group(2)
    return None, None, None


def parse_test_case_priority(task: str) -> str | None:
    """解析测试用例基本信息中的优先级字段。"""
    return parse_test_case_list_field(task, "优先级")


def parse_test_case_module(task: str) -> str | None:
    """解析测试用例基本信息中的关联模块字段。"""
    return parse_test_case_list_field(task, "关联模块")


def parse_test_case_list_field(task: str, field_name: str) -> str | None:
    """解析 Markdown 列表字段，例如：- 优先级：P0。"""
    pattern = re.compile(
        rf"^\s*-\s*{re.escape(field_name)}\s*[:：]\s*(.+?)\s*$",
        re.MULTILINE,
    )
    match = pattern.search(task)
    return match.group(1).strip() if match else None


def parse_test_case(task: str) -> ParsedTestCase:
    """解析单条测试用例文本，保留完整文本作为 agent task。"""
    case_id, title, rule_type = parse_test_case_heading(task)
    priority = parse_test_case_priority(task)
    module = parse_test_case_module(task)
    return ParsedTestCase(
        case_id=case_id,
        title=title,
        rule_type=rule_type,
        priority=priority,
        module=module,
        task=task.strip(),
    )


def parse_test_cases_from_markdown(content: str) -> list[ParsedTestCase]:
    """按测试用例规范从 Markdown 文件中解析所有用例。

    只以二级标题作为用例起始标记，不依赖 --- 分隔线。
    """
    heading_pattern = re.compile(
        r"^##\s+(TC-WEARFIT-[A-Za-z0-9]+-\d{3}-(normal|audio))\s+(.+)$",
        re.MULTILINE,
    )
    matches = list(heading_pattern.finditer(content))
    cases: list[ParsedTestCase] = []

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        block = content[start:end].strip()
        cases.append(parse_test_case(block))

    return cases


def load_test_cases_from_file(file_path: str) -> list[ParsedTestCase]:
    """读取 Markdown 文件并解析测试用例列表。"""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    cases = parse_test_cases_from_markdown(content)
    if not cases:
        raise ValueError(
            "No test cases found. Expected headings like "
            "'## TC-WEARFIT-MODULE-001-normal 用例标题'."
        )
    return cases


def parse_execution_steps(task: str) -> list[ParsedExecutionStep]:
    """解析测试用例中的“### 执行步骤”编号列表。"""
    section_match = re.search(
        r"^###\s+执行步骤\s*$([\s\S]*?)(?=^###\s+|\Z)",
        task,
        re.MULTILINE,
    )
    if not section_match:
        return []

    section = section_match.group(1)
    step_matches = list(re.finditer(r"^\s*(\d+)\.\s+(.+?)\s*$", section, re.MULTILINE))
    steps: list[ParsedExecutionStep] = []
    for pos, match in enumerate(step_matches):
        start = match.start()
        end = step_matches[pos + 1].start() if pos + 1 < len(step_matches) else len(section)
        raw = section[start:end].strip()
        text = match.group(2).strip()
        target_state = parse_step_field(raw, "目标状态")
        activity = parse_step_field(raw, "所在 Activity")
        steps.append(
            ParsedExecutionStep(
                index=int(match.group(1)),
                text=text,
                target_state=target_state,
                activity=activity,
                raw=raw,
            )
        )
    return steps


def parse_step_field(step_block: str, field_name: str) -> str | None:
    """解析单个执行步骤下的固定字段，例如：目标状态：xxx。"""
    pattern = re.compile(rf"^\s*{re.escape(field_name)}\s*[:：]\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(step_block)
    return match.group(1).strip() if match else None


def extract_markdown_section(task: str, title: str) -> str:
    """提取 Markdown 三级标题段落。"""
    match = re.search(
        rf"^###\s+{re.escape(title)}\s*$([\s\S]*?)(?=^###\s+|\Z)",
        task,
        re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def build_step_task(
    test_case: ParsedTestCase,
    step: ParsedExecutionStep,
    all_steps: list[ParsedExecutionStep],
    completed_summaries: list[str],
    external_status_judge: bool = False,
) -> str:
    """构造给模型的单步任务，只要求完成当前测试步骤。"""
    target_state = step.target_state or "按当前步骤要求达到可观察的目标状态。"
    activity = step.activity or "未指定"
    completed_text = "\n".join(completed_summaries) if completed_summaries else "暂无。"
    all_steps_text = format_execution_steps(all_steps)
    goal = extract_markdown_section(test_case.task, "测试目标")
    preconditions = extract_markdown_section(test_case.task, "前置条件")
    failure_conditions = extract_markdown_section(test_case.task, "失败条件")
    expected = extract_markdown_section(test_case.task, "预期结果")
    if external_status_judge:
        finish_requirement = """7. 当前启用了外部文本判定模型，你只负责执行当前步骤和描述当前可观察证据。
8. 当前步骤完成或无法继续时，调用 finish(message="...") 结束；message 只写你看到了什么、做了什么、为什么停止。
9. 不要在 finish message 中输出 STATUS、PASS、SKIPPED、BLOCKED、FAIL、REVIEW；最终状态由外部判定模型统一判断。
10. 如果入口或目标内容找不到，不要无限探索；合理查找后 finish 并说明找不到的原因。
11. 如果当前步骤结果无法明确判断，也 finish 并说明证据不足。
12. 不要进入视频、我的、设置、搜索、小游戏等与当前步骤无关的页面，除非当前步骤明确要求。
13. 不要执行支付、删除、解绑、上传隐私数据、真实发送或配置保存等禁止行为。"""
        conditional_requirement = "3. 如果当前步骤是“如果...则...”的条件步骤，且条件已经不需要执行，调用 finish 并说明条件不成立或无需执行。"
    else:
        finish_requirement = """7. finish 的 message 第一行必须是下面五者之一，且必须大写：
   STATUS: PASS
   STATUS: SKIPPED
   STATUS: BLOCKED
   STATUS: FAIL
   STATUS: REVIEW
8. message 第二行开始写 REASON，说明你完成了什么、看到了什么证据。
9. 如果入口或目标内容找不到，不要无限探索；合理查找后使用 STATUS: BLOCKED 并说明原因。
10. 如果当前步骤结果无法明确判断，但也没有明确失败或阻塞，使用 STATUS: REVIEW。
11. 不要进入视频、我的、设置、搜索、小游戏等与当前步骤无关的页面，除非当前步骤明确要求。
12. 不要执行支付、删除、解绑、上传隐私数据、真实发送或配置保存等禁止行为。"""
        conditional_requirement = "3. 如果当前步骤是“如果...则...”的条件步骤，且条件已经不需要执行，使用 STATUS: SKIPPED。"
    return f"""你正在执行一个自动化测试用例，但本次只允许完成当前这一个测试步骤。

用例 ID：{test_case.case_id or "unknown"}
用例标题：{test_case.title or "unknown"}
规则类型：{test_case.rule_type or "unknown"}
优先级：{test_case.priority or "unknown"}
关联模块：{test_case.module or "unknown"}

测试目标：
{goal or "未提供。"}

前置条件：
{preconditions or "未提供。"}

全局预期结果：
{expected or "未提供。"}

全局失败条件和禁止行为：
{failure_conditions or "未提供。"}

已完成步骤摘要：
{completed_text}

完整执行步骤列表：
{all_steps_text}

步骤衔接要求：
1. 开始当前步骤前，先核对“已完成步骤摘要”是否真的满足当前步骤的前置状态。
2. 如果上一条步骤是 SKIPPED、STEP_LIMIT 或 UNKNOWN，你需要先根据当前截图判断上一条步骤的目标状态是否已经满足。
3. 如果上一条步骤目标状态尚未满足，可以先补足上一条步骤必要动作，再执行当前步骤。
4. 如果上一条步骤目标状态已经满足，不要重复执行上一条步骤，直接执行当前步骤。
5. 仍然不要执行当前步骤之后的后续步骤。

当前执行第 {step.index} 步，共 {len(all_steps)} 步。

当前步骤：
{step.text}

当前步骤目标状态：
{target_state}

当前步骤期望 Activity：
{activity}

执行要求：
1. 只完成当前步骤，不要主动提前执行后续步骤。
2. 第一优先级：如果当前截图已经满足当前步骤目标，立即调用 finish，不要继续点击、滑动或进入详情。
{conditional_requirement}
4. 如果当前步骤目标状态达成后，立即调用 finish，不要继续探索。
5. 当前步骤不要求查找后续步骤入口时，禁止为了后续步骤继续查找、点击或滑动。
6. 如果当前步骤只是点击首页、切换底部导航、返回首页或确认当前页：
   - 当前页面已经显示目标页面核心内容时，直接 finish。
   - 只允许点击底部导航/明确导航控件，不要点击手表卡片、健康数据卡片、右上角图标、列表项或详情入口。
   - 不要把“确认在首页”理解成“进入设备详情”。
{finish_requirement}
"""


def format_execution_steps(steps: list[ParsedExecutionStep]) -> str:
    """Format all parsed execution steps for prompts and judge context."""
    lines: list[str] = []
    for step in steps:
        parts = [f"{step.index}. {step.text}"]
        if step.target_state:
            parts.append(f"目标状态：{step.target_state}")
        if step.activity:
            parts.append(f"Activity：{step.activity}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def classify_step_result(result: str, is_last_step: bool = False) -> str:
    """解析模型显式输出的步骤状态，不再用自然语言关键词判定。

    兜底策略：
    - 模型跑满当前步骤步数时，非最后一步记为 STEP_LIMIT 并继续；
      最后一步记为 BLOCKED。
    - 模型没有按约定输出 STATUS 时，记为 REVIEW，不再自动补成 PASS。
    """
    text = result or ""
    status_match = re.search(
        r"^\s*(?:STATUS|状态)\s*[:：]\s*(PASS|SKIPPED|BLOCKED|FAIL|REVIEW)\b",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if status_match:
        return status_match.group(1).upper()
    if text.strip() == "Max steps reached":
        return "BLOCKED" if is_last_step else "STEP_LIMIT"
    return "REVIEW"


def build_operation_log(reporter: TestRunReporter, test_step_index: int) -> str:
    """Build text-only evidence for the judge model from current step artifacts."""
    case = reporter.current_case
    if not case:
        return ""

    lines: list[str] = []
    for artifact in case.steps:
        if artifact.test_step_index != test_step_index:
            continue
        action = artifact.action or {}
        action_name = action.get("action") or action.get("_metadata") or "unknown"
        action_payload = {
            key: value
            for key, value in action.items()
            if key not in {"_metadata"} and value not in (None, "")
        }
        lines.append(f"- 动作 {artifact.step}: {action_name}")
        if action_payload:
            lines.append(
                f"  参数: {_clip_text(json.dumps(action_payload, ensure_ascii=False), 800)}"
            )
        lines.append(f"  执行结果: {artifact.action_success}")
        if artifact.action_message:
            lines.append(f"  动作消息: {_clip_text(artifact.action_message, 900)}")
        if artifact.current_app:
            lines.append(f"  当前应用: {artifact.current_app}")
        if artifact.current_activity:
            lines.append(f"  当前 Activity: {artifact.current_activity}")
        if artifact.ui_texts:
            lines.append(
                f"  动作前/收尾时 UI 文本: {' | '.join(artifact.ui_texts[:25])}"
            )
    return "\n".join(lines)


def build_agent_text_log(context: list[dict], max_chars: int = 4000) -> str:
    """Extract text-only agent context for the judge model."""
    lines: list[str] = []
    for message in context:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        text_parts: list[str] = []
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text") or ""))
        text = "\n".join(part for part in text_parts if part).strip()
        if text:
            lines.append(f"[{role}]\n{text}")
    joined = "\n\n".join(lines)
    if len(joined) > max_chars:
        return joined[-max_chars:]
    return joined


def _clip_text(value: str, max_chars: int) -> str:
    """Clip long text while preserving the beginning and explicit truncation."""
    text = value or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[truncated {len(text) - max_chars} chars]"


def print_judge_token_summary(reporter: TestRunReporter) -> None:
    """Print total text-judge token usage for the current run."""
    total_tokens = sum(
        step.judge_total_tokens
        for case in reporter.cases
        for attempt in (case.attempts or [case])
        for step in attempt.test_steps
    )
    prompt_tokens = sum(
        step.judge_prompt_tokens
        for case in reporter.cases
        for attempt in (case.attempts or [case])
        for step in attempt.test_steps
    )
    completion_tokens = sum(
        step.judge_completion_tokens
        for case in reporter.cases
        for attempt in (case.attempts or [case])
        for step in attempt.test_steps
    )
    print(
        f"\nTotal Judge Tokens: {total_tokens} "
        f"(prompt={prompt_tokens}, completion={completion_tokens})"
    )


def vision_tokens_for_test_step(reporter: TestRunReporter, test_step_index: int) -> int:
    """Return total vision-model tokens used by one parsed Markdown test step."""
    if not reporter.current_case:
        return 0
    return sum(
        step.vision_total_tokens
        for step in reporter.current_case.steps
        if step.test_step_index == test_step_index
    )


def vision_tokens_for_case(case) -> int:
    """Return total vision-model tokens used by one case."""
    return sum(
        step.vision_total_tokens
        for attempt in (case.attempts or [case])
        for step in attempt.steps
    )


def print_vision_token_summary(reporter: TestRunReporter) -> None:
    """Print total vision-model token usage for the current run."""
    total_tokens = sum(
        step.vision_total_tokens
        for case in reporter.cases
        for attempt in (case.attempts or [case])
        for step in attempt.steps
    )
    print(f"Total Vision Tokens: {total_tokens}")


def filter_test_cases(
    test_cases: list[ParsedTestCase],
    priorities: list[str] | None = None,
    modules: list[str] | None = None,
    start: int = 1,
    limit: int | None = None,
) -> tuple[list[ParsedTestCase], list[str]]:
    """按优先级、关联模块、起始位置和数量上限过滤用例。多个过滤条件之间是 AND 关系。"""
    filtered = test_cases
    messages: list[str] = []

    if priorities:
        selected_priorities = {priority.upper() for priority in priorities}
        filtered = [
            test_case
            for test_case in filtered
            if (test_case.priority or "").upper() in selected_priorities
        ]
        messages.append(f"priority: {', '.join(sorted(selected_priorities))}")

    if modules:
        selected_modules = {module.strip() for module in modules if module.strip()}
        filtered = [
            test_case
            for test_case in filtered
            if (test_case.module or "").strip() in selected_modules
        ]
        messages.append(f"module: {', '.join(sorted(selected_modules))}")

    if start > 1:
        filtered = filtered[start - 1 :]
        messages.append(f"start: {start}")

    if limit is not None:
        filtered = filtered[:limit]
        messages.append(f"limit: {limit}")

    return filtered, messages


def reset_android_wearfit_home(device_id: str | None = None, wait_seconds: int = 8) -> None:
    """每条用例开始前重置 Wearfit Pro 到 launcher 入口。

    使用已验证可启动的 exported SplashActivity，并清理当前 task 栈；
    不使用 force-stop，避免每条用例都完整冷启动导致过慢。
    """
    cmd = ["adb"]
    if device_id:
        cmd.extend(["-s", device_id])
    cmd.extend(
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
            "com.wakeup.howear/com.wakeup.howear.view.app.SplashActivity",
            "-f",
            "0x10008000",
        ]
    )
    print("Resetting Wearfit Pro to home entry...")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=30)
    if result.returncode != 0:
        print("Reset command failed:")
        print((result.stderr or result.stdout).strip())
    print(f"Waiting {wait_seconds}s for app home to settle...")
    time.sleep(wait_seconds)


def prompt_required(value: str | None, label: str) -> str:
    """Return an argument value, or ask interactively when it is omitted."""
    if value:
        return value
    try:
        entered = input(f"请输入{label}: ").strip()
    except EOFError as exc:
        raise RuntimeError(f"缺少{label}，请通过参数传入。") from exc
    if not entered:
        raise RuntimeError(f"缺少{label}，请通过参数传入。")
    return entered


def handle_install_command(args) -> bool:
    """Trigger GitLab build, download the APK artifact, and install it."""
    if not args.install:
        return False
    if args.device_type != "adb":
        raise RuntimeError("--install 当前只支持 Android/ADB APK 安装。")

    gitlab = args.install_gitlab or os.getenv("GITLAB_URL") or DEFAULT_GITLAB
    project = args.install_project or os.getenv("GITLAB_PROJECT") or DEFAULT_PROJECT
    token = args.install_token or os.getenv("GITLAB_TOKEN")
    branch = prompt_required(args.install_branch, "GitLab 分支/ref")
    variant = prompt_required(args.install_variant or args.install_env, "打包环境/BUILD_VARIANT")
    if not token:
        raise RuntimeError("缺少 GitLab PAT。请 export GITLAB_TOKEN 或传入 --install-token。")

    try:
        build_download_install(
            GitLabInstallConfig(
                gitlab=gitlab,
                project=project,
                token=token,
                branch=branch,
                variant=variant,
                output_dir=args.install_output_dir,
                filename=args.install_filename,
                job_id=args.install_job_id,
                keep_zip=args.install_keep_zip,
                poll_interval=args.install_poll_interval,
                timeout=args.install_timeout,
                verify_ssl=not args.install_no_verify_ssl,
                tls_version=args.install_tls_version,
                use_env_proxy=args.install_use_env_proxy,
                device_id=args.device_id,
            )
        )
        return True
    except GitLabInstallError as exc:
        raise RuntimeError(str(exc)) from exc


def print_test_case_heading(task: str) -> None:
    """在执行 agent 前打印解析到的测试用例标题信息。"""
    case_id, title, rule_type = parse_test_case_heading(task)
    if case_id:
        print(f"Parsed Test Case: {case_id} [{rule_type}] - {title}")
    else:
        print("Parsed Test Case: <not found>")


def execute_test_case(
    *,
    agent,
    reporter: TestRunReporter,
    test_case: ParsedTestCase,
    index: int,
    total_tasks: int,
    args,
    device_type: DeviceType,
    status_judge: StepStatusJudge | None,
    attempt: int = 0,
) -> tuple[str, str]:
    """Execute one parsed test case once and return (case_status, result_message)."""
    task = test_case.task
    case_label = test_case.case_id or f"Task {index}"
    title = f" - {test_case.title}" if test_case.title else ""
    rule_type = test_case.rule_type or "unknown"
    priority = test_case.priority or "unknown"
    module = test_case.module or "unknown"
    attempt_text = f" Attempt {attempt + 1}" if attempt else ""
    print(
        f"\nTask {index}/{total_tasks}{attempt_text}: "
        f"{case_label} [{rule_type}, {priority}, {module}]{title}\n"
    )
    print_test_case_heading(task)
    if device_type == DeviceType.ADB:
        reset_android_wearfit_home(args.device_id, wait_seconds=8)

    execution_steps = parse_execution_steps(task) if args.file else []
    if execution_steps:
        reporter.start_case(test_case.task, index, attempt=attempt + 1)
        reporter.set_test_steps(
            [
                {
                    "index": step.index,
                    "text": step.text,
                    "target_state": step.target_state,
                    "activity": step.activity,
                }
                for step in execution_steps
            ]
        )

        completed_summaries: list[str] = []
        case_results: list[str] = []
        step_statuses: list[str] = []
        terminal_status = "PASS"
        for step_position, execution_step in enumerate(execution_steps, start=1):
            is_last_step = step_position == len(execution_steps)
            print(
                f"\nCase {index}/{total_tasks} Step "
                f"{execution_step.index}/{len(execution_steps)}: "
                f"{execution_step.text}"
            )
            reporter.begin_test_step(execution_step.index)
            step_task = build_step_task(
                test_case,
                execution_step,
                execution_steps,
                completed_summaries,
                external_status_judge=bool(status_judge),
            )
            result = agent.run(step_task)
            step_limit_reached = result.strip() == "Max steps reached"
            judge_raw = None
            judge_prompt_tokens = 0
            judge_completion_tokens = 0
            judge_total_tokens = 0
            if status_judge:
                operation_log = build_operation_log(reporter, execution_step.index)
                agent_text_log = build_agent_text_log(agent.context)
                try:
                    judge_result = status_judge.judge(
                        case_id=test_case.case_id,
                        case_title=test_case.title,
                        step_index=execution_step.index,
                        step_text=execution_step.text,
                        target_state=execution_step.target_state,
                        expected_activity=execution_step.activity,
                        case_task=test_case.task,
                        all_steps_text=format_execution_steps(execution_steps),
                        previous_steps=completed_summaries,
                        agent_text_log=agent_text_log,
                        operation_log=operation_log,
                        model_result=result,
                        step_limit_reached=step_limit_reached,
                    )
                    step_status = judge_result.status
                    result = judge_result.message
                    judge_raw = judge_result.raw
                    judge_prompt_tokens = judge_result.prompt_tokens
                    judge_completion_tokens = judge_result.completion_tokens
                    judge_total_tokens = judge_result.total_tokens
                except Exception as exc:
                    step_status = "REVIEW"
                    result = (
                        "STATUS: REVIEW\n"
                        f"REASON: 判定模型调用失败，需要人工复核：{exc}"
                    )
                    judge_raw = str(exc)
            else:
                if step_limit_reached:
                    result = agent.request_finish_status_only(
                        execution_step.text,
                        execution_step.target_state,
                    )
                step_status = classify_step_result(result, is_last_step=is_last_step)
            reporter.finish_test_step(
                execution_step.index,
                step_status,
                result,
                judge_raw=judge_raw,
                judge_prompt_tokens=judge_prompt_tokens,
                judge_completion_tokens=judge_completion_tokens,
                judge_total_tokens=judge_total_tokens,
            )
            step_statuses.append(step_status)
            summary = (
                f"{execution_step.index}. {step_status}: "
                f"{result[:220].replace(chr(10), ' ')}"
            )
            completed_summaries.append(summary)
            case_results.append(summary)
            token_text = (
                f" Judge Tokens: {judge_total_tokens} "
                f"(prompt={judge_prompt_tokens}, completion={judge_completion_tokens})"
                if status_judge
                else ""
            )
            vision_step_tokens = vision_tokens_for_test_step(
                reporter, execution_step.index
            )
            print(
                f"Step Result: {step_status} - {result}{token_text} "
                f"Vision Tokens: {vision_step_tokens}"
            )
            agent.reset()

            if step_status in {"FAIL", "BLOCKED"}:
                terminal_status = step_status
                break

        final_result = "\n".join(case_results)
        if terminal_status == "PASS":
            executed_statuses = [
                status for status in step_statuses if status and status != "PENDING"
            ]
            last_status = executed_statuses[-1] if executed_statuses else "PASS"
            if last_status in {"PASS", "SKIPPED"}:
                terminal_status = "PASS"
            elif last_status == "BLOCKED":
                terminal_status = "BLOCKED"
            elif last_status in {"UNKNOWN", "STEP_LIMIT", "REVIEW"}:
                terminal_status = "REVIEW"
            else:
                terminal_status = "PASS"

        if terminal_status == "PASS":
            final_result = "所有测试步骤已按顺序执行完成。\n" + final_result
        elif terminal_status == "REVIEW":
            final_result = (
                "REVIEW: 存在未明确判定或达到单步步数上限的步骤。\n"
                + final_result
            )
        else:
            final_result = f"{terminal_status}: 测试步骤执行中断。\n" + final_result

        case_status = "UNKNOWN"
        if reporter.current_case:
            case_status = reporter.finish_case(final_result)
        if status_judge and reporter.cases:
            usage = sum(step.judge_total_tokens for step in reporter.cases[-1].test_steps)
            prompt_usage = sum(
                step.judge_prompt_tokens for step in reporter.cases[-1].test_steps
            )
            completion_usage = sum(
                step.judge_completion_tokens for step in reporter.cases[-1].test_steps
            )
            print(
                f"Case Judge Tokens: {usage} "
                f"(prompt={prompt_usage}, completion={completion_usage})"
            )
        if reporter.cases:
            print(f"Case Vision Tokens: {vision_tokens_for_case(reporter.cases[-1])}")
        print(f"\nResult {index}/{total_tasks}: {final_result}")
        return case_status, final_result

    reporter.start_case(test_case.task, index, attempt=attempt + 1)
    result = agent.run(task)
    case_status = "UNKNOWN"
    if reporter.current_case:
        case_status = reporter.finish_case(result)
    if reporter.cases:
        print(f"Case Vision Tokens: {vision_tokens_for_case(reporter.cases[-1])}")
    print(f"\nResult {index}/{total_tasks}: {result}")
    agent.reset()
    return case_status, result


def check_system_requirements(
    device_type: DeviceType = DeviceType.ADB, wda_url: str = "http://localhost:8100"
) -> bool:
    """
    Check system requirements before running the agent.

    Checks:
    1. ADB/HDC/iOS tools installed
    2. At least one device connected
    3. ADB Keyboard installed on the device (for ADB only)
    4. WebDriverAgent running (for iOS only)

    Args:
        device_type: Type of device tool (ADB, HDC, or IOS).
        wda_url: WebDriverAgent URL (for iOS only).

    Returns:
        True if all checks pass, False otherwise.
    """
    print("🔍 Checking system requirements...")
    print("-" * 50)

    all_passed = True

    # Determine tool name and command
    if device_type == DeviceType.IOS:
        tool_name = "libimobiledevice"
        tool_cmd = "idevice_id"
    else:
        tool_name = "ADB" if device_type == DeviceType.ADB else "HDC"
        tool_cmd = "adb" if device_type == DeviceType.ADB else "hdc"

    # Check 1: Tool installed
    print(f"1. Checking {tool_name} installation...", end=" ")
    if shutil.which(tool_cmd) is None:
        print("❌ FAILED")
        print(f"   Error: {tool_name} is not installed or not in PATH.")
        print(f"   Solution: Install {tool_name}:")
        if device_type == DeviceType.ADB:
            print("     - macOS: brew install android-platform-tools")
            print("     - Linux: sudo apt install android-tools-adb")
            print(
                "     - Windows: Download from https://developer.android.com/studio/releases/platform-tools"
            )
        elif device_type == DeviceType.HDC:
            print(
                "     - Download from HarmonyOS SDK or https://gitee.com/openharmony/docs"
            )
            print("     - Add to PATH environment variable")
        else:  # IOS
            print("     - macOS: brew install libimobiledevice")
            print("     - Linux: sudo apt-get install libimobiledevice-utils")
        all_passed = False
    else:
        # Double check by running version command
        try:
            if device_type == DeviceType.ADB:
                version_cmd = [tool_cmd, "version"]
            elif device_type == DeviceType.HDC:
                version_cmd = [tool_cmd, "-v"]
            else:  # IOS
                version_cmd = [tool_cmd, "-ln"]

            result = subprocess.run(
                version_cmd, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.strip().split("\n")[0]
                print(f"✅ OK ({version_line if version_line else 'installed'})")
            else:
                print("❌ FAILED")
                print(f"   Error: {tool_name} command failed to run.")
                all_passed = False
        except FileNotFoundError:
            print("❌ FAILED")
            print(f"   Error: {tool_name} command not found.")
            all_passed = False
        except subprocess.TimeoutExpired:
            print("❌ FAILED")
            print(f"   Error: {tool_name} command timed out.")
            all_passed = False

    # If ADB is not installed, skip remaining checks
    if not all_passed:
        print("-" * 50)
        print("❌ System check failed. Please fix the issues above.")
        return False

    # Check 2: Device connected
    print("2. Checking connected devices...", end=" ")
    try:
        if device_type == DeviceType.ADB:
            result = subprocess.run(
                ["adb", "devices"], capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")
            # Filter out header and empty lines, look for 'device' status
            devices = [
                line for line in lines[1:] if line.strip() and "\tdevice" in line
            ]
        elif device_type == DeviceType.HDC:
            result = subprocess.run(
                ["hdc", "list", "targets"], capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")
            devices = [line for line in lines if line.strip()]
        else:  # IOS
            ios_devices = list_ios_devices()
            devices = [d.device_id for d in ios_devices]

        if not devices:
            print("❌ FAILED")
            print("   Error: No devices connected.")
            print("   Solution:")
            if device_type == DeviceType.ADB:
                print("     1. Enable USB debugging on your Android device")
                print("     2. Connect via USB and authorize the connection")
                print(
                    "     3. Or connect remotely: python main.py --connect <ip>:<port>"
                )
            elif device_type == DeviceType.HDC:
                print("     1. Enable USB debugging on your HarmonyOS device")
                print("     2. Connect via USB and authorize the connection")
                print(
                    "     3. Or connect remotely: python main.py --device-type hdc --connect <ip>:<port>"
                )
            else:  # IOS
                print("     1. Connect your iOS device via USB")
                print("     2. Unlock device and tap 'Trust This Computer'")
                print("     3. Verify: idevice_id -l")
                print("     4. Or connect via WiFi using device IP")
            all_passed = False
        else:
            if device_type == DeviceType.ADB:
                device_ids = [d.split("\t")[0] for d in devices]
            elif device_type == DeviceType.HDC:
                device_ids = [d.strip() for d in devices]
            else:  # IOS
                device_ids = devices
            print(
                f"✅ OK ({len(devices)} device(s): {', '.join(device_ids[:2])}{'...' if len(device_ids) > 2 else ''})"
            )
    except subprocess.TimeoutExpired:
        print("❌ FAILED")
        print(f"   Error: {tool_name} command timed out.")
        all_passed = False
    except Exception as e:
        print("❌ FAILED")
        print(f"   Error: {e}")
        all_passed = False

    # If no device connected, skip ADB Keyboard check
    if not all_passed:
        print("-" * 50)
        print("❌ System check failed. Please fix the issues above.")
        return False

    # Check 3: ADB Keyboard installed (only for ADB) or WebDriverAgent (for iOS)
    if device_type == DeviceType.ADB:
        print("3. Checking ADB Keyboard...", end=" ")
        try:
            result = subprocess.run(
                ["adb", "shell", "ime", "list", "-s"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            ime_list = result.stdout.strip()

            if "com.android.adbkeyboard/.AdbIME" in ime_list:
                print("✅ OK")
            else:
                print("❌ FAILED")
                print("   Error: ADB Keyboard is not installed on the device.")
                print("   Solution:")
                print("     1. Download ADB Keyboard APK from:")
                print(
                    "        https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk"
                )
                print("     2. Install it on your device: adb install ADBKeyboard.apk")
                print(
                    "     3. Enable it in Settings > System > Languages & Input > Virtual Keyboard"
                )
                all_passed = False
        except subprocess.TimeoutExpired:
            print("❌ FAILED")
            print("   Error: ADB command timed out.")
            all_passed = False
        except Exception as e:
            print("❌ FAILED")
            print(f"   Error: {e}")
            all_passed = False
    elif device_type == DeviceType.HDC:
        # For HDC, skip keyboard check as it uses different input method
        print("3. Skipping keyboard check for HarmonyOS...", end=" ")
        print("✅ OK (using native input)")
    else:  # IOS
        # Check WebDriverAgent
        print(f"3. Checking WebDriverAgent ({wda_url})...", end=" ")
        try:
            conn = XCTestConnection(wda_url=wda_url)

            if conn.is_wda_ready():
                print("✅ OK")
                # Get WDA status for additional info
                status = conn.get_wda_status()
                if status:
                    session_id = status.get("sessionId", "N/A")
                    print(f"   Session ID: {session_id}")
            else:
                print("❌ FAILED")
                print("   Error: WebDriverAgent is not running or not accessible.")
                print("   Solution:")
                print("     1. Run WebDriverAgent on your iOS device via Xcode")
                print("     2. For USB: Set up port forwarding: iproxy 8100 8100")
                print(
                    "     3. For WiFi: Use device IP, e.g., --wda-url http://192.168.1.100:8100"
                )
                print("     4. Verify in browser: open http://localhost:8100/status")
                all_passed = False
        except Exception as e:
            print("❌ FAILED")
            print(f"   Error: {e}")
            all_passed = False

    print("-" * 50)

    if all_passed:
        print("✅ All system checks passed!\n")
    else:
        print("❌ System check failed. Please fix the issues above.")

    return all_passed


def check_model_api(base_url: str, model_name: str, api_key: str = "EMPTY") -> bool:
    """
    Check if the model API is accessible and the specified model exists.

    Checks:
    1. Network connectivity to the API endpoint
    2. Model exists in the available models list

    Args:
        base_url: The API base URL
        model_name: The model name to check
        api_key: The API key for authentication

    Returns:
        True if all checks pass, False otherwise.
    """
    print("🔍 Checking model API...")
    print("-" * 50)

    all_passed = True

    # Check 1: Network connectivity using chat API
    print(f"1. Checking API connectivity ({base_url})...", end=" ")
    try:
        # Create OpenAI client
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)

        # Use chat completion to test connectivity (more universally supported than /models)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            temperature=0.0,
            stream=False,
        )

        # Check if we got a valid response
        if response.choices and len(response.choices) > 0:
            print("✅ OK")
        else:
            print("❌ FAILED")
            print("   Error: Received empty response from API")
            all_passed = False

    except Exception as e:
        print("❌ FAILED")
        error_msg = str(e)
        print_model_api_exception_detail(e)

        # Provide more specific error messages
        if "Connection refused" in error_msg or "Connection error" in error_msg:
            print(f"   Error: Cannot connect to {base_url}")
            print("   Solution:")
            print("     1. Check if the model server is running")
            print("     2. Verify the base URL is correct")
            print(f"     3. Try: curl {base_url}/chat/completions")
        elif "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            print(f"   Error: Connection to {base_url} timed out")
            print("   Solution:")
            print("     1. Check your network connection")
            print("     2. Verify the server is responding")
        elif (
            "Name or service not known" in error_msg
            or "nodename nor servname" in error_msg
        ):
            print(f"   Error: Cannot resolve hostname")
            print("   Solution:")
            print("     1. Check the URL is correct")
            print("     2. Verify DNS settings")
        else:
            print(f"   Error: {error_msg}")

        all_passed = False

    print("-" * 50)

    if all_passed:
        print("✅ Model API checks passed!\n")
    else:
        print("❌ Model API check failed. Please fix the issues above.")

    return all_passed


def print_model_api_exception_detail(exc: Exception) -> None:
    """Print useful details from OpenAI-compatible API exceptions."""
    print(f"   Exception: {type(exc).__name__}")
    print(f"   Message: {exc}")

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        print(f"   HTTP status: {status_code}")

    request_id = getattr(exc, "request_id", None)
    if request_id:
        print(f"   Request ID: {request_id}")

    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if response_status is not None and response_status != status_code:
            print(f"   Response status: {response_status}")

        response_text = None
        try:
            response_text = response.text
        except Exception:
            response_text = None
        if response_text:
            print("   Response body:")
            print(_indent_block(_clip_text(response_text.strip(), 2000), "     "))

    body = getattr(exc, "body", None)
    if body:
        print("   Error body:")
        print(_indent_block(_clip_text(str(body), 2000), "     "))


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phone Agent - AI-powered phone automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (Android)
    python main.py

    # Specify model endpoint
    python main.py --base-url http://localhost:8000/v1

    # Use API key for authentication
    python main.py --apikey sk-xxxxx

    # Run with specific device
    python main.py --device-id emulator-5554

    # Connect to remote device
    python main.py --connect 192.168.1.100:5555

    # List connected devices
    python main.py --list-devices

    # Enable TCP/IP on USB device and get connection info
    python main.py --enable-tcpip

    # List supported apps
    python main.py --list-apps

    # iOS specific examples
    # Run with iOS device
    python main.py --device-type ios "Open Safari and search for iPhone tips"

    # Use WiFi connection for iOS
    python main.py --device-type ios --wda-url http://192.168.1.100:8100

    # List connected iOS devices
    python main.py --device-type ios --list-devices

    # Run all test cases from a Markdown file
    python main.py --file docs/wearfit_cases.md

    # Run at most 5 filtered test cases from a Markdown file
    python main.py --file docs/wearfit_cases.md --priority P0 --limit 5

    # Run 10 cases starting from the 21st matched case, retry each non-PASS once
    python main.py --file docs/wearfit_cases.md --module 手表 --start 21 --limit 10 --case-retries 1

    # Build from GitLab, download APK artifacts, install to Android phone, then run tests
    python main.py --install --install-branch develop --install-env wpZhDebug

    # Check WebDriverAgent status
    python main.py --device-type ios --wda-status

    # Pair with iOS device
    python main.py --device-type ios --pair
        """,
    )

    # Model options
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("PHONE_AGENT_BASE_URL", "http://localhost:8000/v1"),
        help="Model API base URL",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("PHONE_AGENT_MODEL", "autoglm-phone-9b"),
        help="Model name",
    )

    parser.add_argument(
        "--apikey",
        type=str,
        default=os.getenv("PHONE_AGENT_API_KEY", "EMPTY"),
        help="API key for model authentication",
    )

    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=os.getenv(
            "PHONE_AGENT_JUDGE_BASE_URL",
            "https://ark.cn-beijing.volces.com/api/v3",
        ),
        help="Text judge model API base URL",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.getenv("PHONE_AGENT_JUDGE_MODEL", "ep-20240813102137-m62n6"),
        help="Text judge model name",
    )

    parser.add_argument(
        "--judge-api-key-env",
        type=str,
        default=os.getenv("PHONE_AGENT_JUDGE_API_KEY_ENV", "ARK_API_KEY"),
        help="Environment variable name that stores the text judge model API key",
    )

    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=os.getenv("PHONE_AGENT_JUDGE_API_KEY"),
        help="Text judge model API key. If omitted, --judge-api-key-env is used.",
    )

    parser.add_argument(
        "--disable-status-judge",
        action="store_true",
        help="Disable text judge model and fall back to local STATUS parsing.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("PHONE_AGENT_MAX_STEPS", "100")),
        help="Maximum steps per task",
    )

    # Device options
    parser.add_argument(
        "--device-id",
        "-d",
        type=str,
        default=os.getenv("PHONE_AGENT_DEVICE_ID"),
        help="ADB device ID",
    )

    parser.add_argument(
        "--connect",
        "-c",
        type=str,
        metavar="ADDRESS",
        help="Connect to remote device (e.g., 192.168.1.100:5555)",
    )

    parser.add_argument(
        "--disconnect",
        type=str,
        nargs="?",
        const="all",
        metavar="ADDRESS",
        help="Disconnect from remote device (or 'all' to disconnect all)",
    )

    parser.add_argument(
        "--list-devices", action="store_true", help="List connected devices and exit"
    )

    parser.add_argument(
        "--enable-tcpip",
        type=int,
        nargs="?",
        const=5555,
        metavar="PORT",
        help="Enable TCP/IP debugging on USB device (default port: 5555)",
    )

    # iOS specific options
    parser.add_argument(
        "--wda-url",
        type=str,
        default=os.getenv("PHONE_AGENT_WDA_URL", "http://localhost:8100"),
        help="WebDriverAgent URL for iOS (default: http://localhost:8100)",
    )

    parser.add_argument(
        "--pair",
        action="store_true",
        help="Pair with iOS device (required for some operations)",
    )

    parser.add_argument(
        "--wda-status",
        action="store_true",
        help="Show WebDriverAgent status and exit (iOS only)",
    )

    # Other options
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    parser.add_argument(
        "--artifact-name",
        type=str,
        default=os.getenv("PHONE_AGENT_ARTIFACT_NAME"),
        help="Name for the test_artifacts run directory",
    )

    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=os.getenv("PHONE_AGENT_ARTIFACT_DIR", "test_artifacts"),
        help="Directory for screenshots, UI dumps, and reports",
    )

    parser.add_argument(
        "--list-apps", action="store_true", help="List supported apps and exit"
    )

    parser.add_argument(
        "--lang",
        type=str,
        choices=["cn", "en"],
        default=os.getenv("PHONE_AGENT_LANG", "cn"),
        help="Language for system prompt (cn or en, default: cn)",
    )

    parser.add_argument(
        "--device-type",
        type=str,
        choices=["adb", "hdc", "ios"],
        default=os.getenv("PHONE_AGENT_DEVICE_TYPE", "adb"),
        help="Device type: adb for Android, hdc for HarmonyOS, ios for iPhone (default: adb)",
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Markdown test case file. Mutually exclusive with positional tasks.",
    )

    parser.add_argument(
        "--priority",
        action="append",
        help=(
            "Only run test cases with the given priority from --file. "
            "Can be used multiple times, e.g. --priority P0 --priority P1."
        ),
    )

    parser.add_argument(
        "--module",
        action="append",
        help=(
            "Only run test cases with the given 关联模块 from --file. "
            "Can be used multiple times, e.g. --module 登录 --module 手表."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of test cases to run from --file after filtering.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help=(
            "1-based start position from --file after priority/module filtering. "
            "Useful for batch execution, e.g. --start 11 --limit 10."
        ),
    )

    parser.add_argument(
        "--case-retries",
        type=int,
        default=int(os.getenv("PHONE_AGENT_CASE_RETRIES", "1")),
        help=(
            "Retry count for each non-PASS test case. Default: 1. "
            "Use 0 to disable retries."
        ),
    )

    # GitLab build/download/install options
    parser.add_argument(
        "--install",
        action="store_true",
        help="Before running tasks/tests, trigger GitLab build, download APK artifacts, and install to Android device.",
    )

    parser.add_argument(
        "--install-branch",
        help="GitLab branch/ref for --install. If omitted, prompts interactively.",
    )

    parser.add_argument(
        "--install-env",
        help=(
            "Build environment/variant for --install, e.g. wpZhDebug. "
            "Alias of --install-variant."
        ),
    )

    parser.add_argument(
        "--install-variant",
        help="BUILD_VARIANT for --install, e.g. wpZhDebug, wpZhRelease.",
    )

    parser.add_argument(
        "--install-token",
        default=os.getenv("GITLAB_TOKEN"),
        help="GitLab PAT for --install. Default reads GITLAB_TOKEN.",
    )

    parser.add_argument(
        "--install-gitlab",
        default=os.getenv("GITLAB_URL"),
        help="GitLab URL for --install. Default uses helper script default.",
    )

    parser.add_argument(
        "--install-project",
        default=os.getenv("GITLAB_PROJECT"),
        help="GitLab project path/id for --install. Default uses helper script default.",
    )

    parser.add_argument(
        "--install-output-dir",
        default="gitlab_artifacts",
        help="Directory to save downloaded APK. Default: gitlab_artifacts.",
    )

    parser.add_argument(
        "--install-filename",
        help="Optional APK filename after download.",
    )

    parser.add_argument(
        "--install-job-id",
        type=int,
        help="Skip automatic artifact job selection and download this GitLab job id.",
    )

    parser.add_argument(
        "--install-keep-zip",
        action="store_true",
        help="Keep downloaded artifacts zip after extracting APK.",
    )

    parser.add_argument(
        "--install-poll-interval",
        type=int,
        default=20,
        help="GitLab pipeline polling interval in seconds. Default: 20.",
    )

    parser.add_argument(
        "--install-timeout",
        type=int,
        default=0,
        help="GitLab pipeline wait timeout in seconds. 0 means wait forever.",
    )

    parser.add_argument(
        "--install-no-verify-ssl",
        action="store_true",
        help="Do not verify GitLab HTTPS certificate for --install.",
    )

    parser.add_argument(
        "--install-tls-version",
        choices=["auto", "1.2", "1.3"],
        default=os.getenv("GITLAB_TLS_VERSION", "1.2"),
        help="GitLab HTTPS TLS version for --install. Default: 1.2.",
    )

    parser.add_argument(
        "--install-use-env-proxy",
        action="store_true",
        help="Allow requests to use HTTP_PROXY/HTTPS_PROXY/NO_PROXY for --install.",
    )

    parser.add_argument(
        "tasks",
        nargs="*",
        type=str,
        help="Tasks to execute one by one (interactive mode if not provided)",
    )

    args = parser.parse_args()
    if args.file and args.tasks:
        parser.error("--file cannot be used together with positional tasks")
    if args.priority and not args.file:
        parser.error("--priority can only be used together with --file")
    if args.module and not args.file:
        parser.error("--module can only be used together with --file")
    if args.start != 1 and not args.file:
        parser.error("--start can only be used together with --file")
    if args.start <= 0:
        parser.error("--start must be a positive integer")
    if args.limit is not None:
        if not args.file:
            parser.error("--limit can only be used together with --file")
        if args.limit <= 0:
            parser.error("--limit must be a positive integer")
    if args.case_retries < 0:
        parser.error("--case-retries must be zero or a positive integer")
    if args.install_env and args.install_variant and args.install_env != args.install_variant:
        parser.error("--install-env and --install-variant cannot have different values")
    return args


def handle_ios_device_commands(args) -> bool:
    """
    Handle iOS device-related commands.

    Returns:
        True if a device command was handled (should exit), False otherwise.
    """
    conn = XCTestConnection(wda_url=args.wda_url)

    # Handle --list-devices
    if args.list_devices:
        devices = list_ios_devices()
        if not devices:
            print("No iOS devices connected.")
            print("\nTroubleshooting:")
            print("  1. Connect device via USB")
            print("  2. Unlock device and trust this computer")
            print("  3. Run: idevice_id -l")
        else:
            print("Connected iOS devices:")
            print("-" * 70)
            for device in devices:
                conn_type = device.connection_type.value
                model_info = f"{device.model}" if device.model else "Unknown"
                ios_info = f"iOS {device.ios_version}" if device.ios_version else ""
                name_info = device.device_name or "Unnamed"

                print(f"  ✓ {name_info}")
                print(f"    UUID: {device.device_id}")
                print(f"    Model: {model_info}")
                print(f"    OS: {ios_info}")
                print(f"    Connection: {conn_type}")
                print("-" * 70)
        return True

    # Handle --pair
    if args.pair:
        print("Pairing with iOS device...")
        success, message = conn.pair_device(args.device_id)
        print(f"{'✓' if success else '✗'} {message}")
        return True

    # Handle --wda-status
    if args.wda_status:
        print(f"Checking WebDriverAgent status at {args.wda_url}...")
        print("-" * 50)

        if conn.is_wda_ready():
            print("✓ WebDriverAgent is running")

            status = conn.get_wda_status()
            if status:
                print(f"\nStatus details:")
                value = status.get("value", {})
                print(f"  Session ID: {status.get('sessionId', 'N/A')}")
                print(f"  Build: {value.get('build', {}).get('time', 'N/A')}")

                current_app = value.get("currentApp", {})
                if current_app:
                    print(f"\nCurrent App:")
                    print(f"  Bundle ID: {current_app.get('bundleId', 'N/A')}")
                    print(f"  Process ID: {current_app.get('pid', 'N/A')}")
        else:
            print("✗ WebDriverAgent is not running")
            print("\nPlease start WebDriverAgent on your iOS device:")
            print("  1. Open WebDriverAgent.xcodeproj in Xcode")
            print("  2. Select your device")
            print("  3. Run WebDriverAgentRunner (Product > Test or Cmd+U)")
            print(f"  4. For USB: Run port forwarding: iproxy 8100 8100")

        return True

    return False


def handle_device_commands(args) -> bool:
    """
    Handle device-related commands.

    Returns:
        True if a device command was handled (should exit), False otherwise.
    """
    device_type = (
        DeviceType.ADB
        if args.device_type == "adb"
        else (DeviceType.HDC if args.device_type == "hdc" else DeviceType.IOS)
    )

    # Handle iOS-specific commands
    if device_type == DeviceType.IOS:
        return handle_ios_device_commands(args)

    device_factory = get_device_factory()
    ConnectionClass = device_factory.get_connection_class()
    conn = ConnectionClass()

    # Handle --list-devices
    if args.list_devices:
        devices = device_factory.list_devices()
        if not devices:
            print("No devices connected.")
        else:
            print("Connected devices:")
            print("-" * 60)
            for device in devices:
                status_icon = "✓" if device.status == "device" else "✗"
                conn_type = device.connection_type.value
                model_info = f" ({device.model})" if device.model else ""
                print(
                    f"  {status_icon} {device.device_id:<30} [{conn_type}]{model_info}"
                )
        return True

    # Handle --connect
    if args.connect:
        print(f"Connecting to {args.connect}...")
        success, message = conn.connect(args.connect)
        print(f"{'✓' if success else '✗'} {message}")
        if success:
            # Set as default device
            args.device_id = args.connect
        return not success  # Continue if connection succeeded

    # Handle --disconnect
    if args.disconnect:
        if args.disconnect == "all":
            print("Disconnecting all remote devices...")
            success, message = conn.disconnect()
        else:
            print(f"Disconnecting from {args.disconnect}...")
            success, message = conn.disconnect(args.disconnect)
        print(f"{'✓' if success else '✗'} {message}")
        return True

    # Handle --enable-tcpip
    if args.enable_tcpip:
        port = args.enable_tcpip
        print(f"Enabling TCP/IP debugging on port {port}...")

        success, message = conn.enable_tcpip(port, args.device_id)
        print(f"{'✓' if success else '✗'} {message}")

        if success:
            # Try to get device IP
            ip = conn.get_device_ip(args.device_id)
            if ip:
                print(f"\nYou can now connect remotely using:")
                print(f"  python main.py --connect {ip}:{port}")
                print(f"\nOr via ADB directly:")
                print(f"  adb connect {ip}:{port}")
            else:
                print("\nCould not determine device IP. Check device WiFi settings.")
        return True

    return False


def main():
    """Main entry point."""
    args = parse_args()

    # Set device type globally based on args
    if args.device_type == "adb":
        device_type = DeviceType.ADB
    elif args.device_type == "hdc":
        device_type = DeviceType.HDC
    else:  # ios
        device_type = DeviceType.IOS

    # Set device type globally for non-iOS devices
    if device_type != DeviceType.IOS:
        set_device_type(device_type)
        if args.device_id and device_type == DeviceType.ADB:
            # Keep plain adb calls inside setup checks pinned to the selected device.
            os.environ["ANDROID_SERIAL"] = args.device_id

    # Enable HDC verbose mode if using HDC
    if device_type == DeviceType.HDC:
        from phone_agent.hdc import set_hdc_verbose

        set_hdc_verbose(True)

    # Handle --list-apps (no system check needed)
    if args.list_apps:
        if device_type == DeviceType.HDC:
            print("Supported HarmonyOS apps:")
            apps = list_harmonyos_apps()
        elif device_type == DeviceType.IOS:
            print("Supported iOS apps:")
            print("\nNote: For iOS apps, Bundle IDs are configured in:")
            print("  phone_agent/config/apps_ios.py")
            print("\nCurrently configured apps:")
            apps = list_ios_apps()
        else:
            print("Supported Android apps:")
            apps = list_supported_apps()

        for app in sorted(apps):
            print(f"  - {app}")

        if device_type == DeviceType.IOS:
            print(
                "\nTo add iOS apps, find the Bundle ID and add to APP_PACKAGES_IOS dictionary."
            )
        return

    # Handle device commands (these may need partial system checks)
    if handle_device_commands(args):
        return

    # Run system requirements check before proceeding
    if not check_system_requirements(
        device_type,
        wda_url=args.wda_url
        if device_type == DeviceType.IOS
        else "http://localhost:8100",
    ):
        sys.exit(1)

    # Optional pre-test install flow: build/download/install first, then continue testing.
    if args.install:
        if device_type != DeviceType.ADB:
            print("❌ --install 当前只支持 Android/ADB APK 安装。", file=sys.stderr)
            sys.exit(1)
        try:
            handle_install_command(args)
        except RuntimeError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            sys.exit(1)

    # Check model API connectivity and model availability
    if not check_model_api(args.base_url, args.model, args.apikey):
        sys.exit(1)

    # Create configurations and agent based on device type
    model_config = ModelConfig(
        base_url=args.base_url,
        model_name=args.model,
        api_key=args.apikey,
        lang=args.lang,
    )

    reporter = TestRunReporter(
        artifact_name=args.artifact_name,
        base_dir=args.artifact_dir,
        device_type=args.device_type,
        device_id=args.device_id,
        model_name=args.model,
        base_url=args.base_url,
        wda_url=args.wda_url if device_type == DeviceType.IOS else None,
    )
    status_judge: StepStatusJudge | None = None
    if args.file and not args.disable_status_judge:
        judge_config = JudgeConfig(
            base_url=args.judge_base_url,
            model_name=args.judge_model,
            api_key_env=args.judge_api_key_env,
            api_key=args.judge_api_key,
        )
        status_judge = StepStatusJudge(judge_config)
        reporter.environment["judge_model"] = judge_config.model_name
        reporter.environment["judge_base_url"] = judge_config.base_url
        reporter.environment["judge_api_key_env"] = judge_config.api_key_env
    external_judge_system_override = ""
    if status_judge:
        external_judge_system_override = (
            "\n\n【外部判定模式】当前测试运行启用了独立文本判定模型。"
            "你作为手机执行 agent 只负责操作和描述可观察证据，不负责判定 PASS/FAIL。"
            "当当前步骤完成、无需执行或无法继续时，直接调用 finish(message=\"...\")，"
            "message 只写事实证据和停止原因。不要输出 STATUS、PASS、SKIPPED、BLOCKED、FAIL、REVIEW。"
        )

    if device_type == DeviceType.IOS:
        # Create iOS agent
        agent_config = IOSAgentConfig(
            max_steps=args.max_steps,
            wda_url=args.wda_url,
            device_id=args.device_id,
            verbose=not args.quiet,
            lang=args.lang,
            reporter=reporter,
            auto_manage_report_case=not bool(args.file),
            require_structured_finish_status=not bool(status_judge),
            system_prompt=None,
        )
        if external_judge_system_override:
            agent_config.system_prompt = (
                agent_config.system_prompt or ""
            ) + external_judge_system_override

        agent = IOSPhoneAgent(
            model_config=model_config,
            agent_config=agent_config,
        )
    else:
        # Create Android/HarmonyOS agent
        agent_config = AgentConfig(
            max_steps=args.max_steps,
            device_id=args.device_id,
            verbose=not args.quiet,
            lang=args.lang,
            reporter=reporter,
            auto_manage_report_case=not bool(args.file),
            require_structured_finish_status=not bool(status_judge),
            system_prompt=None,
        )
        if external_judge_system_override:
            agent_config.system_prompt = (
                agent_config.system_prompt or ""
            ) + external_judge_system_override

        my_rules = load_rules("rules")

        agent = PhoneAgent(
            model_config=model_config,
            agent_config=agent_config,
            custom_rules=my_rules,
        )

    # Print header
    print("=" * 50)
    if device_type == DeviceType.IOS:
        print("Phone Agent iOS - AI-powered iOS automation")
    else:
        print("Phone Agent - AI-powered phone automation")
    print("=" * 50)
    print(f"Vision Model: {model_config.model_name}")
    print(f"Vision Base URL: {model_config.base_url}")
    if status_judge:
        print(f"Judge Model: {status_judge.config.model_name}")
        print(f"Judge Base URL: {status_judge.config.base_url}")
        print(f"Judge API Key Env: {status_judge.config.api_key_env}")
    print(f"Max Steps: {agent_config.max_steps}")
    print(f"Language: {agent_config.lang}")
    print(f"Device Type: {args.device_type.upper()}")

    # Show iOS-specific config
    if device_type == DeviceType.IOS:
        print(f"WDA URL: {args.wda_url}")

    # Show device info
    if device_type == DeviceType.IOS:
        devices = list_ios_devices()
        if agent_config.device_id:
            print(f"Device: {agent_config.device_id}")
        elif devices:
            device = devices[0]
            print(f"Device: {device.device_name or device.device_id[:16]}")
            if device.model and device.ios_version:
                print(f"        {device.model}, iOS {device.ios_version}")
    else:
        device_factory = get_device_factory()
        devices = device_factory.list_devices()
        if agent_config.device_id:
            print(f"Device: {agent_config.device_id}")
        elif devices:
            print(f"Device: {devices[0].device_id} (auto-detected)")

    print("=" * 50)

    # Run with provided tasks, Markdown test case file, or enter interactive mode
    test_cases: list[ParsedTestCase] = []
    if args.file:
        test_cases = load_test_cases_from_file(args.file)
        loaded_count = len(test_cases)
        test_cases, filter_messages = filter_test_cases(
            test_cases,
            priorities=args.priority,
            modules=args.module,
            start=args.start,
            limit=args.limit,
        )
        if filter_messages:
            filter_text = "; ".join(filter_messages)
            print(
                f"\nLoaded {loaded_count} test case(s) from {args.file}; "
                f"{len(test_cases)} matched {filter_text}"
            )
            if not test_cases:
                raise ValueError(f"No test cases matched filters: {filter_text}")
        else:
            print(f"\nLoaded {loaded_count} test case(s) from {args.file}")
    elif args.tasks:
        test_cases = [parse_test_case(task) for task in args.tasks]

    if test_cases:
        total_tasks = len(test_cases)
        try:
            for index, test_case in enumerate(test_cases, start=1):
                final_status = "UNKNOWN"
                final_result = ""
                max_attempts = 1 + args.case_retries
                for attempt in range(max_attempts):
                    if attempt:
                        print(
                            f"\nRetrying case {test_case.case_id or index}: "
                            f"attempt {attempt + 1}/{max_attempts}"
                        )
                    final_status, final_result = execute_test_case(
                        agent=agent,
                        reporter=reporter,
                        test_case=test_case,
                        index=index,
                        total_tasks=total_tasks,
                        args=args,
                        device_type=device_type,
                        status_judge=status_judge,
                        attempt=attempt,
                    )
                    if final_status == "PASS":
                        if attempt:
                            print(
                                f"Retry passed for {test_case.case_id or index} "
                                f"on attempt {attempt + 1}/{max_attempts}."
                            )
                        break
                    if attempt < max_attempts - 1:
                        print(
                            f"Case {test_case.case_id or index} ended as "
                            f"{final_status}; retry will run."
                        )
                    else:
                        print(
                            f"Case {test_case.case_id or index} final status after "
                            f"{max_attempts} attempt(s): {final_status}"
                        )
        except (Exception, KeyboardInterrupt) as exc:
            if reporter.current_case:
                reporter.finish_case(
                    f"REVIEW: Runner interrupted before normal completion.\n"
                    f"{type(exc).__name__}: {exc}"
                )
            reporter.finish_run()
            print_vision_token_summary(reporter)
            if status_judge:
                print_judge_token_summary(reporter)
            print(f"\nHTML report: {reporter.root_dir / 'index.html'}")
            raise
        reporter.finish_run()
        print_vision_token_summary(reporter)
        if status_judge:
            print_judge_token_summary(reporter)
        print(f"\nSummary report: {reporter.root_dir / 'summary.md'}")
        print(f"HTML report: {reporter.root_dir / 'index.html'}")
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.\n")

        while True:
            try:
                task = input("Enter your task: ").strip()

                if task.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                if not task:
                    continue

                print()
                print_test_case_heading(task)
                reporter.start_case(task, len(reporter.cases) + 1)
                result = agent.run(task)
                if reporter.current_case:
                    reporter.finish_case(result)
                print(f"\nResult: {result}\n")
                agent.reset()
                reporter.finish_run()
                print(f"Summary report: {reporter.root_dir / 'summary.md'}\n")
                print(f"HTML report: {reporter.root_dir / 'index.html'}\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
