"""Text-only status judge for Markdown test steps."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


VALID_STEP_STATUSES = {"PASS", "SKIPPED", "BLOCKED", "FAIL", "REVIEW"}


@dataclass
class JudgeConfig:
    """Configuration for the text-only step judge model."""

    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    model_name: str = "ep-20240813102137-m62n6"
    api_key_env: str = "ARK_API_KEY"
    api_key: str | None = None
    max_tokens: int = 800
    temperature: float = 0.0

    def resolve_api_key(self) -> str | None:
        return self.api_key or os.getenv(self.api_key_env)


@dataclass
class JudgeResult:
    """Structured result returned by the judge."""

    status: str
    reason: str
    raw: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def message(self) -> str:
        return f"STATUS: {self.status}\nREASON: {self.reason}"


class StepStatusJudge:
    """Judge one test step status using text-only evidence."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        api_key = config.resolve_api_key()
        if not api_key:
            raise ValueError(
                f"Judge model API key is required. Set {config.api_key_env} "
                "or pass --judge-api-key."
            )
        self.client = OpenAI(base_url=config.base_url, api_key=api_key)

    def judge(
        self,
        *,
        case_id: str | None,
        case_title: str | None,
        step_index: int,
        step_text: str,
        target_state: str | None,
        expected_activity: str | None,
        case_task: str,
        all_steps_text: str,
        previous_steps: list[str],
        agent_text_log: str,
        operation_log: str,
        model_result: str,
        step_limit_reached: bool,
    ) -> JudgeResult:
        """Return PASS/SKIPPED/BLOCKED/FAIL/REVIEW for one test step."""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是自动化测试步骤结果判定器，只根据文本证据判定状态。"
                    "你不是手机操作 agent，不能提出或执行任何点击、滑动、等待、返回、启动应用等动作。"
                    "你只能输出 JSON，不要输出 Markdown、解释、代码块或额外文本。"
                ),
            },
            {
                "role": "user",
                "content": self._build_prompt(
                    case_id=case_id,
                    case_title=case_title,
                    step_index=step_index,
                    step_text=step_text,
                    target_state=target_state,
                    expected_activity=expected_activity,
                    case_task=case_task,
                    all_steps_text=all_steps_text,
                    previous_steps=previous_steps,
                    agent_text_log=agent_text_log,
                    operation_log=operation_log,
                    model_result=model_result,
                    step_limit_reached=step_limit_reached,
                ),
            },
        ]
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=False,
        )
        raw = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        result = parse_judge_response(raw)
        result.prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        result.completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        result.total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return result

    def _build_prompt(
        self,
        *,
        case_id: str | None,
        case_title: str | None,
        step_index: int,
        step_text: str,
        target_state: str | None,
        expected_activity: str | None,
        case_task: str,
        all_steps_text: str,
        previous_steps: list[str],
        agent_text_log: str,
        operation_log: str,
        model_result: str,
        step_limit_reached: bool,
    ) -> str:
        return f"""请判定当前测试步骤最终状态。

输出要求：
只输出一个 JSON 对象，格式必须是：
{{"status":"PASS|SKIPPED|BLOCKED|FAIL|REVIEW","reason":"一句中文原因"}}

状态定义：
- PASS：文本证据显示当前步骤目标已经达成。
- SKIPPED：这是条件步骤，且证据显示条件不成立或无需执行。
- BLOCKED：环境、权限、设备连接、入口不可用、模型达到步骤上限等原因导致无法继续确认。
- FAIL：证据显示 App 崩溃、白屏、黑屏、明确错误、进入错误页面、目标结果明确不符合。
- REVIEW：证据不足，无法可靠判断 PASS/SKIPPED/BLOCKED/FAIL。

判定原则：
1. 不要因为模型文字里出现“成功”“完成”就直接 PASS，必须看是否符合当前步骤和目标状态。
2. 如果 step_limit_reached=true，但操作日志已经清楚显示目标达成，可以 PASS；否则优先 BLOCKED 或 REVIEW。
3. 必须结合“完整执行步骤列表”判断当前步骤结果。如果执行模型多做了后续步骤，但当前状态已经满足当前步骤或推进到了后续步骤所需状态，且没有触发禁止行为，不要判 FAIL，可以判 PASS，并在 reason 里说明“已超前推进但有利于后续步骤”。
4. 只有当执行结果明确破坏后续流程、进入错误且不可恢复的页面、触发禁止行为、App 崩溃/白屏/黑屏/明确错误时，才判 FAIL。
5. 如果当前步骤未严格完成，但当前状态可以让后续步骤继续执行，优先 PASS 或 REVIEW，不要直接 FAIL 阻塞流程。
6. 如果当前步骤是“如果...则...”且条件不需要执行，可以 SKIPPED。
7. 对于“点击底部导航/进入某页面/回到首页”这类导航步骤，如果文本证据显示目标页面的核心内容已经出现，可以判 PASS；不要求证据中逐字出现“Tab 被选中”。
8. 当前步骤操作记录中的 UI 文本通常是动作执行前截图；finish 动作所在记录的 UI 文本代表执行模型决定收尾时的当前状态。
9. 不要输出手机操作建议，不要输出 finish/do 函数。

用例 ID：{case_id or "unknown"}
用例标题：{case_title or "unknown"}
完整测试用例：
{_clip_text(case_task or "未提供", 9000)}

完整执行步骤列表：
{_clip_text(all_steps_text or "未提供", 4000)}

当前步骤序号：{step_index}
当前步骤：{step_text}
目标状态：{target_state or "未提供"}
期望 Activity：{expected_activity or "未提供"}
step_limit_reached：{str(step_limit_reached).lower()}

前序步骤摘要：
{chr(10).join(previous_steps) if previous_steps else "暂无"}

执行模型最终输出：
{model_result or "未提供"}

执行模型文本上下文（已去除图片，只保留文本）：
{_clip_text(agent_text_log or "未提供", 5000)}

当前步骤操作记录：
{_clip_text(operation_log or "无操作记录", 7000)}
"""


def parse_judge_response(raw: str) -> JudgeResult:
    """Parse judge JSON, with a conservative REVIEW fallback."""
    text = (raw or "").strip()
    data: dict[str, Any] | None = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    if data:
        status = str(data.get("status", "")).strip().upper()
        reason = str(data.get("reason", "")).strip()
        if status in VALID_STEP_STATUSES:
            return JudgeResult(status=status, reason=reason or "判定模型未提供原因。", raw=raw)

    status_match = re.search(
        r"^\s*(?:STATUS|状态|status)\s*[:：]\s*(PASS|SKIPPED|BLOCKED|FAIL|REVIEW)\b",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if status_match:
        status = status_match.group(1).upper()
        return JudgeResult(status=status, reason="判定模型未输出标准 JSON，已从文本中提取状态。", raw=raw)

    return JudgeResult(
        status="REVIEW",
        reason="判定模型未输出可解析的状态，需要人工复核。",
        raw=raw,
    )


def _clip_text(value: str, max_chars: int) -> str:
    text = value or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[truncated {len(text) - max_chars} chars]"
