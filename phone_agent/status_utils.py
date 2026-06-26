"""Helpers for structured finish status handling."""

from __future__ import annotations

import re
from typing import Any


VALID_STATUS_PATTERN = re.compile(
    r"^\s*(?:STATUS|状态)\s*[:：]\s*(PASS|SKIPPED|BLOCKED|FAIL|REVIEW)\b",
    re.IGNORECASE | re.MULTILINE,
)


def finish_message_has_status(action: dict[str, Any] | None) -> bool:
    """Return True when a finish action message starts with a valid status."""
    if not action or action.get("_metadata") != "finish":
        return True
    message = str(action.get("message") or "")
    return bool(VALID_STATUS_PATTERN.search(message))


def coerce_finish_to_review(action: dict[str, Any] | None) -> dict[str, Any]:
    """Convert a no-status finish into a structured REVIEW finish."""
    message = ""
    if action:
        message = str(action.get("message") or "")
    if VALID_STATUS_PATTERN.search(message):
        return action or {"_metadata": "finish", "message": message}
    reason = message.strip() or "模型已调用 finish，但未按要求给出结构化 STATUS。"
    return {
        "_metadata": "finish",
        "message": f"STATUS: REVIEW\nREASON: {reason}",
    }


def status_formatter_system_prompt() -> str:
    """System prompt for isolated finish-status formatting requests."""
    return (
        "你是测试步骤状态格式化器，不是手机自动化执行 agent。\n"
        "你的唯一任务：把用户提供的步骤结果改写成一个结构化 finish action。\n"
        "你没有手机操作能力，也不允许继续执行测试。\n"
        "输出必须从第一个字符开始就是 finish(message=...)。\n"
        "禁止输出 do(...)、Tap、Wait、Back、Launch、Swipe、Long Press、Note、Take_over。\n"
        "禁止输出思考过程、解释、Markdown、代码块、项目符号、示例或任何额外文本。\n"
        "STATUS 只能是 PASS、SKIPPED、BLOCKED、FAIL、REVIEW。\n"
        "如果无法可靠判断是否通过，输出 REVIEW。\n"
    )


def status_formatter_user_prompt(
    original_message: str,
    attempt: int = 1,
    max_attempts: int = 2,
    step_text: str | None = None,
    target_state: str | None = None,
) -> str:
    """User prompt for isolated finish-status formatting requests."""
    return (
        f"第 {attempt}/{max_attempts} 次格式化。\n"
        "只把下面的信息改写成结构化 finish action，不要执行任何手机动作。\n"
        f"当前步骤：{step_text or '未提供'}\n"
        f"目标状态：{target_state or '未提供'}\n"
        f"原始结果：{original_message or '未提供'}\n\n"
        "判定规则：\n"
        "1. 明确达成当前步骤目标，STATUS 用 PASS。\n"
        "2. 条件步骤无需执行，STATUS 用 SKIPPED。\n"
        "3. 环境、入口、权限、设备状态等导致无法继续，STATUS 用 BLOCKED。\n"
        "4. 出现明确错误、崩溃、白屏、黑屏、目标结果错误，STATUS 用 FAIL。\n"
        "5. 证据不足或无法可靠判断，STATUS 用 REVIEW。\n\n"
        "最终输出必须严格等于下面这种形态，只允许这一行：\n"
        'finish(message="STATUS: <PASS或SKIPPED或BLOCKED或FAIL或REVIEW>\\nREASON: <一句中文原因>")'
    )


def status_repair_prompt(
    attempt: int = 1,
    max_attempts: int = 2,
    step_text: str | None = None,
    target_state: str | None = None,
) -> str:
    """Prompt used when the model forgot STATUS in finish(message=...)."""
    step_context = ""
    if step_text or target_state:
        step_context = (
            f"当前步骤：{step_text or '未提供'}\n"
            f"目标状态：{target_state or '未提供'}\n"
        )
    return (
        f"上一条输出缺少合格的 finish STATUS。这是第 {attempt}/{max_attempts} 次状态修复。不要执行任何新操作。\n"
        "你现在不是在继续执行测试步骤，只是在修正上一条 finish 的 message。\n"
        f"{step_context}"
        "只允许输出 finish(message=\"...\") 这一种 action。\n"
        "禁止输出 do(...)、Tap、Wait、Back、Launch、Swipe、Long Press、Note、Take_over 或任何其他动作。\n"
        "禁止点击、等待、返回、滑动、启动应用，也不要根据当前页面继续探索。\n"
        "只输出一行 finish action，不要解释、不要复述规则、不要引用示例。\n"
        "STATUS 必须选择一个实际单词：PASS、SKIPPED、BLOCKED、FAIL、REVIEW；禁止输出尖括号或占位符。\n"
        "如果无法确定 PASS/SKIPPED/BLOCKED/FAIL，但需要人工查看当前结果，使用 REVIEW。\n"
        "唯一允许的格式示例：\n"
        'finish(message="STATUS: PASS\\nREASON: 当前步骤目标已达成，证据是当前页面显示目标内容。")'
    )


def step_limit_status_prompt(step_text: str, target_state: str | None = None) -> str:
    """Prompt used after a step reaches its action limit."""
    target = target_state or "当前步骤目标状态"
    return (
        "当前测试步骤已经达到单步操作上限。不要执行任何新操作。\n"
        "你现在不是在继续执行测试步骤，只是在给当前步骤补一个最终状态。\n"
        "请根据当前截图和刚才的操作历史，只判断这个步骤最终状态。\n"
        f"当前步骤：{step_text}\n"
        f"目标状态：{target}\n"
        "只允许输出 finish(message=\"...\") 这一种 action。\n"
        "禁止输出 do(...)、Tap、Wait、Back、Launch、Swipe、Long Press、Note、Take_over 或任何其他动作。\n"
        "禁止点击、等待、返回、滑动、启动应用，也不要根据当前页面继续探索。\n"
        "只输出一行 finish action，不要解释、不要复述规则、不要引用示例。\n"
        "如果目标已经达成，使用 PASS；如果条件步骤已无需执行，使用 SKIPPED；"
        "如果入口/环境阻塞导致无法完成，使用 BLOCKED；如果出现明确错误，使用 FAIL；"
        "如果无法确定但需要人工查看，使用 REVIEW。\n"
        "STATUS 必须选择一个实际单词：PASS、SKIPPED、BLOCKED、FAIL、REVIEW；禁止输出尖括号或占位符。\n"
        "唯一允许的格式示例：\n"
        'finish(message="STATUS: PASS\\nREASON: 当前步骤目标已达成，证据是当前页面显示目标内容。")'
    )
