"""Text-only status judge resolved from a Worker-local model profile."""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from phone_agent.execution.models import ExecutionCase, ExecutionStep
from phone_agent.model import ModelConfig


_STATUSES = {"PASS", "SKIPPED", "BLOCKED", "FAIL", "REVIEW"}


@dataclass(frozen=True, slots=True)
class JudgeResult:
    status: str
    message: str
    raw: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class StructuredStatusJudge:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def judge(
        self,
        *,
        case: ExecutionCase,
        step: ExecutionStep,
        execution_message: str,
    ) -> JudgeResult:
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是手机测试结果判定器，只根据执行证据判定当前步骤。"
                        "只输出 JSON 对象，字段为 status 和 message。status 只能是 "
                        "PASS、SKIPPED、BLOCKED、FAIL、REVIEW；证据不足必须 REVIEW。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用例：{case.display_id} {case.title}\n"
                        f"目标：{case.goal}\n步骤：{step.instruction}\n"
                        f"目标状态：{step.target_state or case.expected_result}\n"
                        f"预期 Activity：{step.expected_activity or '未指定'}\n"
                        f"执行模型返回：\n{execution_message}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=min(self.config.max_tokens, 1000),
        )
        raw = response.choices[0].message.content or ""
        data = _parse_json_object(raw)
        status = str(data.get("status", "REVIEW")).upper()
        if status not in _STATUSES:
            status = "REVIEW"
        message = str(data.get("message") or raw or "判定模型未返回理由")
        usage = response.usage
        return JudgeResult(
            status=status,
            message=message,
            raw=raw,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )


def _parse_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.startswith("json"):
            text = text[4:].lstrip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    return value
