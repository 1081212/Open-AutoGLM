"""从 rules 目录加载 YAML 自定义规则。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from phone_agent.actions.handler import (
    do,
    finish,
    play_audio_while_holding_element,
    tap_element,
    type_into_element,
)
from phone_agent.agent import CustomRule, RuleContext


ACTION_BUILDERS = {
    "do": do,
    "finish": finish,
    "tap_element": tap_element,
    "type_into_element": type_into_element,
    "play_audio_while_holding_element": play_audio_while_holding_element,
}


def load_rules(rules_dir: str | Path = "rules") -> list[CustomRule]:
    """扫描目录下所有 YAML 文件并加载 CustomRule。"""
    root = Path(rules_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    if not root.exists():
        return []

    rules: list[CustomRule] = []
    for path in sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")]):
        rules.extend(_load_rule_file(path))
    print(f"Loaded {len(rules)} custom rule(s) from {root}")
    return rules


def _load_rule_file(path: Path) -> list[CustomRule]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML 顶层必须是对象")
    specs = data.get("rules")
    if not isinstance(specs, list):
        raise ValueError(f"{path}: 缺少 rules 列表")

    loaded: list[CustomRule] = []
    for index, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: rules[{index}] 必须是对象")
        if spec.get("enabled", True) is False:
            continue
        loaded.append(_build_rule(path, index, spec))
    return loaded


def _build_rule(path: Path, index: int, spec: dict[str, Any]) -> CustomRule:
    name = _require_str(path, index, spec, "name")
    activity = _require_str_list(path, index, spec, "activity")
    actions = _require_actions(path, index, spec)
    conditions = spec.get("conditions") or {}
    if not isinstance(conditions, dict):
        raise ValueError(f"{path}: rules[{index}].conditions 必须是对象")

    return CustomRule(
        name=name,
        activity=activity,
        condition=_build_condition(conditions),
        action=actions,
        terminal=bool(spec.get("terminal", False)),
        step_delay=float(spec.get("step_delay", 1.0)),
        max_fires=int(spec.get("max_fires", 1)),
        post_delay=float(spec.get("post_delay", 2.0)),
        context_note=str(spec.get("context_note", "")),
    )


def _build_condition(conditions: dict[str, Any]):
    """生成二级条件函数；Activity 已经在 PhoneAgent 内部优先粗筛。"""

    def condition(ctx: RuleContext) -> bool:
        if not _match_str_or_list(ctx.task, conditions.get("task_contains"), contains=True):
            return False
        if not _match_str_or_list(ctx.task, conditions.get("task_not_contains"), contains=False):
            return False
        if not _match_str_or_list(ctx.app_name, conditions.get("app_name"), exact=True):
            return False
        if "step" in conditions and ctx.step != int(conditions["step"]):
            return False
        return True

    return condition


def _match_str_or_list(
    value: str,
    expected: Any,
    *,
    contains: bool = False,
    exact: bool = False,
) -> bool:
    if expected is None:
        return True
    values = expected if isinstance(expected, list) else [expected]
    values = [str(item) for item in values]
    if contains:
        return any(item in value for item in values)
    if exact:
        return any(value == item for item in values)
    return not any(item in value for item in values)


def _require_actions(path: Path, index: int, spec: dict[str, Any]) -> list[dict[str, Any]]:
    action_specs = spec.get("actions")
    if not isinstance(action_specs, list) or not action_specs:
        raise ValueError(f"{path}: rules[{index}].actions 必须是非空列表")
    return [_build_action(path, index, action_spec) for action_spec in action_specs]


def _build_action(path: Path, index: int, action_spec: Any) -> dict[str, Any]:
    if not isinstance(action_spec, dict):
        raise ValueError(f"{path}: rules[{index}].actions 内每一项必须是对象")
    action_type = action_spec.get("type")
    if not isinstance(action_type, str):
        raise ValueError(f"{path}: rules[{index}].actions 缺少 type")
    builder = ACTION_BUILDERS.get(action_type)
    if builder is None:
        raise ValueError(f"{path}: rules[{index}] 不支持 action type: {action_type}")

    kwargs = {key: value for key, value in action_spec.items() if key != "type"}
    if action_type == "type_into_element":
        input_text = kwargs.pop("input_text", None)
        if input_text is None:
            raise ValueError(f"{path}: rules[{index}].type_into_element 缺少 input_text")
        return builder(str(input_text), **kwargs)
    if action_type == "play_audio_while_holding_element":
        audio_path = kwargs.pop("audio_path", None)
        if audio_path is None:
            raise ValueError(
                f"{path}: rules[{index}].play_audio_while_holding_element 缺少 audio_path"
            )
        return builder(str(audio_path), **kwargs)
    return builder(**kwargs)


def _require_str(path: Path, index: int, spec: dict[str, Any], key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: rules[{index}].{key} 必须是非空字符串")
    return value.strip()


def _require_str_list(path: Path, index: int, spec: dict[str, Any], key: str) -> list[str]:
    value = spec.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: rules[{index}].{key} 必须是非空列表")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{path}: rules[{index}].{key} 只能包含非空字符串")
    return [item.strip() for item in value]
