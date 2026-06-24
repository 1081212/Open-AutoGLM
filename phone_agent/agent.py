"""Main PhoneAgent class for orchestrating phone automation."""

import json
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from phone_agent.actions import ActionHandler
from phone_agent.actions.handler import do, finish, parse_action
from phone_agent.config import get_messages, get_system_prompt
from phone_agent.device_factory import DeviceType
from phone_agent.device_factory import get_device_factory
from phone_agent.model import ModelClient, ModelConfig
from phone_agent.model.client import MessageBuilder

ISSUE_KEYWORDS = (
    "bug",
    "不符合",
    "不合理",
    "异常",
    "错误",
    "失败",
    "没生效",
    "无响应",
    "白屏",
    "黑屏",
    "闪退",
    "卡死",
    "崩溃",
)


@dataclass
class AgentConfig:
    """Configuration for the PhoneAgent."""

    max_steps: int = 100
    device_id: str | None = None
    lang: str = "cn"
    system_prompt: str | None = None
    verbose: bool = True
    reporter: Any | None = None

    def __post_init__(self):
        if self.system_prompt is None:
            self.system_prompt = get_system_prompt(self.lang)


@dataclass
class StepResult:
    """Result of a single agent step."""

    success: bool
    finished: bool
    action: dict[str, Any] | None
    thinking: str
    message: str | None = None


@dataclass
class RuleContext:
    """
    Context passed to each CustomRule condition and action.

    Attributes:
        app_name:  Name of the currently foregrounded app (from APP_PACKAGES mapping),
                   or "System Home" if unknown.
        activity:  Full Android Activity class name of the current screen,
                   e.g. "com.meituan.android.pt.main.MainActivity".
                   Empty string when unavailable (iOS / HDC mode).
        screenshot: The current screenshot object (.base64_data / .width / .height).
        step:      Current step number (1-based).
        task:      The original task string given to agent.run().
        device:    UIAutomator2 device object (u2.Device) for element-based operations.
                   None when not in ADB/UIAutomator2 mode.
    """
    app_name: str
    activity: str         # e.g. "com.meituan.android.pt.main.MainActivity"
    screenshot: Any       # phone_agent.adb.screenshot.Screenshot
    step: int
    task: str
    device: Any | None = None   # uiautomator2.Device, for element-based ops


@dataclass
class CustomRule:
    """在模型调用前执行的自定义规则。

    规则适合处理稳定、重复、低价值的前置流程，例如登录、权限弹窗、
    广告关闭等。规则命中后会先执行 action，再让模型继续原任务。
    """

    name: str
    condition: Callable[["RuleContext"], bool]
    action: (
            dict[str, Any]
            | list[dict[str, Any]]
            | Callable[["RuleContext"], dict[str, Any] | list[dict[str, Any]] | None]
    )
    terminal: bool = False  # True → 规则执行完后任务结束
    step_delay: float = 1.0  # 多个动作之间的间隔秒数
    max_fires: int = 1  # 同一个 run() 内最多触发几次（0 = 不限）
    post_delay: float = 2.0  # 规则执行后等待多少秒（给 app 留时间跳转页面）
    context_note: str = ""  # 规则执行后注入给模型的说明，空则自动生成


class PhoneAgent:
    """
    AI-powered agent for automating Android phone interactions.

    The agent uses a vision-language model to understand screen content
    and decide on actions to complete user tasks.

    Args:
        model_config: Configuration for the AI model.
        agent_config: Configuration for the agent behavior.
        confirmation_callback: Optional callback for sensitive action confirmation.
        takeover_callback: Optional callback for takeover requests.

    Example:
        >>> from phone_agent import PhoneAgent
        >>> from phone_agent.model import ModelConfig
        >>>
        >>> model_config = ModelConfig(base_url="http://localhost:8000/v1")
        >>> agent = PhoneAgent(model_config)
        >>> agent.run("Open WeChat and send a message to John")
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        agent_config: AgentConfig | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
        custom_rules: list[CustomRule] | None = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.agent_config = agent_config or AgentConfig()

        self.model_client = ModelClient(self.model_config)
        self.action_handler = ActionHandler(
            device_id=self.agent_config.device_id,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        self._context: list[dict[str, Any]] = []
        self._step_count = 0
        self._current_task = ""
        self.custom_rules = custom_rules or []
        self._rule_fire_counts: dict[str, int] = {}
        self._remind_task = False

    def add_rule(self, rule: CustomRule) -> None:
        """运行时注册一条自定义规则。"""
        self.custom_rules.append(rule)

    def run(self, task: str) -> str:
        """
        Run the agent to complete a task.

        Args:
            task: Natural language description of the task.

        Returns:
            Final message from the agent.
        """
        self._context = []
        self._step_count = 0
        self._current_task = task
        self._rule_fire_counts = {}
        self._remind_task = False
        if self.agent_config.reporter and not self.agent_config.reporter.current_case:
            self.agent_config.reporter.start_case(
                task, len(self.agent_config.reporter.cases) + 1
            )

        # First step with user prompt
        result = self._execute_step(task, is_first=True)

        if result.finished:
            message = result.message or "Task completed"
            self._finish_report_case(message)
            return message

        # Continue until finished or max steps reached
        while self._step_count < self.agent_config.max_steps:
            result = self._execute_step(is_first=False)

            if result.finished:
                message = result.message or "Task completed"
                self._finish_report_case(message)
                return message

        message = "Max steps reached"
        self._finish_report_case(message, max_steps_reached=True)
        return message

    def step(self, task: str | None = None) -> StepResult:
        """
        Execute a single step of the agent.

        Useful for manual control or debugging.

        Args:
            task: Task description (only needed for first step).

        Returns:
            StepResult with step details.
        """
        is_first = len(self._context) == 0

        if is_first and not task:
            raise ValueError("Task is required for the first step")

        if is_first and task:
            self._current_task = task

        return self._execute_step(task, is_first)

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self._context = []
        self._step_count = 0
        self._current_task = ""
        self._rule_fire_counts = {}
        self._remind_task = False

    def _execute_step(
        self, user_prompt: str | None = None, is_first: bool = False
    ) -> StepResult:
        """Execute a single step of the agent loop."""
        self._step_count += 1

        # Capture current screen state
        device_factory = get_device_factory()
        screenshot = device_factory.get_screenshot(self.agent_config.device_id)
        current_app = device_factory.get_current_app(self.agent_config.device_id)
        if self.agent_config.reporter:
            self.agent_config.reporter.save_step(
                step=self._step_count,
                screenshot_base64=screenshot.base64_data,
                width=screenshot.width,
                height=screenshot.height,
                current_app=current_app,
                sensitive_screenshot=screenshot.is_sensitive,
            )

        # 首步也先放入 system prompt，避免规则在第一步命中后，
        # 下一轮模型上下文里缺少系统提示。
        if is_first and not self._context:
            self._context.append(
                MessageBuilder.create_system_message(self.agent_config.system_prompt)
            )

        # 在调用模型前先尝试命中自定义规则。规则命中后会直接执行动作，
        # 本轮不再请求模型，从而把登录/弹窗等确定性流程从模型决策中剥离。
        rule_result = self._try_custom_rules(current_app, screenshot)
        if rule_result is not None:
            return rule_result

        # Build messages
        if is_first:
            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"{user_prompt}\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )
        else:
            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"** Screen Info **\n\n{screen_info}"
            if self._remind_task and self._current_task:
                text_content += f"\n\n** 原任务（请继续完成）**\n{self._current_task}"
                self._remind_task = False

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )

        # Get model response
        try:
            msgs = get_messages(self.agent_config.lang)
            # The model thinking stream is useful for debugging, but it makes
            # test reports noisy. Keep it in context, do not print it by default.
            # print("\n" + "=" * 50)
            # print(f"💭 {msgs['thinking']}:")
            # print("-" * 50)
            response = self.model_client.request(self._context)
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            return StepResult(
                success=False,
                finished=True,
                action=None,
                thinking="",
                message=f"Model error: {e}",
            )

        # Parse action from response
        try:
            action = parse_action(response.action)
        except ValueError:
            if self.agent_config.verbose:
                traceback.print_exc()
            action = finish(message=response.action)

        action_response = response.action
        action = self._convert_bug_finish_to_note(action)
        if action.get("action") == "Note" and action_response != response.action:
            action_response = self._format_action_for_context(action)

        if self.agent_config.verbose:
            # Detailed action JSON is retained in test_artifacts/report.json.
            # Leaving this disabled keeps stdout focused on final results.
            # print("-" * 50)
            # print(f"🎯 {msgs['action']}:")
            # print(json.dumps(action, ensure_ascii=False, indent=2))
            # print("=" * 50 + "\n")
            pass

        # Remove image from context to save space
        self._context[-1] = MessageBuilder.remove_images_from_message(self._context[-1])

        # Execute action
        try:
            result = self.action_handler.execute(
                action, screenshot.width, screenshot.height
            )
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            result = self.action_handler.execute(
                finish(message=str(e)), screenshot.width, screenshot.height
            )
        if self.agent_config.reporter:
            self.agent_config.reporter.record_action(
                step=self._step_count,
                action=action,
                success=result.success,
                message=result.message,
            )

        # Add assistant response to context
        self._context.append(
            MessageBuilder.create_assistant_message(
                f"<think>{response.thinking}</think><answer>{action_response}</answer>"
            )
        )

        # Check if finished
        finished = action.get("_metadata") == "finish" or result.should_finish

        if finished and self.agent_config.verbose:
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "🎉 " + "=" * 48)
            print(
                f"✅ {msgs['task_completed']}: {result.message or action.get('message', msgs['done'])}"
            )
            print("=" * 50 + "\n")

        return StepResult(
            success=result.success,
            finished=finished,
            action=action,
            thinking=response.thinking,
            message=result.message or action.get("message"),
        )

    def _try_custom_rules(self, current_app: str, screenshot: Any) -> StepResult | None:
        """尝试执行自定义规则；未命中时返回 None，保持原有模型流程。"""
        if not self.custom_rules:
            return None

        u2_device = self._get_u2_device_or_none()
        activity = self._get_current_activity(u2_device)
        ctx = RuleContext(
            app_name=current_app,
            activity=activity,
            screenshot=screenshot,
            step=self._step_count,
            task=self._current_task,
            device=u2_device,
        )

        for rule in self.custom_rules:
            try:
                fires = self._rule_fire_counts.get(rule.name, 0)
                if rule.max_fires > 0 and fires >= rule.max_fires:
                    continue
                if not rule.condition(ctx):
                    continue

                raw_actions = rule.action(ctx) if callable(rule.action) else rule.action
                self._rule_fire_counts[rule.name] = fires + 1

                if raw_actions is None:
                    # callable 规则可直接通过 ctx.device 完成操作。返回 None 表示
                    # 规则自己已经处理完，本框架不再派发 action。
                    if self.agent_config.verbose:
                        print(f'\n[Rule:{rule.name}] fired - callable handled it')
                    if rule.post_delay > 0:
                        time.sleep(rule.post_delay)
                    self._append_rule_context_note(rule, 0)
                    return StepResult(
                        success=True,
                        finished=rule.terminal,
                        action=None,
                        thinking=f"[Custom rule: {rule.name}]",
                        message=None,
                    )

                actions = raw_actions if isinstance(raw_actions, list) else [raw_actions]
                if self.agent_config.verbose:
                    print(f'\n[Rule:{rule.name}] fired - {len(actions)} action(s)')

                last_action: dict[str, Any] | None = None
                last_result = None
                for index, action in enumerate(actions, start=1):
                    last_action = action
                    if self.agent_config.verbose:
                        print(f"  [{index}/{len(actions)}] {action}")
                    last_result = self.action_handler.execute(
                        action, screenshot.width, screenshot.height
                    )
                    if self.agent_config.reporter:
                        self.agent_config.reporter.record_action(
                            step=self._step_count,
                            action=action,
                            success=last_result.success,
                            message=last_result.message,
                        )
                    if last_result.should_finish or action.get("_metadata") == "finish":
                        break
                    if index < len(actions) and rule.step_delay > 0:
                        time.sleep(rule.step_delay)

                if rule.post_delay > 0:
                    time.sleep(rule.post_delay)

                self._append_rule_context_note(rule, len(actions))
                finished = (
                    rule.terminal
                    or (last_action or {}).get("_metadata") == "finish"
                    or (last_result is not None and last_result.should_finish)
                )
                return StepResult(
                    success=last_result.success if last_result else True,
                    finished=finished,
                    action=last_action,
                    thinking=f"[Custom rule: {rule.name}]",
                    message=(last_result.message if last_result else None)
                    or (last_action or {}).get("message"),
                )
            except Exception as exc:
                # 规则失败不能拖垮基础 agent 流程；记录后继续尝试下一条规则。
                if self.agent_config.verbose:
                    print(f"[Rule:{rule.name}] error: {exc}")

        return None

    def _append_rule_context_note(self, rule: CustomRule, action_count: int) -> None:
        """把规则执行结果写入上下文，提醒模型继续原任务。"""
        note = rule.context_note or (
            f'【系统提示】自动规则"{rule.name}"已执行完毕'
            f"（共 {action_count} 个动作），请继续完成原任务。"
        )
        self._context.append(
            MessageBuilder.create_assistant_message(
                f"<think>{note}</think>"
                '<answer>do(action="Wait", duration="0 seconds")</answer>'
            )
        )
        self._remind_task = True

    def _get_u2_device_or_none(self) -> Any | None:
        """懒加载 UIAutomator2；未安装或连接失败时返回 None，不影响普通流程。"""
        try:
            from phone_agent.adb.u2_connection import get_u2_device

            return get_u2_device(self.agent_config.device_id)
        except Exception:
            return None

    def _get_current_activity(self, u2_device: Any | None = None) -> str:
        """获取当前 Activity；优先 UIAutomator2，失败后退回 adb dumpsys。"""
        if u2_device is not None:
            try:
                app_info = u2_device.app_current()
                activity = app_info.get("activity", "") or ""
                package = app_info.get("package", "") or ""
                if activity.startswith(".") and package:
                    activity = package + activity
                if activity:
                    return activity
            except Exception:
                pass

        device_factory = get_device_factory()
        if device_factory.device_type != DeviceType.ADB:
            return ""

        import subprocess

        cmd = ["adb"]
        if self.agent_config.device_id:
            cmd.extend(["-s", self.agent_config.device_id])
        cmd.extend(["shell", "dumpsys", "window"])
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", timeout=10
            )
        except Exception:
            return ""

        for line in result.stdout.splitlines():
            if "mCurrentFocus" not in line and "mFocusedApp" not in line:
                continue
            match = re.search(r"([A-Za-z0-9_$.]+/[A-Za-z0-9_.$]+)", line)
            if not match:
                continue
            package, activity = match.group(1).split("/", 1)
            if activity.startswith("."):
                activity = package + activity
            return activity
        return ""

    def _finish_report_case(
        self, message: str, max_steps_reached: bool = False
    ) -> None:
        if self.agent_config.reporter and self.agent_config.reporter.current_case:
            status = self.agent_config.reporter.finish_case(
                message, max_steps_reached=max_steps_reached
            )
            print(f"\nResult: {status} - {message}")
            print(f"Artifacts: {self.agent_config.reporter.root_dir}\n")

    def _convert_bug_finish_to_note(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a premature bug-reporting finish into Note.

        This only inspects the parsed action payload, not the model thinking, so
        historical mentions of earlier issues do not trigger the guard.
        """
        if action.get("_metadata") != "finish":
            return action

        message = str(action.get("message", ""))
        if any(keyword.lower() in message.lower() for keyword in ISSUE_KEYWORDS):
            return do(action="Note", message=message)

        return action

    def _format_action_for_context(self, action: dict[str, Any]) -> str:
        """Serialize an executed action back into the prompt action format."""
        if action.get("_metadata") == "do":
            args = []
            for key, value in action.items():
                if key == "_metadata":
                    continue
                args.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
            return f"do({', '.join(args)})"

        if action.get("_metadata") == "finish":
            return f"finish(message={json.dumps(action.get('message', ''), ensure_ascii=False)})"

        return str(action)

    @property
    def context(self) -> list[dict[str, Any]]:
        """Get the current conversation context."""
        return self._context.copy()

    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count
