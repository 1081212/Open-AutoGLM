"""iOS PhoneAgent class for orchestrating iOS phone automation."""

import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from phone_agent.actions.handler import do, finish, parse_action
from phone_agent.actions.handler_ios import IOSActionHandler
from phone_agent.config import get_messages
from phone_agent.config.prompts_ios_zh import SYSTEM_PROMPT_IOS_ZH
from phone_agent.model import ModelClient, ModelConfig
from phone_agent.model.client import MessageBuilder
from phone_agent.status_utils import (
    coerce_finish_to_review,
    finish_message_has_status,
    status_formatter_system_prompt,
    status_formatter_user_prompt,
)
from phone_agent.xctest import XCTestConnection, get_current_app, get_screenshot


@dataclass
class IOSAgentConfig:
    """Configuration for the iOS PhoneAgent."""

    max_steps: int = 100
    wda_url: str = "http://localhost:8100"
    session_id: str | None = None
    device_id: str | None = None  # iOS device UDID
    lang: str = "cn"
    system_prompt: str | None = None
    verbose: bool = True
    reporter: Any | None = None
    auto_manage_report_case: bool = True
    require_structured_finish_status: bool = True

    def __post_init__(self):
        if self.system_prompt is None:
            self.system_prompt = SYSTEM_PROMPT_IOS_ZH


@dataclass
class StepResult:
    """Result of a single agent step."""

    success: bool
    finished: bool
    action: dict[str, Any] | None
    thinking: str
    message: str | None = None


class IOSPhoneAgent:
    """
    AI-powered agent for automating iOS phone interactions.

    The agent uses a vision-language model to understand screen content
    and decide on actions to complete user tasks via WebDriverAgent.

    Args:
        model_config: Configuration for the AI model.
        agent_config: Configuration for the iOS agent behavior.
        confirmation_callback: Optional callback for sensitive action confirmation.
        takeover_callback: Optional callback for takeover requests.

    Example:
        >>> from phone_agent.agent_ios import IOSPhoneAgent, IOSAgentConfig
        >>> from phone_agent.model import ModelConfig
        >>>
        >>> model_config = ModelConfig(base_url="http://localhost:8000/v1")
        >>> agent_config = IOSAgentConfig(wda_url="http://localhost:8100")
        >>> agent = IOSPhoneAgent(model_config, agent_config)
        >>> agent.run("Open Safari and search for Apple")
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        agent_config: IOSAgentConfig | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.agent_config = agent_config or IOSAgentConfig()

        self.model_client = ModelClient(self.model_config)

        # Initialize WDA connection and create session if needed
        self.wda_connection = XCTestConnection(wda_url=self.agent_config.wda_url)

        # Auto-create session if not provided
        if self.agent_config.session_id is None:
            success, session_id = self.wda_connection.start_wda_session()
            if success and session_id != "session_started":
                self.agent_config.session_id = session_id
                if self.agent_config.verbose:
                    print(f"✅ Created WDA session: {session_id}")
            elif self.agent_config.verbose:
                print(f"⚠️  Using default WDA session (no explicit session ID)")

        self.action_handler = IOSActionHandler(
            wda_url=self.agent_config.wda_url,
            session_id=self.agent_config.session_id,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        self._context: list[dict[str, Any]] = []
        self._step_count = 0
        self._finish_status_repair_count = 0

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
        self._finish_status_repair_count = 0

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

        return self._execute_step(task, is_first)

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self._context = []
        self._step_count = 0
        self._finish_status_repair_count = 0

    def request_finish_status_only(
        self, step_text: str, target_state: str | None = None
    ) -> str:
        """Ask the model to close the current step with finish STATUS only."""
        return self._repair_finish_status_isolated(
            "当前步骤达到操作上限，主执行 agent 未能在步数限制内完成结构化 finish。",
            step_text,
            target_state,
        )

    def _execute_step(
        self, user_prompt: str | None = None, is_first: bool = False
    ) -> StepResult:
        """Execute a single step of the agent loop."""
        self._step_count += 1

        # Capture current screen state
        screenshot = get_screenshot(
            wda_url=self.agent_config.wda_url,
            session_id=self.agent_config.session_id,
            device_id=self.agent_config.device_id,
        )
        current_app = get_current_app(
            wda_url=self.agent_config.wda_url, session_id=self.agent_config.session_id
        )
        if self.agent_config.reporter:
            self.agent_config.reporter.save_step(
                step=self._step_count,
                screenshot_base64=screenshot.base64_data,
                width=screenshot.width,
                height=screenshot.height,
                current_app=current_app,
                sensitive_screenshot=screenshot.is_sensitive,
            )

        # Build messages
        if is_first:
            self._context.append(
                MessageBuilder.create_system_message(self.agent_config.system_prompt)
            )

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

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )

        # Get model response
        try:
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
        if (
            self.agent_config.require_structured_finish_status
            and
            action.get("_metadata") == "finish"
            and not finish_message_has_status(action)
            and self._finish_status_repair_count < 2
        ):
            original_finish_action = action
            self._context[-1] = MessageBuilder.remove_images_from_message(
                self._context[-1]
            )
            repaired_message = self._repair_finish_status_isolated(
                str(original_finish_action.get("message") or ""),
            )
            action = finish(message=repaired_message)
            action_response = self._format_action_for_context(action)

        if self.agent_config.verbose:
            # Print thinking process
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "=" * 50)
            print(f"💭 {msgs['thinking']}:")
            print("-" * 50)
            print(response.thinking)
            print("-" * 50)
            print(f"🎯 {msgs['action']}:")
            print(json.dumps(action, ensure_ascii=False, indent=2))
            print("=" * 50 + "\n")

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

    def _repair_finish_status_isolated(
        self,
        original_message: str,
        step_text: str | None = None,
        target_state: str | None = None,
    ) -> str:
        """Repair finish STATUS using an isolated status-only model context."""
        fallback_action = coerce_finish_to_review(
            {"_metadata": "finish", "message": original_message}
        )
        fallback_message = str(fallback_action.get("message") or "")
        for attempt in range(1, 3):
            self._finish_status_repair_count += 1
            messages = [
                MessageBuilder.create_system_message(status_formatter_system_prompt()),
                MessageBuilder.create_user_message(
                    status_formatter_user_prompt(
                        original_message,
                        attempt=attempt,
                        max_attempts=2,
                        step_text=step_text,
                        target_state=target_state,
                    )
                ),
            ]
            try:
                response = self.model_client.request(messages)
                action = parse_action(response.action)
            except Exception:
                if self.agent_config.verbose:
                    traceback.print_exc()
                continue
            if action.get("_metadata") == "finish" and finish_message_has_status(action):
                return str(action.get("message") or fallback_message)
        return fallback_message

    @property
    def context(self) -> list[dict[str, Any]]:
        """Get the current conversation context."""
        return self._context.copy()

    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count

    def _finish_report_case(
        self, message: str, max_steps_reached: bool = False
    ) -> None:
        if self.agent_config.reporter and self.agent_config.reporter.current_case:
            if not self.agent_config.auto_manage_report_case:
                return
            status = self.agent_config.reporter.finish_case(
                message, max_steps_reached=max_steps_reached
            )
            print(f"\nResult: {status} - {message}")
            print(f"Artifacts: {self.agent_config.reporter.root_dir}\n")
