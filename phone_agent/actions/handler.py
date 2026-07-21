"""Action handler for processing AI model outputs."""

import ast
import os
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from phone_agent.config.timing import TIMING_CONFIG
from phone_agent.device_factory import get_device_factory
from phone_agent.execution.cancellation import CancellationToken
from phone_agent.execution.errors import ExecutionError


@dataclass
class ActionResult:
    """Result of an action execution."""

    success: bool
    should_finish: bool
    message: str | None = None
    requires_confirmation: bool = False


class ActionHandler:
    """
    Handles execution of actions from AI model output.

    Args:
        device_id: Optional ADB device ID for multi-device setups.
        confirmation_callback: Optional callback for sensitive action confirmation.
            Should return True to proceed, False to cancel.
        takeover_callback: Optional callback for takeover requests (login, captcha).
    """

    def __init__(
        self,
        device_id: str | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
        cancellation_token: CancellationToken | None = None,
    ):
        self.device_id = device_id
        self.confirmation_callback = confirmation_callback or self._default_confirmation
        self.takeover_callback = takeover_callback or self._default_takeover
        self.cancellation_token = cancellation_token or CancellationToken()

    def execute(
        self, action: dict[str, Any], screen_width: int, screen_height: int
    ) -> ActionResult:
        """
        Execute an action from the AI model.

        Args:
            action: The action dictionary from the model.
            screen_width: Current screen width in pixels.
            screen_height: Current screen height in pixels.

        Returns:
            ActionResult indicating success and whether to finish.
        """
        self.cancellation_token.raise_if_cancelled()
        action_type = action.get("_metadata")

        if action_type == "finish":
            return ActionResult(
                success=True, should_finish=True, message=action.get("message")
            )

        if action_type != "do":
            return ActionResult(
                success=False,
                should_finish=True,
                message=f"Unknown action type: {action_type}",
            )

        action_name = action.get("action")
        handler_method = self._get_handler(action_name)

        if handler_method is None:
            return ActionResult(
                success=False,
                should_finish=False,
                message=f"Unknown action: {action_name}",
            )

        try:
            return handler_method(action, screen_width, screen_height)
        except ExecutionError:
            raise
        except Exception as e:
            return ActionResult(
                success=False, should_finish=False, message=f"Action failed: {e}"
            )

    def _get_handler(self, action_name: str) -> Callable | None:
        """Get the handler method for an action."""
        handlers = {
            "Launch": self._handle_launch,
            "Tap": self._handle_tap,
            "TapElement": self._handle_tap_element,
            "TypeIntoElement": self._handle_type_into_element,
            "PlayAudioWhileHoldingElement": self._handle_play_audio_while_holding_element,
            "Type": self._handle_type,
            "Type_Name": self._handle_type,
            "Swipe": self._handle_swipe,
            "Back": self._handle_back,
            "Home": self._handle_home,
            "Double Tap": self._handle_double_tap,
            "Long Press": self._handle_long_press,
            "Wait": self._handle_wait,
            "Take_over": self._handle_takeover,
            "Note": self._handle_note,
            "Call_API": self._handle_call_api,
            "Interact": self._handle_interact,
        }
        return handlers.get(action_name)

    def _convert_relative_to_absolute(
        self, element: list[int], screen_width: int, screen_height: int
    ) -> tuple[int, int]:
        """Convert relative coordinates (0-1000) to absolute pixels."""
        x = int(element[0] / 1000 * screen_width)
        y = int(element[1] / 1000 * screen_height)
        return x, y

    def _handle_launch(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle app launch action."""
        app_name = action.get("app")
        if not app_name:
            return ActionResult(False, False, "No app name specified")

        device_factory = get_device_factory()
        success = device_factory.launch_app(app_name, self.device_id)
        if success:
            return ActionResult(True, False)
        return ActionResult(False, False, f"App not found: {app_name}")

    def _handle_tap(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle tap action."""
        element = action.get("element")
        if not element:
            return ActionResult(False, False, "No element coordinates")

        x, y = self._convert_relative_to_absolute(element, width, height)

        # Check for sensitive operation
        if "message" in action:
            if not self.confirmation_callback(action["message"]):
                return ActionResult(
                    success=False,
                    should_finish=True,
                    message="User cancelled sensitive operation",
                )

        device_factory = get_device_factory()
        device_factory.tap(x, y, self.device_id)
        return ActionResult(True, False)

    def _handle_type(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle text input action."""
        text = action.get("text", "")

        device_factory = get_device_factory()

        # Switch to ADB keyboard
        original_ime = device_factory.detect_and_set_adb_keyboard(self.device_id)
        self._interruptible_wait(TIMING_CONFIG.action.keyboard_switch_delay)

        # Clear existing text and type new text
        device_factory.clear_text(self.device_id)
        self._interruptible_wait(TIMING_CONFIG.action.text_clear_delay)

        # Handle multiline text by splitting on newlines
        device_factory.type_text(text, self.device_id)
        self._interruptible_wait(TIMING_CONFIG.action.text_input_delay)

        # Restore original keyboard
        device_factory.restore_keyboard(original_ime, self.device_id)
        self._interruptible_wait(TIMING_CONFIG.action.keyboard_restore_delay)

        return ActionResult(True, False)

    def _handle_swipe(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle swipe action."""
        start = action.get("start")
        end = action.get("end")

        if not start or not end:
            return ActionResult(False, False, "Missing swipe coordinates")

        start_x, start_y = self._convert_relative_to_absolute(start, width, height)
        end_x, end_y = self._convert_relative_to_absolute(end, width, height)

        device_factory = get_device_factory()
        device_factory.swipe(start_x, start_y, end_x, end_y, device_id=self.device_id)
        return ActionResult(True, False)

    def _handle_back(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle back button action."""
        device_factory = get_device_factory()
        device_factory.back(self.device_id)
        return ActionResult(True, False)

    def _handle_home(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle home button action."""
        device_factory = get_device_factory()
        device_factory.home(self.device_id)
        return ActionResult(True, False)

    def _handle_double_tap(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle double tap action."""
        element = action.get("element")
        if not element:
            return ActionResult(False, False, "No element coordinates")

        x, y = self._convert_relative_to_absolute(element, width, height)
        device_factory = get_device_factory()
        device_factory.double_tap(x, y, self.device_id)
        return ActionResult(True, False)

    def _handle_long_press(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle long press action."""
        element = action.get("element")
        if not element:
            return ActionResult(False, False, "No element coordinates")

        x, y = self._convert_relative_to_absolute(element, width, height)
        device_factory = get_device_factory()
        device_factory.long_press(x, y, device_id=self.device_id)
        return ActionResult(True, False)

    def _handle_wait(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle wait action."""
        duration_str = action.get("duration", "1 seconds")
        try:
            duration = float(duration_str.replace("seconds", "").strip())
        except ValueError:
            duration = 1.0

        if self.cancellation_token.wait(duration):
            self.cancellation_token.raise_if_cancelled()
        return ActionResult(True, False)

    def _interruptible_wait(self, seconds: float) -> None:
        if self.cancellation_token.wait(seconds):
            self.cancellation_token.raise_if_cancelled()

    def _handle_takeover(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle takeover request (login, captcha, etc.)."""
        message = action.get("message", "User intervention required")
        self.takeover_callback(message)
        return ActionResult(True, False, message=f"Take_over auto-skipped: {message}")

    def _handle_note(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle note action for recording a test issue and continuing."""
        message = action.get("message", "Issue noted")
        return ActionResult(True, False, message=message)

    def _handle_call_api(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle API call action (placeholder for summarization)."""
        # This action is typically used for content summarization
        # Implementation depends on specific requirements
        return ActionResult(True, False)

    def _handle_interact(self, action: dict, width: int, height: int) -> ActionResult:
        """Handle interaction request (user choice needed)."""
        # This action signals that user input is needed
        return ActionResult(True, False, message="User interaction required")

    def _build_u2_selector(self, action: dict):
        """根据 action 中的选择器字段构建 UIAutomator2 元素查询对象。

        这是 rule 专用的元素级操作能力。普通模型动作仍然走坐标 Tap/Type，
        只有使用 tap_element/type_into_element 时才会懒加载 uiautomator2。
        """
        try:
            from phone_agent.adb.u2_connection import get_u2_device
        except ImportError as exc:
            raise RuntimeError(
                "tap_element/type_into_element 需要安装可选依赖 uiautomator2"
            ) from exc

        device = get_u2_device(self.device_id)
        selector: dict[str, Any] = {}
        key_map = {
            "resource_id": "resourceId",
            "content_desc": "description",
            "class_name": "className",
        }
        for src, dst in key_map.items():
            if action.get(src) is not None:
                selector[dst] = action[src]
        for key in (
            "text",
            "textContains",
            "textStartsWith",
            "resourceId",
            "description",
            "className",
        ):
            if action.get(key) is not None:
                selector[key] = action[key]

        if not selector:
            raise ValueError(
                "tap_element/type_into_element 至少需要一个选择器："
                "text、resource_id、content_desc 或 class_name"
            )
        return device(**selector)

    def _handle_tap_element(
        self, action: dict, width: int, height: int
    ) -> ActionResult:
        """按 UI 元素选择器点击控件，而不是按坐标点击。"""
        timeout = float(action.get("timeout", 5))
        element = self._build_u2_selector(action)
        if not element.wait(timeout=timeout):
            return ActionResult(
                False, False, f"Element not found: {action} (timeout={timeout}s)"
            )
        element.click()
        return ActionResult(True, False)

    def _handle_type_into_element(
        self, action: dict, width: int, height: int
    ) -> ActionResult:
        """按 UI 元素选择器找到输入框，清空后写入文本。"""
        input_text = action.get("input_text", "")
        timeout = float(action.get("timeout", 5))
        clear = action.get("clear", True)
        selector_action = {
            key: value
            for key, value in action.items()
            if key
            not in {
                "_metadata",
                "action",
                "input_text",
                "timeout",
                "clear",
            }
        }
        element = self._build_u2_selector(selector_action)
        if not element.wait(timeout=timeout):
            return ActionResult(
                False,
                False,
                f"Element not found: {selector_action} (timeout={timeout}s)",
            )
        if clear:
            element.clear_text()
        element.set_text(input_text)
        return ActionResult(True, False)

    def _resolve_audio_path(self, audio_path: str) -> Path:
        """解析音频路径；相对路径默认相对项目根目录。"""
        path = Path(audio_path)
        if path.is_absolute():
            return path
        project_root = Path(__file__).resolve().parents[2]
        return project_root / path

    def _get_wav_duration_seconds(self, path: Path) -> float:
        """读取 wav 时长；读取失败时使用保守默认值。"""
        try:
            with wave.open(str(path), "rb") as wav_file:
                return wav_file.getnframes() / float(wav_file.getframerate())
        except Exception:
            return 6.0

    def _handle_play_audio_while_holding_element(
        self, action: dict, width: int, height: int
    ) -> ActionResult:
        """按住手机端元素，同时在电脑端播放音频。

        这是音频类 CustomRule 使用的动作。参数全部写在 action dict 中，
        便于后续从 YAML 或其它结构化配置生成。
        """
        audio_path = self._resolve_audio_path(action.get("audio_path", "xiaoxiao.wav"))
        if not audio_path.exists():
            return ActionResult(False, False, f"Audio file not found: {audio_path}")

        timeout = float(action.get("timeout", 5))
        press_lead_seconds = float(action.get("press_lead_seconds", 0.3))
        min_hold_seconds = float(action.get("min_hold_seconds", 3.0))
        player = action.get("player", "afplay")

        selector_action = {
            key: value
            for key, value in action.items()
            if key
            not in {
                "_metadata",
                "action",
                "audio_path",
                "timeout",
                "press_lead_seconds",
                "min_hold_seconds",
                "player",
                "hold_seconds",
            }
        }
        element = self._build_u2_selector(selector_action)
        if not element.wait(timeout=timeout):
            return ActionResult(
                False,
                False,
                f"Element not found: {selector_action} (timeout={timeout}s)",
            )

        bounds = element.info.get("bounds") or {}
        if not {"left", "right", "top", "bottom"}.issubset(bounds):
            return ActionResult(False, False, "Element bounds are incomplete")

        hold_seconds = float(
            action.get(
                "hold_seconds",
                max(self._get_wav_duration_seconds(audio_path) + 1.0, min_hold_seconds),
            )
        )

        def hold_button() -> None:
            element.long_click(duration=hold_seconds)

        hold_thread = threading.Thread(target=hold_button, daemon=True)
        hold_thread.start()
        self._interruptible_wait(press_lead_seconds)

        result = self._run_process_cancellable(
            [player, str(audio_path)],
            timeout=max(hold_seconds + 5, 10),
        )
        hold_thread.join(timeout=hold_seconds + 2)
        if result.returncode != 0:
            return ActionResult(
                False,
                False,
                f"Audio player failed: {(result.stderr or result.stdout).strip()}",
            )
        return ActionResult(True, False)

    def _send_keyevent(self, keycode: str) -> None:
        """Send a keyevent to the device."""
        from phone_agent.device_factory import DeviceType, get_device_factory
        from phone_agent.hdc.connection import _run_hdc_command

        device_factory = get_device_factory()

        # Handle HDC devices with HarmonyOS-specific keyEvent command
        if device_factory.device_type == DeviceType.HDC:
            hdc_prefix = ["hdc", "-t", self.device_id] if self.device_id else ["hdc"]
            
            # Map common keycodes to HarmonyOS keyEvent codes
            # KEYCODE_ENTER (66) -> 2054 (HarmonyOS Enter key code)
            if keycode == "KEYCODE_ENTER" or keycode == "66":
                _run_hdc_command(
                    hdc_prefix + ["shell", "uitest", "uiInput", "keyEvent", "2054"],
                    capture_output=True,
                    text=True,
                )
            else:
                # For other keys, try to use the numeric code directly
                # If keycode is a string like "KEYCODE_ENTER", convert it
                try:
                    # Try to extract numeric code from string or use as-is
                    if keycode.startswith("KEYCODE_"):
                        # For now, only handle ENTER, other keys may need mapping
                        if "ENTER" in keycode:
                            _run_hdc_command(
                                hdc_prefix + ["shell", "uitest", "uiInput", "keyEvent", "2054"],
                                capture_output=True,
                                text=True,
                            )
                        else:
                            # Fallback to ADB-style command for unsupported keys
                            subprocess.run(
                                hdc_prefix + ["shell", "input", "keyevent", keycode],
                                capture_output=True,
                                text=True,
                                timeout=15,
                            )
                    else:
                        # Assume it's a numeric code
                        _run_hdc_command(
                            hdc_prefix + ["shell", "uitest", "uiInput", "keyEvent", str(keycode)],
                            capture_output=True,
                            text=True,
                        )
                except Exception:
                    # Fallback to ADB-style command
                    subprocess.run(
                        hdc_prefix + ["shell", "input", "keyevent", keycode],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
        else:
            # ADB devices use standard input keyevent command
            if os.getenv("AUTOGLM_WORKER_CHILD") == "1":
                if not self.device_id:
                    raise ValueError("Worker keyevent requires an explicit ADB serial")
                from phone_agent.adb.command import AdbCommandAdapter

                AdbCommandAdapter().run(
                    self.device_id,
                    ["shell", "input", "keyevent", keycode],
                    timeout=15,
                )
            else:
                cmd_prefix = ["adb", "-s", self.device_id] if self.device_id else ["adb"]
                subprocess.run(
                    cmd_prefix + ["shell", "input", "keyevent", keycode],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )

    def _run_process_cancellable(self, argv: list[str], *, timeout: float) -> subprocess.CompletedProcess:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if self.cancellation_token.wait(0.1):
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                self.cancellation_token.raise_if_cancelled()
            if time.monotonic() >= deadline:
                process.kill()
                stdout, stderr = process.communicate()
                raise TimeoutError(f"Process timed out: {argv[0]}: {stderr or stdout}")
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)

    @staticmethod
    def _default_confirmation(message: str) -> bool:
        """Default confirmation callback using console input."""
        response = input(f"Sensitive operation: {message}\nConfirm? (Y/N): ")
        return response.upper() == "Y"

    @staticmethod
    def _default_takeover(message: str) -> None:
        """Default takeover callback for unattended test runs."""
        print(f"Take_over requested, auto-skipping manual intervention: {message}")


def parse_action(response: str) -> dict[str, Any]:
    """
    Parse action from model response.

    Args:
        response: Raw response string from the model.

    Returns:
        Parsed action dictionary.

    Raises:
        ValueError: If the response cannot be parsed.
    """
    print(f"Parsing action: {response}")
    try:
        response = response.strip()
        if response.startswith('do(action="Type"') or response.startswith(
            'do(action="Type_Name"'
        ):
            text = response.split("text=", 1)[1][1:-2]
            action = {"_metadata": "do", "action": "Type", "text": text}
            return action
        elif response.startswith("do"):
            # Use AST parsing instead of eval for safety
            try:
                # Escape special characters (newlines, tabs, etc.) for valid Python syntax
                response = response.replace('\n', '\\n')
                response = response.replace('\r', '\\r')
                response = response.replace('\t', '\\t')

                tree = ast.parse(response, mode="eval")
                if not isinstance(tree.body, ast.Call):
                    raise ValueError("Expected a function call")

                call = tree.body
                # Extract keyword arguments safely
                action = {"_metadata": "do"}
                for keyword in call.keywords:
                    key = keyword.arg
                    value = ast.literal_eval(keyword.value)
                    action[key] = value

                if "action" in action:
                    action["action"] = normalize_action_name(action["action"])

                return action
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Failed to parse do() action: {e}")

        elif response.startswith("finish"):
            action = {
                "_metadata": "finish",
                "message": response.replace("finish(message=", "")[1:-2],
            }
        else:
            raise ValueError(f"Failed to parse action: {response}")
        return action
    except Exception as e:
        raise ValueError(f"Failed to parse action: {e}")


def do(**kwargs) -> dict[str, Any]:
    """Helper function for creating 'do' actions."""
    kwargs["_metadata"] = "do"
    if "action" in kwargs:
        kwargs["action"] = normalize_action_name(kwargs["action"])
    return kwargs


def finish(**kwargs) -> dict[str, Any]:
    """Helper function for creating 'finish' actions."""
    kwargs["_metadata"] = "finish"
    return kwargs


def tap_element(**kwargs) -> dict[str, Any]:
    """构造按元素选择器点击的 action。

    支持的选择器字段：
    - text：按控件显示文字精确匹配
    - resource_id：按 Android resource-id 匹配
    - content_desc：按无障碍描述匹配
    - class_name：按控件 className 匹配
    - textContains/textStartsWith：按文字包含/前缀匹配
    """
    kwargs["_metadata"] = "do"
    kwargs["action"] = "TapElement"
    return kwargs


def type_into_element(input_text: str, **kwargs) -> dict[str, Any]:
    """构造按元素选择器输入文本的 action。

    input_text 是要输入的文本；其它参数作为选择器或控制项使用。
    默认会先清空控件文本，可传 clear=False 关闭。
    """
    kwargs["_metadata"] = "do"
    kwargs["action"] = "TypeIntoElement"
    kwargs["input_text"] = input_text
    return kwargs


def play_audio_while_holding_element(audio_path: str, **kwargs) -> dict[str, Any]:
    """构造“按住手机元素并播放电脑音频”的 action。

    示例：
        play_audio_while_holding_element(
            audio_path="xiaoxiao.wav",
            textContains="按住",
            timeout=5,
        )
    """
    kwargs["_metadata"] = "do"
    kwargs["action"] = "PlayAudioWhileHoldingElement"
    kwargs["audio_path"] = audio_path
    return kwargs


def normalize_action_name(action_name: Any) -> Any:
    """Normalize common model variants to executable action names."""
    if not isinstance(action_name, str):
        return action_name

    aliases = {
        "note": "Note",
        "takeover": "Take_over",
        "take_over": "Take_over",
        "take over": "Take_over",
        "tack_over": "Take_over",
        "tackover": "Take_over",
        "tap_element": "TapElement",
        "tapelement": "TapElement",
        "type_into_element": "TypeIntoElement",
        "typeintoelement": "TypeIntoElement",
        "play_audio_while_holding_element": "PlayAudioWhileHoldingElement",
        "playaudiowhileholdingelement": "PlayAudioWhileHoldingElement",
    }
    return aliases.get(action_name.strip().lower(), action_name)
