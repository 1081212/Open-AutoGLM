"""Model client for AI inference using OpenAI-compatible API."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI

@dataclass
class ModelConfig:
    """Configuration for the AI model."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = "autoglm-phone-9b"
    max_tokens: int = 3000
    temperature: float = 0.0
    top_p: float = 0.85
    frequency_penalty: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)
    lang: str = "cn"  # Language for UI messages: 'cn' or 'en'
    timeout_seconds: float = 180.0


@dataclass
class ModelResponse:
    """Response from the AI model."""

    thinking: str
    action: str
    raw_content: str
    # Performance metrics
    time_to_first_token: float | None = None  # Time to first token (seconds)
    time_to_thinking_end: float | None = None  # Time to thinking end (seconds)
    total_time: float | None = None  # Total inference time (seconds)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0


class ModelClient:
    """
    Client for interacting with OpenAI-compatible vision-language models.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
        )
        self._active_stream: Any | None = None

    def request(
        self,
        messages: list[dict[str, Any]],
        cancellation_check: Callable[[], None] | None = None,
    ) -> ModelResponse:
        """
        Send a request to the model.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            ModelResponse containing thinking and action.

        Raises:
            ValueError: If the response cannot be parsed.
        """
        # Start timing
        start_time = time.time()
        time_to_first_token = None
        time_to_thinking_end = None

        stream = self.client.chat.completions.create(
            messages=messages,
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            extra_body=self.config.extra_body,
            stream_options={"include_usage": True},
            stream=True,
        )
        self._active_stream = stream

        raw_content = ""
        usage = None
        buffer = ""  # Buffer to hold content that might be part of a marker
        action_markers = ["finish(message=", "do(action="]
        in_action_phase = False  # Track if we've entered the action phase
        first_token_received = False

        try:
            for chunk in stream:
                if cancellation_check:
                    cancellation_check()
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                if len(chunk.choices) == 0:
                    continue
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    raw_content += content

                    # Record time to first token
                    if not first_token_received:
                        time_to_first_token = time.time() - start_time
                        first_token_received = True

                    if in_action_phase:
                        continue

                    buffer += content

                    marker_found = False
                    for marker in action_markers:
                        if marker in buffer:
                            # Thinking remains in the response/context, not stdout.
                            in_action_phase = True
                            marker_found = True

                            if time_to_thinking_end is None:
                                time_to_thinking_end = time.time() - start_time

                            break

                    if marker_found:
                        continue

                    is_potential_marker = False
                    for marker in action_markers:
                        for i in range(1, len(marker)):
                            if buffer.endswith(marker[:i]):
                                is_potential_marker = True
                                break
                        if is_potential_marker:
                            break

                    if not is_potential_marker:
                        buffer = ""
        finally:
            self._active_stream = None

        # Calculate total time
        total_time = time.time() - start_time

        # Parse thinking and action from response
        thinking, action = self._parse_response(raw_content)

        # Print performance metrics
        # Performance metrics are still returned on ModelResponse. Keeping them
        # out of stdout prevents them from polluting functional test reports.
        # print()
        # print("=" * 50)
        # print(f"⏱️  {get_message('performance_metrics', lang)}:")
        # print("-" * 50)
        # if time_to_first_token is not None:
        #     print(
        #         f"{get_message('time_to_first_token', lang)}: {time_to_first_token:.3f}s"
        #     )
        # if time_to_thinking_end is not None:
        #     print(
        #         f"{get_message('time_to_thinking_end', lang)}:        {time_to_thinking_end:.3f}s"
        #     )
        # print(
        #     f"{get_message('total_inference_time', lang)}:          {total_time:.3f}s"
        # )
        # print("=" * 50)

        return ModelResponse(
            thinking=thinking,
            action=action,
            raw_content=raw_content,
            time_to_first_token=time_to_first_token,
            time_to_thinking_end=time_to_thinking_end,
            total_time=total_time,
            prompt_tokens=_usage_value(usage, "prompt_tokens"),
            completion_tokens=_usage_value(usage, "completion_tokens"),
            cached_tokens=_cached_tokens(usage),
            total_tokens=_usage_value(usage, "total_tokens"),
        )

    def close_active_stream(self) -> None:
        stream = self._active_stream
        if stream is not None and hasattr(stream, "close"):
            stream.close()

    def _parse_response(self, content: str) -> tuple[str, str]:
        """
        Parse the model response into thinking and action parts.

        Parsing rules:
        1. If content contains 'finish(message=', everything before is thinking,
           everything from 'finish(message=' onwards is action.
        2. If rule 1 doesn't apply but content contains 'do(action=',
           everything before is thinking, everything from 'do(action=' onwards is action.
        3. Fallback: If content contains '<answer>', use legacy parsing with XML tags.
        4. Otherwise, return empty thinking and full content as action.

        Args:
            content: Raw response content.

        Returns:
            Tuple of (thinking, action).
        """
        # Rule 1: Check for finish(message=
        if "finish(message=" in content:
            parts = content.split("finish(message=", 1)
            thinking = parts[0].strip()
            action = "finish(message=" + parts[1]
            return thinking, action

        # Rule 2: Check for do(action=
        if "do(action=" in content:
            parts = content.split("do(action=", 1)
            thinking = parts[0].strip()
            action = "do(action=" + parts[1]
            return thinking, action

        # Rule 3: Fallback to legacy XML tag parsing
        if "<answer>" in content:
            parts = content.split("<answer>", 1)
            thinking = parts[0].replace("<think>", "").replace("</think>", "").strip()
            action = parts[1].replace("</answer>", "").strip()
            return thinking, action

        # Rule 4: No markers found, return content as action
        return "", content


def _usage_value(usage: Any, field: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(field) or 0)
    return int(getattr(usage, field, 0) or 0)


def _cached_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    details = (
        usage.get("prompt_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "prompt_tokens_details", None)
    )
    if not details:
        return 0
    if isinstance(details, dict):
        return int(details.get("cached_tokens") or 0)
    return int(getattr(details, "cached_tokens", 0) or 0)


class MessageBuilder:
    """Helper class for building conversation messages."""

    @staticmethod
    def create_system_message(content: str) -> dict[str, Any]:
        """Create a system message."""
        return {"role": "system", "content": content}

    @staticmethod
    def create_user_message(
        text: str, image_base64: str | None = None
    ) -> dict[str, Any]:
        """
        Create a user message with optional image.

        Args:
            text: Text content.
            image_base64: Optional base64-encoded image.

        Returns:
            Message dictionary.
        """
        content = []

        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )

        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    @staticmethod
    def create_assistant_message(content: str) -> dict[str, Any]:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}

    @staticmethod
    def remove_images_from_message(message: dict[str, Any]) -> dict[str, Any]:
        """
        Remove image content from a message to save context space.

        Args:
            message: Message dictionary.

        Returns:
            Message with images removed.
        """
        if isinstance(message.get("content"), list):
            message["content"] = [
                item for item in message["content"] if item.get("type") == "text"
            ]
        return message

    @staticmethod
    def build_screen_info(current_app: str, **extra_info) -> str:
        """
        Build screen info string for the model.

        Args:
            current_app: Current app name.
            **extra_info: Additional info to include.

        Returns:
            JSON string with screen info.
        """
        info = {"current_app": current_app, **extra_info}
        return json.dumps(info, ensure_ascii=False)
