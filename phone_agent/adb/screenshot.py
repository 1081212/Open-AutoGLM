"""Screenshot utilities for capturing Android device screen."""

import base64
import logging
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from phone_agent.adb.command import AdbCommandAdapter
from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode

logger = logging.getLogger(__name__)
_WORKER_SCREENSHOT_ATTEMPTS = 3


@dataclass
class Screenshot:
    """Represents a captured screenshot."""

    base64_data: str
    width: int
    height: int
    is_sensitive: bool = False


def get_screenshot(device_id: str | None = None, timeout: int = 10) -> Screenshot:
    """
    Capture a screenshot from the connected Android device.

    Args:
        device_id: Optional ADB device ID for multi-device setups.
        timeout: Timeout in seconds for screenshot operations.

    Returns:
        Screenshot object containing base64 data and dimensions.

    Note:
        If the screenshot fails (e.g., on sensitive screens like payment pages),
        a black fallback image is returned with is_sensitive=True.
    """
    if os.getenv("AUTOGLM_WORKER_CHILD") == "1":
        return _get_strict_worker_screenshot(device_id, timeout)

    temp_path = os.path.join(tempfile.gettempdir(), f"screenshot_{uuid.uuid4()}.png")
    adb_prefix = _get_adb_prefix(device_id)

    try:
        # Execute screenshot command
        result = subprocess.run(
            adb_prefix + ["shell", "screencap", "-p", "/sdcard/tmp.png"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Check for screenshot failure (sensitive screen)
        output = result.stdout + result.stderr
        if "Status: -1" in output or "Failed" in output:
            return _create_fallback_screenshot(is_sensitive=True)

        # Pull screenshot to local temp path
        subprocess.run(
            adb_prefix + ["pull", "/sdcard/tmp.png", temp_path],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if not os.path.exists(temp_path):
            return _create_fallback_screenshot(is_sensitive=False)

        # Read and encode image
        img = Image.open(temp_path)
        width, height = img.size

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Cleanup
        os.remove(temp_path)

        return Screenshot(
            base64_data=base64_data, width=width, height=height, is_sensitive=False
        )

    except Exception as e:
        print(f"Screenshot error: {e}")
        return _create_fallback_screenshot(is_sensitive=False)


def _get_adb_prefix(device_id: str | None) -> list:
    """Get ADB command prefix with optional device specifier."""
    if device_id:
        return ["adb", "-s", device_id]
    return ["adb"]


def _create_fallback_screenshot(is_sensitive: bool) -> Screenshot:
    """Create a black fallback image when screenshot fails."""
    default_width, default_height = 1080, 2400

    black_img = Image.new("RGB", (default_width, default_height), color="black")
    buffered = BytesIO()
    black_img.save(buffered, format="PNG")
    base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return Screenshot(
        base64_data=base64_data,
        width=default_width,
        height=default_height,
        is_sensitive=is_sensitive,
    )


def _get_strict_worker_screenshot(device_id: str | None, timeout: int) -> Screenshot:
    if not device_id:
        raise ExecutionError(
            ExecutionErrorCode.DEVICE_NOT_FOUND,
            "Worker screenshot requires an explicitly selected ADB serial",
        )
    adapter = AdbCommandAdapter()
    last_error: ExecutionError | None = None
    for attempt in range(1, _WORKER_SCREENSHOT_ATTEMPTS + 1):
        try:
            result = adapter.run_bytes(
                device_id,
                ["exec-out", "screencap", "-p"],
                timeout=timeout,
            )
            if not result.stdout:
                raise ExecutionError(
                    ExecutionErrorCode.DEVICE_DISCONNECTED,
                    "ADB screenshot returned no data",
                    retryable=True,
                )
            return _decode_worker_screenshot(result.stdout)
        except ExecutionError as error:
            last_error = error
        try:
            adapter.get_state(device_id, timeout=min(5, timeout))
        except ExecutionError as state_error:
            raise ExecutionError(
                ExecutionErrorCode.DEVICE_DISCONNECTED,
                f"ADB device became unavailable after screenshot failure: "
                f"{state_error.message}",
                retryable=True,
            ) from state_error
        if attempt < _WORKER_SCREENSHOT_ATTEMPTS:
            logger.warning(
                "ADB screenshot failed while device remains online; retrying "
                "adb_serial=%s attempt=%d/%d error_code=%s",
                device_id,
                attempt,
                _WORKER_SCREENSHOT_ATTEMPTS,
                last_error.code.value,
            )
            time.sleep(0.25 * attempt)
    assert last_error is not None
    raise ExecutionError(
        last_error.code,
        f"{last_error.message} after {_WORKER_SCREENSHOT_ATTEMPTS} attempts "
        "while device remained online",
        retryable=True,
    ) from last_error


def _decode_worker_screenshot(payload: bytes) -> Screenshot:
    try:
        image = Image.open(BytesIO(payload))
        image.load()
        width, height = image.size
        buffered = BytesIO()
        image.save(buffered, format="PNG")
    except Exception as exc:
        raise ExecutionError(
            ExecutionErrorCode.ACTION_ERROR,
            "ADB screenshot data is not a valid image",
        ) from exc
    return Screenshot(
        base64_data=base64.b64encode(buffered.getvalue()).decode("ascii"),
        width=width,
        height=height,
        is_sensitive=False,
    )
