from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from phone_agent.adb.screenshot import get_screenshot
from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode


class FakeAdapter:
    payload = b""
    run_calls = 0
    state_calls = 0

    def run_bytes(self, serial, args, timeout):
        type(self).run_calls += 1
        assert serial == "SERIAL"
        assert args == ["exec-out", "screencap", "-p"]
        assert timeout == 10
        return SimpleNamespace(stdout=self.payload)

    def get_state(self, serial, timeout):
        type(self).state_calls += 1
        assert serial == "SERIAL"
        assert timeout == 5
        return "device"


def test_worker_screenshot_never_falls_back_to_black(monkeypatch):
    monkeypatch.setenv("AUTOGLM_WORKER_CHILD", "1")
    monkeypatch.setattr("phone_agent.adb.screenshot.AdbCommandAdapter", FakeAdapter)
    monkeypatch.setattr("phone_agent.adb.screenshot.time.sleep", lambda _delay: None)
    FakeAdapter.payload = b"not-an-image"
    FakeAdapter.run_calls = 0
    FakeAdapter.state_calls = 0

    with pytest.raises(ExecutionError, match="after 3 attempts"):
        get_screenshot("SERIAL")

    assert FakeAdapter.run_calls == 3
    assert FakeAdapter.state_calls == 3


def test_worker_screenshot_uses_explicit_serial(monkeypatch):
    monkeypatch.setenv("AUTOGLM_WORKER_CHILD", "1")
    monkeypatch.setattr("phone_agent.adb.screenshot.AdbCommandAdapter", FakeAdapter)
    buffer = BytesIO()
    Image.new("RGB", (12, 34), color="white").save(buffer, format="PNG")
    FakeAdapter.payload = buffer.getvalue()

    screenshot = get_screenshot("SERIAL")

    assert (screenshot.width, screenshot.height) == (12, 34)
    assert screenshot.is_sensitive is False


def test_worker_screenshot_retries_empty_result_after_online_probe(monkeypatch):
    monkeypatch.setenv("AUTOGLM_WORKER_CHILD", "1")
    monkeypatch.setattr("phone_agent.adb.screenshot.time.sleep", lambda _delay: None)
    buffer = BytesIO()
    Image.new("RGB", (20, 30), color="white").save(buffer, format="PNG")

    class RetryAdapter:
        attempts = 0
        state_calls = 0

        def run_bytes(self, _serial, _args, _timeout=None, **_kwargs):
            type(self).attempts += 1
            return SimpleNamespace(
                stdout=b"" if self.attempts == 1 else buffer.getvalue()
            )

        def get_state(self, _serial, timeout):
            assert timeout == 5
            type(self).state_calls += 1
            return "device"

    monkeypatch.setattr("phone_agent.adb.screenshot.AdbCommandAdapter", RetryAdapter)

    screenshot = get_screenshot("SERIAL")

    assert (screenshot.width, screenshot.height) == (20, 30)
    assert RetryAdapter.attempts == 2
    assert RetryAdapter.state_calls == 1


def test_worker_screenshot_stops_retry_when_device_probe_fails(monkeypatch):
    monkeypatch.setenv("AUTOGLM_WORKER_CHILD", "1")

    class OfflineAdapter:
        attempts = 0

        def run_bytes(self, _serial, _args, _timeout=None, **_kwargs):
            type(self).attempts += 1
            return SimpleNamespace(stdout=b"")

        def get_state(self, _serial, timeout):
            assert timeout == 5
            raise ExecutionError(
                ExecutionErrorCode.DEVICE_DISCONNECTED,
                "device offline",
                retryable=True,
            )

    monkeypatch.setattr("phone_agent.adb.screenshot.AdbCommandAdapter", OfflineAdapter)

    with pytest.raises(ExecutionError, match="device became unavailable") as caught:
        get_screenshot("SERIAL")

    assert caught.value.code is ExecutionErrorCode.DEVICE_DISCONNECTED
    assert OfflineAdapter.attempts == 1
