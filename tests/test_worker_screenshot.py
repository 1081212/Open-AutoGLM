from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from phone_agent.adb.screenshot import get_screenshot
from phone_agent.execution.errors import ExecutionError


class FakeAdapter:
    payload = b""

    def run_bytes(self, serial, args, timeout):
        assert serial == "SERIAL"
        assert args == ["exec-out", "screencap", "-p"]
        assert timeout == 10
        return SimpleNamespace(stdout=self.payload)


def test_worker_screenshot_never_falls_back_to_black(monkeypatch):
    monkeypatch.setenv("AUTOGLM_WORKER_CHILD", "1")
    monkeypatch.setattr("phone_agent.adb.screenshot.AdbCommandAdapter", FakeAdapter)
    FakeAdapter.payload = b"not-an-image"

    with pytest.raises(ExecutionError, match="not a valid image"):
        get_screenshot("SERIAL")


def test_worker_screenshot_uses_explicit_serial(monkeypatch):
    monkeypatch.setenv("AUTOGLM_WORKER_CHILD", "1")
    monkeypatch.setattr("phone_agent.adb.screenshot.AdbCommandAdapter", FakeAdapter)
    buffer = BytesIO()
    Image.new("RGB", (12, 34), color="white").save(buffer, format="PNG")
    FakeAdapter.payload = buffer.getvalue()

    screenshot = get_screenshot("SERIAL")

    assert (screenshot.width, screenshot.height) == (12, 34)
    assert screenshot.is_sensitive is False
