"""Typed ADB command adapter used by worker discovery and execution probes."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode


@dataclass(frozen=True, slots=True)
class AdbCommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class AdbBytesResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    duration_ms: int


class AdbCommandAdapter:
    def __init__(self, adb_path: str = "adb") -> None:
        self.adb_path = adb_path

    def run(
        self,
        serial: str,
        args: list[str] | tuple[str, ...],
        *,
        timeout: float = 15,
        check: bool = True,
    ) -> AdbCommandResult:
        serial = _normalize_serial(serial)
        argv = (self.adb_path, "-s", serial, *args)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(
                ExecutionErrorCode.DEVICE_COMMAND_TIMEOUT,
                f"ADB command timed out after {timeout:g}s",
                retryable=True,
            ) from exc
        duration_ms = round((time.monotonic() - started) * 1000)
        result = AdbCommandResult(
            argv=tuple(argv),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=duration_ms,
        )
        if check and completed.returncode != 0:
            raise _command_error(result)
        return result

    def get_state(self, serial: str, *, timeout: float = 5) -> str:
        result = self.run(serial, ["get-state"], timeout=timeout)
        state = result.stdout.strip()
        if state != "device":
            raise ExecutionError(
                ExecutionErrorCode.DEVICE_LOST,
                f"ADB device state is {state or 'unknown'}",
                retryable=False,
            )
        return state

    def run_bytes(
        self,
        serial: str,
        args: list[str] | tuple[str, ...],
        *,
        timeout: float = 15,
    ) -> AdbBytesResult:
        serial = _normalize_serial(serial)
        argv = (self.adb_path, "-s", serial, *args)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(
                ExecutionErrorCode.DEVICE_COMMAND_TIMEOUT,
                f"ADB command timed out after {timeout:g}s",
                retryable=True,
            ) from exc
        result = AdbBytesResult(
            argv=tuple(argv),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        if completed.returncode != 0:
            text_result = AdbCommandResult(
                result.argv,
                result.returncode,
                result.stdout.decode("utf-8", errors="replace"),
                result.stderr.decode("utf-8", errors="replace"),
                result.duration_ms,
            )
            raise _command_error(text_result)
        return result

    def list_devices(self, *, timeout: float = 5) -> list[tuple[str, str, dict[str, str]]]:
        argv = (self.adb_path, "devices", "-l")
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(
                ExecutionErrorCode.DEVICE_COMMAND_TIMEOUT,
                "adb devices timed out",
                retryable=True,
            ) from exc
        if completed.returncode != 0:
            result = AdbCommandResult(tuple(argv), completed.returncode, completed.stdout, completed.stderr, 0)
            raise _command_error(result)
        devices = []
        for line in completed.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            metadata = {}
            for token in parts[2:]:
                if ":" in token:
                    key, value = token.split(":", 1)
                    metadata[key] = value
            devices.append((_normalize_serial(parts[0]), parts[1], metadata))
        return devices


def _normalize_serial(serial: str) -> str:
    if "\x00" in serial:
        raise ValueError("ADB serial must not contain NUL")
    normalized = serial.strip(" \t\r\n\f\v")
    if not normalized:
        raise ValueError("ADB serial must not be empty")
    return normalized


def _command_error(result: AdbCommandResult) -> ExecutionError:
    detail = (result.stderr or result.stdout or "ADB command failed").strip()
    lowered = detail.lower()
    if any(marker in lowered for marker in ("offline", "unauthorized", "no devices", "device not found")):
        code = ExecutionErrorCode.DEVICE_DISCONNECTED
        retryable = True
    else:
        code = ExecutionErrorCode.ACTION_ERROR
        retryable = False
    return ExecutionError(code, detail, retryable=retryable)
