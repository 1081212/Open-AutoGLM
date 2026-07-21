#!/usr/bin/env python3
"""
Phone Agent Web Server - Expose phone agent as a web service.

Legacy local-only entry point. It is not the platform Worker.

Usage:
    python phoneagent_server.py

Environment Variables:
    PHONE_AGENT_BASE_URL: Model API base URL (default: http://localhost:8000/v1)
    PHONE_AGENT_MODEL: Model name (default: autoglm-phone-9b)
    PHONE_AGENT_API_KEY: API key for model authentication (default: EMPTY)
    PHONE_AGENT_MAX_STEPS: Maximum steps per task (default: 100)
    PHONE_AGENT_DEVICE_ID: ADB device ID for multi-device setups
    PHONE_AGENT_DEVICE_TYPE: Device type (adb/hdc/ios, default: adb)
    PHONE_AGENT_WDA_URL: WebDriverAgent URL for iOS (default: http://localhost:8100)
"""

import os
import sys
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
from uuid import uuid4

from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.agent_ios import IOSAgentConfig, IOSPhoneAgent
from phone_agent.device_factory import DeviceType, set_device_type
from phone_agent.model import ModelConfig
from phone_agent.reporting import TestRunReporter

# Import check functions from main
from main import check_system_requirements, check_model_api


# Request/Response models
class RunRequest(BaseModel):
    prompt: str


class RunResponse(BaseModel):
    result: str
    success: bool


# Initialize FastAPI app
app = FastAPI(
    title="Phone Agent API",
    description="AI-powered phone automation API",
    version="1.0.0"
)


# Global agent instance
_agent = None
_agent_lock: asyncio.Lock | None = None


def initialize_agent():
    """Initialize the phone agent based on environment variables."""
    global _agent

    # Get configuration from environment
    base_url = os.getenv("PHONE_AGENT_BASE_URL", "http://localhost:8000/v1")
    model_name = os.getenv("PHONE_AGENT_MODEL", "autoglm-phone-9b")
    api_key = os.getenv("PHONE_AGENT_API_KEY", "EMPTY")
    max_steps = int(os.getenv("PHONE_AGENT_MAX_STEPS", "100"))
    device_id = os.getenv("PHONE_AGENT_DEVICE_ID")
    device_type_str = os.getenv("PHONE_AGENT_DEVICE_TYPE", "adb")
    lang = os.getenv("PHONE_AGENT_LANG", "cn")
    wda_url = os.getenv("PHONE_AGENT_WDA_URL", "http://localhost:8100")

    # Determine device type
    if device_type_str == "adb":
        device_type = DeviceType.ADB
    elif device_type_str == "hdc":
        device_type = DeviceType.HDC
    else:
        device_type = DeviceType.IOS

    # Set device type globally for non-iOS devices
    if device_type != DeviceType.IOS:
        set_device_type(device_type)

    # Enable HDC verbose mode if using HDC
    if device_type == DeviceType.HDC:
        from phone_agent.hdc import set_hdc_verbose
        set_hdc_verbose(True)

    print("=" * 50)
    print("Phone Agent Server - Initializing")
    print("=" * 50)

    # Check system requirements (device connection, tools, etc.)
    print("\n[1/3] Checking system requirements...")
    if not check_system_requirements(device_type, wda_url):
        print("\n❌ System requirements check failed!")
        sys.exit(1)
    print("✅ System requirements check passed")

    # Check model API connection
    print("\n[2/3] Checking model API connection...")
    if not check_model_api(base_url, model_name, api_key):
        print("\n❌ Model API check failed!")
        sys.exit(1)
    print("✅ Model API check passed")

    # Create model config
    model_config = ModelConfig(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        lang=lang,
    )

    # Create agent based on device type
    print("\n[3/3] Creating agent instance...")
    if device_type == DeviceType.IOS:
        agent_config = IOSAgentConfig(
            max_steps=max_steps,
            wda_url=wda_url,
            device_id=device_id,
            verbose=True,
            lang=lang,
        )
        _agent = IOSPhoneAgent(
            model_config=model_config,
            agent_config=agent_config,
        )
    else:
        agent_config = AgentConfig(
            max_steps=max_steps,
            device_id=device_id,
            verbose=True,
            lang=lang,
        )
        _agent = PhoneAgent(
            model_config=model_config,
            agent_config=agent_config,
        )

    print("\n" + "=" * 50)
    print("✅ Phone Agent Server - Ready")
    print("=" * 50)
    print(f"Model: {model_config.model_name}")
    print(f"Base URL: {model_config.base_url}")
    print(f"Max Steps: {agent_config.max_steps}")
    print(f"Language: {agent_config.lang}")
    print(f"Device Type: {device_type_str.upper()}")
    if device_type == DeviceType.IOS:
        print(f"WDA URL: {wda_url}")
    print("=" * 50 + "\n")


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup."""
    global _agent_lock
    _agent_lock = asyncio.Lock()
    initialize_agent()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Phone Agent API",
        "status": "running",
        "version": "1.0.0",
        "legacy": True,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_initialized": _agent is not None
    }


@app.post("/run", response_model=RunResponse)
async def run_task(request: RunRequest):
    """
    Run a phone agent task (non-streaming).

    Args:
        request: RunRequest with prompt field

    Returns:
        RunResponse with result and success status
    """
    if _agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    lock = _agent_lock
    if lock is None:
        raise HTTPException(status_code=503, detail="Agent lock not initialized")
    async with lock:
        agent = _agent
        reporter = _new_legacy_reporter(agent)
        original_auto_manage = agent.agent_config.auto_manage_report_case
        agent.agent_config.reporter = reporter
        agent.agent_config.auto_manage_report_case = True
        try:
            print(f"\n>>> Received task: {request.prompt}")
            result = await asyncio.to_thread(agent.run, request.prompt)
            print(f">>> Task completed: {result}\n")
            return RunResponse(result=result, success=True)
        except Exception as e:
            error_msg = str(e)
            print(f">>> Task failed: {error_msg}\n")
            return RunResponse(result=f"Error: {error_msg}", success=False)
        finally:
            reporter.finish_run()
            agent.agent_config.reporter = None
            agent.agent_config.auto_manage_report_case = original_auto_manage
            agent.reset()


@app.post("/run/stream")
async def run_task_stream(request: RunRequest):
    """
    Run a phone agent task with streaming progress updates (SSE).

    Args:
        request: RunRequest with prompt field

    Returns:
        Server-Sent Events stream with progress updates
    """
    if _agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    def sse_event(event: dict) -> str:
        """Format one Server-Sent Event payload."""
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    async def event_generator():
        """Generate SSE events for each agent step."""
        lock = _agent_lock
        if lock is None:
            yield sse_event({
                "type": "error",
                "message": "Agent lock not initialized",
                "success": False,
            })
            return

        if lock.locked():
            yield sse_event({
                "type": "queued",
                "message": "Another task is running. Waiting for the agent to become available.",
            })

        async with lock:
            agent = _agent
            if agent is None:
                yield sse_event({
                    "type": "error",
                    "message": "Agent not initialized",
                    "success": False,
                })
                return

            agent.reset()
            reporter = _new_legacy_reporter(agent)
            original_auto_manage = agent.agent_config.auto_manage_report_case
            agent.agent_config.reporter = reporter
            agent.agent_config.auto_manage_report_case = False
            reporter.start_case(request.prompt, 1)

            step_no = 0

            yield sse_event({
                "type": "start",
                "prompt": request.prompt,
            })

            try:
                while step_no < agent.agent_config.max_steps:
                    step_no += 1
                    yield sse_event({
                        "type": "step_start",
                        "step": step_no,
                    })

                    result = await asyncio.to_thread(
                        agent.step,
                        request.prompt if step_no == 1 else None,
                    )

                    yield sse_event({
                        "type": "step",
                        "step": step_no,
                        "thinking": result.thinking,
                        "action": result.action,
                        "message": result.message,
                        "success": result.success,
                        "finished": result.finished,
                    })

                    if result.finished:
                        reporter.finish_case(result.message or "Task completed")
                        yield sse_event({
                            "type": "done",
                            "result": result.message or "Task completed",
                            "success": result.success,
                            "steps": step_no,
                        })
                        return

                yield sse_event({
                    "type": "done",
                    "result": "Max steps reached",
                    "success": False,
                    "steps": step_no,
                })
                if reporter.current_case:
                    reporter.finish_case("Max steps reached", max_steps_reached=True)
            except Exception as e:
                error_msg = str(e)
                print(f">>> Task failed: {error_msg}\n")
                yield sse_event({
                    "type": "error",
                    "message": error_msg,
                    "success": False,
                    "steps": step_no,
                })
            finally:
                if reporter.current_case:
                    reporter.finish_case("STATUS: REVIEW\nREASON: Legacy stream interrupted")
                reporter.finish_run()
                agent.agent_config.reporter = None
                agent.agent_config.auto_manage_report_case = original_auto_manage
                agent.reset()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


def _new_legacy_reporter(agent) -> TestRunReporter:
    return TestRunReporter(
        artifact_name=f"legacy-server-{uuid4()}",
        base_dir=os.getenv("PHONE_AGENT_ARTIFACT_DIR", "test_artifacts"),
        device_type=os.getenv("PHONE_AGENT_DEVICE_TYPE", "adb"),
        device_id=agent.agent_config.device_id,
        model_name=agent.model_config.model_name,
        base_url=agent.model_config.base_url,
        wda_url=getattr(agent.agent_config, "wda_url", None),
    )


def main():
    """Start the web server."""
    port = int(os.getenv("PHONE_AGENT_PORT", "8000"))
    host = os.getenv("PHONE_AGENT_HOST", "0.0.0.0")

    print(f"\nStarting Phone Agent Server on {host}:{port}")
    print(f"API docs available at: http://{host}:{port}/docs\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
