"""Command-line runtime for the single-capacity Open-AutoGLM Worker."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

from phone_agent.worker.api_client import WorkerApiClient
from phone_agent.worker.artifact_uploader import ArtifactOutboxPump
from phone_agent.worker.child_process import ChildProcessPlanExecutor
from phone_agent.worker.config import WorkerConfig
from phone_agent.worker.device_discovery import DeviceDiscoveryCache
from phone_agent.worker.heartbeat import ControlHeartbeatLoop, WorkerRuntimeState
from phone_agent.worker.identity import load_or_create_worker_id, uuid7
from phone_agent.worker.outbox import DurableOutbox, LocalSealer
from phone_agent.worker.platform_events import OutboxPump
from phone_agent.worker.recovery import StartupRecovery
from phone_agent.worker.redis_notifier import RedisDispatchNotifier
from phone_agent.worker.spool import PlanSpool
from phone_agent.worker.supervisor import WorkerSupervisor
from phone_agent.worker.models import WorkerActivity


def run_worker(config: WorkerConfig, *, once: bool = False) -> int:
    config.spool_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    worker_id = load_or_create_worker_id(config.worker_id_path, config.worker_id)
    instance_id = uuid7()
    api = WorkerApiClient(
        config.platform_base_url,
        config.worker_credential,
        config.runtime_environment,
    )
    outbox = DurableOutbox(config.spool_root / "worker.db")
    sealer = LocalSealer(config.sealing_key_path)
    state = WorkerRuntimeState(activity=WorkerActivity.STARTING)
    stop = threading.Event()

    def request_stop(_signum=None, _frame=None):
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        StartupRecovery(
            api=api,
            outbox=outbox,
            sealer=sealer,
            spool_root=config.spool_root,
            worker_id=worker_id,
            instance_id=instance_id,
            runtime_environment=config.runtime_environment,
        ).recover()
        discovery = DeviceDiscoveryCache(
            worker_id=worker_id,
            allowlist=config.device_allowlist,
        )
        heartbeat = ControlHeartbeatLoop(
            api=api,
            worker_id=worker_id,
            instance_id=instance_id,
            runtime_environment=config.runtime_environment,
            state=state,
            device_snapshot=lambda busy: discovery.heartbeat_snapshot(worker_busy=busy),
            outbox_pending=outbox.pending_count,
            spool_root=config.spool_root,
            interval_seconds=config.heartbeat_seconds,
        )
        # Environment/profile binding must be accepted before any ADB discovery,
        # Redis consumption, or claim. This first heartbeat uses an empty cache.
        heartbeat.send_once()
        discovery.discover(worker_busy=False)
        state.set_activity(WorkerActivity.IDLE)
        notifier = RedisDispatchNotifier(
            config.redis_url,
            worker_id,
            instance_id,
            config.runtime_environment,
        )
        executor = ChildProcessPlanExecutor(
            python_executable=sys.executable,
            model_profiles_path=config.model_profiles_path,
            report_root=config.spool_root / "local-reports",
            outbox_db_path=config.spool_root / "worker.db",
            sealing_key_path=config.sealing_key_path,
            platform_base_url=config.platform_base_url,
            runtime_environment=config.runtime_environment,
            active_probe_callback=lambda payload: discovery.record_active_probe(
                payload["device_uid"], payload["adb_serial"]
            ),
        )
        supervisor = WorkerSupervisor(
            worker_id=worker_id,
            instance_id=instance_id,
            api=api,
            notifier=notifier,
            discovery=discovery,
            spool=PlanSpool(config.spool_root),
            outbox=outbox,
            sealer=sealer,
            state=state,
            device_lock_dir=config.spool_root / "device-locks",
            execute_plan=executor,
        )
        event_pump = OutboxPump(outbox, api, sealer)
        artifact_pump = ArtifactOutboxPump(api, outbox, sealer)

        def flush_outbox() -> None:
            while not stop.is_set():
                try:
                    for item in outbox.due(20):
                        if item.kind == "ARTIFACT_UPLOAD":
                            artifact_pump.flush_item(item)
                    event_pump.flush_once(100)
                except Exception:
                    # Individual durable items retain their own error/retry state.
                    # A transient DB/API failure is retried by the next pass.
                    pass
                stop.wait(0.5)

        pump_thread = threading.Thread(
            target=flush_outbox,
            name="worker-outbox-pump",
            daemon=True,
        )
        heartbeat.start()
        pump_thread.start()
        last_discovery = 0.0
        while not stop.is_set():
            activity, _, _ = state.snapshot()
            now = time.monotonic()
            if activity is WorkerActivity.IDLE and now - last_discovery >= config.discovery_seconds:
                discovery.discover(worker_busy=False)
                last_discovery = now
            if config.claim_enabled and activity is WorkerActivity.IDLE:
                handled = supervisor.process_one(block_ms=1000)
                if once and handled:
                    break
            elif stop.wait(0.2):
                break
            if once and not config.claim_enabled:
                break
        heartbeat.stop()
        stop.set()
        pump_thread.join(timeout=3)
        notifier.close()
        return 0
    finally:
        outbox.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Open-AutoGLM platform Worker")
    parser.add_argument("--once", action="store_true", help="process at most one dispatch")
    args = parser.parse_args()
    return run_worker(WorkerConfig.from_env(), once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
