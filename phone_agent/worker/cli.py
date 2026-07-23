"""Command-line runtime for the single-capacity Open-AutoGLM Worker."""

from __future__ import annotations

import argparse
import logging
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
from phone_agent.worker.pre_test_install import (
    FrozenGitLabApkInstaller,
    WorkerGitLabConfig,
)
from phone_agent.worker.recovery import StartupRecovery
from phone_agent.worker.redis_notifier import RedisDispatchNotifier
from phone_agent.worker.spool import PlanSpool
from phone_agent.worker.spool_gc import SpoolGarbageCollector
from phone_agent.worker.supervisor import WorkerSupervisor
from phone_agent.worker.models import WorkerActivity
from phone_agent.worker.logging_config import configure_worker_logging
from phone_agent.execution.errors import ExecutionErrorCode

logger = logging.getLogger(__name__)


def run_worker(config: WorkerConfig, *, once: bool = False) -> int:
    config.spool_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    configure_worker_logging(spool_root=config.spool_root)
    worker_id = load_or_create_worker_id(config.worker_id_path, config.worker_id)
    instance_id = uuid7()
    logger.info(
        "Worker starting environment=%s worker_id=%s instance_id=%s "
        "claim_enabled=%s once=%s spool_root=%s",
        config.runtime_environment,
        worker_id,
        instance_id,
        config.claim_enabled,
        once,
        config.spool_root,
    )
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
        logger.info("Worker stop requested signal=%s", _signum)
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        logger.info("Startup recovery checking durable task state")
        StartupRecovery(
            api=api,
            outbox=outbox,
            sealer=sealer,
            spool_root=config.spool_root,
            worker_id=worker_id,
            instance_id=instance_id,
            runtime_environment=config.runtime_environment,
        ).recover()
        logger.info("Startup recovery completed")
        suspended_uploads = outbox.suspend_terminal_run_artifact_uploads()
        if suspended_uploads:
            logger.info(
                "Suspended recovered terminal-Run Artifact uploads count=%d",
                suspended_uploads,
            )
        spool_gc = SpoolGarbageCollector(
            spool_root=config.spool_root,
            outbox=outbox,
            retention_days=config.spool_retention_days,
            max_bytes=config.spool_max_bytes,
            min_free_bytes=config.spool_min_free_bytes,
        )
        gc_result = spool_gc.collect()
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
        logger.info("Initial control heartbeat accepted")
        devices = discovery.discover(worker_busy=False)
        logger.info(
            "Initial ADB discovery completed device_count=%d devices=%s",
            len(devices),
            ",".join(f"{device.device_uid}:{device.cached_state}" for device in devices)
            or "-",
        )
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
            pre_test_installer=FrozenGitLabApkInstaller(
                config=WorkerGitLabConfig(
                    base_url=config.gitlab_base_url,
                    token=config.gitlab_token,
                    verify_ssl=config.gitlab_verify_ssl,
                    use_env_proxy=config.gitlab_use_env_proxy,
                    timeout_seconds=config.gitlab_download_timeout_seconds,
                    max_artifact_bytes=config.gitlab_max_artifact_bytes,
                    max_apk_bytes=config.gitlab_max_apk_bytes,
                    min_free_bytes=config.gitlab_min_free_bytes,
                    apk_metadata_tool=config.android_apk_metadata_tool,
                ),
                outbox=outbox,
            ),
        )
        event_pump = OutboxPump(outbox, api, sealer)
        artifact_pump = ArtifactOutboxPump(api, outbox, sealer)

        def flush_outbox() -> None:
            while not stop.is_set():
                try:
                    uploaded = 0
                    for item in outbox.due(20):
                        if item.kind == "ARTIFACT_UPLOAD":
                            uploaded += int(artifact_pump.flush_item(item))
                    events = event_pump.flush_once(100)
                    if uploaded or events:
                        logger.info(
                            "Outbox flush completed artifacts=%d events=%d",
                            uploaded,
                            events,
                        )
                except Exception as error:
                    # Individual durable items retain their own error/retry state.
                    # A transient DB/API failure is retried by the next pass.
                    logger.warning(
                        "Outbox flush failed error_type=%s pending=%d",
                        type(error).__name__,
                        outbox.pending_count(),
                    )
                stop.wait(0.5)

        pump_thread = threading.Thread(
            target=flush_outbox,
            name="worker-outbox-pump",
            daemon=True,
        )
        heartbeat.start()
        pump_thread.start()
        logger.info("Worker is online and waiting for dispatch")
        last_discovery = 0.0
        last_gc = time.monotonic()
        while not stop.is_set():
            activity, _, _ = state.snapshot()
            now = time.monotonic()
            if (
                activity is WorkerActivity.IDLE
                and now - last_gc >= config.spool_gc_interval_seconds
            ):
                gc_result = spool_gc.collect()
                last_gc = now
            if (
                activity is WorkerActivity.IDLE
                and now - last_discovery >= config.discovery_seconds
            ):
                discovery.discover(worker_busy=False)
                last_discovery = now
            if (
                config.claim_enabled
                and activity is WorkerActivity.IDLE
                and gc_result.can_claim
            ):
                handled = supervisor.process_one(block_ms=1000)
                if handled:
                    gc_result = spool_gc.collect()
                    last_gc = time.monotonic()
                if once and handled:
                    break
            elif (
                config.claim_enabled
                and activity is WorkerActivity.IDLE
                and not gc_result.can_claim
            ):
                state.set_activity(
                    WorkerActivity.DEGRADED,
                    last_error_code=ExecutionErrorCode.OUTBOX_FULL.value,
                )
                logger.error(
                    "Worker claim paused because spool capacity is unsafe "
                    "spool_bytes=%d free_bytes=%d",
                    gc_result.spool_bytes,
                    gc_result.free_bytes,
                )
                if stop.wait(min(30, config.spool_gc_interval_seconds)):
                    break
                state.set_activity(
                    WorkerActivity.IDLE,
                    last_error_code=ExecutionErrorCode.OUTBOX_FULL.value,
                )
                gc_result = spool_gc.collect()
                last_gc = time.monotonic()
            elif stop.wait(0.2):
                break
            if once and not config.claim_enabled:
                break
        logger.info("Worker main loop stopping")
        heartbeat.stop()
        stop.set()
        pump_thread.join(timeout=3)
        notifier.close()
        logger.info("Worker stopped cleanly")
        return 0
    except Exception:
        logger.exception("Worker stopped because of an unhandled error")
        raise
    finally:
        outbox.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Open-AutoGLM platform Worker")
    parser.add_argument(
        "--once", action="store_true", help="process at most one dispatch"
    )
    args = parser.parse_args()
    return run_worker(WorkerConfig.from_env(), once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
