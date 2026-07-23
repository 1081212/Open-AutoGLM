"""IDLE device discovery cache; heartbeat reads this cache without ADB."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from uuid import UUID

from phone_agent.adb.command import AdbCommandAdapter
from phone_agent.worker.identity import derive_device_uid

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DiscoveredDevice:
    device_uid: str
    adb_serial: str
    device_type: str
    cached_state: str
    schedulable: bool
    last_probe_at: str
    model: str | None = None
    product: str | None = None
    transport_id: str | None = None

    def as_heartbeat_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DeviceDiscoveryCache:
    worker_id: UUID
    adapter: AdbCommandAdapter = field(default_factory=AdbCommandAdapter)
    allowlist: frozenset[str] | None = None
    _devices: dict[str, DiscoveredDevice] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)

    def discover(self, *, worker_busy: bool = False) -> tuple[DiscoveredDevice, ...]:
        discovered: dict[str, DiscoveredDevice] = {}
        now = datetime.now(timezone.utc).isoformat()
        for serial, adb_state, metadata in self.adapter.list_devices():
            if self.allowlist is not None and serial not in self.allowlist:
                continue
            device_uid = str(derive_device_uid(self.worker_id, serial))
            online = adb_state == "device"
            discovered[device_uid] = DiscoveredDevice(
                device_uid=device_uid,
                adb_serial=serial,
                device_type="adb",
                cached_state="ONLINE" if online else adb_state.upper(),
                schedulable=online and not worker_busy,
                last_probe_at=now,
                model=metadata.get("model"),
                product=metadata.get("product"),
                transport_id=metadata.get("transport_id"),
            )
        with self._lock:
            self._devices = discovered
            logger.debug(
                "ADB discovery refreshed worker_id=%s busy=%s device_count=%d",
                self.worker_id,
                worker_busy,
                len(discovered),
            )
            return tuple(self._devices.values())

    def heartbeat_snapshot(self, *, worker_busy: bool) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(
                {
                    **device.as_heartbeat_dict(),
                    "schedulable": device.cached_state == "ONLINE" and not worker_busy,
                }
                for device in self._devices.values()
            )

    def resolve_serial(self, device_uid: UUID | str) -> str | None:
        with self._lock:
            device = self._devices.get(str(device_uid))
            return (
                device.adb_serial
                if device and device.cached_state == "ONLINE"
                else None
            )

    def probe_active(self, device_uid: UUID | str) -> None:
        serial = self.resolve_serial(device_uid)
        if serial is None:
            raise LookupError(f"device_uid is not online: {device_uid}")
        self.adapter.get_state(serial)

    def record_active_probe(self, device_uid: UUID | str, adb_serial: str) -> None:
        """Refresh only the active cached device after a child Case-boundary probe."""
        key = str(device_uid)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            existing = self._devices.get(key)
            self._devices[key] = DiscoveredDevice(
                device_uid=key,
                adb_serial=adb_serial,
                device_type="adb",
                cached_state="ONLINE",
                schedulable=False,
                last_probe_at=now,
                model=existing.model if existing else None,
                product=existing.product if existing else None,
                transport_id=existing.transport_id if existing else None,
            )
