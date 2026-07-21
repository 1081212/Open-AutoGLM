from __future__ import annotations

from uuid import UUID

from phone_agent.worker.device_discovery import DeviceDiscoveryCache


class FakeAdapter:
    def __init__(self):
        self.get_state_calls = []

    def list_devices(self):
        return [
            ("SERIAL-A", "device", {"model": "Pixel"}),
            ("SERIAL-B", "offline", {"model": "Other"}),
        ]

    def get_state(self, serial):
        self.get_state_calls.append(serial)
        return "device"


def test_discovery_applies_allowlist_and_busy_capacity():
    adapter = FakeAdapter()
    cache = DeviceDiscoveryCache(
        worker_id=UUID("018f47a0-7b5c-7a05-8c00-000000000005"),
        adapter=adapter,
        allowlist=frozenset({"SERIAL-A"}),
    )
    devices = cache.discover()
    assert len(devices) == 1
    assert devices[0].schedulable is True
    busy = cache.heartbeat_snapshot(worker_busy=True)
    assert busy[0]["schedulable"] is False
    assert adapter.get_state_calls == []


def test_heartbeat_snapshot_does_not_call_adb():
    adapter = FakeAdapter()
    cache = DeviceDiscoveryCache(
        worker_id=UUID("018f47a0-7b5c-7a05-8c00-000000000005"),
        adapter=adapter,
    )
    cache.discover()
    before = list(adapter.get_state_calls)
    cache.heartbeat_snapshot(worker_busy=False)
    assert adapter.get_state_calls == before


def test_child_case_boundary_probe_refreshes_cache_without_parent_adb():
    adapter = FakeAdapter()
    cache = DeviceDiscoveryCache(
        worker_id=UUID("018f47a0-7b5c-7a05-8c00-000000000005"),
        adapter=adapter,
    )
    device = cache.discover()[0]
    old_probe = device.last_probe_at

    cache.record_active_probe(device.device_uid, device.adb_serial)
    snapshot = next(
        item for item in cache.heartbeat_snapshot(worker_busy=True)
        if item["device_uid"] == device.device_uid
    )

    assert snapshot["last_probe_at"] >= old_probe
    assert snapshot["schedulable"] is False
    assert adapter.get_state_calls == []
