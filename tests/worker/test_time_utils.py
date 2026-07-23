from __future__ import annotations

from datetime import timezone

import pytest

from phone_agent.worker.time_utils import parse_aware_iso8601


@pytest.mark.parametrize(
    "value,microsecond",
    [
        ("2026-07-23T06:22:32Z", 0),
        ("2026-07-23T06:22:32.1Z", 100000),
        ("2026-07-23T06:22:32.19851Z", 198510),
        ("2026-07-23T06:22:32.198510Z", 198510),
        ("2026-07-23T06:22:32.198510123Z", 198510),
        ("2026-07-23T14:22:32.19851+08:00", 198510),
    ],
)
def test_parse_aware_iso8601_accepts_platform_fractional_precision(value, microsecond):
    parsed = parse_aware_iso8601(value, "timestamp")

    assert parsed.microsecond == microsecond
    assert parsed.utcoffset() is not None


def test_parse_aware_iso8601_rejects_missing_timezone():
    with pytest.raises(ValueError, match="timezone"):
        parse_aware_iso8601("2026-07-23T06:22:32.19851", "timestamp")


def test_parse_aware_iso8601_preserves_the_instant():
    parsed = parse_aware_iso8601(
        "2026-07-23T14:22:32.19851+08:00",
        "timestamp",
    )

    assert parsed.astimezone(timezone.utc).isoformat() == (
        "2026-07-23T06:22:32.198510+00:00"
    )
