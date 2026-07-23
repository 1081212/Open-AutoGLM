"""ISO-8601 helpers compatible with the Worker's Python 3.10 runtime."""

from __future__ import annotations

import re
from datetime import datetime


def parse_aware_iso8601(value: str, field_name: str) -> datetime:
    """Parse valid fractional precision without relying on Python 3.11 fixes."""
    normalized = normalize_fractional_seconds(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 date-time") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def normalize_fractional_seconds(value: str) -> str:
    """Pad or truncate ISO-8601 fractions to Python 3.10 microseconds."""
    match = re.fullmatch(
        r"(?P<prefix>.+T\d{2}:\d{2}:\d{2})\.(?P<fraction>\d+)"
        r"(?P<timezone>Z|[+-]\d{2}:\d{2})?",
        value,
    )
    if match is None:
        return value
    fraction = (match.group("fraction") + "000000")[:6]
    return f"{match.group('prefix')}.{fraction}{match.group('timezone') or ''}"
