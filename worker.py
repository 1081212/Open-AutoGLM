#!/usr/bin/env python3
"""Compatibility entry point for the platform Worker."""

from phone_agent.worker.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
