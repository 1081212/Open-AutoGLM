"""
Phone Agent - An AI-powered phone automation framework.

This package provides tools for automating Android and iOS phone interactions
using AI models for visual understanding and decision making.
"""

from phone_agent.actions.handler import (
    do,
    finish,
    play_audio_while_holding_element,
    tap_element,
    type_into_element,
)
from phone_agent.agent import CustomRule, PhoneAgent, RuleContext
from phone_agent.agent_ios import IOSPhoneAgent

__version__ = "0.1.0"
__all__ = [
    "PhoneAgent",
    "IOSPhoneAgent",
    "CustomRule",
    "RuleContext",
    "do",
    "finish",
    "tap_element",
    "type_into_element",
    "play_audio_while_holding_element",
]
