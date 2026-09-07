"""Shared Anthropic client helpers for locate filtering and quiz generation."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


def anthropic_client() -> Anthropic:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("replace-with"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put a real key in .env."
        )
    return Anthropic(api_key=api_key)


def response_text(response: Any) -> str:
    """Join text blocks from a Messages API response."""

    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return JSON: {exc}\n{text}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Model JSON must be an object.")
    return payload
