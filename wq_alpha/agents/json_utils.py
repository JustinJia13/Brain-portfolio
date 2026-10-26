"""Small JSON helpers for LLM responses."""

from __future__ import annotations

import json
from typing import Any


def extract_json_text(raw_text: str) -> str:
    """Return the most likely JSON object/array substring from an LLM response."""
    text = raw_text.strip()
    if "```" in text:
        for part in text.split("```"):
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{") or stripped.startswith("["):
                return stripped

    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [idx for idx in (start_obj, start_arr) if idx >= 0]
    if not starts:
        return text

    start = min(starts)
    end_char = "}" if text[start] == "{" else "]"
    end = text.rfind(end_char)
    return text[start : end + 1] if end >= start else text


def parse_json_object(raw_text: str) -> dict[str, Any] | None:
    """Parse a JSON object from raw LLM text."""
    try:
        payload = json.loads(extract_json_text(raw_text))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def parse_json_array(raw_text: str, *, allow_object: bool = False) -> list[Any]:
    """Parse a JSON array from raw LLM text.

    ``allow_object=True`` wraps a single parsed object in a one-item list, which
    is useful for agents where the expected output is an array but older prompts
    sometimes returned one object.
    """
    try:
        payload = json.loads(extract_json_text(raw_text))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if isinstance(payload, list):
        return payload
    if allow_object and isinstance(payload, dict):
        return [payload]
    return []


__all__ = ["extract_json_text", "parse_json_array", "parse_json_object"]
