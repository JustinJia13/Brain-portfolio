"""Anthropic client with the .messages.create() interface used across the repo.

Drop-in for `import anthropic` in every agent file:

    from wq_alpha import llm as anthropic
    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(model=..., max_tokens=..., system=..., messages=[...])
    text = response.content[0].text

Default model is Anthropic Opus 4.8 (claude-opus-4-8). Calls that pass a legacy
model string (e.g. "gpt-5.5") are routed to DEFAULT_MODEL; an explicit "claude-*"
model id is honored as-is, so callers can opt into Sonnet/Haiku per request.

Key resolution order: explicit api_key → ANTHROPIC_API_KEY env var →
antropicapi.txt (existing file) → anthropicapi.txt.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Existing key file is spelled `antropicapi.txt`; also accept the correct spelling.
_KEY_FILES = (ROOT / "antropicapi.txt", ROOT / "anthropicapi.txt")
_ENV_VAR = "ANTHROPIC_API_KEY"

DEFAULT_MODEL = "claude-opus-4-8"  # Anthropic Opus 4.8
# Anthropic requires an explicit max_tokens. Callers that pass None ("uncapped")
# fall back to this — generous enough for the largest theory/decompose JSON
# batches without truncating structured output.
DEFAULT_MAX_TOKENS = 16384

# Anthropic create() kwargs we forward when a caller supplies them.
_PASSTHROUGH_KWARGS = {"temperature", "top_p", "top_k", "stop_sequences", "thinking"}


def load_api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get(_ENV_VAR, "")
    if not key:
        for path in _KEY_FILES:
            if path.exists():
                key = path.read_text().strip()
                if key:
                    break
    return key


def load_openai_key(api_key: str | None = None) -> str:
    """OpenAI key resolver — only for embeddings (Anthropic has no embeddings API).

    Resolution: explicit api_key → OPENAI_API_KEY env → openaiapi.txt.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    openai_file = ROOT / "openaiapi.txt"
    if not key and openai_file.exists():
        key = openai_file.read_text().strip()
    return key


def _resolve_model(model: str | None) -> str:
    """Honor an explicit claude-* id; route everything else to the default."""
    if model and str(model).startswith("claude-"):
        return str(model)
    return DEFAULT_MODEL


class _Content:
    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, text: str, stop_reason: str | None = None) -> None:
        self.content = [_Content(text)]
        self.stop_reason = stop_reason


class _Messages:
    def __init__(self, anthropic_client: Any) -> None:
        self._client = anthropic_client

    def create(
        self,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        system: str = "",
        messages: list[dict],
        **kwargs: Any,
    ) -> _Response:
        # Accept either max_tokens or the legacy max_completion_tokens; Anthropic
        # mandates a concrete cap, so None -> DEFAULT_MAX_TOKENS.
        cap = max_tokens if max_tokens is not None else max_completion_tokens
        if cap is None:
            cap = DEFAULT_MAX_TOKENS

        params: dict[str, Any] = {
            "model": _resolve_model(model),
            "max_tokens": cap,
            "messages": messages,
        }
        if system:
            params["system"] = system
        for name in _PASSTHROUGH_KWARGS:
            if name in kwargs and kwargs[name] is not None:
                params[name] = kwargs[name]

        resp = self._client.messages.create(**params)

        # Concatenate text blocks (ignore any non-text block, e.g. thinking).
        text = "".join(
            getattr(block, "text", "") for block in (resp.content or [])
        ).strip()
        return _Response(text, stop_reason=getattr(resp, "stop_reason", None))


class Anthropic:
    """Real Anthropic client behind the repo's message API surface."""

    def __init__(self, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The `anthropic` package is not installed in this Python environment. "
                "Install it with `pip install anthropic` into the active env."
            ) from exc
        resolved = load_api_key(api_key)
        if not resolved:
            raise ValueError(
                "No Anthropic API key found. "
                f"Set {_ENV_VAR} env var or place the key in antropicapi.txt"
            )
        self._anthropic = anthropic.Anthropic(api_key=resolved)
        self.messages = _Messages(self._anthropic)
