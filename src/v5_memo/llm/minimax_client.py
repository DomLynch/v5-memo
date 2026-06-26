"""MiniMax Anthropic-compatible transport and configuration."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"
MINIMAX_MODEL = "MiniMax-M3"
MINIMAX_KEY_ENV = "MINIMAX_API_KEY"
MINIMAX_V5_KEY_ENV = "V5_MEMO_MINIMAX_API_KEY"
MINIMAX_BASE_URL_ENV = "V5_MEMO_MINIMAX_BASE_URL"
MINIMAX_MODEL_ENV = "V5_MEMO_MINIMAX_MODEL"
MINIMAX_TIMEOUT_ENV = "V5_MEMO_MINIMAX_TIMEOUT_SECONDS"
MINIMAX_MAX_TOKENS_ENV = "V5_MEMO_MINIMAX_MAX_TOKENS"
MINIMAX_KEY_FILE = Path.home() / ".codex" / "secrets" / "minimax_api_key"
_MINIMAX_RETRY_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class HttpResponse(Protocol):
    def __enter__(self) -> HttpResponse: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    def read(self) -> bytes: ...


class RequestOpener(Protocol):
    def __call__(self, request: Request, timeout: float) -> HttpResponse: ...


def call_minimax_m3(
    *,
    api_key: str,
    prompt: str,
    system: str,
    temperature: float,
    max_tokens: int,
    base_url: str = MINIMAX_BASE_URL,
    model: str = MINIMAX_MODEL,
    timeout: float = 60.0,
    opener: RequestOpener | None = None,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    request_opener = opener or cast(RequestOpener, urlopen)
    for attempt in range(3):
        try:
            with request_opener(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            return _anthropic_text(data)
        except HTTPError as exc:
            if exc.code not in _MINIMAX_RETRY_HTTP_STATUS or attempt == 2:
                raise
            time.sleep(0.75 * (attempt + 1))
        except (TimeoutError, URLError):
            if attempt == 2:
                raise
            time.sleep(0.75 * (attempt + 1))
    raise RuntimeError("unreachable MiniMax retry state")


def load_minimax_api_key() -> str:
    for env_name in (MINIMAX_V5_KEY_ENV, MINIMAX_KEY_ENV):
        key = os.environ.get(env_name, "").strip()
        if key:
            return key
    if MINIMAX_KEY_FILE.exists():
        return MINIMAX_KEY_FILE.read_text().strip()
    return ""


def _anthropic_text(data: Mapping[str, Any]) -> str:
    content = data.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ):
            parts.append(item["text"])
    return "\n".join(parts).strip()
