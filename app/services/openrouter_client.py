"""OpenRouter Vision client with provider pinning and low-cost keepalive."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)
_client_lock = Lock()
_write_lock = Lock()
_state_lock = Lock()
_thread_lock = Lock()
_http_client: httpx.Client | None = None
_keepalive_thread: Thread | None = None
_stop_event = Event()
_state: dict[str, Any] = {
    "last_keepalive_at": None,
    "last_keepalive_ok": None,
    "last_keepalive_latency_ms": None,
    "last_provider": None,
    "last_error": None,
}


def enabled() -> bool:
    return bool(
        settings.OPENROUTER_ENABLED
        and settings.OPENROUTER_BASE_URL
        and settings.OPENROUTER_API_KEY
    )


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        with _client_lock:
            if _http_client is None:
                _http_client = httpx.Client()
    return _http_client


def _providers() -> list[str]:
    values = [settings.OPENROUTER_PRIMARY_PROVIDER]
    values.extend(settings.OPENROUTER_FALLBACK_PROVIDERS.split(","))
    result: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # A keepalive must reach the upstream provider rather than OpenRouter's
        # response cache, and production classifications should be auditable.
        "X-OpenRouter-Cache": "false",
    }


def _record(
    *,
    kind: str,
    status: str,
    provider_requested: str,
    duration_ms: float,
    provider_returned: str | None = None,
    reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> None:
    usage = usage or {}
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "model": settings.OPENROUTER_MODEL,
        "provider_requested": provider_requested,
        "provider_returned": provider_returned,
        "status": status,
        "reason": reason,
        "duration_ms": round(duration_ms, 1),
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cost_usd": float(usage.get("cost", 0) or 0),
    }
    path = Path(settings.OPENROUTER_USAGE_LOG_PATH)
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("OpenRouter usage log skipped: %s", exc)


def _request(
    *,
    provider: str,
    content: str | list[dict[str, Any]],
    max_tokens: int,
    timeout: float,
    kind: str,
) -> str | None:
    body = {
        "model": settings.OPENROUTER_MODEL,
        "provider": {"only": [provider], "allow_fallbacks": False},
        "session_id": settings.OPENROUTER_SESSION_ID,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "reasoning": {"effort": "none"},
    }
    if kind == "classification":
        body["response_format"] = {"type": "json_object"}
    started = time.perf_counter()
    try:
        response = _get_http_client().post(
            settings.OPENROUTER_BASE_URL.rstrip("/") + "/chat/completions",
            headers=_headers(),
            json=body,
            timeout=max(0.1, timeout),
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty OpenRouter content")
        duration_ms = (time.perf_counter() - started) * 1000
        returned = payload.get("provider")
        _record(
            kind=kind,
            status="ok",
            provider_requested=provider,
            provider_returned=returned,
            duration_ms=duration_ms,
            usage=payload.get("usage"),
        )
        with _state_lock:
            _state["last_provider"] = returned or provider
            _state["last_error"] = None
        return answer
    except httpx.HTTPStatusError as exc:
        reason = f"http_{exc.response.status_code}"
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        reason = type(exc).__name__
    duration_ms = (time.perf_counter() - started) * 1000
    _record(
        kind=kind,
        status="fallback",
        provider_requested=provider,
        duration_ms=duration_ms,
        reason=reason,
    )
    with _state_lock:
        _state["last_error"] = reason
    logger.warning("OpenRouter %s failed via %s (%s)", kind, provider, reason)
    return None


def generate(prompt: str, images: list[str]) -> str | None:
    """Return schema-constrained Vision JSON, trying only configured providers."""
    if not enabled() or not images:
        return None
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"},
    } for image in images)
    deadline = time.perf_counter() + settings.OPENROUTER_TOTAL_TIMEOUT_SEC
    for provider in _providers():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        answer = _request(
            provider=provider,
            content=content,
            max_tokens=settings.OPENROUTER_MAX_TOKENS,
            timeout=min(settings.OPENROUTER_ATTEMPT_TIMEOUT_SEC, remaining),
            kind="classification",
        )
        if answer is not None:
            return answer
    return None


def _within_keepalive_window(now: datetime | None = None) -> bool:
    current = now or datetime.now(ZoneInfo("Asia/Seoul"))
    if current.tzinfo is None:
        current = current.replace(tzinfo=ZoneInfo("Asia/Seoul"))
    else:
        current = current.astimezone(ZoneInfo("Asia/Seoul"))
    return (
        settings.OPENROUTER_KEEPALIVE_START_HOUR
        <= current.hour
        < settings.OPENROUTER_KEEPALIVE_END_HOUR
    )


def keepalive_once(
    *,
    now: datetime | None = None,
    force: bool = False,
) -> bool | None:
    """Reach the primary upstream with a minimal uncached generation."""
    if not enabled():
        return False
    if not force and not _within_keepalive_window(now):
        return None
    started = time.perf_counter()
    answer = _request(
        provider=settings.OPENROUTER_PRIMARY_PROVIDER.strip().lower(),
        content="Reply with OK.",
        max_tokens=1,
        timeout=settings.OPENROUTER_ATTEMPT_TIMEOUT_SEC,
        kind="keepalive",
    )
    now = datetime.now(timezone.utc).isoformat()
    latency_ms = (time.perf_counter() - started) * 1000
    with _state_lock:
        _state["last_keepalive_at"] = now
        _state["last_keepalive_ok"] = answer is not None
        _state["last_keepalive_latency_ms"] = round(latency_ms, 1)
    return answer is not None


def _keepalive_loop() -> None:
    while not _stop_event.is_set():
        keepalive_once()
        if _stop_event.wait(settings.OPENROUTER_KEEPALIVE_INTERVAL_SEC):
            break


def start_keepalive() -> bool:
    global _keepalive_thread
    if not enabled():
        return False
    with _thread_lock:
        if _keepalive_thread is not None and _keepalive_thread.is_alive():
            return True
        _stop_event.clear()
        _keepalive_thread = Thread(
            target=_keepalive_loop,
            name="openrouter-keepalive",
            daemon=True,
        )
        _keepalive_thread.start()
    return True


def health_status() -> dict[str, Any]:
    with _state_lock:
        state = dict(_state)
    if not enabled():
        return {
            "enabled": False,
            "status": "disabled",
            "reachable": False,
            "model": settings.OPENROUTER_MODEL,
            "model_available": False,
            "accelerator": "unknown",
            "size_vram_bytes": None,
            "latency_ms": None,
            "keepalive_window_active": _within_keepalive_window(),
            **state,
        }
    keepalive_ok = state["last_keepalive_ok"]
    window_active = _within_keepalive_window()
    return {
        "enabled": True,
        "status": (
            "ok" if keepalive_ok is True
            else "unavailable" if window_active
            else "scheduled_idle"
        ),
        "reachable": keepalive_ok is True,
        "model": settings.OPENROUTER_MODEL,
        "model_available": keepalive_ok is True,
        "accelerator": "unknown",
        "size_vram_bytes": None,
        "latency_ms": state["last_keepalive_latency_ms"],
        "keepalive_window_active": window_active,
        **state,
    }


def shutdown() -> None:
    global _http_client, _keepalive_thread
    _stop_event.set()
    with _thread_lock:
        thread = _keepalive_thread
        _keepalive_thread = None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
    with _client_lock:
        if _http_client is not None:
            _http_client.close()
            _http_client = None
