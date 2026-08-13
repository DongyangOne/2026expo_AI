"""Spring 서버 결과 전송.

하드웨어 응답에는 영향을 주지 않는 백그라운드 작업이지만, 순간적인 네트워크 오류와
Spring 5xx 응답은 제한적으로 재시도하고 각 전송 결과를 JSONL로 남긴다.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import httpx

from app.core.config import settings
from app.schemas.response import DetectResponse

logger = logging.getLogger(__name__)


def _append_jsonl(filename: str, entry: dict) -> None:
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_path = os.path.join(settings.LOG_DIR, filename)
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log_result(result: DetectResponse) -> None:
    """판정 결과를 logs/results.jsonl 에 한 줄 JSON으로 추가."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result.model_dump(mode="json"),
        }
        _append_jsonl("results.jsonl", entry)
    except Exception:
        logger.exception("결과 로그 기록 실패")


def _log_callback(
    result: DetectResponse,
    *,
    outcome: str,
    attempt: int,
    status_code: int | None = None,
    error: str | None = None,
) -> None:
    """Spring 전송 성공/재시도/최종 실패를 client_id 기준으로 추적한다."""
    if not settings.LOG_RESULTS:
        return
    try:
        _append_jsonl(
            "callbacks.jsonl",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "client_id": result.client_id,
                "outcome": outcome,
                "attempt": attempt,
                "status_code": status_code,
                "error": error,
            },
        )
    except Exception:
        logger.exception("Spring 콜백 결과 로그 기록 실패")


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 425, 429} or status_code >= 500


async def notify(result: DetectResponse) -> None:
    if settings.LOG_RESULTS:
        _log_result(result)

    url = settings.SPRING_CALLBACK_URL
    if not url:
        return

    # Spring DTO의 선택 필드는 nullable이 아니라 미포함으로 정의돼 있다.
    payload = result.model_dump(mode="json", exclude_none=True)
    max_attempts = max(1, settings.SPRING_MAX_ATTEMPTS)
    base_backoff = max(0.0, settings.SPRING_RETRY_BACKOFF_SEC)

    try:
        async with httpx.AsyncClient(timeout=settings.SPRING_TIMEOUT_SEC) as client:
            for attempt in range(1, max_attempts + 1):
                try:
                    response = await client.post(url, json=payload)
                except httpx.RequestError as exc:
                    error = type(exc).__name__
                    if attempt >= max_attempts:
                        logger.warning(
                            "Spring 콜백 최종 실패 (%s/%s, %s): %s",
                            attempt,
                            max_attempts,
                            error,
                            url,
                        )
                        _log_callback(result, outcome="failed", attempt=attempt, error=error)
                        return
                    logger.warning(
                        "Spring 콜백 재시도 예정 (%s/%s, %s): %s",
                        attempt,
                        max_attempts,
                        error,
                        url,
                    )
                    _log_callback(result, outcome="retry", attempt=attempt, error=error)
                else:
                    if 200 <= response.status_code < 300:
                        logger.info(
                            "Spring 콜백 전송 완료: client_id=%s HTTP %s (%s/%s)",
                            result.client_id,
                            response.status_code,
                            attempt,
                            max_attempts,
                        )
                        _log_callback(
                            result,
                            outcome="delivered",
                            attempt=attempt,
                            status_code=response.status_code,
                        )
                        return

                    response_text = getattr(response, "text", "")[:200]
                    retryable = _is_retryable_status(response.status_code)
                    if not retryable or attempt >= max_attempts:
                        logger.warning(
                            "Spring 콜백 오류 응답: HTTP %s (%s/%s) — %s",
                            response.status_code,
                            attempt,
                            max_attempts,
                            response_text,
                        )
                        _log_callback(
                            result,
                            outcome="failed",
                            attempt=attempt,
                            status_code=response.status_code,
                            error=response_text or None,
                        )
                        return
                    logger.warning(
                        "Spring 콜백 재시도 예정: HTTP %s (%s/%s) — %s",
                        response.status_code,
                        attempt,
                        max_attempts,
                        response_text,
                    )
                    _log_callback(
                        result,
                        outcome="retry",
                        attempt=attempt,
                        status_code=response.status_code,
                        error=response_text or None,
                    )

                await asyncio.sleep(base_backoff * (2 ** (attempt - 1)))
    except Exception:
        logger.exception("Spring 콜백 예기치 못한 오류")
        _log_callback(result, outcome="failed", attempt=0, error="UnexpectedError")
