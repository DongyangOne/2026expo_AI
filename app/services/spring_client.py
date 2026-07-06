"""Spring 서버 결과 전송 (fire-and-forget). 실패해도 메인 응답에 영향 없음."""

import json
import logging
import os
from datetime import datetime, timezone

import httpx

from app.core.config import settings
from app.schemas.response import DetectResponse

logger = logging.getLogger(__name__)


def _log_result(result: DetectResponse) -> None:
    """판정 결과를 logs/results.jsonl 에 한 줄 JSON으로 추가."""
    try:
        os.makedirs(settings.LOG_DIR, exist_ok=True)
        log_path = os.path.join(settings.LOG_DIR, "results.jsonl")
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result.model_dump(mode="json"),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("결과 로그 기록 실패")


async def notify(result: DetectResponse) -> None:
    if settings.LOG_RESULTS:
        _log_result(result)

    url = settings.SPRING_CALLBACK_URL
    if not url:
        return

    payload = result.model_dump(mode="json")
    try:
        async with httpx.AsyncClient(timeout=settings.SPRING_TIMEOUT_SEC) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            logger.warning("Spring 콜백 오류 응답: HTTP %s — %s", resp.status_code, resp.text[:200])
        else:
            logger.debug("Spring 콜백 전송 완료: HTTP %s", resp.status_code)
    except httpx.TimeoutException:
        logger.warning("Spring 콜백 타임아웃 (%.1fs): %s", settings.SPRING_TIMEOUT_SEC, url)
    except httpx.ConnectError:
        logger.warning("Spring 서버 연결 실패: %s", url)
    except Exception:
        logger.exception("Spring 콜백 예기치 못한 오류")
