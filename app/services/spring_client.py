"""Spring 서버 결과 전송 (fire-and-forget). 실패해도 메인 응답에 영향 없음."""

import logging

import httpx

from app.core.config import settings
from app.schemas.response import DetectResponse

logger = logging.getLogger(__name__)


async def notify(result: DetectResponse) -> None:
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
