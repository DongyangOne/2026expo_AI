"""Authenticated operations monitor for captures and inference status."""

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, HTMLResponse

from app.core.config import settings
from app.services.monitoring import build_monitor_snapshot, find_capture_image


router = APIRouter()
_PAGE_PATH = Path(__file__).resolve().parents[1] / "web" / "monitor.html"


@router.get("/monitor", include_in_schema=False, response_class=HTMLResponse)
async def monitor_page() -> HTMLResponse:
    return HTMLResponse(_PAGE_PATH.read_text(encoding="utf-8"))


@router.get(
    "/api/v1/monitor/summary",
    include_in_schema=False,
)
async def monitor_summary(limit: Annotated[int, Query(ge=1, le=500)] = 100) -> dict:
    # Image metadata and audit JSONL parsing are disk-bound. Keep them away from
    # the event loop so opening the monitor cannot delay a classification request.
    return await asyncio.to_thread(
        build_monitor_snapshot,
        Path(settings.CAPTURE_DIR),
        limit=limit,
        audit_log_path=Path(settings.LOCAL_LLM_SHADOW_LOG_PATH),
    )


@router.get(
    "/api/v1/monitor/captures/{capture_id}/image",
    include_in_schema=False,
)
async def monitor_capture_image(capture_id: str) -> FileResponse:
    path = find_capture_image(Path(settings.CAPTURE_DIR), capture_id)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CAPTURE_NOT_FOUND", "message": "캡처 이미지를 찾을 수 없습니다."},
        )
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media_type)
