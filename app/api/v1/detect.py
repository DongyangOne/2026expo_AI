import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from app.core.security import verify_api_key
from app.models.registry import ModelRegistry
from app.schemas.request import DetectFormData
from app.schemas.response import DetectResponse, ErrorResponse
from app.services import pipeline, spring_client

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_registry(request: Request) -> ModelRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "MODEL_NOT_READY", "message": "모델이 로드되지 않았습니다. 서버 로그를 확인하세요."},
        )
    return registry


@router.post(
    "/detect",
    response_model=DetectResponse,
    responses={
        401: {"model": ErrorResponse, "description": "유효하지 않은 API 키"},
        422: {"model": ErrorResponse, "description": "이미지 디코딩 실패"},
        503: {"model": ErrorResponse, "description": "모델 미로드"},
        500: {"model": ErrorResponse, "description": "추론 오류"},
    },
    summary="쓰레기 분류",
    description=(
        "이미지와 무게 값으로 9-class 분류 후 상태(압착/라벨/무게)를 검사합니다.\n\n"
        "- `status=ALLOWED`: 재활용 허용 (페트/플라스틱/캔/종이 + 조건 충족)\n"
        "- `status=REJECTED`: 조건 불충족(`guidance` 재처리 안내) 또는 완전거부(`rejection`, 유리·건전지 등)\n"
        "- `status=GENERAL_WASTE`: 일반쓰레기 (`general` — 비닐/저신뢰/미분류)\n"
        "- `status=NOT_DETECTED`: 미감지\n"
        "- `client_id`: 하드웨어가 보낸 사용자/피드백 구분 ID를 응답과 Spring 콜백에 그대로 포함\n"
        "- `conditions`: is_dented(페트·캔 압착), has_label(페트·플라스틱 라벨)"
    ),
)
async def detect(
    _: Annotated[None, Depends(verify_api_key)],
    form: Annotated[DetectFormData, Depends()],
    background_tasks: BackgroundTasks,
    registry: Annotated[ModelRegistry, Depends(_get_registry)],
) -> DetectResponse:
    try:
        result = await pipeline.run(form.image, form.weight_g, form.client_id, registry)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "INVALID_IMAGE", "message": str(exc)},
        )
    except Exception:
        logger.exception("파이프라인 오류")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INFERENCE_ERROR", "message": "추론 중 오류가 발생했습니다."},
        )

    background_tasks.add_task(spring_client.notify, result)
    return result
