import logging
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status

from app.core.security import verify_api_key
from app.models.registry import ModelRegistry
from app.schemas.request import DetectFormData
from app.schemas.response import DetectResponse, ErrorResponse
from app.core.config import settings
from app.services import pipeline, request_capture, spring_client

logger = logging.getLogger(__name__)

router = APIRouter()


_DETECT_DESCRIPTION = """
이미지와 무게 값으로 단일 투입 재활용품을 판정합니다. 운영 `primary` 모드에서는
YOLO가 객체 위치를 검출하고 해당 crop을 NAS 로컬 Vision LLM이 품목·라벨·압착·외부
이물질까지 최종 판정합니다. YOLO가 bbox를 찾지 못하면 LLM이 원본 전체 프레임을 한 번만
판정합니다. 무게 및 안내 코드는 기존 규칙으로
결정합니다. LLM이 단독 일반폐기물(예: 빨대)을 확정하면 `GENERAL_WASTE / GENERAL_WASTE`로
일반함 처리합니다. LLM이 응답하지 않거나, 저신뢰이거나, 단일 주 물체가 아니면 YOLO 품목으로
대체하지 않고 `GENERAL_WASTE / LOW_CONFIDENCE`로 보류합니다.

## 요청 형식

- Content-Type: `multipart/form-data`
- Header `X-API-Key` (string, 필수): 서버의 `API_KEY`와 일치하는 인증 키
- Form `image` (file, 필수): 분류할 JPG 또는 PNG 이미지
- Form `client_id` (string, 필수, 1~128자): 사용자·피드백·하드웨어 요청 구분 ID
- Form `weight_g` (number, 선택, 0 이상): 무게 센서의 그램값. 생략하면 무게 이상 검사를 하지 않음

## 응답 시간 확인

- HTTP 응답 헤더 `X-Process-Time-Ms`: 이미지 수신 뒤 AI가 최종 판정을 만들 때까지의 서버 처리 시간(밀리초)
- Spring callback은 비동기 후속 작업이므로 이 헤더에 포함되지 않는다. Pi AI 로그에는 `YOLO 검출 시간`,
  `NAS LLM 분류 시간`, `Spring 콜백 전송 완료 ... callback_ms`가 각각 기록된다.

## 응답 필드

| 필드 | 타입 | 반환 조건 | 설명 |
|---|---|---|---|
| `client_id` | string | 항상 | 요청에서 받은 검사 식별자. 1~128자이며 변경하지 않고 Spring 콜백에도 전달 |
| `status` | enum | 항상 | `ALLOWED`, `REJECTED`, `GENERAL_WASTE`, `NOT_DETECTED` 중 하나. 최우선 분기값 |
| `classification` | object | 최종 품목 확정 시 | `class_id`, `class_name`, `confidence`를 포함. 미감지 또는 LLM 보류 시 필드 생략 |
| `conditions` | object | 항상 | 상태 판정값. 대상이 아닌 항목은 내부 필드를 생략하므로 `{}`일 수 있음 |
| `weight` | object | 항상 | `anomaly`는 항상 포함, 무게 미입력 시 `value_g` 생략 |
| `guidance` | array | 항상 | 재처리 안내 목록. 통과 또는 완전거부 시 빈 배열 |
| `rejection` | object | 완전거부 시 | 유리·건전지·형광등·스티로폼의 거부 코드와 메시지. 그 외에는 필드 생략 |
| `general` | object | `GENERAL_WASTE` 시 | 확정 일반폐기물 또는 저신뢰·미분류 코드와 메시지. 그 외에는 필드 생략 |
| `bbox` | number[4] | 품목 확정 시 | 원본 이미지 픽셀 기준 `[x1, y1, x2, y2]`. YOLO 미감지 LLM fallback은 전체 프레임 범위 |

선택 필드는 값이 없을 때 JSON `null`을 보내지 않고 **필드 자체를 생략**합니다.
Spring DTO에만 있는 `image_url`은 AI 서버가 전송하지 않습니다.

## 중첩 필드

### `classification`

| 필드 | 타입/범위 | 설명 |
|---|---|---|
| `class_id` | integer, 0~8 | 외부 클래스 ID. 내부 PET(1)는 외부에서 반드시 `plastic/3`으로 변환 |
| `class_name` | enum | `can`, `paper`, `plastic`, `styrofoam`, `vinyl`, `glass`, `battery`, `fluorescent` 중 실제 반환값 |
| `confidence` | number, 0~1 | 주 분류 신뢰도 |

### `conditions`

| 필드 | 타입 | 반환 조건 | 의미 |
|---|---|---|---|
| `has_label` | boolean | 내부 PET·플라스틱 상태 검사 시 | `true`: 라벨 있음, `false`: 라벨 없음 |
| `is_dented` | boolean | 내부 PET·캔 상태 검사 시 | `true`: 압착됨, `false`: 미압착 |

`conditions.has_foreign_material`은 Spring 계약에 없으므로 반환하지 않습니다.
PET는 외부 분류가 `plastic/3`이어도 내부 PET 상태 기준으로 라벨과 압착을 검사합니다.

### `weight`

| 필드 | 타입 | 설명 |
|---|---|---|
| `value_g` | number | 요청에서 받은 0 이상의 그램값. 미입력 시 생략 |
| `anomaly` | boolean | `true`: 해당 품목의 정상 무게 범위를 벗어남, `false`: 정상 또는 미측정 |

### 코드/메시지 객체

`guidance[]`, `rejection`, `general`은 모두 `code`와 `message`를 가집니다.
하드웨어와 Spring은 변경 가능한 한국어 `message`가 아니라 **`code`로 분기**해야 합니다.

## `status`별 필드 조합

| status | 의미 | 주요 필드 |
|---|---|---|
| `ALLOWED` | 지정 함 투입 허용 | `classification`, `conditions`, `weight`, 빈 `guidance`, `bbox` |
| `REJECTED` + guidance | 조건 불충족, 재처리 후 재투입 | `classification`, `conditions`, `weight`, 1개 이상의 `guidance`, `bbox` |
| `REJECTED` + rejection | 기기 수거 불가 | `classification`, `rejection`, 빈 `guidance`, `bbox` |
| `GENERAL_WASTE` | 단독 일반폐기물 확정 또는 LLM 장애·저신뢰·복수 물체 보류 | `general`, 빈 `guidance`, `bbox`; `classification`은 생략 |
| `NOT_DETECTED` | YOLO와 전체프레임 LLM 모두 품목을 확정하지 못함 | `client_id`, `status`, 빈 `conditions`, `weight.anomaly=false`, 빈 `guidance` |

## 전체 분기표

아래 표는 하드웨어 즉시 응답과 Spring 콜백에 같은 형식으로 적용됩니다. `client_id`는 모든 행에서
그대로 포함됩니다. 여러 재처리 조건이 동시에 맞으면 `guidance`에 코드를 함께 넣습니다.

| 우선순위 | 판정 조건 | status | class_id / class_name | 추가 필드·코드 |
|---:|---|---|---|---|
| 1 | `weight_g`가 설정된 하한 미만(기본 1g) | `NOT_DETECTED` | 생략 | 빈 물체 오탐 방지. `weight.anomaly=false` |
| 2 | YOLO bbox 미감지 + 전체프레임 LLM 단일 품목 확정 | 최종 품목 기준 | LLM 결과 | bbox=원본 전체 범위. LLM이 일반폐기물을 확정하면 `GENERAL_WASTE` |
| 3 | YOLO bbox 미감지 + LLM 장애·JSON 오류·저신뢰·복수 주 물체 | `NOT_DETECTED` | 생략 | `weight.anomaly=false` |
| 4 | LLM이 단독 일반폐기물로 확정 (예: 빨대) | `GENERAL_WASTE` | 생략 | `general.code=GENERAL_WASTE`, bbox 유지, 일반함 처리 |
| 5 | LLM 장애·JSON 오류·저신뢰·복수 주 물체 (YOLO bbox 있음) | `GENERAL_WASTE` | 생략 | `general.code=LOW_CONFIDENCE`, bbox 유지, YOLO 품목 fallback 없음 |
| 6 | 캔: 무게 정상·외부 이물질 없음·압착됨 | `ALLOWED` | `0 / can` | `conditions.is_dented=true` |
| 7 | 캔: 무게 이상/내용물 | `REJECTED` | `0 / can` | `guidance=EMPTY_CONTENTS` |
| 8 | 캔: 미압착 | `REJECTED` | `0 / can` | `guidance=COMPRESS` |
| 9 | PET 또는 플라스틱: 무게 정상·라벨 없음·외부 이물질 없음 (PET는 압착됨) | `ALLOWED` | `3 / plastic` | PET도 외부 `plastic/3`으로 통합 |
| 10 | PET 또는 플라스틱: 무게 이상/내용물 | `REJECTED` | `3 / plastic` | `guidance=EMPTY_CONTENTS` |
| 11 | PET 또는 플라스틱: 라벨 미제거 | `REJECTED` | `3 / plastic` | `guidance=REMOVE_LABEL` |
| 12 | PET: 미압착 | `REJECTED` | `3 / plastic` | `guidance=COMPRESS` |
| 13 | 종이: 무게 정상·외부 이물질 없음 | `ALLOWED` | `2 / paper` | `conditions={}` |
| 14 | 종이: 무게 이상 | `REJECTED` | `2 / paper` | `guidance=WEIGHT_ANOMALY` |
| 15 | 비닐: 무게 정상·외부 이물질 없음 | `ALLOWED` | `5 / vinyl` | 정상 비닐도 `ALLOWED` |
| 16 | 비닐: 무게 이상 | `REJECTED` | `5 / vinyl` | `guidance=WEIGHT_ANOMALY` |
| 17 | 허용 대상에서 다른 재질 이물질/혼합 부착물 감지 | `REJECTED` | 최종 품목 유지 | `guidance=FOREIGN_MATERIAL` |
| 18 | 유리 | `REJECTED` | `6 / glass` | `rejection.code=GLASS` |
| 19 | 건전지 | `REJECTED` | `7 / battery` | `rejection.code=BATTERY` |
| 20 | 형광등 | `REJECTED` | `8 / fluorescent` | `rejection.code=FLUORESCENT` |
| 21 | 스티로폼 | `REJECTED` | `4 / styrofoam` | `rejection.code=STYROFOAM` |

`FOREIGN_MATERIAL`은 재활용 주 품목에 붙은 다른 재질의 부착물·혼합 이물질에 적용합니다. 단독 빨대는
`GENERAL_WASTE / GENERAL_WASTE`로 일반함 처리하며, 컵에 붙은 빨대는 컵 분류를 유지한 `FOREIGN_MATERIAL`입니다.
카페 음료컵은 빨대·컵홀더 제거 후 `plastic`으로, 단독 컵홀더는 `paper`로 판정합니다. 그 밖의 재활용
9종에 명확히 속하지 않는 단독 생활폐기물도 `GENERAL_WASTE / GENERAL_WASTE`로 일반함 처리합니다.
표의 6~16은 한 항목에 둘 이상 적용되면 하나의 `REJECTED` 응답에 guidance 배열로 합쳐집니다.

## 모델/외부 클래스 매핑

| 모델 class_id/name | 외부 응답 class_id/name | 처리 |
|---|---|---|
| `0 / can` | `0 / can` | 조건 충족 시 `ALLOWED` |
| `1 / pet` | **`3 / plastic`** | PET 외부 분기 미사용, 플라스틱으로 통합 |
| `2 / paper` | `2 / paper` | 조건 충족 시 `ALLOWED` |
| `3 / plastic` | `3 / plastic` | 조건 충족 시 `ALLOWED` |
| `4 / styrofoam` | `4 / styrofoam` | `REJECTED` + `rejection.code=STYROFOAM` |
| `5 / vinyl` | `5 / vinyl` | 조건 충족 시 `ALLOWED` |
| `6 / glass` | `6 / glass` | `REJECTED` + `rejection.code=GLASS` |
| `7 / battery` | `7 / battery` | `REJECTED` + `rejection.code=BATTERY` |
| `8 / fluorescent` | `8 / fluorescent` | `REJECTED` + `rejection.code=FLUORESCENT` |

## 재처리 `guidance` 코드

여러 조건을 동시에 위반하면 여러 항목이 배열로 함께 반환될 수 있습니다.

| code | 발생 조건 |
|---|---|
| `EMPTY_CONTENTS` | 내부 PET·플라스틱·캔의 무게 이상 또는 내용물 존재 추정 |
| `WEIGHT_ANOMALY` | 종이·비닐의 무게 이상 |
| `REMOVE_LABEL` | 내부 PET·플라스틱에 라벨이 있음 |
| `COMPRESS` | 내부 PET·캔이 미압착 상태 |
| `FOREIGN_MATERIAL` | 로컬 Vision LLM이 다른 재질의 부착물·혼합 이물질을 감지함. 같은 재질 부속품(예: 플라스틱 빨대)은 이물질로 보지 않음 |

## 완전거부 및 일반분류 코드

- `rejection.code`: `GLASS`, `BATTERY`, `FLUORESCENT`, `STYROFOAM`
- `general.code`: 확정 일반폐기물(예: 단독 빨대)은 `GENERAL_WASTE`; LLM 저신뢰·장애 보류는 `LOW_CONFIDENCE`; `UNCLASSIFIED`와 `VINYL`은 하위 호환 값
- 정상 비닐은 `GENERAL_WASTE`가 아니라 `ALLOWED / class_id=5 / class_name=vinyl`

## 오류 응답

오류는 항상 `{"error":{"code":"...","message":"..."}}` 형식입니다.

| HTTP | code | 의미 |
|---:|---|---|
| 401 | `UNAUTHORIZED` | 전달한 `X-API-Key`가 올바르지 않음 |
| 422 | `VALIDATION_ERROR` | 인증 헤더·필수 폼 누락, 빈 `client_id`, 음수 무게 등 요청 검증 실패 |
| 422 | `INVALID_IMAGE` | 빈 파일, 손상 파일 또는 지원하지 않는 이미지 |
| 503 | `MODEL_NOT_READY` | 서버에서 모델을 로드하지 못한 상태 |
| 500 | `INFERENCE_ERROR` | 추론 파이프라인 실행 실패 |
| 500 | `INTERNAL_ERROR` | 처리되지 않은 서버 내부 오류 |

동일 JSON을 하드웨어에 즉시 반환하고 Spring
`POST /api/v1/feedback-detail/result`로 백그라운드 전송합니다.
Spring 콜백 실패는 하드웨어에 이미 반환한 HTTP 응답을 바꾸지 않습니다.
""".strip()


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
    response_model_exclude_none=True,
    responses={
        401: {"model": ErrorResponse, "description": "유효하지 않은 API 키"},
        422: {"model": ErrorResponse, "description": "요청 검증 또는 이미지 디코딩 실패"},
        503: {"model": ErrorResponse, "description": "모델 미로드"},
        500: {"model": ErrorResponse, "description": "추론 오류"},
    },
    summary="쓰레기 분류",
    description=_DETECT_DESCRIPTION,
)
async def detect(
    _: Annotated[None, Depends(verify_api_key)],
    form: Annotated[DetectFormData, Depends()],
    background_tasks: BackgroundTasks,
    registry: Annotated[ModelRegistry, Depends(_get_registry)],
    response: Response,
) -> DetectResponse:
    started_at = perf_counter()
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

    elapsed_ms = (perf_counter() - started_at) * 1000
    # 기존 JSON/Spring 계약은 바꾸지 않는다. 하드웨어·프론트는 이 헤더로 요청 단위
    # 분류 지연시간을 확인할 수 있고, 콜백은 백그라운드라 이 시간에 포함되지 않는다.
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.info(
        "분류 완료: client_id=%s status=%s process_ms=%.1f",
        form.client_id,
        result.status.value,
        elapsed_ms,
    )

    if settings.CAPTURE_REQUESTS:
        try:
            # 파이프라인에서 소비한 UploadFile을 되감아 원본 바이트를 보존한다.
            # 제한보다 1바이트 더 읽어 request_capture에서 초과 여부를 판별한다.
            await form.image.seek(0)
            image_bytes = await form.image.read(settings.CAPTURE_MAX_IMAGE_BYTES + 1)
            background_tasks.add_task(
                request_capture.save_capture,
                image_bytes=image_bytes,
                original_filename=form.image.filename,
                content_type=form.image.content_type,
                client_id=form.client_id,
                weight_g=form.weight_g,
                result=result,
            )
        except Exception:
            # 캡처는 보조 기능이므로 실패해도 추론 응답은 정상 반환한다.
            logger.exception("요청 이미지 캡처 준비 실패")

    # BackgroundTasks는 등록 순서대로 실행한다. 재학습 근거를 먼저 로컬에 보존한 뒤
    # Spring 콜백을 전송해 순간적인 네트워크 장애가 있어도 요청/판정 원본은 남긴다.
    background_tasks.add_task(spring_client.notify, result)

    return result
