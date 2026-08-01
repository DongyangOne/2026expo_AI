"""
감지 파이프라인 (흐름 제어 전담).

흐름:
  이미지 디코드
    → 주 YOLO 감지 (inference.run_main)
    → status 판정:
        미감지            → NOT_DETECTED
        저신뢰            → GENERAL_WASTE (일반쓰레기)
        거부품목(유리 등)  → REJECTED (완전 거부)
        비닐              → 상태·무게 검사
                            이상 있음 → REJECTED (재처리 guidance)
                            이상 없음 → ALLOWED (비닐함)
        허용품목          → 멀티헤드 상태(inference.run_state) + 무게 검사
                            조건 충족  → ALLOWED
                            조건 불충족 → REJECTED (재처리 guidance)
        PET 감지          → 내부 상태 검사는 PET 기준, 외부 분류는 PLASTIC으로 정규화
    → DetectResponse 조립

추론은 services.inference, 도메인 규칙은 services.guidance, 무게 판정은 services.weight_check 에 위임.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.config import settings
from app.models.registry import ModelRegistry
from app.schemas.enums import DetectionStatus, GeneralWasteCode, WasteClass
from app.schemas.response import Classification, DetectResponse, WeightInfo
from app.services import guidance, inference, verifier_shadow
from app.services.weight_check import is_anomaly

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1)

# 주 모델 class_id → WasteClass (학습 시 순서와 일치)
_CLASS_BY_ID: dict[int, WasteClass] = {
    0: WasteClass.CAN,       1: WasteClass.PET,        2: WasteClass.PAPER,
    3: WasteClass.PLASTIC,   4: WasteClass.STYROFOAM,  5: WasteClass.VINYL,
    6: WasteClass.GLASS,     7: WasteClass.BATTERY,    8: WasteClass.FLUORESCENT,
}

_PLASTIC_CLASS_ID = 3


def _build_classification(
    model_class_id: int,
    cls: WasteClass | None,
    confidence: float,
) -> Classification | None:
    """모델의 PET 클래스를 외부 계약에서는 PLASTIC 하나로 통합한다."""
    if cls is None:
        return None
    if cls is WasteClass.PET:
        return Classification(
            class_id=_PLASTIC_CLASS_ID,
            class_name=WasteClass.PLASTIC,
            confidence=round(confidence, 4),
        )
    return Classification(
        class_id=model_class_id,
        class_name=cls,
        confidence=round(confidence, 4),
    )


def shutdown() -> None:
    _executor.shutdown(wait=True)
    verifier_shadow.shutdown()


async def _read_image(upload: UploadFile) -> np.ndarray:
    raw = await upload.read()
    if not raw:
        raise ValueError("빈 이미지 파일입니다.")
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩에 실패했습니다. 지원 형식: jpg, png")
    return img


async def run(
    upload: UploadFile,
    weight_g: Optional[float],
    client_id: str,
    registry: ModelRegistry,
) -> DetectResponse:
    loop = asyncio.get_running_loop()

    img = await _read_image(upload)
    detection = await loop.run_in_executor(_executor, inference.run_main, registry, img)

    # ── 미감지 ──────────────────────────────────────────────────────────────────
    if detection is None:
        return DetectResponse(
            client_id=client_id,
            status=DetectionStatus.NOT_DETECTED,
            weight=WeightInfo(value_g=weight_g),
        )

    class_id, confidence, bbox = detection
    verifier_session = (
        registry.verifier() if hasattr(registry, "verifier") else None
    )
    verifier_shadow.submit(
        verifier_session, img, bbox, class_id, confidence, client_id
    )
    cls = _CLASS_BY_ID.get(class_id)
    bbox_rounded = [round(v, 1) for v in bbox]
    weight_info = WeightInfo(value_g=weight_g)
    classification = _build_classification(class_id, cls, confidence)

    # ── 저신뢰 → 일반쓰레기 ──────────────────────────────────────────────────────
    if cls is None or confidence < settings.TRUST_CONF:
        return DetectResponse(
            client_id=client_id,
            status=DetectionStatus.GENERAL_WASTE,
            classification=classification,
            weight=weight_info,
            general=guidance.build_general(GeneralWasteCode.LOW_CONFIDENCE),
            bbox=bbox_rounded,
        )

    # ── 완전 거부 (유리/건전지/형광등/스티로폼) ──────────────────────────────────
    if guidance.is_rejected(cls):
        return DetectResponse(
            client_id=client_id,
            status=DetectionStatus.REJECTED,
            classification=classification,
            weight=weight_info,
            rejection=guidance.build_rejection(cls),
            bbox=bbox_rounded,
        )

    # ── 비닐 — 정상일 때만 비닐함 허용, 이상이면 재처리 안내 ────────────────────
    if guidance.is_vinyl(cls):
        conditions = await loop.run_in_executor(
            _executor, inference.run_state, registry.state(), img, bbox, cls
        )
        weight_info.anomaly = (
            settings.WEIGHT_ANOMALY_ENABLED
            and weight_g is not None
            and is_anomaly(cls.value, weight_g, bbox=bbox, img_area=float(img.shape[0] * img.shape[1]))
        )
        guide = guidance.build_guidance(cls, conditions, weight_info.anomaly)
        if guide:
            return DetectResponse(
                client_id=client_id,
                status=DetectionStatus.REJECTED,
                classification=classification,
                conditions=conditions,
                weight=weight_info,
                guidance=guide,
                bbox=bbox_rounded,
            )
        return DetectResponse(
            client_id=client_id,
            status=DetectionStatus.ALLOWED,
            classification=classification,
            conditions=conditions,
            weight=weight_info,
            bbox=bbox_rounded,
        )

    # ── 허용 (플라스틱/PET/캔/종이) → 상태·무게 조건 검사 ────────────────────────
    conditions = await loop.run_in_executor(
        _executor, inference.run_state, registry.state(), img, bbox, cls
    )
    weight_info.anomaly = (
        settings.WEIGHT_ANOMALY_ENABLED
        and weight_g is not None
        and is_anomaly(cls.value, weight_g, bbox=bbox, img_area=float(img.shape[0] * img.shape[1]))
    )

    guide = guidance.build_guidance(cls, conditions, weight_info.anomaly)
    # 안내가 있으면 조건 불충족 → 재처리 거부, 없으면 충족 → 수거 허용
    status = DetectionStatus.REJECTED if guide else DetectionStatus.ALLOWED

    return DetectResponse(
        client_id=client_id,
        status=status,
        classification=classification,
        conditions=conditions,
        weight=weight_info,
        guidance=guide,
        bbox=bbox_rounded,
    )
