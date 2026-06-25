"""
감지 파이프라인 (흐름 제어 전담).

흐름:
  이미지 디코드
    → 주 YOLO 감지 (inference.run_main)
    → status 판정:
        미감지            → NOT_DETECTED
        저신뢰            → GENERAL_WASTE (일반쓰레기)
        거부품목(유리 등)  → REJECTED (완전 거부)
        비닐              → GENERAL_WASTE
        허용품목          → 멀티헤드 상태(inference.run_state) + 무게 검사
                            조건 충족  → ALLOWED
                            조건 불충족 → REJECTED (재처리 guidance)
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
from app.services import guidance, inference
from app.services.weight_check import is_anomaly

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1)

# 주 모델 class_id → WasteClass (학습 시 순서와 일치)
_CLASS_BY_ID: dict[int, WasteClass] = {
    0: WasteClass.CAN,       1: WasteClass.PET,        2: WasteClass.PAPER,
    3: WasteClass.PLASTIC,   4: WasteClass.STYROFOAM,  5: WasteClass.VINYL,
    6: WasteClass.GLASS,     7: WasteClass.BATTERY,    8: WasteClass.FLUORESCENT,
}


def shutdown() -> None:
    _executor.shutdown(wait=True)


async def _read_image(upload: UploadFile) -> np.ndarray:
    raw = await upload.read()
    if not raw:
        raise ValueError("빈 이미지 파일입니다.")
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩에 실패했습니다. 지원 형식: jpg, png")
    return img


async def run(upload: UploadFile, weight_g: Optional[float], registry: ModelRegistry) -> DetectResponse:
    loop = asyncio.get_running_loop()

    img = await _read_image(upload)
    detection = await loop.run_in_executor(_executor, inference.run_main, registry, img)

    # ── 미감지 ──────────────────────────────────────────────────────────────────
    if detection is None:
        return DetectResponse(status=DetectionStatus.NOT_DETECTED, weight=WeightInfo(value_g=weight_g))

    class_id, confidence, bbox = detection
    cls = _CLASS_BY_ID.get(class_id)
    bbox_rounded = [round(v, 1) for v in bbox]
    weight_info = WeightInfo(value_g=weight_g)
    classification = (
        Classification(class_id=class_id, class_name=cls, confidence=round(confidence, 4))
        if cls is not None else None
    )

    # ── 저신뢰 → 일반쓰레기 ──────────────────────────────────────────────────────
    if cls is None or confidence < settings.TRUST_CONF:
        return DetectResponse(
            status=DetectionStatus.GENERAL_WASTE,
            classification=classification,
            weight=weight_info,
            general=guidance.build_general(GeneralWasteCode.LOW_CONFIDENCE),
            bbox=bbox_rounded,
        )

    # ── 완전 거부 (유리/건전지/형광등/스티로폼) ──────────────────────────────────
    if guidance.is_rejected(cls):
        return DetectResponse(
            status=DetectionStatus.REJECTED,
            classification=classification,
            weight=weight_info,
            rejection=guidance.build_rejection(cls),
            bbox=bbox_rounded,
        )

    # ── 일반쓰레기 (비닐) ────────────────────────────────────────────────────────
    if guidance.is_general(cls):
        return DetectResponse(
            status=DetectionStatus.GENERAL_WASTE,
            classification=classification,
            weight=weight_info,
            general=guidance.build_general(GeneralWasteCode.VINYL),
            bbox=bbox_rounded,
        )

    # ── 허용 (페트/플라스틱/캔/종이) → 상태·무게 조건 검사 ────────────────────────
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
        status=status,
        classification=classification,
        conditions=conditions,
        weight=weight_info,
        guidance=guide,
        bbox=bbox_rounded,
    )
