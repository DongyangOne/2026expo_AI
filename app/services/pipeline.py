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
from app.schemas.response import Classification, Conditions, DetectResponse, WeightInfo
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
_PET_MODEL_CLASS_ID = 1
_VINYL_MODEL_CLASS_ID = 5


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _find_vinyl_candidate(
    class_id: int,
    confidence: float,
    bbox: list[float],
    candidates: list[tuple[int, float, list[float]]],
) -> tuple[int, float, list[float]] | None:
    """저신뢰 PET/PLASTIC과 같은 물체를 가리키는 VINYL 보조 후보를 찾는다."""
    if (
        not settings.VINYL_CORRECTION_ENABLED
        or class_id not in {_PET_MODEL_CLASS_ID, _PLASTIC_CLASS_ID}
        or confidence >= settings.TRUST_CONF
    ):
        return None

    matches = [
        candidate
        for candidate in candidates
        if candidate[0] == _VINYL_MODEL_CLASS_ID
        and candidate[1] >= settings.VINYL_CANDIDATE_CONF
        and candidate[1] >= confidence * settings.VINYL_CANDIDATE_RATIO
        and _bbox_iou(bbox, candidate[2]) >= settings.VINYL_CANDIDATE_IOU
    ]
    return max(matches, key=lambda candidate: candidate[1]) if matches else None


def _verifier_supports_vinyl(prediction: dict | None, yolo_confidence: float) -> bool:
    if prediction is None:
        return False
    material = prediction.get("material") or {}
    verifier_confidence = float(material.get("confidence", 0.0))
    return (
        material.get("class_id") == _VINYL_MODEL_CLASS_ID
        and verifier_confidence >= settings.VINYL_VERIFIER_CONF
        and verifier_confidence - yolo_confidence >= settings.VINYL_VERIFIER_MARGIN
    )


def _apply_verifier_label(conditions: Conditions, prediction: dict | None) -> Conditions:
    """라벨 판정만 검증기 헤드로 대체한다.

    같은 하드웨어 crop에서 구형 state 모델은 라벨을 놓치는 쪽으로 틀렸고
    검증기 헤드는 틀리지 않았다. dent는 하드웨어 정답에 양성 표본이 없어 그대로 둔다.
    """
    if conditions.has_label is None or prediction is None:
        return conditions
    head = (prediction.get("heads") or {}).get("label")
    if head is None:
        return conditions
    return Conditions(has_label=bool(head["value"]), is_dented=conditions.is_dented)


def _response_class_id(model_class_id: int | None) -> int | None:
    """PET는 응답 계약에서 PLASTIC과 같은 통이므로 모델 비교도 같은 기준으로 한다."""
    if model_class_id is None:
        return None
    return _PLASTIC_CLASS_ID if model_class_id == _PET_MODEL_CLASS_ID else model_class_id


def _materials_disagree(model_class_id: int, prediction: dict | None) -> bool:
    """검증기 결과가 없으면 판단 근거가 없으므로 불일치로 보지 않는다."""
    if prediction is None:
        return False
    material = prediction.get("material") or {}
    return _response_class_id(material.get("class_id")) != _response_class_id(model_class_id)


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

    if len(detection) == 3:
        class_id, confidence, bbox = detection
        candidates: list[tuple[int, float, list[float]]] = []
    else:
        class_id, confidence, bbox, candidates = detection

    yolo_class_id = class_id
    yolo_confidence = confidence
    verifier_session = (
        registry.verifier() if hasattr(registry, "verifier") else None
    )
    vinyl_candidate = _find_vinyl_candidate(
        class_id, confidence, bbox, candidates
    )
    verifier_prediction = None
    correction_applied = False
    if (
        verifier_session is not None
        and vinyl_candidate is not None
        and not inference.verifier_is_shadow_only(verifier_session)
    ):
        verifier_prediction = await loop.run_in_executor(
            _executor, inference.run_verifier, verifier_session, img, bbox
        )
        if _verifier_supports_vinyl(verifier_prediction, confidence):
            verifier_confidence = float(
                verifier_prediction["material"]["confidence"]
            )
            logger.info(
                "저신뢰 %s(%.4f)를 vinyl(%.4f)로 교정: client_id=%s",
                _CLASS_BY_ID.get(class_id),
                confidence,
                verifier_confidence,
                client_id,
            )
            class_id = _VINYL_MODEL_CLASS_ID
            confidence = verifier_confidence
            correction_applied = True

    # ── 저신뢰 구제 — 일반쓰레기로 버려질 건을 검증기가 확신하면 되살린다 ─────────────
    if (
        settings.VERIFIER_RESCUE_ENABLED
        and not correction_applied
        and confidence < settings.TRUST_CONF
        and verifier_session is not None
        and not inference.verifier_is_shadow_only(verifier_session)
    ):
        if verifier_prediction is None:
            verifier_prediction = await loop.run_in_executor(
                _executor, inference.run_verifier, verifier_session, img, bbox
            )
        material = (verifier_prediction or {}).get("material") or {}
        rescued_id = material.get("class_id")
        rescued_confidence = float(material.get("confidence", 0.0))
        if (
            rescued_id in _CLASS_BY_ID
            and rescued_confidence >= settings.VERIFIER_RESCUE_CONF
        ):
            logger.info(
                "저신뢰 %s(%.4f)를 검증기 %s(%.4f)로 구제: client_id=%s",
                _CLASS_BY_ID.get(class_id),
                confidence,
                _CLASS_BY_ID.get(rescued_id),
                rescued_confidence,
                client_id,
            )
            class_id = rescued_id
            confidence = rescued_confidence
            correction_applied = True

    # ── 라벨 판정용 검증기 확보 — 고신뢰 PET/플라스틱도 label 헤드가 필요하다 ────────
    if (
        settings.VERIFIER_LABEL_HEAD_ENABLED
        and verifier_prediction is None
        and class_id in {_PET_MODEL_CLASS_ID, _PLASTIC_CLASS_ID}
        and verifier_session is not None
        and not inference.verifier_is_shadow_only(verifier_session)
    ):
        verifier_prediction = await loop.run_in_executor(
            _executor, inference.run_verifier, verifier_session, img, bbox
        )

    # ── 합의 게이트 — 두 모델이 갈리면 어느 쪽도 확정하지 않는다 ────────────────────
    gate_defers = False
    if (
        settings.VERIFIER_AGREEMENT_GATE_ENABLED
        and verifier_session is not None
        and not inference.verifier_is_shadow_only(verifier_session)
    ):
        if verifier_prediction is None:
            verifier_prediction = await loop.run_in_executor(
                _executor, inference.run_verifier, verifier_session, img, bbox
            )
        gate_defers = _materials_disagree(class_id, verifier_prediction)
        if gate_defers:
            logger.info(
                "합의 실패로 확정 보류: yolo=%s(%.4f) verifier=%s client_id=%s",
                _CLASS_BY_ID.get(class_id),
                confidence,
                (verifier_prediction.get("material") or {}).get("class_name"),
                client_id,
            )

    if verifier_prediction is None:
        verifier_shadow.submit(
            verifier_session, img, bbox, yolo_class_id, yolo_confidence, client_id
        )
    else:
        verifier_shadow.submit_precomputed(
            verifier_session,
            img,
            bbox,
            yolo_class_id,
            yolo_confidence,
            client_id,
            verifier_prediction,
            correction_applied,
        )
    cls = _CLASS_BY_ID.get(class_id)
    bbox_rounded = [round(v, 1) for v in bbox]
    weight_info = WeightInfo(value_g=weight_g)
    classification = _build_classification(class_id, cls, confidence)

    # ── 저신뢰 또는 합의 실패 → 일반쓰레기 ───────────────────────────────────────
    if cls is None or confidence < settings.TRUST_CONF or gate_defers:
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
        state_prediction = await loop.run_in_executor(
            _executor, inference.run_state, registry.state(), img, bbox, cls
        )
        conditions = state_prediction.conditions
        weight_info.anomaly = (
            settings.WEIGHT_ANOMALY_ENABLED
            and weight_g is not None
            and is_anomaly(cls.value, weight_g, bbox=bbox, img_area=float(img.shape[0] * img.shape[1]))
        )
        guide = guidance.build_guidance(
            cls,
            conditions,
            weight_info.anomaly,
            state_prediction.has_foreign_material,
        )
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
    state_prediction = await loop.run_in_executor(
        _executor, inference.run_state, registry.state(), img, bbox, cls
    )
    conditions = state_prediction.conditions
    if settings.VERIFIER_LABEL_HEAD_ENABLED:
        conditions = _apply_verifier_label(conditions, verifier_prediction)
    weight_info.anomaly = (
        settings.WEIGHT_ANOMALY_ENABLED
        and weight_g is not None
        and is_anomaly(cls.value, weight_g, bbox=bbox, img_area=float(img.shape[0] * img.shape[1]))
    )

    guide = guidance.build_guidance(
        cls,
        conditions,
        weight_info.anomaly,
        state_prediction.has_foreign_material,
    )
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
