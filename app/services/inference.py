"""
모델 추론 실행 — 전처리 + 세션 호출만 담당.

흐름 제어(pipeline)와 도메인 해석(class_id→WasteClass, 허용/거부 판정)은 호출 측이 맡는다.
전처리 상수는 extract_crops.py / train_classifier.py 와 동일해야 학습-추론이 일치한다.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import settings
from app.models.registry import ModelRegistry
from app.schemas.enums import WasteClass
from app.schemas.response import Conditions

# 멀티헤드 전처리 상수 (학습 파이프라인과 동일)
_STATE_SIZE = 224
_PAD = 0.12
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 멀티헤드 헤드 대상 품목 (모델 구조 지식)
_DENT_CLASSES = (WasteClass.PET, WasteClass.CAN)        # 압착 검사
_LABEL_CLASSES = (WasteClass.PET, WasteClass.PLASTIC)   # 라벨 검사
VERIFIER_CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)


@dataclass(frozen=True)
class StatePrediction:
    """상태 모델 내부 결과.

    Spring 외부 DTO에는 ``has_foreign_material`` 필드가 없으므로 해당 값은
    안내 코드를 만드는 동안에만 보존한다.
    """

    conditions: Conditions
    has_foreign_material: bool | None = None


def run_main(registry: ModelRegistry, img: np.ndarray):
    """
    주 9-class YOLO.

    최종 감지는 기존 ``DETECT_CONF`` 이상만 인정한다. 다만 저신뢰
    PET/PLASTIC↔VINYL 혼동 교정에 쓸 같은 bbox의 보조 후보를 보존하기 위해 모델에는
    더 낮은 ``VINYL_CANDIDATE_CONF``를 전달한다.

    Returns:
        (class_id, confidence, bbox, candidates) | None
        candidates: [(class_id, confidence, bbox), ...]
    """
    candidate_conf = min(settings.DETECT_CONF, settings.VINYL_CANDIDATE_CONF)
    results = registry.main()(
        img,
        imgsz=settings.IMG_SIZE,
        conf=candidate_conf,
        device=settings.DEVICE,
        verbose=False,
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    candidates = [
        (int(boxes.cls[index]), float(boxes.conf[index]), boxes.xyxy[index].tolist())
        for index in range(len(boxes))
    ]
    eligible = [
        index for index, (_, confidence, _) in enumerate(candidates)
        if confidence >= settings.DETECT_CONF
    ]
    if not eligible:
        return None
    best = max(eligible, key=lambda index: candidates[index][1])
    class_id, confidence, bbox = candidates[best]
    return class_id, confidence, bbox, candidates


def _letterbox(crop: np.ndarray, size: int = _STATE_SIZE) -> np.ndarray:
    """비율 유지 resize + 회색(114) 패딩으로 정사각. (extract_crops 와 동일)"""
    h, w = crop.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = r
    return canvas


def run_state(session, img: np.ndarray, bbox: list[float], cls: WasteClass) -> StatePrediction:
    """
    bbox 크롭 → 멀티헤드 추론 → 내부 StatePrediction.
    세션 미탑재(None) 또는 헤드 비대상 품목은 해당 값 None.
    기존 모델은 dent/label 2헤드이며, foreign_material 출력이 있는 모델은 자동 사용한다.
    """
    if session is None:
        return StatePrediction(Conditions())

    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * _PAD; y1 -= bh * _PAD
    x2 += bw * _PAD; y2 += bh * _PAD
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 <= x1 or y2 <= y1:
        return StatePrediction(Conditions())

    crop = _letterbox(img[y1:y2, x1:x2])
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    arr = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None]  # (1,3,224,224)

    output_names = {output.name for output in session.get_outputs()}
    requested_outputs = ["dent", "label"]
    if "foreign_material" in output_names:
        requested_outputs.append("foreign_material")
    output_values = session.run(requested_outputs, {"img": arr})
    outputs = dict(zip(requested_outputs, output_values))

    dent_out = outputs["dent"]
    label_out = outputs["label"]
    is_dented = bool(dent_out[0].argmax()) if cls in _DENT_CLASSES else None
    has_label = bool(label_out[0].argmax()) if cls in _LABEL_CLASSES else None
    foreign_out = outputs.get("foreign_material")
    has_foreign_material = bool(foreign_out[0].argmax()) if foreign_out is not None else None
    return StatePrediction(
        conditions=Conditions(is_dented=is_dented, has_label=has_label),
        has_foreign_material=has_foreign_material,
    )


def _confidence(logits: np.ndarray) -> tuple[int, float]:
    values = np.asarray(logits[0], dtype=np.float64)
    shifted = values - values.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    class_id = int(probabilities.argmax())
    return class_id, float(probabilities[class_id])


def run_verifier(session, img: np.ndarray, bbox: list[float]) -> dict | None:
    """YOLO bbox를 임시 320px 검증기로 재판정한다. 운영 응답에는 직접 사용하지 않는다."""
    if session is None:
        return None

    height, width = img.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    padding = 0.08
    x1 = max(0, int(x1 - box_w * padding))
    y1 = max(0, int(y1 - box_h * padding))
    x2 = min(width, int(x2 + box_w * padding))
    y2 = min(height, int(y2 + box_h * padding))
    if x2 <= x1 or y2 <= y1:
        return None

    model_input = session.get_inputs()[0]
    input_size = model_input.shape[-1]
    if not isinstance(input_size, int):
        input_size = 320
    crop = _letterbox(img[y1:y2, x1:x2], size=input_size)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    array = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None]

    output_names = ("material", "dent", "label", "foreign_material")
    available = {output.name for output in session.get_outputs()}
    missing = set(output_names) - available
    if missing:
        raise RuntimeError(f"crop verifier outputs missing: {sorted(missing)}")
    values = session.run(list(output_names), {model_input.name: array})
    predictions = {
        name: _confidence(value) for name, value in zip(output_names, values)
    }
    material_id, material_confidence = predictions["material"]
    return {
        "material": {
            "class_id": material_id,
            "class_name": VERIFIER_CLASS_NAMES[material_id],
            "confidence": round(material_confidence, 6),
        },
        "heads": {
            name: {"value": bool(class_id), "confidence": round(confidence, 6)}
            for name, (class_id, confidence) in predictions.items()
            if name != "material"
        },
        "input_size": input_size,
    }
