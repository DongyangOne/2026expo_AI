"""
모델 추론 실행 — 전처리 + 세션 호출만 담당.

흐름 제어(pipeline)와 도메인 해석(class_id→WasteClass, 허용/거부 판정)은 호출 측이 맡는다.
전처리 상수는 extract_crops.py / train_classifier.py 와 동일해야 학습-추론이 일치한다.
"""

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


def run_main(registry: ModelRegistry, img: np.ndarray):
    """주 9-class YOLO. Returns (class_id, conf, [x1,y1,x2,y2]) | None."""
    results = registry.main()(
        img,
        imgsz=settings.IMG_SIZE,
        conf=settings.DETECT_CONF,
        device=settings.DEVICE,
        verbose=False,
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = int(boxes.conf.argmax())
    return int(boxes.cls[best]), float(boxes.conf[best]), boxes.xyxy[best].tolist()


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


def run_state(session, img: np.ndarray, bbox: list[float], cls: WasteClass) -> Conditions:
    """
    bbox 크롭 → 멀티헤드 추론 → Conditions.
    세션 미탑재(None) 또는 헤드 비대상 품목은 해당 값 None.
    기존 모델은 dent/label 2헤드이며, foreign_material 출력이 있는 모델은 자동 사용한다.
    """
    if session is None:
        return Conditions()

    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * _PAD; y1 -= bh * _PAD
    x2 += bw * _PAD; y2 += bh * _PAD
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 <= x1 or y2 <= y1:
        return Conditions()

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
    return Conditions(
        is_dented=is_dented,
        has_label=has_label,
        has_foreign_material=has_foreign_material,
    )
