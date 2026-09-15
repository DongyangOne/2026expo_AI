"""NAS Ollama Vision LLM crop classifier.

This is deliberately a crop classifier, not a replacement detector: YOLO still
owns object localisation and NOT_DETECTED.  A malformed, unavailable, or
low-confidence LLM answer is never allowed to fail an API request.
"""

from __future__ import annotations

import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import httpx
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)
_shadow_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-llm-shadow")
_write_lock = Lock()

CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)
CLASS_ID_BY_NAME = {name: index for index, name in enumerate(CLASS_NAMES)}

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "material", "confidence", "has_label", "is_dented",
        "has_foreign_material", "is_single_primary_item",
    ],
    "properties": {
        "material": {"type": "string", "enum": list(CLASS_NAMES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "has_label": {"type": "boolean"},
        "is_dented": {"type": "boolean"},
        "has_foreign_material": {"type": "boolean"},
        "is_single_primary_item": {"type": "boolean"},
    },
}

_PROMPT = """You classify exactly one recycling item in this cropped image.
Choose only one material from: can, pet, paper, plastic, styrofoam, vinyl, glass, battery, fluorescent.
Treat an attached item made of a different material (for example a paper sleeve on a plastic cup) as foreign material.
Do not treat same-material accessories such as a plastic straw as foreign material.
has_label means a removable recycling label is still attached. is_dented means a can or PET bottle is compressed.
Return only JSON matching the supplied schema; do not add explanation."""


@dataclass(frozen=True)
class LocalLLMPrediction:
    class_id: int
    class_name: str
    confidence: float
    has_label: bool
    is_dented: bool
    has_foreign_material: bool
    is_single_primary_item: bool


def enabled() -> bool:
    return (
        settings.LOCAL_LLM_MODE in {"shadow", "primary"}
        and bool(settings.LOCAL_LLM_BASE_URL)
    )


def primary_enabled() -> bool:
    return enabled() and settings.LOCAL_LLM_MODE == "primary"


def _crop_as_jpeg(img: np.ndarray, bbox: list[float]) -> str | None:
    height, width = img.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - box_w * 0.08)); y1 = max(0, int(y1 - box_h * 0.08))
    x2 = min(width, int(x2 + box_w * 0.08)); y2 = min(height, int(y2 + box_h * 0.08))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    longest = max(crop.shape[:2])
    if longest > settings.LOCAL_LLM_MAX_IMAGE_SIDE:
        scale = settings.LOCAL_LLM_MAX_IMAGE_SIDE / longest
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for quality in (90, 80, 70, 60):
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(encoded) <= settings.LOCAL_LLM_MAX_IMAGE_BYTES:
            return base64.b64encode(encoded.tobytes()).decode("ascii")
    return None


def _parse(content: str) -> LocalLLMPrediction:
    parsed: Any = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != set(_SCHEMA["required"]):
        raise ValueError("LLM JSON contract mismatch")
    material = parsed["material"]
    confidence = parsed["confidence"]
    if material not in CLASS_ID_BY_NAME or type(confidence) not in {int, float}:
        raise ValueError("LLM material/confidence invalid")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("LLM confidence out of range")
    flags = ("has_label", "is_dented", "has_foreign_material", "is_single_primary_item")
    if any(type(parsed[name]) is not bool for name in flags):
        raise ValueError("LLM boolean contract invalid")
    return LocalLLMPrediction(
        class_id=CLASS_ID_BY_NAME[material], class_name=material,
        confidence=float(confidence), has_label=parsed["has_label"],
        is_dented=parsed["is_dented"],
        has_foreign_material=parsed["has_foreign_material"],
        is_single_primary_item=parsed["is_single_primary_item"],
    )


def classify(img: np.ndarray, bbox: list[float]) -> LocalLLMPrediction | None:
    """Synchronously query local Ollama; callers run this outside the event loop."""
    if not enabled():
        return None
    image = _crop_as_jpeg(img, bbox)
    if image is None:
        return None
    headers = {"Content-Type": "application/json"}
    if settings.LOCAL_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LOCAL_LLM_API_KEY}"
    body = {
        "model": settings.LOCAL_LLM_MODEL,
        "stream": False,
        "format": _SCHEMA,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": _PROMPT, "images": [image]}],
    }
    url = settings.LOCAL_LLM_BASE_URL.rstrip("/") + "/api/chat"
    try:
        with httpx.Client(timeout=settings.LOCAL_LLM_TIMEOUT_SEC) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
        payload = response.json()
        return _parse(payload["message"]["content"])
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("NAS local LLM crop classification skipped: %s", exc)
        return None


def submit_shadow(
    img: np.ndarray, bbox: list[float], yolo_class_id: int,
    yolo_confidence: float, client_id: str,
) -> None:
    """Best-effort shadow record. It must never affect an API response."""
    if settings.LOCAL_LLM_MODE != "shadow" or not enabled():
        return

    def task() -> None:
        prediction = classify(img, bbox)
        if prediction is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "client_id": client_id,
            "bbox": [round(float(value), 1) for value in bbox],
            "yolo": {"class_id": yolo_class_id, "confidence": round(float(yolo_confidence), 6)},
            "local_llm": prediction.__dict__,
            "material_agreement": prediction.class_id == yolo_class_id,
            "mode": "shadow",
        }
        path = Path(settings.LOCAL_LLM_SHADOW_LOG_PATH)
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    try:
        _shadow_executor.submit(task)
    except RuntimeError:
        logger.warning("local LLM shadow executor is shutting down")


def record_primary(
    *, bbox: list[float], yolo_class_id: int, yolo_confidence: float,
    prediction: LocalLLMPrediction | None, selected: bool, reason: str,
    client_id: str,
) -> None:
    """Persist a primary decision audit record without storing an image or secrets."""
    if settings.LOCAL_LLM_MODE != "primary":
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_id": client_id,
        "bbox": [round(float(value), 1) for value in bbox],
        "yolo": {"class_id": yolo_class_id, "confidence": round(float(yolo_confidence), 6)},
        "local_llm": prediction.__dict__ if prediction else None,
        "selected": selected,
        "reason": reason,
        "mode": "primary",
    }
    path = Path(settings.LOCAL_LLM_SHADOW_LOG_PATH)
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("local LLM primary audit log skipped: %s", exc)


def shutdown() -> None:
    _shadow_executor.shutdown(wait=True)
