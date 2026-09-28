"""NAS Ollama Vision LLM classifier used after YOLO localisation.

YOLO normally supplies the crop.  When YOLO finds no box, the caller may send
the full frame as a conservative fallback.  A malformed, unavailable, or
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
    "vinyl", "glass", "battery", "fluorescent", "general_waste",
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

_PROMPT = """Act as the final visual inspector for a Korean smart waste-sorting bin.
Classify the ONE primary disposal item by its physical material and form, not by the product printed on it.

Image usage:
- Image 1 is the complete camera frame and provides overall shape and scene context.
- Image 2, when present, is the padded YOLO crop and provides surface and material detail.
- The crop is authoritative for which item to inspect. Use the full frame only to recover its complete shape and context.
- Ignore the bin tray, floor, background, shadows, hands, fixtures, cables, and objects that are not attached to the primary item.

Choose exactly one material:
- can: metal beverage or food can.
- pet: a lightweight molded plastic bottle with a bottle neck, including transparent, colored, tinted, or opaque PET-style bottles. Do not use for rigid cups, lids, trays, or non-bottle containers. A dark or amber bottle is not glass merely because its color hides transparency; require clear glass cues such as a thick rigid wall/base and glass-like reflections before choosing glass.
- paper: paper, cardboard, carton, paper cup, or a loose paper cup sleeve/holder. A matte opaque disposable drink cup with a rolled paper rim or side seam is paper unless there is clear rigid-polymer evidence.
- plastic: rigid molded polymer bottle (other than PET beverage bottle), cup, lid, tray, tub, squeeze tube, or container. Printed paper or film wrapped around a rigid molded cup/container is a label or contaminant; it does not make the primary item paper. A printed flexible squeeze tube is plastic, not paper.
- styrofoam: expanded-polystyrene foam, including white or black lightweight molded foam food trays and packaging. The surface may look smooth and the foam beads may be subtle, especially on dark trays; an embossed material mark and molded ribs can be stronger evidence than visible beads.
- vinyl: thin flexible film, bag, wrapper, or pouch that bends, folds, wrinkles, crumples, or has heat-sealed edges. Printed snack, food, or candy packaging remains vinyl when its body is a flexible film pouch or wrapper; printing does not make it paper.
- glass: an item whose body is clearly glass, such as a thick rigid glass bottle, jar, or glass object. Do not choose glass for a lightweight molded plastic/PET bottle only because it is glossy, transparent, or amber-colored.
- battery: household battery or battery pack.
- fluorescent: any discarded electric lamp or light bulb handled by the bin's lamp-rejection route, including a fluorescent tube, compact fluorescent lamp, globe bulb, LED bulb, or incandescent-style bulb. This product taxonomy intentionally groups bulb-shaped lamps together even when the visible diffuser resembles glass.
- general_waste: a loose straw, used tissue, food waste, hygiene waste, or another ordinary non-recyclable item outside the nine recyclable classes.

State rules:
- A cafe cup is plastic only after its loose straw and paper sleeve/holder are removed. A loose straw is general_waste; a loose sleeve/holder is paper.
- For the cup itself, distinguish material: an opaque matte cup with a rolled paper rim or paper seam is paper; a clearly translucent, glossy, injection-molded cup is plastic.
- has_foreign_material=true only when a different material is physically attached to, inside, or mixed with the primary item. Background objects are not foreign material.
- has_label=true only when a removable recycling label remains attached to a plastic/PET container.
- is_dented=true only when a can or PET beverage bottle is visibly compressed enough for disposal.
- is_single_primary_item=false when multiple separate disposal items are presented together.
- If visual evidence is genuinely ambiguous, lower confidence instead of defaulting to plastic or paper.

Return only one JSON object matching the supplied schema. Do not add Markdown or explanation."""

_VINYL_PLASTIC_PROMPT = """Re-evaluate only the primary disposal item as either vinyl or plastic.
Image 1 is the complete camera frame. Image 2, when present, is the padded YOLO crop and identifies the target item.
Choose vinyl for thin flexible film, a bag, wrapper, or pouch that bends, folds, wrinkles, or crumples.
Choose plastic only for a rigid bottle, cup, lid, tray, tub, or container that keeps its shape.
Judge physical flexibility and three-dimensional form, not color, transparency, printed branding, or the fact that both materials are polymers.
Ignore the bin tray, floor, background, shadows, hands, fixtures, and anything not attached to the target.
Set has_foreign_material=true only if another material is physically attached to, inside, or mixed with the primary item.
If the evidence is ambiguous, lower confidence rather than defaulting to plastic.
Return only one JSON object matching the supplied schema. Do not add Markdown or explanation."""

_REJECTION_CONFLICT_PROMPT = """Re-evaluate the ONE primary disposal item because two visual classifiers disagree.
Choose exactly one of these two materials: {yolo_material} or {llm_material}.
Do not blindly trust either prior classifier; inspect the complete frame and padded target crop.
Safety-specific visual rules:
- fluorescent means any complete discarded electric lamp or bulb routed to lamp rejection, including globe, LED, incandescent-style, compact fluorescent, or tube lamps. A smooth round diffuser does not make a complete lamp plastic.
- styrofoam includes lightweight molded expanded-polystyrene food trays, including smooth black trays with subtle beads, molded ribs, or embossed material marks.
- glass requires a thick rigid glass wall or base and glass-like reflections; glossy, transparent, tinted, or amber plastic alone is not glass.
- battery requires visible battery-cell, terminal, pack, or battery-label evidence.
- plastic is an ordinary rigid molded polymer item and must not be chosen for a complete lamp or molded foam tray.
Ignore hands, the bin, fixtures, cables, floor, and background. If uncertain, lower confidence.
Return only one JSON object matching the supplied schema. Do not add Markdown or explanation."""

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


def _encode_as_jpeg(img: np.ndarray, max_side: int) -> str | None:
    if img.size == 0:
        return None
    longest = max(img.shape[:2])
    if longest > max_side:
        scale = max_side / longest
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for quality in (90, 80, 70, 60):
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(encoded) <= settings.LOCAL_LLM_MAX_IMAGE_BYTES:
            return base64.b64encode(encoded.tobytes()).decode("ascii")
    return None


def _crop_as_jpeg(img: np.ndarray, bbox: list[float]) -> str | None:
    height, width = img.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - box_w * 0.10)); y1 = max(0, int(y1 - box_h * 0.10))
    x2 = min(width, int(x2 + box_w * 0.10)); y2 = min(height, int(y2 + box_h * 0.10))
    if x2 <= x1 or y2 <= y1:
        return None
    return _encode_as_jpeg(img[y1:y2, x1:x2], settings.LOCAL_LLM_MAX_IMAGE_SIDE)


def _images_as_jpeg(
    img: np.ndarray, bbox: list[float], *, crop_only: bool = False,
) -> list[str]:
    """Return full-frame context followed by the target crop when it is distinct."""
    crop = _crop_as_jpeg(img, bbox)
    if crop is None:
        return []
    if crop_only:
        return [crop]

    full_frame = _encode_as_jpeg(img, settings.LOCAL_LLM_FULL_IMAGE_SIDE)
    if full_frame is None:
        return []

    height, width = img.shape[:2]
    x1, y1, x2, y2 = bbox
    covers_frame = (
        x1 <= width * 0.02 and y1 <= height * 0.02
        and x2 >= width * 0.98 and y2 >= height * 0.98
    )
    return [full_frame] if covers_frame else [full_frame, crop]


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


def _classify_with_prompt(
    img: np.ndarray, bbox: list[float], prompt: str, *, crop_only: bool = False,
) -> LocalLLMPrediction | None:
    """Synchronously query local Ollama; callers run this outside the event loop."""
    if not enabled():
        return None
    images = _images_as_jpeg(img, bbox, crop_only=crop_only)
    if not images:
        return None
    headers = {"Content-Type": "application/json"}
    if settings.LOCAL_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LOCAL_LLM_API_KEY}"
    body = {
        "model": settings.LOCAL_LLM_MODEL,
        "stream": False,
        # 매 요청마다 모델을 다시 올리면 NAS의 cold start가 하드웨어 HTTP timeout을 유발한다.
        "keep_alive": settings.LOCAL_LLM_KEEP_ALIVE,
        # Qwen의 reasoning 텍스트는 이 엄격한 JSON 계약에 필요하지 않다. 이를 끄고
        # 출력 상한을 두어 NAS GPU를 오래 점유하거나 HTTP timeout에 빠지지 않게 한다.
        "think": False,
        "format": _SCHEMA,
        "options": {"temperature": 0, "num_predict": settings.LOCAL_LLM_MAX_TOKENS},
        "messages": [{"role": "user", "content": prompt, "images": images}],
    }
    url = settings.LOCAL_LLM_BASE_URL.rstrip("/") + "/api/chat"
    try:
        with httpx.Client(timeout=settings.LOCAL_LLM_TIMEOUT_SEC) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
        payload = response.json()
        return _parse(payload["message"]["content"])
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("NAS local LLM classification skipped: %s", exc)
        return None


def classify(img: np.ndarray, bbox: list[float]) -> LocalLLMPrediction | None:
    """Classify a YOLO crop or a full-frame fallback."""
    return _classify_with_prompt(img, bbox, _PROMPT)


def reclassify_vinyl_plastic(
    img: np.ndarray, bbox: list[float],
) -> LocalLLMPrediction | None:
    """Resolve only the flexible-vinyl versus rigid-plastic ambiguity."""
    prediction = _classify_with_prompt(img, bbox, _VINYL_PLASTIC_PROMPT)
    if prediction is not None and prediction.class_name in {"vinyl", "plastic"}:
        return prediction
    if prediction is not None:
        logger.warning(
            "NAS local LLM vinyl/plastic recheck returned invalid material: %s",
            prediction.class_name,
        )
    return None


def reclassify_rejection_conflict(
    img: np.ndarray,
    bbox: list[float],
    yolo_material: str,
    llm_material: str,
) -> LocalLLMPrediction | None:
    """Adjudicate a high-confidence YOLO rejection versus a benign LLM result."""
    if yolo_material not in CLASS_ID_BY_NAME or llm_material not in CLASS_ID_BY_NAME:
        return None
    prompt = _REJECTION_CONFLICT_PROMPT.format(
        yolo_material=yolo_material,
        llm_material=llm_material,
    )
    prediction = _classify_with_prompt(img, bbox, prompt)
    allowed = {yolo_material, llm_material}
    if prediction is not None and prediction.class_name in allowed:
        return prediction
    if prediction is not None:
        logger.warning(
            "NAS local LLM rejection-conflict recheck returned invalid material: %s",
            prediction.class_name,
        )
    return None


def recheck_foreign_material(
    img: np.ndarray, bbox: list[float], material: str,
) -> LocalLLMPrediction | None:
    """Recheck a possible contaminant using only the target crop.

    Full-frame context helps material recognition but can make a fixed camera
    fixture look attached to the item.  This second pass is intentionally
    conditional and crop-only, so normal requests do not pay its latency.
    Reusing the primary prompt avoids biasing the model toward finding a
    contaminant merely because a recheck was requested.
    """
    if material not in CLASS_ID_BY_NAME:
        return None
    prediction = _classify_with_prompt(
        img,
        bbox,
        _PROMPT,
        crop_only=True,
    )
    if prediction is not None and prediction.class_name == material:
        return prediction
    if prediction is not None:
        logger.warning(
            "NAS local LLM foreign-material recheck changed material: %s -> %s",
            material,
            prediction.class_name,
        )
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
    *, bbox: list[float], yolo_class_id: int | None, yolo_confidence: float | None,
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
        "yolo": (
            {"class_id": yolo_class_id, "confidence": round(float(yolo_confidence), 6)}
            if yolo_class_id is not None and yolo_confidence is not None else None
        ),
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
