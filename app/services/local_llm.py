"""NAS Ollama Vision LLM classifier used after YOLO localisation.

YOLO normally supplies the crop.  When YOLO finds no box, the caller may send
the full frame as a conservative fallback.  A malformed, unavailable, or
low-confidence LLM answer is never allowed to fail an API request.
"""

from __future__ import annotations

import base64
import json
import logging
import time
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
_client_lock = Lock()
_http_client: httpx.Client | None = None

CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent", "general_waste",
)
CLASS_ID_BY_NAME = {name: index for index, name in enumerate(CLASS_NAMES)}

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "material", "confidence", "has_label", "compression_required", "is_dented",
        "has_foreign_material", "is_single_primary_item",
    ],
    "properties": {
        "material": {"type": "string", "enum": list(CLASS_NAMES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "has_label": {"type": "boolean"},
        "compression_required": {"type": "boolean"},
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
- has_foreign_material=true when a removable straw or sleeve is attached, or when a different material is physically attached to, inside, or mixed with the primary item. A plastic straw attached to a plastic cup is still removable foreign material. Background objects are not foreign material.
- has_label=true only when a removable recycling label remains attached to a plastic/PET container.
- compression_required=true for every can and PET beverage bottle. For ordinary plastic, use true only for a thin-walled hollow plastic bottle or container that a person can safely flatten by hand.
- compression_required=false for cafe takeaway cups, disposable drink cups, rigid cups, lids, trays, tubs, thick storage containers, small plastic parts, and hard or brittle plastic that is difficult or unsafe to flatten by hand.
- is_dented=true only when compression_required=true and the item is visibly flattened or crushed enough to reduce its volume. Otherwise set it to false.
- is_single_primary_item=false when multiple separate disposal items are presented together.
- If visual evidence is genuinely ambiguous, lower confidence instead of defaulting to plastic or paper.

Return only one JSON object matching the supplied schema. Do not add Markdown or explanation."""

_FORM_REASONING_GUIDANCE = """Decision order:
1. Identify the complete object's form first: bottle, cup, tray, wrapper, can, battery, or lamp.
2. Inspect observable material cues: wall thickness, molded seams, rolled or crimped rim, foam cells, paper fibers, film wrinkles, heat seals, glass refraction, and electrical base or socket.
3. Choose the class that best explains the complete object, not just its most visually prominent surface.
"""

_FORM_REASONING_PROMPT = _PROMPT.replace(
    "Choose exactly one material:",
    _FORM_REASONING_GUIDANCE + "\nChoose exactly one material:",
)

_VINYL_PLASTIC_PROMPT = """Re-evaluate only the primary disposal item as either vinyl or plastic.
Image 1 is the complete camera frame. Image 2, when present, is the padded YOLO crop and identifies the target item.
Choose vinyl for thin flexible film, a bag, wrapper, or pouch that bends, folds, wrinkles, or crumples.
Choose plastic only for a rigid bottle, cup, lid, tray, tub, or container that keeps its shape.
Judge physical flexibility and three-dimensional form, not color, transparency, printed branding, or the fact that both materials are polymers.
This is form classification, not polymer chemistry: a loose translucent shopping or packaging bag, or a film sheet draped over any support, MUST be vinyl and never plastic; ignore the support underneath it.
Ignore the bin tray, floor, background, shadows, hands, fixtures, and anything not attached to the target.
Set has_foreign_material=true if a removable straw or sleeve is attached to the item, or if another material is physically attached to, inside, or mixed with it. A plastic straw attached to a plastic cup still counts as removable foreign material.
Set compression_required=true only for a thin-walled hollow plastic bottle or container that can safely be flattened by hand. Cafe takeaway cups, rigid cups, lids, trays, tubs, and hard or brittle plastic are not compression-required.
Set is_dented=true only when compression_required=true and the item is visibly flattened enough to reduce its volume.
If the evidence is ambiguous, lower confidence rather than defaulting to plastic.
Return only one JSON object matching the supplied schema. Do not add Markdown or explanation."""

_PLASTIC_MATERIAL_PROMPT = """Act as a material specialist for one disposal item.
Choose exactly one: plastic, paper, or styrofoam.

Use physical construction rather than color or product type:
- plastic includes dense rigid injection-molded or thermoformed cups, tubs, divided food trays, inserts, bottles, plant pots, and packaging. Thin rigid walls, continuous molded rims, ribs, cavities, seams, and PP/PET/PE/PS recycling marks support plastic. A white opaque object is not paper or foam merely because it is white.
- styrofoam means expanded or foamed polystyrene. Require visible bead/cell/porous texture, a foam fracture, unusually thick lightweight foam walls, or an EPS/PSP/foamed-PS mark. A smooth dense molded tray or insert without foam evidence is plastic.
- paper requires fibrous, layered, folded, rolled-rim, glued-seam, cardboard, or torn-paper evidence. Printing or a matte white surface alone does not make an item paper.

Image 1 is the full frame; Image 2, when present, is the target crop. Ignore the ground and background. A removable straw or sleeve attached to the item counts as foreign material even when the straw and cup are both plastic. If uncertain, lower confidence rather than guessing from color.
Set compression_required=true only for a thin-walled hollow plastic bottle or container that can safely be flattened by hand. Cafe takeaway cups, rigid cups, lids, trays, tubs, and hard or brittle plastic are not compression-required. Set is_dented=true only when compression_required=true and the item is visibly flattened enough to reduce its volume.
Return only one JSON object matching the supplied schema. Do not add Markdown or explanation."""

_REJECTION_CONFLICT_PROMPT = """Re-evaluate the ONE primary disposal item because two visual classifiers disagree.
Choose exactly one of these two materials: {yolo_material} or {llm_material}.
Do not blindly trust either prior classifier; inspect the target crop and use manufacturing evidence, not color.
Pair-specific evidence:
{pair_rules}
Safety-specific visual rules:
- fluorescent means any complete discarded electric lamp or bulb routed to lamp rejection, including globe, LED, incandescent-style, compact fluorescent, or tube lamps. A smooth round diffuser does not make a complete lamp plastic.
- styrofoam includes lightweight molded expanded-polystyrene food trays, including smooth black trays with subtle beads, molded ribs, or embossed material marks. Embossed PSP, EPS, PS, or foamed-polystyrene markings are strong styrofoam evidence; PSP specifically means foamed polystyrene.
- pet includes lightweight molded bottles with thin walls, a threaded neck or collar, molded shoulders or seams, even when opaque, tinted, or amber. Dark brown color does not imply glass.
- glass requires decisive evidence such as a thick heavy wall or base and glass-like refraction; gloss, transparency, tint, or amber color alone is not glass.
- battery requires visible battery-cell, terminal, pack, or battery-label evidence.
- plastic is an ordinary dense rigid molded polymer item and must not be chosen for a complete lamp or a molded foam tray with foam or PSP/EPS evidence.
Set compression_required=true for can and PET. For plastic, use true only for a thin-walled hollow bottle or container that can safely be flattened by hand; cafe takeaway cups, rigid cups, lids, trays, tubs, and hard or brittle plastic use false. Other materials also use false. Set is_dented=true only when compression_required=true and the item is visibly flattened enough to reduce its volume.
Ignore hands, the bin, fixtures, cables, floor, and background. A removable straw or sleeve attached to the target counts as foreign material even when it is made from the same material as the target. If uncertain, lower confidence.
Return only one JSON object matching the supplied schema. Do not add Markdown or explanation."""

_PAIR_RULES = {
    frozenset(("can", "plastic")): (
        "A metal can or tin has a rolled/crimped metal rim, stamped lid or base, "
        "metal seam, or metallic reflection. A shallow printed food tin remains can "
        "even when only its circular lid is prominent. Plastic has a molded polymer rim."
    ),
    frozenset(("plastic", "paper")): (
        "Plastic cups and containers have a continuous molded wall, injection-molded rim, "
        "polymer sheen, or one-piece base. Paper has a rolled paper lip, glued side seam, "
        "fibrous/torn edge, or layered cardboard construction. Printing alone proves neither."
    ),
    frozenset(("plastic", "styrofoam")): (
        "Styrofoam/PSP/EPS is visibly foamed: porous or cellular texture, thick lightweight "
        "walls, bead structure, foam fracture, or an explicit PSP/EPS/foamed-PS mark. "
        "A smooth thin dense molded tray without foam evidence is plastic, even when white or black."
    ),
    frozenset(("vinyl", "paper")): (
        "Vinyl film shows heat-sealed edges, crinkles, flexible folds, stretched highlights, "
        "or a thin laminated wrapper body. Paper shows fibers, a tear edge, stiffness, creases "
        "that hold shape, or a paper seam. Printed graphics do not make flexible film paper."
    ),
    frozenset(("glass", "plastic")): (
        "Glass needs a thick rigid base or wall, glass refraction, sharp specular highlights, "
        "or a heavy bottle/jar construction. Plastic needs thin molded walls, squeeze deformation, "
        "a polymer seam, or lightweight bottle construction. Use measured weight only as support."
    ),
    frozenset(("pet", "glass")): (
        "PET has thin molded walls, a lightweight threaded neck/collar, shoulder seams, or squeeze "
        "deformation. Glass has a thick rigid base/wall and glass refraction. Dark tint alone is not glass."
    ),
    frozenset(("fluorescent", "glass")): (
        "Any complete electric bulb or lamp with a screw/bayonet base, socket, electrodes, tube, "
        "or lamp housing is fluorescent in this product taxonomy. Its glass envelope is only a component."
    ),
    frozenset(("fluorescent", "plastic")): (
        "Any complete electric bulb or lamp with an electrical base, socket, LED housing, tube, "
        "or diffuser is fluorescent in this product taxonomy, even when its diffuser is plastic."
    ),
}

_FULL_FRAME_REASONING_DIRECTIONS = {
    ("glass", "pet"),
    ("glass", "plastic"),
    ("vinyl", "paper"),
    ("styrofoam", "paper"),
}


def _with_runtime_context(
    prompt: str,
    weight_g: float | None = None,
    yolo_material: str | None = None,
    yolo_confidence: float | None = None,
) -> str:
    """Attach non-authoritative sensor/detector hints to an internal LLM prompt."""
    context = [
        "Runtime evidence (supporting hints, never ground truth):",
        (
            f"- Measured scale weight: {weight_g:.2f} g. The value may include residue or "
            "contents; use it only when it is physically consistent with the visible object's size."
            if weight_g is not None
            else "- Measured scale weight: unavailable."
        ),
        (
            f"- YOLO proposal: {yolo_material} at confidence {yolo_confidence:.4f}. "
            "YOLO mainly localizes the target and may be wrong on a new camera domain."
            if yolo_material is not None and yolo_confidence is not None
            else "- YOLO proposal: unavailable; judge the full visible item."
        ),
    ]
    return prompt + "\n\n" + "\n".join(context)

@dataclass(frozen=True)
class LocalLLMPrediction:
    class_id: int
    class_name: str
    confidence: float
    has_label: bool
    is_dented: bool
    has_foreign_material: bool
    is_single_primary_item: bool
    compression_required: bool = False


def enabled() -> bool:
    return (
        settings.LOCAL_LLM_MODE in {"shadow", "primary"}
        and bool(settings.LOCAL_LLM_BASE_URL)
    )


def primary_enabled() -> bool:
    return enabled() and settings.LOCAL_LLM_MODE == "primary"


def health_status() -> dict[str, Any]:
    """Check the NAS gateway without exposing its URL or credentials."""
    base = {
        "enabled": enabled(),
        "required": primary_enabled(),
        "status": "disabled",
        "reachable": False,
        "model": settings.LOCAL_LLM_MODEL,
        "model_available": False,
        "latency_ms": None,
    }
    if not enabled():
        return base

    headers: dict[str, str] = {}
    if settings.LOCAL_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LOCAL_LLM_API_KEY}"
    started = time.perf_counter()
    try:
        response = _get_http_client().get(
            settings.LOCAL_LLM_BASE_URL.rstrip("/") + "/api/tags",
            headers=headers,
            timeout=3.0,
        )
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        names = {
            str(item.get("name", ""))
            for item in models
            if isinstance(item, dict)
        }
        model_available = settings.LOCAL_LLM_MODEL in names
        base.update({
            "status": "ok" if model_available else "model_missing",
            "reachable": True,
            "model_available": model_available,
        })
    except (httpx.HTTPError, TypeError, ValueError, json.JSONDecodeError):
        base["status"] = "unavailable"
    finally:
        base["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return base


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


def _crop_as_jpeg(
    img: np.ndarray, bbox: list[float], *, max_side: int | None = None,
) -> str | None:
    height, width = img.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - box_w * 0.10)); y1 = max(0, int(y1 - box_h * 0.10))
    x2 = min(width, int(x2 + box_w * 0.10)); y2 = min(height, int(y2 + box_h * 0.10))
    if x2 <= x1 or y2 <= y1:
        return None
    return _encode_as_jpeg(
        img[y1:y2, x1:x2],
        max_side or settings.LOCAL_LLM_MAX_IMAGE_SIDE,
    )


def _images_as_jpeg(
    img: np.ndarray,
    bbox: list[float],
    *,
    crop_only: bool = False,
    crop_max_side: int | None = None,
) -> list[str]:
    """Return full-frame context followed by the target crop when it is distinct."""
    crop = _crop_as_jpeg(img, bbox, max_side=crop_max_side)
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
    flags = (
        "has_label", "compression_required", "is_dented",
        "has_foreign_material", "is_single_primary_item",
    )
    if any(type(parsed[name]) is not bool for name in flags):
        raise ValueError("LLM boolean contract invalid")
    return LocalLLMPrediction(
        class_id=CLASS_ID_BY_NAME[material], class_name=material,
        confidence=float(confidence), has_label=parsed["has_label"],
        compression_required=parsed["compression_required"],
        is_dented=parsed["is_dented"],
        has_foreign_material=parsed["has_foreign_material"],
        is_single_primary_item=parsed["is_single_primary_item"],
    )


def _get_http_client() -> httpx.Client:
    """Return one process-wide connection pool for the NAS gateway."""
    global _http_client
    if _http_client is None:
        with _client_lock:
            if _http_client is None:
                _http_client = httpx.Client(timeout=settings.LOCAL_LLM_TIMEOUT_SEC)
    return _http_client


def _classify_with_prompt(
    img: np.ndarray,
    bbox: list[float],
    prompt: str,
    *,
    crop_only: bool = False,
    crop_max_side: int | None = None,
) -> LocalLLMPrediction | None:
    """Synchronously query local Ollama; callers run this outside the event loop."""
    if not enabled():
        return None
    images = _images_as_jpeg(
        img, bbox, crop_only=crop_only, crop_max_side=crop_max_side,
    )
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
        response = _get_http_client().post(url, headers=headers, json=body)
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
    weight_g: float | None = None,
    yolo_confidence: float | None = None,
) -> LocalLLMPrediction | None:
    """Adjudicate a bounded material conflict while keeping the LLM authoritative."""
    if yolo_material not in CLASS_ID_BY_NAME or llm_material not in CLASS_ID_BY_NAME:
        return None
    pair = frozenset((yolo_material, llm_material))
    plastic_detail_recheck = (
        yolo_material == "plastic" and llm_material in {"paper", "styrofoam"}
    )
    if plastic_detail_recheck:
        # This replaces the existing second pass rather than adding a third call.
        # It is selected only behind the high-confidence detector gate in pipeline.py.
        prompt = _PLASTIC_MATERIAL_PROMPT
        crop_only = False
    elif (yolo_material, llm_material) in _FULL_FRAME_REASONING_DIRECTIONS:
        # These pairs need the complete silhouette and the richer taxonomy.
        # Keep this prompt out of the normal path because applying it globally
        # over-calls paper/foam on rigid plastic trays.
        prompt = _FORM_REASONING_PROMPT
        crop_only = False
    else:
        prompt = _REJECTION_CONFLICT_PROMPT.format(
            yolo_material=yolo_material,
            llm_material=llm_material,
            pair_rules=_PAIR_RULES.get(
                pair,
                "Compare the complete object's physical construction and choose only the better-supported candidate.",
            ),
        )
        prompt = _with_runtime_context(
            prompt, weight_g, yolo_material, yolo_confidence,
        )
        crop_only = (
            pair == frozenset(("styrofoam", "plastic"))
            or (yolo_material, llm_material) == ("pet", "glass")
        )
    prediction = _classify_with_prompt(
        img,
        bbox,
        prompt,
        crop_only=crop_only,
        crop_max_side=(
            settings.LOCAL_LLM_PLASTIC_RECHECK_IMAGE_SIDE
            if plastic_detail_recheck else None
        ),
    )
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
    global _http_client
    _shadow_executor.shutdown(wait=True)
    with _client_lock:
        if _http_client is not None:
            _http_client.close()
            _http_client = None
