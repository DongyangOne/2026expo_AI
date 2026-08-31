"""One frozen crop/preprocessing implementation for offline v3 artifacts.

The proposal builder, operational-crop materializer, and policy-evidence
builder must produce the same pixels.  Keeping their rounding/interpolation in
one pure helper prevents a verifier from being trained on one crop geometry and
gated on another.  The production runtime intentionally does not import this
module until the candidate passes its independent hardware gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


CONTRACT_VERSION = "offline_verifier_crop.v1"
BBOX_ROUNDING = {"min_edges": "floor", "max_edges": "ceil"}
RESIZE_ROUNDING = "nearest_ties_to_even"
RESIZE_INTERPOLATION = {
    "downscale": "INTER_AREA",
    "upscale": "INTER_LINEAR",
    "equal": "identity",
}
LETTERBOX_ALIGNMENT = {"horizontal": "center_floor", "vertical": "center_floor"}
COLOR_CONVERSION = "BGR_TO_RGB"
TENSOR_LAYOUT = "NCHW"
TENSOR_DTYPE = "float32"


@dataclass(frozen=True)
class VerifierCropContract:
    padding_ratio: float
    size: int
    fill: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    input_scale: float


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _triplet(value: object, *, field: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three values")
    result = tuple(_finite(item, field=field) for item in value)
    return result  # type: ignore[return-value]


def validate_crop_preprocessing_spec(spec: Mapping[str, Any]) -> VerifierCropContract:
    """Validate every pixel-affecting choice in the frozen inference spec."""

    crop = spec.get("crop")
    if not isinstance(crop, Mapping):
        raise ValueError("inference spec crop must be an object")
    exact = {
        "source": "selected_detector_bbox",
        "clip_to_source": True,
        "resize": "aspect_preserving_letterbox",
        "preprocessing_contract_version": CONTRACT_VERSION,
        "bbox_rounding": BBOX_ROUNDING,
        "resize_rounding": RESIZE_ROUNDING,
        "resize_interpolation": RESIZE_INTERPOLATION,
        "letterbox_alignment": LETTERBOX_ALIGNMENT,
        "color_conversion": COLOR_CONVERSION,
    }
    for name, expected in exact.items():
        if crop.get(name) != expected:
            raise ValueError(
                f"unsupported crop preprocessing {name}={crop.get(name)!r}; "
                f"expected {expected!r}"
            )
    padding = _finite(crop.get("padding_ratio"), field="crop.padding_ratio")
    if not 0 <= padding <= 1:
        raise ValueError("crop.padding_ratio must be between zero and one")
    size = crop.get("letterbox_size")
    fill = crop.get("letterbox_fill")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("crop.letterbox_size must be a positive integer")
    if not isinstance(fill, int) or isinstance(fill, bool) or not 0 <= fill <= 255:
        raise ValueError("crop.letterbox_fill must be an integer in 0..255")
    normalization = crop.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("crop.normalization must be an object")
    if normalization.get("layout") != TENSOR_LAYOUT:
        raise ValueError(f"crop.normalization.layout must be {TENSOR_LAYOUT}")
    if normalization.get("dtype") != TENSOR_DTYPE:
        raise ValueError(f"crop.normalization.dtype must be {TENSOR_DTYPE}")
    input_scale = _finite(
        normalization.get("input_scale"), field="crop.normalization.input_scale"
    )
    if input_scale <= 0:
        raise ValueError("crop.normalization.input_scale must be positive")
    mean = _triplet(normalization.get("mean"), field="crop.normalization.mean")
    std = _triplet(normalization.get("std"), field="crop.normalization.std")
    if any(value <= 0 for value in std):
        raise ValueError("crop.normalization.std values must be positive")
    return VerifierCropContract(
        padding_ratio=padding,
        size=size,
        fill=fill,
        mean=mean,
        std=std,
        input_scale=input_scale,
    )


def padded_clipped_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    """Use floor on min edges and ceil on max edges, then clip to source."""

    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    coordinates = tuple(_finite(value, field="bbox") for value in bbox)
    x1, y1, x2, y2 = coordinates
    if width <= 0 or height <= 0:
        raise ValueError("source width and height must be positive")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive area")
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        raise ValueError("bbox lies outside the source image")
    padding_value = _finite(padding, field="padding")
    if not 0 <= padding_value <= 1:
        raise ValueError("padding must be between zero and one")
    box_width, box_height = x2 - x1, y2 - y1
    left = max(0, math.floor(x1 - box_width * padding_value))
    top = max(0, math.floor(y1 - box_height * padding_value))
    right = min(width, math.ceil(x2 + box_width * padding_value))
    bottom = min(height, math.ceil(y2 + box_height * padding_value))
    if right <= left or bottom <= top:
        raise ValueError("bbox is empty after padding and clipping")
    return left, top, right, bottom


def letterbox_bgr(
    image: np.ndarray,
    *,
    size: int,
    fill: int,
) -> tuple[np.ndarray, float, int, int, int, int]:
    """Letterbox with ties-to-even resize dimensions and centered floor offsets."""

    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("letterbox requires a nonempty image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("letterbox image must have BGR shape HxWx3")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("letterbox size must be a positive integer")
    if not isinstance(fill, int) or isinstance(fill, bool) or not 0 <= fill <= 255:
        raise ValueError("letterbox fill must be an integer in 0..255")
    height, width = image.shape[:2]
    scale = min(size / float(width), size / float(height))
    # Python round is IEEE nearest with ties-to-even; the spec freezes that
    # choice rather than leaving language-dependent `int(x + .5)` behavior.
    resized_width = max(1, min(size, int(round(width * scale))))
    resized_height = max(1, min(size, int(round(height * scale))))
    if scale < 1.0:
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
    elif scale > 1.0:
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
    else:
        resized = image.copy()
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas, scale, resized_width, resized_height, offset_x, offset_y


def crop_and_letterbox_bgr(
    image: np.ndarray,
    bbox: Sequence[float],
    *,
    padding: float,
    size: int,
    fill: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    bounds = padded_clipped_bbox(
        bbox, width=width, height=height, padding=padding
    )
    left, top, right, bottom = bounds
    letterboxed, *_ = letterbox_bgr(
        image[top:bottom, left:right], size=size, fill=fill
    )
    return letterboxed, bounds


def verifier_input_from_bgr(
    image: np.ndarray,
    bbox: Sequence[float],
    *,
    contract: VerifierCropContract,
) -> np.ndarray:
    """Return the exact float32 NCHW tensor named by the frozen spec."""

    letterboxed, _ = crop_and_letterbox_bgr(
        image,
        bbox,
        padding=contract.padding_ratio,
        size=contract.size,
        fill=contract.fill,
    )
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb /= np.float32(contract.input_scale)
    mean = np.asarray(contract.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(contract.std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (rgb - mean) / std
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
