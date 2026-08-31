from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import materialize_operational_verifier_crops as materializer
from scripts import prepare_proposal_verifier_dataset as proposal_builder
from scripts.verifier_preprocessing_contract import (
    VerifierCropContract,
    letterbox_bgr,
    padded_clipped_bbox,
    validate_crop_preprocessing_spec,
    verifier_input_from_bgr,
)


SPEC_PATH = Path(__file__).parents[1] / "configs" / "detector_inference_v3.json"


def test_frozen_spec_validates_every_pixel_affecting_choice():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    contract = validate_crop_preprocessing_spec(spec)

    assert contract == VerifierCropContract(
        padding_ratio=0.08,
        size=320,
        fill=114,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        input_scale=255.0,
    )


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("bbox_rounding", {"min_edges": "truncate", "max_edges": "ceil"}, "bbox_rounding"),
        ("resize_rounding", "floor", "resize_rounding"),
        (
            "resize_interpolation",
            {"downscale": "INTER_LINEAR", "upscale": "INTER_LINEAR", "equal": "identity"},
            "resize_interpolation",
        ),
        ("color_conversion", "BGR_TO_BGR", "color_conversion"),
    ],
)
def test_frozen_spec_rejects_preprocessing_drift(field, value, message):
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    spec["crop"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_crop_preprocessing_spec(spec)


def test_bbox_uses_floor_for_min_and_ceil_for_max_edges():
    assert padded_clipped_bbox(
        [1.9, 2.1, 8.1, 9.2], width=20, height=20, padding=0.1
    ) == (1, 1, 9, 10)


def test_resize_rounding_is_nearest_ties_to_even_and_offsets_use_floor():
    image = np.zeros((1, 4, 3), dtype=np.uint8)

    _, scale, resized_width, resized_height, offset_x, offset_y = letterbox_bgr(
        image, size=10, fill=114
    )

    assert scale == 2.5
    assert (resized_width, resized_height) == (10, 2)  # round(2.5) -> even 2
    assert (offset_x, offset_y) == (0, 4)


def test_resize_interpolation_is_area_down_linear_up_and_identity_equal(monkeypatch):
    real_resize = cv2.resize
    calls: list[int] = []

    def capture(image, size, *, interpolation):
        calls.append(interpolation)
        return real_resize(image, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", capture)
    letterbox_bgr(np.zeros((20, 10, 3), dtype=np.uint8), size=10, fill=0)
    letterbox_bgr(np.zeros((2, 1, 3), dtype=np.uint8), size=10, fill=0)
    letterbox_bgr(np.zeros((10, 10, 3), dtype=np.uint8), size=10, fill=0)

    assert calls == [cv2.INTER_AREA, cv2.INTER_LINEAR]


def test_tensor_contract_converts_bgr_to_rgb_then_scales_and_normalizes():
    image = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
    contract = VerifierCropContract(
        padding_ratio=0.0,
        size=1,
        fill=0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        input_scale=10.0,
    )

    tensor = verifier_input_from_bgr(image, [0, 0, 1, 1], contract=contract)

    assert tensor.shape == (1, 3, 1, 1)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[:, :, 0, 0], [[3.0, 2.0, 1.0]])


def test_prepare_and_materialize_delegate_to_the_shared_geometry_helper():
    image = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
    shared, *shared_meta = letterbox_bgr(image, size=20, fill=114)

    np.testing.assert_array_equal(proposal_builder.letterbox(image, 20), shared)
    materialized, *materialized_meta = materializer._letterbox(
        image, size=20, fill=114
    )
    np.testing.assert_array_equal(materialized, shared)
    assert materialized_meta == shared_meta
    expected_bounds = padded_clipped_bbox(
        [1.2, 0.7, 11.1, 8.4], width=13, height=9, padding=0.08
    )
    assert proposal_builder._crop_bounds(
        [1.2, 0.7, 11.1, 8.4], 13, 9, 0.08
    ) == expected_bounds
    assert materializer._padded_clipped_bbox(
        (1.2, 0.7, 11.1, 8.4), width=13, height=9, padding=0.08
    ) == expected_bounds
