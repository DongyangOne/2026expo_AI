from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ.setdefault("API_KEY", "test-key")

from app.schemas.enums import DetectionStatus
from app.schemas.response import Conditions
from app.services import inference, pipeline


class _Boxes:
    def __init__(self, rows: list[tuple[int, float, list[float]]]) -> None:
        self.cls = np.asarray([row[0] for row in rows], dtype=np.float32)
        self.conf = np.asarray([row[1] for row in rows], dtype=np.float32)
        self.xyxy = np.asarray([row[2] for row in rows], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.cls)


class _Registry:
    def __init__(self, rows: list[tuple[int, float, list[float]]]) -> None:
        self._boxes = _Boxes(rows)

    def main(self):
        def predict(_image, **_kwargs):
            return [type("Result", (), {"boxes": self._boxes})()]

        return predict

    def state(self):
        return None


def test_run_main_ignores_higher_confidence_box_outside_platform() -> None:
    registry = _Registry(
        [
            (3, 0.95, [52.5, 205.0, 523.5, 480.0]),
            (2, 0.70, [235.0, 110.0, 335.0, 350.0]),
        ]
    )

    detected = inference.run_main(
        registry, np.zeros((480, 640, 3), dtype=np.uint8)
    )

    assert detected is not None
    assert detected is not getattr(inference, "PLATFORM_ROI_REJECTED", None)
    assert detected[0] == 2
    assert detected[2] == [235.0, 110.0, 335.0, 350.0]


def test_run_main_marks_only_off_platform_detection_as_rejected() -> None:
    registry = _Registry([(3, 0.95, [52.5, 205.0, 523.5, 480.0])])

    detected = inference.run_main(
        registry, np.zeros((480, 640, 3), dtype=np.uint8)
    )
    marker = getattr(inference, "PLATFORM_ROI_REJECTED", None)

    assert marker is not None
    assert detected is marker


def test_platform_observation_crop_keeps_platform_and_removes_side_background() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    cropped = inference.platform_observation_crop(image)

    assert cropped.shape == (360, 336, 3)


def test_pipeline_uses_platform_crop_llm_when_yolo_only_finds_off_platform_box(
    monkeypatch,
) -> None:
    marker = getattr(inference, "PLATFORM_ROI_REJECTED", object())
    received_shape = None
    prediction = pipeline.local_llm.LocalLLMPrediction(
        class_id=5,
        class_name="vinyl",
        confidence=0.95,
        has_label=False,
        is_dented=False,
        has_foreign_material=False,
        is_single_primary_item=True,
    )

    async def fake_read_image(_upload):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def classify_platform(image, _bbox):
        nonlocal received_shape
        received_shape = image.shape
        return prediction

    monkeypatch.setattr(pipeline, "_read_image", fake_read_image)
    monkeypatch.setattr(inference, "run_main", lambda *_args: marker)
    monkeypatch.setattr(
        inference,
        "run_state",
        lambda *_args: inference.StatePrediction(conditions=Conditions()),
    )
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(pipeline.local_llm, "primary_enabled", lambda: True)
    monkeypatch.setattr(pipeline.local_llm, "classify", classify_platform)
    monkeypatch.setattr(pipeline.local_llm, "record_primary", lambda **_kwargs: None)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            monkeypatch.setattr(pipeline, "_executor", executor)
            result = asyncio.run(
                pipeline.run(None, 0.0, "off-platform-apparatus", _Registry([]))
            )
    except Exception as exc:  # RED: the old pipeline tries to unpack the marker.
        result = exc

    assert not isinstance(result, Exception)
    assert result.status is DetectionStatus.ALLOWED
    assert result.classification is not None
    assert result.classification.class_name.value == "vinyl"
    assert received_shape == (360, 336, 3)
