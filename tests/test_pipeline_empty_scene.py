import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ.setdefault("API_KEY", "test-key")

from app.schemas.enums import DetectionStatus
from app.services import inference, pipeline


class _Registry:
    def state(self):
        return None


async def _fake_read_image(_upload):
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_yolo가_잘못된_bbox를_잡아도_llm_빈장면은_not_detected(monkeypatch):
    prediction = pipeline.local_llm.LocalLLMPrediction(
        class_id=-1,
        class_name="empty_scene",
        confidence=0.99,
        has_label=False,
        is_dented=False,
        has_foreign_material=False,
        is_single_primary_item=False,
    )
    recorded = {}
    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(
        inference,
        "run_main",
        lambda *_args: (2, 0.91, [10.0, 10.0, 90.0, 90.0]),
    )
    monkeypatch.setattr(pipeline.local_llm, "primary_enabled", lambda: True)
    monkeypatch.setattr(pipeline.local_llm, "classify", lambda *_args: prediction)
    monkeypatch.setattr(
        pipeline.local_llm,
        "record_primary",
        lambda **kwargs: recorded.update(kwargs),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(
            pipeline.run(None, 0.0, "empty-with-false-bbox", _Registry())
        )

    assert result.status is DetectionStatus.NOT_DETECTED
    assert result.classification is None
    assert recorded["reason"] == "empty_scene"
