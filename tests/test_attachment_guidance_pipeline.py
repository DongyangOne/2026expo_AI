import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ.setdefault("API_KEY", "test-key")

from app.schemas.enums import DetectionStatus, GuidanceCode
from app.schemas.response import Conditions
from app.services import inference, pipeline


class _Registry:
    def state(self):
        return None


async def _fake_read_image(_upload):
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_plastic_cup_attachment_flags_reach_guidance(monkeypatch):
    prediction = pipeline.local_llm.LocalLLMPrediction(
        class_id=3,
        class_name="plastic",
        confidence=0.97,
        has_label=False,
        compression_required=False,
        is_dented=False,
        has_foreign_material=False,
        has_straw=True,
        has_cup_holder=True,
        is_single_primary_item=True,
    )
    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(
        inference,
        "run_main",
        lambda *_args: (3, 0.94, [10.0, 10.0, 90.0, 90.0]),
    )
    monkeypatch.setattr(
        inference,
        "run_state",
        lambda *_args: inference.StatePrediction(Conditions()),
    )
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(pipeline.local_llm, "primary_enabled", lambda: True)
    monkeypatch.setattr(pipeline.local_llm, "classify", lambda *_args: prediction)
    monkeypatch.setattr(pipeline.local_llm, "record_primary", lambda **_kwargs: None)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(pipeline.run(None, 20.0, "cup-attachments", _Registry()))

    assert result.status is DetectionStatus.REJECTED
    assert [item.code for item in result.guidance] == [
        GuidanceCode.REMOVE_STRAW,
        GuidanceCode.REMOVE_CUP_HOLDER,
    ]
