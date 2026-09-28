"""Regression tests for the bounded material-conflict and vinyl fast paths."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ.setdefault("API_KEY", "test-key")

from app.schemas.enums import DetectionStatus, WasteClass
from app.schemas.response import Conditions
from app.services import inference, pipeline


class _Registry:
    def state(self):
        return None


async def _image(_upload):
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _prediction(class_id: int, class_name: str):
    return pipeline.local_llm.LocalLLMPrediction(
        class_id=class_id,
        class_name=class_name,
        confidence=0.95,
        has_label=False,
        is_dented=True,
        has_foreign_material=False,
        is_single_primary_item=True,
    )


def test_only_high_confidence_pet_glass_reverse_conflict_is_rechecked():
    assert pipeline._needs_material_conflict_recheck(1, 0.96, 6)
    assert not pipeline._needs_material_conflict_recheck(1, 0.84, 6)
    assert not pipeline._needs_material_conflict_recheck(3, 0.96, 6)
    assert pipeline._needs_material_conflict_recheck(4, 0.91, 3)


def test_vinyl_specialist_requires_light_weight_and_detector_support(monkeypatch):
    monkeypatch.setattr(
        pipeline.settings, "LOCAL_LLM_VINYL_SPECIALIST_MAX_WEIGHT_G", 5.0,
    )
    monkeypatch.setattr(
        pipeline.settings, "LOCAL_LLM_VINYL_SPECIALIST_MIN_YOLO_CONFIDENCE", 0.5,
    )
    assert pipeline._use_vinyl_specialist(5, 0.54, 1.1)
    assert not pipeline._use_vinyl_specialist(5, 0.49, 1.1)
    assert not pipeline._use_vinyl_specialist(5, 0.54, 6.0)
    assert not pipeline._use_vinyl_specialist(3, 0.95, 1.1)


def test_light_yolo_vinyl_uses_one_specialist_pass(monkeypatch):
    calls = {"generic": 0, "specialist": 0}
    resolved = _prediction(5, "vinyl")
    recorded = {}
    monkeypatch.setattr(pipeline, "_read_image", _image)
    monkeypatch.setattr(
        inference, "run_main",
        lambda *_args: (5, 0.54, [0.0, 20.0, 100.0, 95.0]),
    )
    monkeypatch.setattr(inference, "run_state", lambda *_args: inference.StatePrediction(Conditions()))
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(pipeline.local_llm, "primary_enabled", lambda: True)
    monkeypatch.setattr(
        pipeline.local_llm, "classify",
        lambda *_args: calls.__setitem__("generic", calls["generic"] + 1),
    )

    def specialist(*_args):
        calls["specialist"] += 1
        return resolved

    monkeypatch.setattr(pipeline.local_llm, "reclassify_vinyl_plastic", specialist)
    monkeypatch.setattr(pipeline.local_llm, "record_primary", lambda **kw: recorded.update(kw))

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(pipeline.run(None, 1.1, "vinyl-specialist", _Registry()))

    assert result.status is DetectionStatus.ALLOWED
    assert result.classification.class_name is WasteClass.VINYL
    assert calls == {"generic": 0, "specialist": 1}
    assert recorded["reason"] == "vinyl_plastic_specialist"


def test_high_confidence_pet_llm_glass_gets_bounded_recheck(monkeypatch):
    recorded = {}
    monkeypatch.setattr(pipeline, "_read_image", _image)
    monkeypatch.setattr(
        inference, "run_main",
        lambda *_args: (1, 0.96, [10.0, 10.0, 90.0, 90.0]),
    )
    monkeypatch.setattr(inference, "run_state", lambda *_args: inference.StatePrediction(Conditions()))
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(pipeline.local_llm, "primary_enabled", lambda: True)
    monkeypatch.setattr(pipeline.local_llm, "classify", lambda *_args: _prediction(6, "glass"))
    monkeypatch.setattr(
        pipeline.local_llm, "reclassify_rejection_conflict",
        lambda *_args: _prediction(1, "pet"),
    )
    monkeypatch.setattr(pipeline.local_llm, "record_primary", lambda **kw: recorded.update(kw))

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(pipeline.run(None, 20.0, "pet-glass-conflict", _Registry()))

    assert result.status is DetectionStatus.ALLOWED
    assert result.classification.class_name is WasteClass.PLASTIC
    assert recorded["reason"] == "rejection_conflict_recheck"
