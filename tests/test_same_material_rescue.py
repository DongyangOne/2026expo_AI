"""같은 재질 복수(꺾인 캔, 포장지 여러 개)가 '복수 물체'로 버려지지 않게 하는 보조 판정 테스트."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

os.environ.setdefault("API_KEY", "test-key")

from app.schemas.enums import DetectionStatus, GeneralWasteCode
from app.schemas.response import Conditions
from app.services import inference, local_llm, pipeline


class _Registry:
    def state(self):
        return None


async def _fake_read_image(_upload):
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _prediction(name="can", class_id=0, confidence=0.95, single=False, **kw):
    return local_llm.LocalLLMPrediction(
        class_id=class_id,
        class_name=name,
        confidence=confidence,
        has_label=False,
        compression_required=kw.get("compression_required", True),
        is_dented=kw.get("is_dented", True),
        has_foreign_material=False,
        has_straw=False,
        has_cup_holder=False,
        is_single_primary_item=single,
    )


def _run(monkeypatch, first, recheck, *, yolo_box=True):
    """first: 첫 판정, recheck: 재판정(None이면 호출되면 실패). 반환: (응답, 재판정 호출 횟수)."""
    calls = {"recheck": 0}

    def fake_recheck(*_args, **_kwargs):
        calls["recheck"] += 1
        assert recheck is not None, "재판정이 호출되면 안 되는 경우입니다"
        return recheck

    marker = getattr(inference, "PLATFORM_ROI_REJECTED", object())
    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(
        inference, "run_main",
        (lambda *_a: (0, 0.6, [200.0, 150.0, 440.0, 330.0])) if yolo_box else (lambda *_a: marker),
    )
    monkeypatch.setattr(
        inference, "run_state", lambda *_a: inference.StatePrediction(conditions=Conditions()),
    )
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *_a, **_k: False)
    monkeypatch.setattr(pipeline.local_llm, "primary_enabled", lambda: True)
    monkeypatch.setattr(pipeline.local_llm, "classify", lambda *_a, **_k: first)
    monkeypatch.setattr(pipeline.local_llm, "reclassify_same_material", fake_recheck)
    monkeypatch.setattr(pipeline.local_llm, "record_primary", lambda **_k: None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(pipeline.run(None, 9.6, "same-material", _Registry()))
    return result, calls["recheck"]


@pytest.mark.parametrize("yolo_box", [True, False])
def test_crushed_can_flagged_as_multiple_is_rescued(monkeypatch, yolo_box):
    first = _prediction(single=False)
    second = _prediction(single=True)
    result, calls = _run(monkeypatch, first, second, yolo_box=yolo_box)
    assert calls == 1
    assert result.status is DetectionStatus.ALLOWED
    assert result.classification.class_name.value == "can"


def test_rescue_is_dropped_when_recheck_still_sees_multiple_items(monkeypatch):
    result, calls = _run(monkeypatch, _prediction(single=False), _prediction(single=False))
    assert calls == 1
    assert result.status is DetectionStatus.GENERAL_WASTE
    assert result.general.code is GeneralWasteCode.LOW_CONFIDENCE


def test_rescue_is_dropped_when_recheck_names_a_different_material(monkeypatch):
    # 캔과 컵이 같이 있어 재판정이 다른 품목으로 흔들리면 같은 재질이라고 볼 수 없다.
    other = _prediction(name="plastic", class_id=3, single=True, compression_required=False, is_dented=False)
    result, calls = _run(monkeypatch, _prediction(single=False), other)
    assert calls == 1
    assert result.status is DetectionStatus.GENERAL_WASTE


def test_rescue_is_dropped_when_recheck_is_unavailable(monkeypatch):
    result, calls = _run(monkeypatch, _prediction(single=False), _prediction(single=True))
    # 위와 같은 입력이지만 재판정 실패(None)를 흉내 내려면 별도 스텁이 필요하다.
    monkeypatch.setattr(pipeline.local_llm, "reclassify_same_material", lambda *_a, **_k: None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        again = asyncio.run(pipeline.run(None, 9.6, "recheck-down", _Registry()))
    assert result.status is DetectionStatus.ALLOWED
    assert again.status is DetectionStatus.GENERAL_WASTE


@pytest.mark.parametrize(
    "first",
    [
        _prediction(single=True),                              # 이미 단일 → 평소 경로, 추가 호출 없음
        _prediction(single=False, confidence=0.7),             # 저신뢰 → 살리지 않는다
        _prediction(name="empty_scene", class_id=0, single=False),
        _prediction(name="general_waste", class_id=0, single=False),
    ],
)
def test_recheck_is_not_called_for_ordinary_or_hopeless_results(monkeypatch, first):
    result, calls = _run(monkeypatch, first, None)
    assert calls == 0


def test_same_material_prompt_differs_from_main_prompt_by_one_rule():
    assert local_llm._SINGLE_ITEM_RULE in local_llm._PROMPT
    assert local_llm._SAME_MATERIAL_RULE not in local_llm._PROMPT
    assert (
        local_llm._SAME_MATERIAL_PROMPT.replace(local_llm._SAME_MATERIAL_RULE, local_llm._SINGLE_ITEM_RULE)
        == local_llm._PROMPT
    )


def test_reclassify_same_material_sends_prompt_with_runtime_context(monkeypatch):
    seen = {}

    def fake(img, bbox, prompt, **_kw):
        seen["prompt"] = prompt
        return None

    monkeypatch.setattr(local_llm, "_classify_with_prompt", fake)
    local_llm.reclassify_same_material(np.zeros((10, 10, 3), np.uint8), [0, 0, 10, 10], 9.6, "can", 0.49)
    assert "several pieces of the same material" in seen["prompt"]
    assert "9.60 g" in seen["prompt"]
