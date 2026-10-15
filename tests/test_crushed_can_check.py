"""찌그러진 캔이 plastic/vinyl로 보이는 오분류를 모순 신호가 있을 때만 이분법으로 다시 확인하는지 검증한다."""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

os.environ.setdefault("API_KEY", "test-key")

from app.schemas.enums import DetectionStatus
from app.schemas.response import Conditions
from app.services import inference, local_llm, openrouter_client, pipeline


class _Registry:
    def state(self):
        return None


async def _fake_read_image(_upload):
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _pred(name, class_id, confidence=0.9, *, dented=False, compress=False):
    return local_llm.LocalLLMPrediction(
        class_id=class_id, class_name=name, confidence=confidence, has_label=False,
        compression_required=compress, is_dented=dented, has_foreign_material=False,
        has_straw=False, has_cup_holder=False, is_single_primary_item=True,
    )


PLASTIC = dict(name="plastic", class_id=3)
VINYL = dict(name="vinyl", class_id=5)


def _run(monkeypatch, first, check, *, yolo_box=True):
    """first: 첫 판정, check: 확인 결과(None이면 호출되면 실패). 반환: (응답, 호출 횟수)."""
    calls = {"n": 0}

    def fake_check(*_args, **_kwargs):
        calls["n"] += 1
        assert check is not None, "캔 확인이 호출되면 안 되는 경우입니다"
        return check

    marker = getattr(inference, "PLATFORM_ROI_REJECTED", object())
    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(
        inference, "run_main",
        (lambda *_a: (3, 0.6, [200.0, 150.0, 440.0, 330.0])) if yolo_box else (lambda *_a: marker),
    )
    monkeypatch.setattr(inference, "run_state", lambda *_a: inference.StatePrediction(conditions=Conditions()))
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *_a, **_k: False)
    monkeypatch.setattr(pipeline.local_llm, "primary_enabled", lambda: True)
    monkeypatch.setattr(pipeline.local_llm, "classify", lambda *_a, **_k: first)
    monkeypatch.setattr(pipeline.local_llm, "confirm_crushed_can", fake_check)
    monkeypatch.setattr(pipeline.local_llm, "record_primary", lambda **_k: None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(pipeline.run(None, 14.0, "can-check", _Registry()))
    return result, calls["n"]


def _check(is_can=True, crushed=True, confidence=0.9):
    return local_llm.CanCheck(is_can=is_can, crushed=crushed, confidence=confidence, runtime=None)


@pytest.mark.parametrize("yolo_box", [True, False])
def test_crushed_can_called_plastic_becomes_an_allowed_can(monkeypatch, yolo_box):
    first = _pred(**PLASTIC, confidence=0.9, dented=True, compress=False)  # 모순 신호
    result, calls = _run(monkeypatch, first, _check(), yolo_box=yolo_box)
    assert calls == 1
    assert result.status is DetectionStatus.ALLOWED
    assert result.classification.class_name.value == "can"


def test_low_confidence_vinyl_is_rescued_as_a_can(monkeypatch):
    first = _pred(**VINYL, confidence=0.7)  # 16:04 사례
    result, calls = _run(monkeypatch, first, _check())
    assert calls == 1
    assert result.status is DetectionStatus.ALLOWED
    assert result.classification.class_name.value == "can"


def test_can_that_is_not_crushed_is_asked_to_compress(monkeypatch):
    first = _pred(**PLASTIC, confidence=0.9, dented=True, compress=False)
    result, _ = _run(monkeypatch, first, _check(crushed=False))
    assert result.status is DetectionStatus.REJECTED
    assert result.classification.class_name.value == "can"


def test_plastic_answer_keeps_the_first_result(monkeypatch):
    first = _pred(**PLASTIC, confidence=0.9, dented=True, compress=False)
    result, calls = _run(monkeypatch, first, _check(is_can=False))
    assert calls == 1
    assert result.classification.class_name.value == "plastic"


def test_low_confidence_can_answer_is_not_trusted(monkeypatch):
    first = _pred(**PLASTIC, confidence=0.9, dented=True, compress=False)
    result, _ = _run(monkeypatch, first, _check(confidence=0.6))
    assert result.classification.class_name.value == "plastic"


def test_unavailable_check_keeps_the_first_result(monkeypatch):
    first = _pred(**PLASTIC, confidence=0.9, dented=True, compress=False)
    _run(monkeypatch, first, _check())  # 정상 경로를 먼저 구성해 둔다
    monkeypatch.setattr(pipeline.local_llm, "confirm_crushed_can", lambda *_a, **_k: None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        again = asyncio.run(pipeline.run(None, 14.0, "check-down", _Registry()))
    assert again.status is DetectionStatus.ALLOWED
    assert again.classification.class_name.value == "plastic"


@pytest.mark.parametrize(
    "first",
    [
        _pred(**PLASTIC, confidence=0.9, dented=False, compress=False),  # 평범한 플라스틱
        _pred(name="pet", class_id=1, confidence=0.95, dented=True, compress=True),  # 찌그러진 페트병(정상)
        _pred(name="can", class_id=0, confidence=0.95, dented=True, compress=True),  # 이미 캔
        _pred(name="paper", class_id=2, confidence=0.9, dented=True, compress=False),  # 종이는 대상 아님
        _pred(**VINYL, confidence=0.95),  # 고신뢰 비닐
    ],
)
def test_check_is_not_called_for_ordinary_results(monkeypatch, first):
    _, calls = _run(monkeypatch, first, None)
    assert calls == 0


class _Generation:
    def __init__(self, content):
        self.content = content

    def runtime(self):
        return {"backend": "openrouter", "model": "test"}


def _ask(monkeypatch, content):
    monkeypatch.setattr(openrouter_client, "enabled", lambda: True)
    monkeypatch.setattr(local_llm, "_images_as_jpeg", lambda *_a, **_k: ["img"])
    monkeypatch.setattr(
        openrouter_client, "generate_with_metadata",
        lambda *_a, **_k: None if content is None else _Generation(content),
    )
    return local_llm.confirm_crushed_can(np.zeros((10, 10, 3), np.uint8), [0, 0, 10, 10])


def test_confirm_crushed_can_parses_the_answer(monkeypatch):
    got = _ask(monkeypatch, json.dumps({"material": "can", "crushed": True, "confidence": 0.92}))
    assert got.is_can and got.crushed and got.confidence == 0.92
    assert got.runtime["backend"] == "openrouter"
    plastic = _ask(monkeypatch, json.dumps({"material": "plastic", "crushed": False, "confidence": 0.9}))
    assert plastic.is_can is False


@pytest.mark.parametrize(
    "content",
    [
        None,
        "not json",
        json.dumps({"material": "metal", "confidence": 0.9}),
        json.dumps({"material": "can", "confidence": 1.5}),
        json.dumps({"material": "can", "confidence": "high"}),
        json.dumps({"material": "can", "crushed": "yes", "confidence": 0.9}),
    ],
)
def test_confirm_crushed_can_rejects_answers_outside_the_contract(monkeypatch, content):
    assert _ask(monkeypatch, content) is None


def test_confirm_crushed_can_is_off_when_the_provider_is_disabled(monkeypatch):
    monkeypatch.setattr(openrouter_client, "enabled", lambda: False)
    assert local_llm.confirm_crushed_can(np.zeros((10, 10, 3), np.uint8), [0, 0, 10, 10]) is None


def test_can_check_prompt_does_not_depend_on_the_main_prompt():
    assert local_llm._CAN_OR_PLASTIC_PROMPT not in local_llm._PROMPT
    assert "is_single_primary_item" not in local_llm._CAN_OR_PLASTIC_PROMPT
