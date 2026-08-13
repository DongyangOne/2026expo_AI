"""pipeline 분기 테스트 — 비닐 허용과 PET/플라스틱 통합 계약."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ.setdefault("API_KEY", "test-key")

from app.schemas.enums import DetectionStatus, GuidanceCode, WasteClass
from app.schemas.response import Conditions
from app.services import inference, pipeline
from app.services import verifier_shadow


class _Registry:
    def state(self):
        return None


class _VerifierRegistry(_Registry):
    def verifier(self):
        return "vinyl-verifier-session"


async def _fake_read_image(_upload):
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _fake_vinyl_detection(_registry, _img):
    return 5, 0.90, [10.0, 10.0, 90.0, 90.0]


def _fake_pet_detection(_registry, _img):
    return 1, 0.94, [10.0, 10.0, 90.0, 90.0]


def _fake_ambiguous_vinyl_detection(_registry, _img):
    bbox = [10.0, 10.0, 90.0, 90.0]
    return 1, 0.307, bbox, [
        (1, 0.307, bbox),
        (5, 0.167, [9.0, 9.0, 94.0, 94.0]),
    ]


def _fake_clear_plastic_detection(_registry, _img):
    bbox = [10.0, 10.0, 90.0, 90.0]
    return 1, 0.391, bbox, [(1, 0.391, bbox)]


def test_vinyl_무게이상은_rejected_weight_anomaly(monkeypatch):
    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(inference, "run_main", _fake_vinyl_detection)
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *args, **kwargs: True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(pipeline.run(None, 500.0, "vinyl-heavy", _Registry()))

    assert result.status is DetectionStatus.REJECTED
    assert result.weight.anomaly is True
    assert [item.code for item in result.guidance] == [GuidanceCode.EMPTY_CONTENTS]
    assert result.general is None


def test_vinyl_무게정상은_vinyl만_allowed(monkeypatch):
    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(inference, "run_main", _fake_vinyl_detection)
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *args, **kwargs: False)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(pipeline.run(None, 20.0, "vinyl-normal", _Registry()))

    assert result.status is DetectionStatus.ALLOWED
    assert result.weight.anomaly is False
    assert result.guidance == []
    assert result.general is None


def test_pet은_상태검사후_plastic으로_응답(monkeypatch):
    captured = {}

    def fake_run_state(_session, _img, _bbox, cls):
        captured["state_class"] = cls
        return inference.StatePrediction(Conditions(has_label=False, is_dented=True))

    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(inference, "run_main", _fake_pet_detection)
    monkeypatch.setattr(inference, "run_state", fake_run_state)
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *args, **kwargs: False)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(pipeline.run(None, 20.0, "pet-normalized", _Registry()))

    assert result.status is DetectionStatus.ALLOWED
    assert result.classification is not None
    assert result.classification.class_id == 3
    assert result.classification.class_name is WasteClass.PLASTIC
    assert captured["state_class"] is WasteClass.PET
    assert result.guidance == []


def test_저신뢰_pet에_동일_bbox_vinyl후보와_검증기합의가_있으면_vinyl로_교정(monkeypatch):
    captured = {}

    def fake_run_verifier(session, _img, _bbox):
        assert session == "vinyl-verifier-session"
        return {
            "material": {"class_id": 5, "class_name": "vinyl", "confidence": 0.714},
            "heads": {},
            "input_size": 320,
        }

    def fake_submit_precomputed(
        session, _img, _bbox, class_id, confidence, client_id,
        prediction, correction_applied,
    ):
        captured.update({
            "session": session,
            "class_id": class_id,
            "confidence": confidence,
            "client_id": client_id,
            "prediction": prediction,
            "correction_applied": correction_applied,
        })

    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(inference, "run_main", _fake_ambiguous_vinyl_detection)
    monkeypatch.setattr(inference, "run_verifier", fake_run_verifier)
    monkeypatch.setattr(verifier_shadow, "submit_precomputed", fake_submit_precomputed)
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *args, **kwargs: False)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(
            pipeline.run(None, None, "vinyl-correction-001", _VerifierRegistry())
        )

    assert result.status is DetectionStatus.ALLOWED
    assert result.classification is not None
    assert result.classification.class_id == 5
    assert result.classification.class_name is WasteClass.VINYL
    assert result.classification.confidence == 0.714
    assert captured["class_id"] == 1
    assert captured["confidence"] == 0.307
    assert captured["correction_applied"] is True


def test_비닐보조후보가_없는_저신뢰_pet은_plastic_low_confidence를_유지(monkeypatch):
    captured = {}

    def fail_if_verifier_runs(*_args, **_kwargs):
        raise AssertionError("비닐 보조 후보가 없으면 검증기를 동기 실행하면 안 됩니다.")

    def fake_submit(_session, _img, _bbox, class_id, confidence, client_id):
        captured.update({
            "class_id": class_id,
            "confidence": confidence,
            "client_id": client_id,
        })

    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(inference, "run_main", _fake_clear_plastic_detection)
    monkeypatch.setattr(inference, "run_verifier", fail_if_verifier_runs)
    monkeypatch.setattr(verifier_shadow, "submit", fake_submit)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(
            pipeline.run(None, None, "clear-plastic-001", _VerifierRegistry())
        )

    assert result.status is DetectionStatus.GENERAL_WASTE
    assert result.classification is not None
    assert result.classification.class_id == 3
    assert result.classification.class_name is WasteClass.PLASTIC
    assert result.classification.confidence == 0.391
    assert captured == {
        "class_id": 1,
        "confidence": 0.391,
        "client_id": "clear-plastic-001",
    }


def test_shadow_verifier_receives_yolo_result_without_changing_response(monkeypatch):
    captured = {}

    class _ShadowRegistry(_Registry):
        def verifier(self):
            return "temporary-verifier-session"

    def fake_submit(session, img, bbox, class_id, confidence, client_id):
        captured.update({
            "session": session,
            "bbox": bbox,
            "class_id": class_id,
            "confidence": confidence,
            "client_id": client_id,
        })

    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(inference, "run_main", _fake_vinyl_detection)
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *args, **kwargs: False)
    monkeypatch.setattr(verifier_shadow, "submit", fake_submit)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(
            pipeline.run(None, 20.0, "hardware-shadow-001", _ShadowRegistry())
        )

    assert result.status is DetectionStatus.ALLOWED
    assert captured == {
        "session": "temporary-verifier-session",
        "bbox": [10.0, 10.0, 90.0, 90.0],
        "class_id": 5,
        "confidence": 0.90,
        "client_id": "hardware-shadow-001",
    }
