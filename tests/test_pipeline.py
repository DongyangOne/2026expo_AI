"""pipeline 분기 테스트 — 비닐 무게 이상 guidance 계약."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ.setdefault("API_KEY", "test-key")

from app.schemas.enums import DetectionStatus, GeneralWasteCode, GuidanceCode
from app.services import inference, pipeline


class _Registry:
    def state(self):
        return None


async def _fake_read_image(_upload):
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _fake_vinyl_detection(_registry, _img):
    return 5, 0.90, [10.0, 10.0, 90.0, 90.0]


def test_vinyl_무게이상은_rejected_weight_anomaly(monkeypatch):
    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(inference, "run_main", _fake_vinyl_detection)
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *args, **kwargs: True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(pipeline.run(None, 500.0, "vinyl-heavy", _Registry()))

    assert result.status is DetectionStatus.REJECTED
    assert result.weight.anomaly is True
    assert [item.code for item in result.guidance] == [GuidanceCode.WEIGHT_ANOMALY]
    assert result.general is None


def test_vinyl_무게정상은_기존_general_waste(monkeypatch):
    monkeypatch.setattr(pipeline, "_read_image", _fake_read_image)
    monkeypatch.setattr(inference, "run_main", _fake_vinyl_detection)
    monkeypatch.setattr(pipeline, "is_anomaly", lambda *args, **kwargs: False)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pipeline, "_executor", executor)
        result = asyncio.run(pipeline.run(None, 20.0, "vinyl-normal", _Registry()))

    assert result.status is DetectionStatus.GENERAL_WASTE
    assert result.weight.anomaly is False
    assert result.guidance == []
    assert result.general is not None
    assert result.general.code is GeneralWasteCode.VINYL
