import os

os.environ.setdefault("API_KEY", "test-key")

import json

import numpy as np

from app.services import local_llm


def test_parse_accepts_exact_vision_contract():
    answer = json.dumps({
        "material": "vinyl", "confidence": 0.91, "has_label": False,
        "is_dented": False, "has_foreign_material": False,
        "is_single_primary_item": True,
    })
    result = local_llm._parse(answer)
    assert result.class_id == 5
    assert result.class_name == "vinyl"


def test_parse_rejects_extra_or_invalid_fields():
    invalid = json.dumps({
        "material": "vinyl", "confidence": 0.91, "has_label": False,
        "is_dented": False, "has_foreign_material": False,
        "is_single_primary_item": True, "explanation": "extra",
    })
    try:
        local_llm._parse(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid LLM contract accepted")


def test_crop_encoding_is_bounded(monkeypatch):
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_SIDE", 64)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_BYTES", 100_000)
    encoded = local_llm._crop_as_jpeg(np.zeros((300, 600, 3), dtype=np.uint8), [0, 0, 600, 300])
    assert encoded is not None
