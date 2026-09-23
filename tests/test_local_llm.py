import os

os.environ.setdefault("API_KEY", "test-key")

import json

import numpy as np
import pytest

from app.core.config import Settings
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


def test_parse_accepts_llm_only_general_waste_class():
    answer = json.dumps({
        "material": "general_waste", "confidence": 0.91, "has_label": False,
        "is_dented": False, "has_foreign_material": False,
        "is_single_primary_item": True,
    })
    result = local_llm._parse(answer)
    assert result.class_id == 9
    assert result.class_name == "general_waste"


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


def test_local_llm_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="LOCAL_LLM_MODE"):
        Settings(API_KEY="test-key", LOCAL_LLM_MODE="unsafe")


def test_local_llm_defaults_bound_json_generation():
    settings = Settings(API_KEY="test-key")
    assert settings.LOCAL_LLM_MAX_TOKENS == 80
    assert settings.LOCAL_LLM_MAX_IMAGE_SIDE == 448
    assert settings.LOCAL_LLM_KEEP_ALIVE == "24h"


def test_prompt_limits_foreign_material_to_the_primary_item():
    assert "physically attached to, inside, or mixed with the primary item" in local_llm._PROMPT
    assert "surrounding scene" in local_llm._PROMPT


def test_crop_encoding_is_bounded(monkeypatch):
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_SIDE", 64)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_BYTES", 100_000)
    encoded = local_llm._crop_as_jpeg(np.zeros((300, 600, 3), dtype=np.uint8), [0, 0, 600, 300])
    assert encoded is not None
