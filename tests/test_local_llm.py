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
    assert settings.LOCAL_LLM_MODEL == "minicpm-v4.5:8b"
    assert settings.LOCAL_LLM_MAX_TOKENS == 96
    assert settings.LOCAL_LLM_FULL_IMAGE_SIDE == 768
    assert settings.LOCAL_LLM_MAX_IMAGE_SIDE == 640
    assert settings.LOCAL_LLM_KEEP_ALIVE == "24h"


def test_prompt_limits_foreign_material_to_the_primary_item():
    assert "physically attached to, inside, or mixed with the primary item" in local_llm._PROMPT
    assert "Background objects are not foreign material" in local_llm._PROMPT


def test_prompt_defines_material_by_physical_form():
    assert "thin flexible film" in local_llm._PROMPT
    assert "rigid polymer" in local_llm._PROMPT
    assert "Image 1 is the complete camera frame" in local_llm._PROMPT


def test_crop_encoding_is_bounded(monkeypatch):
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_SIDE", 64)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_BYTES", 100_000)
    encoded = local_llm._crop_as_jpeg(np.zeros((300, 600, 3), dtype=np.uint8), [0, 0, 600, 300])
    assert encoded is not None


def test_two_images_are_sent_for_a_distinct_detection(monkeypatch):
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_FULL_IMAGE_SIDE", 64)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_SIDE", 64)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_BYTES", 100_000)
    images = local_llm._images_as_jpeg(
        np.zeros((200, 300, 3), dtype=np.uint8), [50, 40, 250, 180],
    )
    assert len(images) == 2


def test_full_frame_fallback_is_not_duplicated(monkeypatch):
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_FULL_IMAGE_SIDE", 64)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_SIDE", 64)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_BYTES", 100_000)
    images = local_llm._images_as_jpeg(
        np.zeros((200, 300, 3), dtype=np.uint8), [0, 0, 300, 200],
    )
    assert len(images) == 1


def test_foreign_material_recheck_uses_only_target_crop(monkeypatch):
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_FULL_IMAGE_SIDE", 64)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_SIDE", 64)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MAX_IMAGE_BYTES", 100_000)
    images = local_llm._images_as_jpeg(
        np.zeros((200, 300, 3), dtype=np.uint8),
        [50, 40, 250, 180],
        crop_only=True,
    )
    assert len(images) == 1
    assert "bin fixture or clamp" in local_llm._FOREIGN_MATERIAL_PROMPT
