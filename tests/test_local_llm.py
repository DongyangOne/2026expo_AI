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
    assert settings.LOCAL_LLM_PLASTIC_RECHECK_IMAGE_SIDE == 896
    assert settings.LOCAL_LLM_KEEP_ALIVE == "24h"


def test_prompt_limits_foreign_material_to_the_primary_item():
    assert "physically attached to, inside, or mixed with the primary item" in local_llm._PROMPT
    assert "Background objects are not foreign material" in local_llm._PROMPT


def test_prompt_defines_material_by_physical_form():
    assert "thin flexible film" in local_llm._PROMPT
    assert "rigid molded polymer" in local_llm._PROMPT
    assert "Image 1 is the complete camera frame" in local_llm._PROMPT
    assert "black lightweight molded foam food trays" in local_llm._PROMPT
    assert "groups bulb-shaped lamps together" in local_llm._PROMPT
    assert "does not make the primary item paper" in local_llm._PROMPT
    assert "dark or amber bottle is not glass" in local_llm._PROMPT
    assert "rolled paper rim" in local_llm._PROMPT
    assert "heat-sealed edges" in local_llm._PROMPT
    assert "printed flexible squeeze tube" in local_llm._PROMPT
    assert "film sheet draped over any support" in local_llm._VINYL_PLASTIC_PROMPT
    assert "Decision order:" not in local_llm._PROMPT
    assert "Decision order:" in local_llm._FORM_REASONING_PROMPT


def test_runtime_context_marks_weight_and_yolo_as_non_authoritative():
    prompt = local_llm._with_runtime_context(
        "base prompt", weight_g=12.5, yolo_material="paper", yolo_confidence=0.9732,
    )
    assert "Measured scale weight: 12.50 g" in prompt
    assert "YOLO proposal: paper at confidence 0.9732" in prompt
    assert "may be wrong on a new camera domain" in prompt


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
    assert "Background objects are not foreign material" in local_llm._PROMPT


def test_rejection_conflict_recheck_accepts_only_the_two_candidates(monkeypatch):
    fluorescent = local_llm.LocalLLMPrediction(
        class_id=8,
        class_name="fluorescent",
        confidence=0.96,
        has_label=False,
        is_dented=False,
        has_foreign_material=False,
        is_single_primary_item=True,
    )
    captured = {}

    def fake_classify(_img, _bbox, prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return fluorescent

    monkeypatch.setattr(local_llm, "_classify_with_prompt", fake_classify)
    result = local_llm.reclassify_rejection_conflict(
        np.zeros((100, 100, 3), dtype=np.uint8),
        [0.0, 0.0, 100.0, 100.0],
        "fluorescent",
        "plastic",
    )

    assert result == fluorescent
    assert "fluorescent or plastic" in captured["prompt"]
    assert "smooth round diffuser" in captured["prompt"]
    assert captured["kwargs"]["crop_only"] is False


def test_pet_glass_conflict_keeps_crop_only_direction(monkeypatch):
    pet = local_llm.LocalLLMPrediction(
        class_id=1,
        class_name="pet",
        confidence=0.95,
        has_label=True,
        is_dented=False,
        has_foreign_material=False,
        is_single_primary_item=True,
    )
    captured = {}

    def fake_classify(_img, _bbox, prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return pet

    monkeypatch.setattr(local_llm, "_classify_with_prompt", fake_classify)
    result = local_llm.reclassify_rejection_conflict(
        np.zeros((100, 100, 3), dtype=np.uint8),
        [10.0, 10.0, 90.0, 90.0],
        "pet",
        "glass",
    )

    assert result == pet
    assert "threaded neck or collar" in captured["prompt"]
    assert "Decision order:" not in captured["prompt"]
    assert captured["kwargs"]["crop_only"] is True


def test_glass_pet_conflict_uses_full_form_reasoning(monkeypatch):
    glass = local_llm.LocalLLMPrediction(
        class_id=6,
        class_name="glass",
        confidence=0.95,
        has_label=False,
        is_dented=False,
        has_foreign_material=False,
        is_single_primary_item=True,
    )
    captured = {}

    def fake_classify(_img, _bbox, prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return glass

    monkeypatch.setattr(local_llm, "_classify_with_prompt", fake_classify)
    result = local_llm.reclassify_rejection_conflict(
        np.zeros((100, 100, 3), dtype=np.uint8),
        [10.0, 10.0, 90.0, 90.0],
        "glass",
        "pet",
        weight_g=20.0,
        yolo_confidence=0.97,
    )

    assert result == glass
    assert "dark or amber bottle is not glass" in captured["prompt"]
    assert "Decision order:" in captured["prompt"]
    assert captured["kwargs"]["crop_only"] is False


def test_vinyl_paper_conflict_uses_full_form_reasoning(monkeypatch):
    vinyl = local_llm.LocalLLMPrediction(
        class_id=5,
        class_name="vinyl",
        confidence=0.94,
        has_label=False,
        is_dented=False,
        has_foreign_material=False,
        is_single_primary_item=True,
    )
    captured = {}

    def fake_classify(_img, _bbox, prompt, **_kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = _kwargs
        return vinyl

    monkeypatch.setattr(local_llm, "_classify_with_prompt", fake_classify)
    result = local_llm.reclassify_rejection_conflict(
        np.zeros((100, 100, 3), dtype=np.uint8),
        [5.0, 5.0, 95.0, 95.0],
        "vinyl",
        "paper",
        weight_g=2.3,
        yolo_confidence=0.97,
    )

    assert result == vinyl
    assert "heat-sealed edges" in captured["prompt"]
    assert "Decision order:" in captured["prompt"]
    assert "Measured scale weight" not in captured["prompt"]
    assert captured["kwargs"]["crop_only"] is False


@pytest.mark.parametrize("llm_material", ["paper", "styrofoam"])
def test_plastic_conflict_uses_larger_detail_crop(monkeypatch, llm_material):
    plastic = local_llm.LocalLLMPrediction(
        class_id=3,
        class_name="plastic",
        confidence=0.95,
        has_label=False,
        is_dented=False,
        has_foreign_material=False,
        is_single_primary_item=True,
    )
    captured = {}

    def fake_classify(_img, _bbox, prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return plastic

    monkeypatch.setattr(local_llm, "_classify_with_prompt", fake_classify)
    result = local_llm.reclassify_rejection_conflict(
        np.zeros((100, 100, 3), dtype=np.uint8),
        [10.0, 10.0, 90.0, 90.0],
        "plastic",
        llm_material,
    )

    assert result == plastic
    assert captured["crop_max_side"] == 896
    assert "dense rigid injection-molded or thermoformed" in captured["prompt"]
    assert captured["crop_only"] is False


def test_http_client_is_reused(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(local_llm.httpx, "Client", FakeClient)
    monkeypatch.setattr(local_llm, "_http_client", None)

    assert local_llm._get_http_client() is local_llm._get_http_client()
    assert created == [{"timeout": local_llm.settings.LOCAL_LLM_TIMEOUT_SEC}]
