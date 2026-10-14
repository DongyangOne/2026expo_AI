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
        "compression_required": False, "is_dented": False,
        "has_straw": True, "has_cup_holder": False,
        "has_foreign_material": False,
        "is_single_primary_item": True,
    })
    result = local_llm._parse(answer)
    assert result.class_id == 5
    assert result.class_name == "vinyl"
    assert result.compression_required is False
    assert result.has_straw is True
    assert result.has_cup_holder is False


def test_parse_requires_compression_required_decision():
    answer = json.dumps({
        "material": "plastic", "confidence": 0.91, "has_label": False,
        "has_straw": False, "has_cup_holder": False,
        "is_dented": False, "has_foreign_material": False,
        "is_single_primary_item": True,
    })
    with pytest.raises(ValueError):
        local_llm._parse(answer)


def test_parse_accepts_llm_only_general_waste_class():
    answer = json.dumps({
        "material": "general_waste", "confidence": 0.91, "has_label": False,
        "compression_required": False, "is_dented": False,
        "has_straw": False, "has_cup_holder": False,
        "has_foreign_material": False,
        "is_single_primary_item": True,
    })
    result = local_llm._parse(answer)
    assert result.class_id == 9
    assert result.class_name == "general_waste"


def test_parse_normalizes_empty_scene_to_non_item_sentinel():
    answer = json.dumps({
        "material": "empty_scene", "confidence": 0.98, "has_label": False,
        "compression_required": False, "is_dented": False,
        "has_straw": False, "has_cup_holder": False,
        "has_foreign_material": False,
        "is_single_primary_item": True,
    })
    result = local_llm._parse(answer)
    assert result.class_id == -1
    assert result.class_name == "empty_scene"
    assert result.is_single_primary_item is False


def test_parse_rejects_extra_or_invalid_fields():
    invalid = json.dumps({
        "material": "vinyl", "confidence": 0.91, "has_label": False,
        "compression_required": False, "is_dented": False,
        "has_straw": False, "has_cup_holder": False,
        "has_foreign_material": False,
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
    assert settings.LOCAL_LLM_MAX_TOKENS == 128
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
    assert "Image 1 is the authoritative scene view" in local_llm._PROMPT
    assert "non-authoritative detail view" in local_llm._PROMPT
    assert "ignore Image 2" in local_llm._PROMPT
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


def test_prompt_keeps_crushed_molded_cups_out_of_vinyl():
    assert "remains plastic even when crushed" in local_llm._PROMPT
    assert "continuous molded circular rim" in local_llm._PROMPT
    assert "outrank wrinkles caused by crushing" in local_llm._PROMPT
    assert "has no molded cup rim or formed base" in local_llm._PROMPT


def test_prompt_can_explicitly_reject_an_empty_platform():
    assert "empty_scene" in local_llm._SCHEMA["properties"]["material"]["enum"]
    assert "When no disposal item is present on the platform" in local_llm._PROMPT
    assert "is_single_primary_item=false" in local_llm._PROMPT


def test_prompt_spells_out_the_json_contract_for_json_object_mode():
    assert "Return exactly these keys" in local_llm._PROMPT
    for field in local_llm._SCHEMA["required"]:
        assert field in local_llm._PROMPT


def test_prompt_accepts_visibly_crushed_can_as_compressed():
    assert "buckled, folded, shortened, collapsed, wrinkled" in local_llm._PROMPT
    assert "flattened into a puck-like or irregular compact shape" in local_llm._PROMPT
    assert "smooth, full-height cylinder" in local_llm._PROMPT


def test_prompt_keeps_glossy_folded_boxes_in_paper_class():
    assert "folded box, envelope, gift bag, or cardboard package" in local_llm._PROMPT
    assert "glossy, laminated, coated, or heavily printed" in local_llm._PROMPT
    assert "folded flaps, scored creases, glued seams" in local_llm._PROMPT


def test_prompt_requires_pet_sleeve_removal_and_substantial_compression():
    assert "printed sleeve or wrap around a PET or plastic bottle body" in local_llm._PROMPT
    assert "transparent, translucent, colored, or opaque" in local_llm._PROMPT
    assert "sidewall is visibly folded inward or remains pressed in" in local_llm._PROMPT
    assert "silhouette, height, or width is visibly reduced" in local_llm._PROMPT
    assert "shallow wrinkles or small local dimples" in local_llm._PROMPT


def test_runtime_context_marks_weight_and_yolo_as_non_authoritative():
    prompt = local_llm._with_runtime_context(
        "base prompt", weight_g=12.5, yolo_material="paper", yolo_confidence=0.9732,
    )
    assert "Measured scale weight: 12.50 g" in prompt
    assert "YOLO proposal: paper at confidence 0.9732" in prompt
    assert "may be wrong on a new camera domain" in prompt


def test_primary_classification_includes_weight_as_supporting_evidence(monkeypatch):
    captured = {}

    def fake_classify_with_prompt(img, bbox, prompt, **kwargs):
        captured["prompt"] = prompt
        return None

    monkeypatch.setattr(local_llm, "_classify_with_prompt", fake_classify_with_prompt)

    local_llm.classify(
        np.zeros((32, 32, 3), dtype=np.uint8),
        [0.0, 0.0, 32.0, 32.0],
        weight_g=37.2,
    )

    assert "Measured scale weight: 37.20 g" in captured["prompt"]
    assert "supporting hints, never ground truth" in captured["prompt"]


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


def test_openrouter_valid_json_is_used_without_calling_ollama(monkeypatch):
    answer = json.dumps({
        "material": "paper", "confidence": 0.96, "has_label": False,
        "compression_required": False, "is_dented": False, "has_straw": False,
        "has_cup_holder": False, "has_foreign_material": False,
        "is_single_primary_item": True,
    })
    monkeypatch.setattr(local_llm, "enabled", lambda: True)
    monkeypatch.setattr(local_llm.openrouter_client, "enabled", lambda: True)
    generation = local_llm.openrouter_client.OpenRouterGeneration(
        content=answer, model="qwen/qwen3.8-27b",
        provider_requested="deepinfra", provider_returned="DeepInfra",
        duration_ms=850.0, input_tokens=120, output_tokens=20,
        total_tokens=140, cost_usd=0.0004,
    )
    monkeypatch.setattr(
        local_llm.openrouter_client, "generate_with_metadata", lambda *_args: generation,
    )
    monkeypatch.setattr(
        local_llm, "_query_ollama",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama called")),
    )

    result = local_llm._classify_with_prompt(
        np.zeros((32, 32, 3), dtype=np.uint8), [0, 0, 32, 32], "prompt"
    )

    assert result is not None
    assert result.class_name == "paper"
    assert result.runtime == {
        "backend": "openrouter", "model": "qwen/qwen3.8-27b",
        "provider_requested": "deepinfra", "provider_returned": "DeepInfra",
        "duration_ms": 850.0, "input_tokens": 120, "output_tokens": 20,
        "total_tokens": 140, "cost_usd": 0.0004,
    }


def test_openrouter_failure_falls_back_to_existing_ollama(monkeypatch):
    fallback = local_llm.LocalLLMPrediction(
        class_id=5, class_name="vinyl", confidence=0.91, has_label=False,
        compression_required=False, is_dented=False, has_straw=False,
        has_cup_holder=False, has_foreign_material=False, is_single_primary_item=True,
    )
    monkeypatch.setattr(local_llm, "enabled", lambda: True)
    monkeypatch.setattr(local_llm.openrouter_client, "enabled", lambda: True)
    monkeypatch.setattr(
        local_llm.openrouter_client, "generate_with_metadata", lambda *_args: None,
    )
    monkeypatch.setattr(local_llm, "_query_ollama", lambda *_args, **_kwargs: fallback)

    result = local_llm._classify_with_prompt(
        np.zeros((32, 32, 3), dtype=np.uint8), [0, 0, 32, 32], "prompt"
    )

    assert result == fallback


def test_openrouter_failure_without_local_backend_returns_none(monkeypatch):
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MODE", "disabled")
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_BASE_URL", None)
    monkeypatch.setattr(local_llm.openrouter_client, "enabled", lambda: True)
    monkeypatch.setattr(
        local_llm.openrouter_client, "generate_with_metadata", lambda *_args: None,
    )

    result = local_llm._classify_with_prompt(
        np.zeros((32, 32, 3), dtype=np.uint8), [0, 0, 32, 32], "prompt"
    )

    assert result is None


class _HealthResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _HealthClient:
    def __init__(self, payloads):
        self._payloads = iter(payloads)

    def get(self, *_args, **_kwargs):
        return _HealthResponse(next(self._payloads))


def test_health_status_reports_gpu_residency(monkeypatch):
    monkeypatch.setattr(local_llm, "enabled", lambda: True)
    monkeypatch.setattr(local_llm, "primary_enabled", lambda: True)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MODEL", "minicpm-v4.5:8b")
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_BASE_URL", "https://naco.example")
    monkeypatch.setattr(local_llm, "_get_http_client", lambda: _HealthClient([
        {"models": [{"name": "minicpm-v4.5:8b"}]},
        {"models": [{"name": "minicpm-v4.5:8b", "size_vram": 6818260582}]},
    ]))

    status = local_llm.health_status()

    assert status["status"] == "ok"
    assert status["accelerator"] == "gpu"
    assert status["size_vram_bytes"] == 6818260582


def test_health_status_rejects_loaded_cpu_fallback(monkeypatch):
    monkeypatch.setattr(local_llm, "enabled", lambda: True)
    monkeypatch.setattr(local_llm, "primary_enabled", lambda: True)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MODEL", "minicpm-v4.5:8b")
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_BASE_URL", "https://naco.example")
    monkeypatch.setattr(local_llm, "_get_http_client", lambda: _HealthClient([
        {"models": [{"name": "minicpm-v4.5:8b"}]},
        {"models": [{"name": "minicpm-v4.5:8b", "size_vram": 0}]},
    ]))

    status = local_llm.health_status()

    assert status["status"] == "cpu_fallback"
    assert status["accelerator"] == "cpu"
    assert status["size_vram_bytes"] == 0


def test_health_status_reports_model_not_loaded(monkeypatch):
    monkeypatch.setattr(local_llm, "enabled", lambda: True)
    monkeypatch.setattr(local_llm, "primary_enabled", lambda: True)
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_MODEL", "minicpm-v4.5:8b")
    monkeypatch.setattr(local_llm.settings, "LOCAL_LLM_BASE_URL", "https://naco.example")
    monkeypatch.setattr(local_llm, "_get_http_client", lambda: _HealthClient([
        {"models": [{"name": "minicpm-v4.5:8b"}]},
        {"models": []},
    ]))

    status = local_llm.health_status()

    assert status["status"] == "model_not_loaded"
    assert status["accelerator"] == "not_loaded"
    assert status["size_vram_bytes"] is None
