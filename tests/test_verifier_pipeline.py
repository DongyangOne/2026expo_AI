import csv
import os

os.environ.setdefault("API_KEY", "test-key")

import numpy as np
import pytest
import torch

from app.services.inference import run_verifier
from scripts.audit_verifier_dataset import audit_manifest
from scripts.audit_pseudo_status import audit_pseudo_status
from scripts.extract_verifier_crops import (
    CLASS_NAMES,
    _cap_tasks_per_class,
    category_id,
    check_storage_limits,
    decode_source_path,
    encode_source_path,
    letterbox,
    make_source_key,
    should_use_annotations,
)


def test_crop_task_cap_is_applied_per_material():
    tasks = []
    for material in (0, 1):
        for index in range(7):
            task = [None] * 9
            task[0] = f"image-{material}-{index}.jpg"
            task[3] = material
            tasks.append(tuple(task))

    selected = _cap_tasks_per_class(tasks, 5)

    assert len(selected) == 10
    assert sum(task[3] == 0 for task in selected) == 5
    assert sum(task[3] == 1 for task in selected) == 5
from scripts.train_verifier import CropVerifier, enabled_tasks_for
from scripts.pseudo_label_status_qwen import (
    _resize_max_side,
    _round_robin,
    _select_candidates,
    accepted_status,
    consensus_teacher,
    needs_adaptive_second_pass,
    ollama_url_for_row,
    parse_teacher_output,
)


def test_adaptive_consensus_only_rechecks_risky_or_audited_samples():
    row = {
        "source_id": "clean", "filepath": "clean.jpg", "raw_dirtiness": "오염없음"
    }
    clean = {
        "decision": "neither", "is_single_primary_item": True, "confidence": 0.99,
    }
    assert needs_adaptive_second_pass(clean, row, 0.97, 0.0) is False

    dirty = dict(row, raw_dirtiness="이물질(외부)")
    assert needs_adaptive_second_pass(clean, dirty, 0.97, 0.0) is False

    low_confidence = dict(clean, confidence=0.95)
    assert needs_adaptive_second_pass(low_confidence, row, 0.97, 0.0) is True

    positive = dict(clean, decision="foreign_only")
    assert needs_adaptive_second_pass(positive, row, 0.97, 0.0) is True
    label = dict(clean, decision="label_only")
    assert needs_adaptive_second_pass(label, dict(row, category="can"), 0.97, 0.0) is False
    assert needs_adaptive_second_pass(label, dict(row, category="pet"), 0.97, 0.0) is True
    assert needs_adaptive_second_pass(clean, row, 0.97, 1.0) is True


def test_compact_teacher_wire_format_expands_to_stable_record_contract():
    parsed = parse_teacher_output(
        '{"d":"label_only","s":true,"c":0.96,"e":"label"}'
    )
    assert parsed == {
        "decision": "label_only",
        "has_removable_label": True,
        "has_true_foreign_material": False,
        "same_material_accessory_only": False,
        "is_single_primary_item": True,
        "confidence": 0.96,
        "reason": "label",
    }

    repaired = parse_teacher_output(
        '{"d":"neither","l":false,"f":true,"a":false,'
        '"s":true,"c":0.85,"e":"contamination"}'
    )
    assert repaired["decision"] == "foreign_only"
    assert repaired["has_true_foreign_material"] is True

    accessory = parse_teacher_output(
        '{"d":"neither","s":true,"c":0.99,"e":"same_material_accessory"}'
    )
    assert accessory["same_material_accessory_only"] is True


def test_ollama_multi_instance_routing_is_stable_per_sample():
    urls = "http://ollama-a:11434,http://ollama-b:11434"
    rows = [
        {"source_id": f"item-{index}", "filepath": f"{index}.jpg"}
        for index in range(20)
    ]
    routed = [ollama_url_for_row(urls, row) for row in rows]
    assert set(routed) == {"http://ollama-a:11434", "http://ollama-b:11434"}
    assert routed == [ollama_url_for_row(urls, row) for row in rows]


def test_teacher_retry_selects_only_previous_errors():
    base = [
        {
            "source_id": "ok", "filepath": "ok.jpg", "split": "training",
            "category": "pet", "raw_dirtiness": "",
        },
        {
            "source_id": "error", "filepath": "error.jpg", "split": "validation",
            "category": "can", "raw_dirtiness": "",
        },
        {
            "source_id": "pending", "filepath": "pending.jpg", "split": "training",
            "category": "paper", "raw_dirtiness": "",
        },
    ]
    previous = {
        ("ok", "ok.jpg"): {"accepted": {"status_eligible": 1}},
        ("error", "error.jpg"): {"error": "timeout"},
    }

    selected = _select_candidates(base, previous, retry_errors_only=True)

    assert [(row["source_id"], row["filepath"]) for row in selected] == [
        ("error", "error.jpg")
    ]


def test_teacher_resize_limits_only_large_images():
    large = np.zeros((1200, 800, 3), dtype=np.uint8)
    small = np.zeros((320, 240, 3), dtype=np.uint8)

    resized = _resize_max_side(large, 640)

    assert resized.shape == (640, 427, 3)
    assert _resize_max_side(small, 640) is small


def test_teacher_round_robin_balances_dirtiness_groups():
    rows = []
    for category in ("can", "paper", "pet"):
        for dirtiness in ("오염없음", "이물질(외부)"):
            rows.append({
                "split": "training", "category": category,
                "raw_dirtiness": dirtiness, "id": f"{category}-{dirtiness}",
            })

    ordered = _round_robin(rows)

    assert [row["category"] for row in ordered[:3]] == ["can", "paper", "pet"]
    assert len({row["raw_dirtiness"] for row in ordered[:3]}) == 2


def test_automatic_consensus_ignores_auxiliary_accessory_disagreement():
    base = {
        "decision": "neither",
        "has_removable_label": False,
        "has_true_foreign_material": False,
        "is_single_primary_item": True,
        "confidence": 0.97,
        "reason": "same target",
    }
    without_accessory = {**base, "same_material_accessory_only": False}
    with_accessory = {**base, "same_material_accessory_only": True}

    consensus = consensus_teacher([without_accessory, with_accessory])

    assert consensus["decision"] == "neither"
    assert consensus["same_material_accessory_only"] is True


def test_category_mapping_covers_all_verifier_materials():
    folder_names = [
        "TL_2.직접촬영_01.금속캔_001.철캔",
        "TL_2.직접촬영_03.페트병_001.무색단일",
        "TL_2.직접촬영_02.종이_001.종이",
        "TL_2.직접촬영_04.플라스틱_001.PE",
        "TL_2.직접촬영_05.스티로폼_001.스티로폼",
        "TL_2.직접촬영_06.비닐_001.비닐",
        "TL_2.직접촬영_07.유리병_003.투명",
        "TL_2.직접촬영_08.건전지_001.건전지",
        "TL_2.직접촬영_09.형광등_001.형광등",
    ]

    assert [category_id(name) for name in folder_names] == list(range(len(CLASS_NAMES)))


def test_letterbox_preserves_full_crop_with_padding():
    image = np.full((40, 80, 3), 255, dtype=np.uint8)

    result = letterbox(image, 100)

    assert result.shape == (100, 100, 3)
    assert np.all(result[25:75] == 255)
    assert np.all(result[:25] == 114)


def test_source_key_accepts_filesystem_surrogate_names():
    key = make_source_key("Validation", "VL_2.직접촬영", "broken_\udcc7.json")

    assert len(key) == 20
    assert all(character in "0123456789abcdef" for character in key)


def test_source_path_roundtrips_filesystem_surrogate_names():
    source = "broken_\udcc7/image.jpg"

    restored = decode_source_path(encode_source_path(source))

    assert os.fsencode(str(restored)) == os.fsencode(str(type(restored)(source)))


def test_storage_guards_reject_low_free_space_and_large_output():
    gib = 1024 ** 3
    check_storage_limits(600 * gib, 10 * gib, min_free_gb=500, max_output_gb=20)

    with pytest.raises(RuntimeError, match="free space guard"):
        check_storage_limits(499 * gib, 10 * gib, min_free_gb=500, max_output_gb=20)
    with pytest.raises(RuntimeError, match="output guard"):
        check_storage_limits(600 * gib, 21 * gib, min_free_gb=500, max_output_gb=20)


def test_pseudo_label_parser_and_same_material_accessory_allowance():
    teacher = parse_teacher_output(
        '{"decision":"foreign_only","has_removable_label":false,'
        '"has_true_foreign_material":true,"same_material_accessory_only":false,'
        '"is_single_primary_item":true,"confidence":0.96,"reason":"paper attached"}'
    )

    assert accepted_status(teacher, "pet", 0.90) == {
        "label": 0,
        "foreign_material": 1,
        "status_eligible": 1,
    }

    excluded = parse_teacher_output(
        '{"decision":"neither","has_removable_label":false,'
        '"has_true_foreign_material":false,"same_material_accessory_only":true,'
        '"is_single_primary_item":true,"confidence":0.99,"reason":"plastic straw"}'
    )
    assert accepted_status(excluded, "pet", 0.90) == {
        "label": 0,
        "foreign_material": 0,
        "status_eligible": 1,
    }


def test_non_label_material_keeps_label_masked():
    teacher = parse_teacher_output(
        '{"decision":"neither","has_removable_label":false,'
        '"has_true_foreign_material":false,"same_material_accessory_only":false,'
        '"is_single_primary_item":true,"confidence":0.95,"reason":"clean can"}'
    )

    assert accepted_status(teacher, "can", 0.90) == {
        "label": -1,
        "foreign_material": 0,
        "status_eligible": 1,
    }


def test_automatic_consensus_rejects_disagreeing_views():
    clean = parse_teacher_output(
        '{"decision":"neither","has_removable_label":false,'
        '"has_true_foreign_material":false,"same_material_accessory_only":false,'
        '"is_single_primary_item":true,"confidence":0.97,"reason":"clean"}'
    )
    foreign = parse_teacher_output(
        '{"decision":"foreign_only","has_removable_label":false,'
        '"has_true_foreign_material":true,"same_material_accessory_only":false,'
        '"is_single_primary_item":true,"confidence":0.95,"reason":"paper"}'
    )

    consensus = consensus_teacher([clean, foreign])

    assert consensus["decision"] == "ambiguous"
    assert accepted_status(consensus, "pet", 0.90)["status_eligible"] == 0


def test_single_object_filter_rejects_multiple_annotations():
    assert should_use_annotations([{"ID": "1"}]) is True
    assert should_use_annotations([{"ID": "1"}, {"ID": "2"}]) is False
    assert should_use_annotations([], single_object_only=False) is False


def test_crop_verifier_exposes_material_and_three_state_heads():
    model = CropVerifier("mobilenet_v3_small", pretrained=False).eval()

    with torch.inference_mode():
        material, dent, label, foreign = model(torch.randn(1, 3, 64, 64))

    assert material.shape == (1, 9)
    assert dent.shape == (1, 2)
    assert label.shape == (1, 2)
    assert foreign.shape == (1, 2)


def test_runtime_verifier_uses_model_input_size_and_four_named_outputs():
    class _Node:
        def __init__(self, name, shape=None):
            self.name = name
            self.shape = shape

    class _Session:
        def __init__(self):
            self.input_shape = None

        def get_inputs(self):
            return [_Node("img", [1, 3, 320, 320])]

        def get_outputs(self):
            return [_Node(name) for name in ("material", "dent", "label", "foreign_material")]

        def run(self, names, inputs):
            assert names == ["material", "dent", "label", "foreign_material"]
            self.input_shape = inputs["img"].shape
            return [
                np.array([[0, 0, 8, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
                np.array([[0, 5]], dtype=np.float32),
                np.array([[4, 0]], dtype=np.float32),
                np.array([[0, 6]], dtype=np.float32),
            ]

    session = _Session()
    result = run_verifier(
        session, np.zeros((480, 640, 3), dtype=np.uint8), [100, 80, 500, 420]
    )

    assert session.input_shape == (1, 3, 320, 320)
    assert result["material"]["class_name"] == "paper"
    assert result["heads"]["dent"]["value"] is True
    assert result["heads"]["label"]["value"] is False
    assert result["heads"]["foreign_material"]["value"] is True


def test_audit_manifest_accepts_complete_masked_dataset(tmp_path):
    manifest = tmp_path / "manifest.csv"
    source = tmp_path / "high_resolution_source.jpg"
    source.write_bytes(b"high resolution image")
    lines = [
        "filepath,split,source_id,material,category,dent,label,foreign_material,"
        "label_proxy,raw_dirtiness,source_object_count,source_path_b64,source_bbox_x,"
        "source_bbox_y,source_bbox_w,source_bbox_h,source_width,source_height"
    ]
    for split in ("training", "validation"):
        for material, class_name in enumerate(CLASS_NAMES):
            relative = f"{split}/{class_name}/{material}.jpg"
            image = tmp_path / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"image")
            lines.append(
                f"{relative},{split},{split}-{material},{material},{class_name},"
                f"-1,-1,-1,-1,,1,{encode_source_path(source)},10,20,100,200,1920,1080"
            )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = audit_manifest(
        manifest, require_masked_status=True, require_source_references=True
    )

    assert result["ok"] is True
    assert result["rows"] == 18
    assert result["split_overlap_sources"] == 0
    assert result["source_object_counts"] == [1]
    assert result["missing_source_images"] == 0


def test_status_heads_require_both_classes_in_train_and_validation():
    base = []
    for split in ("training", "validation"):
        for material in range(9):
            base.append(
                {
                    "split": split,
                    "material": material,
                    "dent": material % 2 if material in (0, 1) else -1,
                    "label": -1,
                    "foreign_material": -1,
                }
            )

    assert enabled_tasks_for(base) == ["material", "dent"]

    for split in ("training", "validation"):
        for value in (0, 1):
            base.append(
                {
                    "split": split,
                    "material": 1,
                    "dent": value,
                    "label": value,
                    "foreign_material": value,
                }
            )

    assert enabled_tasks_for(base) == ["material", "dent", "label", "foreign_material"]


def test_pseudo_status_audit_accepts_fully_automatic_balanced_heads(tmp_path):
    manifest = tmp_path / "manifest_with_pseudo_status.csv"
    fields = [
        "split", "category", "label", "foreign_material", "status_eligible",
        "teacher_status", "teacher_confidence", "teacher_model", "teacher_rejected",
    ]
    rows = []
    for split in ("training", "validation"):
        for value, decision in ((0, "neither"), (1, "both")):
            rows.append({
                "split": split,
                "category": "pet",
                "label": value,
                "foreign_material": value,
                "status_eligible": 1,
                "teacher_status": decision,
                "teacher_confidence": 0.97,
                "teacher_model": "qwen3.5:9b-q4_K_M",
                "teacher_rejected": 0,
            })
    with open(manifest, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    result = audit_pseudo_status(manifest, min_coverage=0.90, require_ready_heads=True)

    assert result["ok"] is True
    assert result["head_ready"] == {"label": True, "foreign_material": True}


def test_pseudo_status_audit_blocks_pending_and_inconsistent_rows(tmp_path):
    manifest = tmp_path / "manifest_with_pseudo_status.csv"
    manifest.write_text(
        "split,category,label,foreign_material,status_eligible,teacher_status,"
        "teacher_confidence,teacher_model,teacher_rejected\n"
        "training,pet,1,-1,1,label_only,0.99,qwen3.5:9b-q4_K_M,0\n"
        "validation,can,-1,-1,,,,,\n",
        encoding="utf-8",
    )

    result = audit_pseudo_status(manifest, require_complete=True)

    assert result["ok"] is False
    assert result["pending_rows"] == 1
    assert result["invalid_rows"] == 1


def test_pseudo_status_audit_can_tolerate_bounded_teacher_errors(tmp_path):
    manifest = tmp_path / "manifest_with_pseudo_status.csv"
    fields = [
        "split", "category", "label", "foreign_material", "status_eligible",
        "teacher_status", "teacher_confidence", "teacher_model", "teacher_rejected",
    ]
    rows = []
    for split in ("training", "validation"):
        for value, decision in ((0, "neither"), (1, "both")):
            rows.append({
                "split": split, "category": "pet", "label": value,
                "foreign_material": value, "status_eligible": 1,
                "teacher_status": decision, "teacher_confidence": 0.97,
                "teacher_model": "qwen3.5:9b-q4_K_M", "teacher_rejected": 0,
            })
    rows.append({
        "split": "training", "category": "pet", "label": -1,
        "foreign_material": -1, "status_eligible": 0,
        "teacher_status": "error", "teacher_confidence": "",
        "teacher_model": "qwen3.5:9b-q4_K_M", "teacher_rejected": 1,
    })
    with open(manifest, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    strict = audit_pseudo_status(manifest, require_ready_heads=True)
    tolerant = audit_pseudo_status(
        manifest, require_ready_heads=True, max_teacher_error_rate=0.20
    )

    assert strict["ok"] is False
    assert tolerant["ok"] is True
    assert tolerant["teacher_error_rate"] == 0.2
