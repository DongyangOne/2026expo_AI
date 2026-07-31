import os

os.environ.setdefault("API_KEY", "test-key")

import numpy as np
import torch

from scripts.audit_verifier_dataset import audit_manifest
from scripts.extract_verifier_crops import (
    CLASS_NAMES,
    category_id,
    letterbox,
    make_source_key,
    should_use_annotations,
)
from scripts.train_verifier import CropVerifier, enabled_tasks_for


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


def test_audit_manifest_accepts_complete_masked_dataset(tmp_path):
    manifest = tmp_path / "manifest.csv"
    lines = [
        "filepath,split,source_id,material,category,dent,label,foreign_material,label_proxy,raw_dirtiness,source_object_count"
    ]
    for split in ("training", "validation"):
        for material, class_name in enumerate(CLASS_NAMES):
            relative = f"{split}/{class_name}/{material}.jpg"
            image = tmp_path / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"image")
            lines.append(
                f"{relative},{split},{split}-{material},{material},{class_name},-1,-1,-1,-1,,1"
            )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = audit_manifest(manifest, require_masked_status=True)

    assert result["ok"] is True
    assert result["rows"] == 18
    assert result["split_overlap_sources"] == 0
    assert result["source_object_counts"] == [1]


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
