import os

os.environ.setdefault("API_KEY", "test-key")

import numpy as np
import torch

from scripts.extract_verifier_crops import CLASS_NAMES, category_id, letterbox, make_source_key
from scripts.train_verifier import CropVerifier


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


def test_crop_verifier_exposes_material_and_three_state_heads():
    model = CropVerifier("mobilenet_v3_small", pretrained=False).eval()

    with torch.inference_mode():
        material, dent, label, foreign = model(torch.randn(1, 3, 64, 64))

    assert material.shape == (1, 9)
    assert dent.shape == (1, 2)
    assert label.shape == (1, 2)
    assert foreign.shape == (1, 2)
