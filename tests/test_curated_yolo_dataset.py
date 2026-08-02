import base64
import csv
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from scripts.select_curated_yolo_dataset import (
    CLASS_NAMES,
    build_dataset,
    diverse_order,
)


def _b64(path: Path) -> str:
    return base64.urlsafe_b64encode(str(path).encode()).decode().rstrip("=")


def _row(path: Path, category: str, index: int, split: str = "training"):
    return {
        "filepath": f"{split}/{category}/{index}.jpg",
        "split": split,
        "source_id": f"{category}-{index}",
        "material": str(CLASS_NAMES.index(category)),
        "category": category,
        "dent": str(index % 2),
        "label": "-1",
        "foreign_material": "-1",
        "label_proxy": "-1",
        "raw_dirtiness": "오염없음" if index % 2 else "이물질(외부)",
        "source_object_count": "1",
        "source_path_b64": _b64(path),
        "source_bbox_x": "40",
        "source_bbox_y": "30",
        "source_bbox_w": "240",
        "source_bbox_h": "300",
        "source_width": "400",
        "source_height": "400",
    }


def test_diverse_order_is_deterministic_and_keeps_all_rows(tmp_path):
    image = tmp_path / "a.jpg"
    rows = [_row(image, "can", index) for index in range(8)]
    first = diverse_order(rows, 7)
    second = diverse_order(rows, 7)
    assert [row["source_id"] for row in first] == [row["source_id"] for row in second]
    assert {row["source_id"] for row in first} == {row["source_id"] for row in rows}


def test_build_dataset_selects_classes_and_keeps_hardware_holdout_out(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows = []
    for class_index, category in enumerate(CLASS_NAMES):
        for index in range(2):
            image = np.full((400, 400, 3), 50 + class_index * 15, dtype=np.uint8)
            cv2.line(image, (10 + index, 20), (350, 360 - index), (255, 255, 255), 5)
            path = source_dir / f"{category}_{index}.jpg"
            assert cv2.imwrite(str(path), image)
            rows.append(_row(path, category, index))
    validation = np.full((400, 400, 3), 128, dtype=np.uint8)
    validation_path = source_dir / "validation.jpg"
    assert cv2.imwrite(str(validation_path), validation)
    rows.append(_row(validation_path, "can", 99, split="validation"))

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    hardware = tmp_path / "hardware"
    (hardware / "images" / "train").mkdir(parents=True)
    (hardware / "labels" / "train").mkdir(parents=True)
    assert cv2.imwrite(str(hardware / "images" / "train" / "negative.jpg"), validation)
    (hardware / "labels" / "train" / "negative.txt").write_text("", encoding="utf-8")

    output = tmp_path / "output"
    args = Namespace(
        manifest=str(manifest), output_dir=str(output),
        hardware_yolo_dir=str(hardware), hardware_repeats=2,
        validation_images="/app/original/val/images", per_class=1,
        imgsz=320, workers=2, seed=1, min_area_ratio=0.02,
        max_area_ratio=0.95, min_focus=1.0, min_brightness=1.0,
        max_brightness=254.0, min_free_gb=0.0,
    )
    summary = build_dataset(args)

    assert summary["selected_original"] == 9
    assert summary["hardware_repeated"] == 2
    assert len(list((output / "images" / "train").iterdir())) == 11
    assert len(list((output / "labels" / "train").iterdir())) == 11
    yaml = (output / "dataset.yaml").read_text(encoding="utf-8")
    assert "val: /app/original/val/images" in yaml
    assert "8: fluorescent" in yaml
