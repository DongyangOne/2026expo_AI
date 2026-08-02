import csv
from pathlib import Path

import pytest

from scripts.merge_pseudo_status_manifest import merge_manifests
from scripts.select_curated_verifier_manifest import CLASS_NAMES, select_manifest
from scripts.train_verifier import read_manifest


def _base_row(category: str, source_id: str, split: str = "training"):
    return {
        "filepath": f"{split}/{category}/{source_id}.jpg",
        "split": split,
        "source_id": source_id,
        "material": str(CLASS_NAMES.index(category)),
        "category": category,
        "dent": "1" if source_id.endswith("1") else "0",
        "label": "-1",
        "foreign_material": "-1",
        "label_proxy": "-1",
        "raw_dirtiness": "오염없음",
        "source_object_count": "1",
        "source_path_b64": "L3RtcC9hLmpwZw",
        "source_bbox_x": "10",
        "source_bbox_y": "10",
        "source_bbox_w": "200",
        "source_bbox_h": "200",
        "crop_x1": "0",
        "crop_y1": "0",
        "crop_x2": "220",
        "crop_y2": "220",
        "source_width": "400",
        "source_height": "400",
        "crop_bytes": "10000",
    }


def _write(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_merge_requires_all_processed_rows_to_match(tmp_path):
    base = [_base_row("pet", "p1"), _base_row("plastic", "x1")]
    pseudo = []
    for row in base:
        item = dict(row)
        item.update({
            "label": "1", "foreign_material": "0", "status_eligible": "1",
            "teacher_status": "label_only", "teacher_confidence": "0.99",
            "teacher_reason": "label", "teacher_model": "qwen", "teacher_rejected": "0",
        })
        pseudo.append(item)
    base_path = tmp_path / "base.csv"
    pseudo_path = tmp_path / "pseudo.csv"
    output = tmp_path / "merged.csv"
    _write(base_path, base)
    _write(pseudo_path, pseudo)

    summary = merge_manifests(base_path, pseudo_path, output, require_processed=2)
    assert summary["matched_processed"] == 2
    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert all(row["label"] == "1" for row in rows)

    with pytest.raises(RuntimeError):
        merge_manifests(base_path, pseudo_path, tmp_path / "bad.csv", require_processed=3)


def test_curated_manifest_balances_common_and_keeps_rare(tmp_path):
    rows = []
    for split in ("training", "validation"):
        for category in CLASS_NAMES:
            count = 3 if category not in {"battery", "fluorescent"} else 2
            for index in range(count):
                row = _base_row(category, f"{category}-{split}-{index}", split)
                if category == "plastic" and index == 0:
                    row["foreign_material"] = "1"
                rows.append(row)
    source = tmp_path / "source.csv"
    output = tmp_path / "selected.csv"
    _write(source, rows)

    summary = select_manifest(source, output, 2, 1, 4_000, 7)
    assert summary["counts"]["training/can"] == 2
    assert summary["counts"]["training/battery"] == 2
    assert summary["counts"]["validation/can"] == 1
    with output.open(encoding="utf-8", newline="") as file:
        chosen = list(csv.DictReader(file))
    assert any(row["category"] == "plastic" and row["foreign_material"] == "1" for row in chosen)


def test_validation_round_robins_positive_and_negative_status(tmp_path):
    rows = []
    for index in range(10):
        row = _base_row("plastic", f"p{index}", "validation")
        row["label"] = str(index % 2)
        row["foreign_material"] = "0"
        rows.append(row)
    # 모든 필수 클래스도 최소 한 행씩 유지한다.
    for split in ("training", "validation"):
        for category in CLASS_NAMES:
            if split == "validation" and category == "plastic":
                continue
            rows.append(_base_row(category, f"{category}-{split}", split))
    source = tmp_path / "source.csv"
    output = tmp_path / "selected.csv"
    _write(source, rows)

    select_manifest(source, output, 1, 4, 4_000, 11)
    with output.open(encoding="utf-8", newline="") as file:
        selected = [
            row for row in csv.DictReader(file)
            if row["split"] == "validation" and row["category"] == "plastic"
        ]
    assert {row["label"] for row in selected} == {"0", "1"}


def test_oversample_manifest_repeats_training_only(tmp_path):
    rows = [
        _base_row("pet", "train", "training"),
        _base_row("pet", "val", "validation"),
    ]
    manifest = tmp_path / "hardware.csv"
    _write(manifest, rows)
    loaded = read_manifest([], False, 0.25, [str(manifest)], 4)
    assert sum(row["split"] == "training" for row in loaded) == 4
    assert sum(row["split"] == "validation" for row in loaded) == 1
