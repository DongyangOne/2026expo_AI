from pathlib import Path

from scripts.prepare_mixed_replay_yolo_list import (
    CLASS_NAMES,
    prepare_mixed_replay_list,
)


def _sample(root: Path, stem: str, label: str) -> Path:
    image_dir = root / "train" / "images"
    label_dir = root / "train" / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = image_dir / f"{stem}.jpg"
    image.write_bytes(b"jpeg")
    (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")
    return image


def test_prepare_mixed_replay_is_balanced_single_object_and_deterministic(tmp_path):
    base = tmp_path / "base"
    for class_id, name in enumerate(CLASS_NAMES):
        for index in range(3):
            _sample(base, f"{name}-{index}", f"{class_id} 0.5 0.5 0.4 0.4\n")
    _sample(base, "empty", "")
    _sample(base, "multi", "0 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n")
    _sample(base, "invalid", "2 1.5 0.5 0.4 0.4\n")

    commercial = tmp_path / "commercial.txt"
    commercial.write_text("/app/commercial/a.jpg\n/app/commercial/b.jpg\n", encoding="utf-8")
    negative = tmp_path / "negative"
    negative_path = _sample(negative, "clean-frame", "")

    first = prepare_mixed_replay_list(
        base_dataset_dir=base,
        commercial_list=commercial,
        output_dir=tmp_path / "out1",
        validation_images="/app/base/val/images",
        target_per_class=2,
        rare_target_per_class=2,
        seed=7,
        trusted_negative_dir=negative,
        trusted_negative_repeats=2,
    )
    second = prepare_mixed_replay_list(
        base_dataset_dir=base,
        commercial_list=commercial,
        output_dir=tmp_path / "out2",
        validation_images="/app/base/val/images",
        target_per_class=2,
        rare_target_per_class=2,
        seed=7,
        trusted_negative_dir=negative,
        trusted_negative_repeats=2,
    )

    assert first["replay_entries"] == 18
    assert first["commercial_entries"] == 2
    assert first["trusted_negative_entries"] == 2
    assert first["mixed_entries"] == 22
    assert first["base_rejected"] == {
        "empty_label": 1,
        "not_single_object": 1,
        "invalid_bbox": 1,
    }
    assert first["replay_entries_by_class"] == {name: 2 for name in CLASS_NAMES}
    list1 = (tmp_path / "out1" / "train_mixed.txt").read_text(encoding="utf-8")
    list2 = (tmp_path / "out2" / "train_mixed.txt").read_text(encoding="utf-8")
    assert list1 == list2
    assert list1.splitlines().count(negative_path.as_posix()) == 2
    assert second["mixed_entries"] == first["mixed_entries"]
