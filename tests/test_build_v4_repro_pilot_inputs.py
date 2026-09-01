from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.build_v4_repro_pilot_inputs import (
    ScannedSource,
    _selection_rows,
    _source_score,
    build_pilot_inputs,
)


CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)


def _write_source(
    root: Path,
    *,
    split: str,
    name: str,
    content: bytes,
    label: str | None,
    decodable: bool = True,
) -> Path:
    image = root / split / "images" / f"{name}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    if decodable:
        digest = hashlib.sha256(content).digest()
        pixels = np.frombuffer(digest * 24, dtype=np.uint8).reshape(16, 16, 3)
        encoded_ok, encoded = cv2.imencode(
            ".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        assert encoded_ok
        image.write_bytes(encoded.tobytes())
    else:
        image.write_bytes(content)
    if label is not None:
        label_path = root / split / "labels" / f"{name}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(label, encoding="utf-8")
    return image


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    root = tmp_path / "dataset"
    sources = {
        "train_can_a": _write_source(
            root, split="train", name="can_a", content=b"can-a", label="0 .5 .5 .4 .4\n"
        ),
        "train_can_b": _write_source(
            root, split="train", name="can_b", content=b"can-b", label="0 .5 .5 .4 .4\n"
        ),
        "train_paper": _write_source(
            root, split="train", name="paper", content=b"paper", label="2 .5 .5 .4 .4\n"
        ),
        "train_pet": _write_source(
            root, split="train", name="pet", content=b"pet-t", label="1 .5 .5 .4 .4\n"
        ),
        "train_empty": _write_source(
            root, split="train", name="empty", content=b"empty-t", label=""
        ),
        "train_multi": _write_source(
            root,
            split="train",
            name="multi",
            content=b"multi",
            label="0 .5 .5 .4 .4\n2 .5 .5 .2 .2\n",
        ),
        "train_malformed": _write_source(
            root, split="train", name="malformed", content=b"bad", label="oops\n"
        ),
        "train_missing": _write_source(
            root, split="train", name="missing", content=b"missing", label=None
        ),
        "train_dup_a": _write_source(
            root, split="train", name="dup_a", content=b"same-train", label="3 .5 .5 .4 .4\n"
        ),
        "train_dup_b": _write_source(
            root, split="train", name="dup_b", content=b"same-train", label="3 .5 .5 .4 .4\n"
        ),
        "train_cross": _write_source(
            root, split="train", name="cross", content=b"cross", label="4 .5 .5 .4 .4\n"
        ),
        "train_styrofoam": _write_source(
            root, split="train", name="styrofoam", content=b"styrofoam-t", label="4 .5 .5 .4 .4\n"
        ),
        "train_vinyl": _write_source(
            root, split="train", name="vinyl", content=b"vinyl-t", label="5 .5 .5 .4 .4\n"
        ),
        "train_glass": _write_source(
            root, split="train", name="glass", content=b"glass-t", label="6 .5 .5 .4 .4\n"
        ),
        "train_battery": _write_source(
            root, split="train", name="battery", content=b"battery-t", label="7 .5 .5 .4 .4\n"
        ),
        "train_fluorescent": _write_source(
            root, split="train", name="fluorescent", content=b"fluorescent-t", label="8 .5 .5 .4 .4\n"
        ),
        "train_unreadable": _write_source(
            root,
            split="train",
            name="unreadable",
            content=b"not-an-image",
            label="1 .5 .5 .4 .4\n",
            decodable=False,
        ),
        "val_can": _write_source(
            root, split="val", name="can", content=b"can-v", label="0 .5 .5 .4 .4\n"
        ),
        "val_pet": _write_source(
            root, split="val", name="pet", content=b"pet-v", label="1 .5 .5 .4 .4\n"
        ),
        "val_paper": _write_source(
            root, split="val", name="paper", content=b"paper-v", label="2 .5 .5 .4 .4\n"
        ),
        "val_plastic": _write_source(
            root, split="val", name="plastic", content=b"plastic-v", label="3 .5 .5 .4 .4\n"
        ),
        "val_styrofoam": _write_source(
            root, split="val", name="styrofoam", content=b"styrofoam-v", label="4 .5 .5 .4 .4\n"
        ),
        "val_vinyl": _write_source(
            root, split="val", name="vinyl", content=b"vinyl-v", label="5 .5 .5 .4 .4\n"
        ),
        "val_glass": _write_source(
            root, split="val", name="glass", content=b"glass-v", label="6 .5 .5 .4 .4\n"
        ),
        "val_battery": _write_source(
            root, split="val", name="battery", content=b"battery-v", label="7 .5 .5 .4 .4\n"
        ),
        "val_fluorescent": _write_source(
            root, split="val", name="fluorescent", content=b"fluorescent-v", label="8 .5 .5 .4 .4\n"
        ),
        "val_empty": _write_source(
            root, split="val", name="empty", content=b"empty-v", label=""
        ),
        "val_cross": _write_source(
            root, split="val", name="cross", content=b"cross", label="4 .5 .5 .4 .4\n"
        ),
    }
    train_list = root / "train.txt"
    train_list.write_text(
        "\n".join(path.as_posix() for key, path in sources.items() if key.startswith("train_"))
        + "\n",
        encoding="utf-8",
    )
    yaml = root / "dataset.yaml"
    yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                f"train: {train_list.as_posix()}",
                f"val: {(root / 'val' / 'images').as_posix()}",
                "names:",
                *(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml, root, sources


def _inventory(output: Path) -> dict:
    return json.loads((output / "selection_inventory.json").read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_deterministic_selection_and_invalid_source_quarantine(tmp_path: Path) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    first = tmp_path / "pilot-a"
    second = tmp_path / "pilot-b"

    build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=first,
        seed=77,
        training_quota=1,
        validation_quota=1,
    )
    build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=second,
        seed=77,
        training_quota=1,
        validation_quota=1,
    )

    assert (first / "train_pilot.txt").read_bytes() == (
        second / "train_pilot.txt"
    ).read_bytes()
    assert (first / "validation_pilot.txt").read_bytes() == (
        second / "validation_pilot.txt"
    ).read_bytes()
    first_selected = _inventory(first)["selected_sources"]
    second_selected = _inventory(second)["selected_sources"]
    assert first_selected == second_selected

    selected_paths = {row["path"] for row in first_selected}
    assert sources["train_cross"].resolve().as_posix() not in selected_paths
    assert sources["val_cross"].resolve().as_posix() not in selected_paths
    assert sources["train_multi"].resolve().as_posix() not in selected_paths
    assert sources["train_malformed"].resolve().as_posix() not in selected_paths
    assert sources["train_missing"].resolve().as_posix() not in selected_paths
    duplicate_paths = {
        sources["train_dup_a"].resolve().as_posix(),
        sources["train_dup_b"].resolve().as_posix(),
    }
    assert len(selected_paths & duplicate_paths) == 1

    counts = _inventory(first)["rejections"]["counts"]
    assert counts["duplicate_source_content_cross_split"] == 2
    assert counts["duplicate_source_content_same_split"] == 1
    assert counts["multi_object_label"] == 1
    assert counts["malformed_label/invalid_column_count"] == 1
    assert counts["missing_label_file"] == 1
    assert counts["unreadable_image"] == 1


def test_historical_drift_anchor_is_selection_only_and_has_priority(tmp_path: Path) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    can_sources = [sources["train_can_a"], sources["train_can_b"]]
    default_output = tmp_path / "without-anchor"
    build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=default_output,
        seed=11,
        training_quota=1,
        validation_quota=1,
    )
    chosen_can = next(
        row for row in _inventory(default_output)["selected_sources"]
        if row["split"] == "training" and row["stratum"] == "can"
    )
    anchor_path = next(
        path for path in can_sources if path.resolve().as_posix() != chosen_can["path"]
    )
    anchor_sha = _sha(anchor_path)

    old_manifest = tmp_path / "historical.csv"
    with old_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "split", "category"])
        writer.writeheader()
        for path in can_sources:
            writer.writerow(
                {"source_id": _sha(path), "split": "training", "category": "can"}
            )
    drift = tmp_path / "drift.json"
    unrelated_sha = _sha(next(path for path in can_sources if path != anchor_path))
    drift.write_text(
        json.dumps(
            {
                "source_id": unrelated_sha,
                "replay": {
                    "hard_semantic_mismatch_examples": {
                        "crop_bounds_changed": [{"source_id": anchor_sha}]
                    },
                    "unrelated_nested_diagnostic": {"source_id": unrelated_sha},
                    "fixed_threshold_diagnostics": {"crossing_counts": {}},
                },
            }
        ),
        encoding="utf-8",
    )

    anchored_output = tmp_path / "with-anchor"
    build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=anchored_output,
        seed=11,
        training_quota=1,
        validation_quota=1,
        old_manifest=old_manifest,
        drift_report=drift,
    )
    inventory = _inventory(anchored_output)
    selected_can = next(
        row for row in inventory["selected_sources"]
        if row["split"] == "training" and row["stratum"] == "can"
    )
    assert selected_can["source_sha256"] == anchor_sha
    assert selected_can["drift_anchor"] is True
    assert selected_can["historical_categories_selection_only"] == ["can"]
    assert inventory["historical_selection_evidence"]["ground_truth_authority"] is False
    assert inventory["historical_selection_evidence"]["drift_report"]["anchor_source_ids"] == 1
    assert inventory["authority"]["training_authorized"] is False
    assert inventory["authority"]["production_deployment_authorized"] is False


def test_anchor_priority_is_capped_and_remainder_uses_original_blake_order(
    tmp_path: Path,
) -> None:
    records = []
    for index in range(10):
        source_sha = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        records.append(
            ScannedSource(
                path=tmp_path / f"source-{index}" / "images" / "item.jpg",
                split="training",
                source_sha256=source_sha,
                label_path=tmp_path / f"source-{index}" / "labels" / "item.txt",
                label_sha256=hashlib.sha256(f"label-{index}".encode()).hexdigest(),
                stratum="can",
                gt_class_id=0,
                gt_xywhn=(0.5, 0.5, 0.4, 0.4),
                historical_categories=("can",),
                anchor=index < 5,
            )
        )

    selected, _, _, _ = _selection_rows(
        records, seed=123, training_quota=5, validation_quota=1
    )
    selected_can = [
        row for row in selected
        if row["split"] == "training" and row["stratum"] == "can"
    ]
    assert len(selected_can) == 5
    assert sum(
        row["selection_reason"] == "drift_anchor_priority" for row in selected_can
    ) == 1

    def score(record: ScannedSource) -> tuple[str, str, str]:
        return (
            _source_score(
                seed=123,
                split="training",
                stratum="can",
                source_sha256=record.source_sha256,
            ),
            record.source_sha256,
            record.path.resolve().as_posix(),
        )

    priority = sorted((record for record in records if record.anchor), key=score)[:1]
    expected = [
        *priority,
        *sorted((record for record in records if record not in priority), key=score)[:4],
    ]
    assert {row["source_sha256"] for row in selected_can} == {
        record.source_sha256 for record in expected
    }


def test_ready_marker_binds_every_input_and_is_published_last(tmp_path: Path) -> None:
    data, dataset_dir, _ = _fixture(tmp_path)
    output = tmp_path / "pilot"
    ready = build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=output,
        seed=99,
        training_quota=1,
        validation_quota=1,
    )

    marker = output / "inputs.sha256"
    parsed = {}
    for line in marker.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        parsed[name] = digest
        assert _sha(output / name) == digest
    assert parsed == ready["bindings"]["artifacts"]
    assert _sha(marker) == ready["bindings"]["inputs_marker_sha256"]
    on_disk = json.loads((output / "input_ready.json").read_text(encoding="utf-8"))
    assert on_disk == ready
    assert on_disk["full_quota_met"] is True
    assert len(on_disk["selected_counts"]) == 20
    assert set(on_disk["selected_counts"].values()) == {1}
    assert on_disk["validator_authority"] is False
    assert on_disk["training_authorized"] is False
    assert on_disk["blind_test_authorized"] is False
    assert on_disk["production_deployment_authorized"] is False
    assert not (output / "failed.txt").exists()


def test_failure_after_directory_creation_publishes_failed_without_ready(
    tmp_path: Path,
) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    # Leave validation with only a source that is quarantined across splits.
    val_list = dataset_dir / "val-only-cross.txt"
    val_list.write_text(sources["val_cross"].as_posix() + "\n", encoding="utf-8")
    text = data.read_text(encoding="utf-8")
    data.write_text(
        text.replace((dataset_dir / "val" / "images").as_posix(), val_list.as_posix()),
        encoding="utf-8",
    )
    output = tmp_path / "failed-pilot"

    with pytest.raises(RuntimeError, match="balanced pilot quota shortage"):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            training_quota=1,
            validation_quota=1,
        )

    assert (output / "failed.txt").is_file()
    assert not (output / "input_ready.json").exists()


def test_unreadable_required_source_causes_shortage_failure(tmp_path: Path) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    valid = sources["val_fluorescent"]
    valid.unlink()
    valid.with_name(valid.name).parent.parent.joinpath("labels", "fluorescent.txt").unlink()
    _write_source(
        dataset_dir,
        split="val",
        name="fluorescent_unreadable",
        content=b"broken-required-image",
        label="8 .5 .5 .4 .4\n",
        decodable=False,
    )
    output = tmp_path / "unreadable-shortage"

    with pytest.raises(RuntimeError, match="validation/fluorescent=1"):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            training_quota=1,
            validation_quota=1,
        )

    assert (output / "failed.txt").is_file()
    assert not (output / "input_ready.json").exists()


def test_ready_publish_race_also_publishes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.build_v4_repro_pilot_inputs as builder

    data, dataset_dir, _ = _fixture(tmp_path)
    output = tmp_path / "ready-race"
    original = builder._publish_exclusive

    def publish_then_fail(path: Path, content: bytes) -> None:
        original(path, content)
        if path.name == "input_ready.json":
            raise RuntimeError("simulated failure after ready publication")

    monkeypatch.setattr(builder, "_publish_exclusive", publish_then_fail)
    with pytest.raises(RuntimeError, match="simulated failure"):
        builder.build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            training_quota=1,
            validation_quota=1,
        )

    assert (output / "input_ready.json").is_file()
    assert (output / "failed.txt").is_file()
    assert not (
        (output / "input_ready.json").is_file()
        and not (output / "failed.txt").exists()
    )


def test_existing_output_directory_is_never_reused(tmp_path: Path) -> None:
    data, dataset_dir, _ = _fixture(tmp_path)
    output = tmp_path / "already-there"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            training_quota=1,
            validation_quota=1,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output / "input_ready.json").exists()
    assert not (output / "failed.txt").exists()
