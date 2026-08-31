import builtins
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.prepare_proposal_verifier_dataset as proposal_dataset

from scripts.prepare_proposal_verifier_dataset import (
    BACKGROUND_CLASS_ID,
    Candidate,
    GroundTruth,
    PredictedFrame,
    Proposal,
    SourceRecord,
    assign_proposal,
    bbox_iou,
    build_proposal_verifier_dataset,
    collect_sources,
    eager_initialize_cuda_context,
    parse_yolo_label_text,
    select_deterministic_candidates,
)


def test_eager_cuda_context_creates_real_tensor_and_synchronizes(monkeypatch):
    events = []

    class FakeTensor:
        def __add__(self, value):
            events.append(("add", value))
            return self

        def item(self):
            return 2

    class FakeCuda:
        @staticmethod
        def is_available():
            events.append("available")
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def synchronize(device):
            events.append(("synchronize", device))

        @staticmethod
        def get_device_name(device):
            events.append(("name", device))
            return "fake-gpu"

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def ones(size, *, device):
            events.append(("ones", size, device))
            return FakeTensor()

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return FakeTorch()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    guard = eager_initialize_cuda_context("0")

    assert isinstance(guard, FakeTensor)
    assert events == [
        "available",
        ("ones", 1, "cuda:0"),
        ("add", 1),
        ("synchronize", 0),
        ("name", 0),
    ]


@pytest.mark.parametrize("device", ["cpu", "mps", "1", "0,1"])
def test_eager_cuda_context_ignores_non_qnap_gpu0_devices(device, monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("non-QNAP-GPU0 path must not import torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert eager_initialize_cuda_context(device) is None


def test_iou_assignment_boundaries_are_inclusive():
    gt = (0.0, 0.0, 10.0, 10.0)
    # A 5x10 intersection inside the GT has IoU exactly 0.5.
    positive_box = (0.0, 0.0, 5.0, 10.0)
    material, overlap, reason = assign_proposal(
        positive_box,
        gt_bbox=gt,
        gt_class_id=2,
        positive_iou=0.5,
        negative_iou=0.1,
    )
    assert bbox_iou(positive_box, gt) == pytest.approx(0.5)
    assert (material, reason) == (2, "positive_iou")
    assert overlap == pytest.approx(0.5)

    # A 1x10 intersection inside the GT has IoU exactly 0.1.
    negative_box = (0.0, 0.0, 1.0, 10.0)
    material, overlap, reason = assign_proposal(
        negative_box,
        gt_bbox=gt,
        gt_class_id=2,
        positive_iou=0.5,
        negative_iou=0.1,
    )
    assert overlap == pytest.approx(0.1)
    assert (material, reason) == (BACKGROUND_CLASS_ID, "low_iou")

    material, overlap, reason = assign_proposal(
        (0.0, 0.0, 3.0, 10.0),
        gt_bbox=gt,
        gt_class_id=2,
        positive_iou=0.5,
        negative_iou=0.1,
    )
    assert material is None
    assert overlap == pytest.approx(0.3)
    assert reason == "ambiguous_iou"


def test_multi_object_source_is_excluded_but_empty_negative_is_valid():
    ground_truth, reason = parse_yolo_label_text(
        "0 0.25 0.25 0.2 0.2\n1 0.75 0.75 0.2 0.2\n"
    )
    assert ground_truth is None
    assert reason == "not_single_object"

    ground_truth, reason = parse_yolo_label_text("\n")
    assert ground_truth is None
    assert reason is None


def test_collect_sources_rejects_missing_label_sidecar(tmp_path):
    dataset = tmp_path / "dataset"
    image_path = dataset / "images" / "train" / "unlabelled.jpg"
    image_path.parent.mkdir(parents=True)
    assert cv2.imwrite(
        str(image_path), np.full((24, 24, 3), 127, dtype=np.uint8)
    )

    records, rejected = collect_sources(
        {"training": [image_path], "validation": []}, dataset
    )

    assert records == []
    assert rejected == {"missing_label_file": 1}


def test_no_ground_truth_proposal_is_background():
    material, overlap, reason = assign_proposal(
        (10.0, 20.0, 80.0, 100.0),
        gt_bbox=None,
        gt_class_id=None,
        positive_iou=0.5,
        negative_iou=0.1,
    )
    assert material == BACKGROUND_CLASS_ID
    assert overlap == 0.0
    assert reason == "no_ground_truth"


def test_strict_background_policy_uses_final_padded_crop_and_gt_margin():
    gt = GroundTruth(2, (0.5, 0.5, 0.2, 0.2))  # pixel bbox 40..60
    accepted_source = SourceRecord(
        Path("accepted.jpg"), "training", "accepted-source", gt
    )
    rejected_source = SourceRecord(
        Path("rejected.jpg"), "training", "rejected-source", gt
    )
    frames = (
        PredictedFrame(
            accepted_source,
            100,
            100,
            (Proposal(3, 0.8, (0.0, 0.0, 20.0, 20.0), "plastic"),),
        ),
        PredictedFrame(
            rejected_source,
            100,
            100,
            # Raw IoU is low, but the final padded crop reaches the GT safety
            # margin and therefore must not become an automatic background.
            (Proposal(3, 0.8, (20.0, 20.0, 42.0, 42.0), "plastic"),),
        ),
    )
    policy_stats = proposal_dataset.Counter()

    candidates = list(
        proposal_dataset.candidates_from_frames(
            frames,
            positive_iou=0.5,
            negative_iou=0.1,
            background_policy="strict-zero-intersection",
            verifier_crop_padding=0.08,
            background_gt_margin=0.10,
            policy_stats=policy_stats,
        )
    )

    assert [candidate.source.source_id for candidate in candidates] == [
        "accepted-source"
    ]
    assert candidates[0].material == BACKGROUND_CLASS_ID
    assert policy_stats["background_accepted_zero_intersection"] == 1
    assert policy_stats["background_rejected_gt_intersection"] == 1


def test_hard_negative_manifest_distinguishes_source_and_crop_object_counts(
    tmp_path,
):
    image_path = tmp_path / "source.jpg"
    assert cv2.imwrite(
        str(image_path), np.full((100, 100, 3), 180, dtype=np.uint8)
    )
    gt = GroundTruth(2, (0.5, 0.5, 0.2, 0.2))
    source = SourceRecord(image_path, "training", "source-with-object", gt)
    candidate = Candidate(
        source=source,
        proposal_index=0,
        proposal=Proposal(3, 0.9, (0.0, 0.0, 20.0, 20.0), "plastic"),
        material=BACKGROUND_CLASS_ID,
        category="background",
        matched_iou=0.0,
        assignment="low_iou",
        gt_bbox=gt.xyxy(100, 100),
    )

    rows, _, rejected = proposal_dataset.write_selected_crops(
        [candidate],
        tmp_path / "crops",
        crop_size=32,
        padding=0.08,
        jpeg_quality=90,
        min_free_gb=0,
        max_output_gb=0,
    )

    assert rejected == {}
    assert len(rows) == 1
    assert rows[0]["source_object_count"] == 1
    assert rows[0]["crop_object_count"] == 0


def _candidate(index: int, material: int, split: str = "training") -> Candidate:
    gt = None if material == BACKGROUND_CLASS_ID else GroundTruth(material, (0.5, 0.5, 0.5, 0.5))
    source = SourceRecord(Path(f"image-{index}.jpg"), split, f"source-{index}", gt)
    return Candidate(
        source=source,
        proposal_index=0,
        proposal=Proposal(index % 9, 0.5 + index / 1000, (1.0, 1.0, 10.0, 10.0), "pred"),
        material=material,
        category="background" if material == BACKGROUND_CLASS_ID else f"class-{material}",
        matched_iou=0.0 if material == BACKGROUND_CLASS_ID else 1.0,
        assignment="low_iou" if material == BACKGROUND_CLASS_ID else "positive_iou",
        gt_bbox=None if gt is None else (0.0, 0.0, 10.0, 10.0),
    )


def test_deterministic_caps_are_independent_of_input_order():
    candidates = [
        *(_candidate(index, 2) for index in range(8)),
        *(_candidate(index + 100, BACKGROUND_CLASS_ID) for index in range(8)),
        *(_candidate(index + 200, 2, "validation") for index in range(5)),
        *(_candidate(index + 300, BACKGROUND_CLASS_ID, "validation") for index in range(5)),
    ]
    shuffled = list(candidates)
    random.Random(42).shuffle(shuffled)
    kwargs = dict(
        max_per_class=3,
        val_max_per_class=2,
        max_background=4,
        val_max_background=1,
        seed=7,
    )
    first = select_deterministic_candidates(candidates, **kwargs)
    second = select_deterministic_candidates(shuffled, **kwargs)

    assert [item.identity for item in first] == [item.identity for item in second]
    assert sum(item.material == 2 and item.source.split == "training" for item in first) == 3
    assert sum(item.material == BACKGROUND_CLASS_ID and item.source.split == "training" for item in first) == 4
    assert sum(item.material == 2 and item.source.split == "validation" for item in first) == 2
    assert sum(item.material == BACKGROUND_CLASS_ID and item.source.split == "validation" for item in first) == 1


def test_nonempty_output_refuses_before_loading_yolo(tmp_path, monkeypatch):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("do not overwrite", encoding="utf-8")

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise AssertionError("YOLO must not load before the output guard")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(FileExistsError, match="not empty"):
        build_proposal_verifier_dataset(
            model_path=tmp_path / "missing.pt",
            data_path=tmp_path / "missing.yaml",
            dataset_dir=tmp_path / "missing-dataset",
            output_dir=output_dir,
            device="cpu",
            batch=1,
            imgsz=32,
            conf=0.25,
            positive_iou=0.5,
            negative_iou=0.1,
            crop_size=32,
            padding=0.08,
            max_per_class=1,
            val_max_per_class=1,
            max_background=1,
            val_max_background=1,
            seed=1,
            min_free_gb=0,
            max_output_gb=0,
            jpeg_quality=90,
        )
    assert marker.read_text(encoding="utf-8") == "do not overwrite"


def test_fake_predictions_write_split_preserving_audit_manifest(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    for split_index, split in enumerate(("train", "val")):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image = np.full((100, 120, 3), 200 + split_index, dtype=np.uint8)
        assert cv2.imwrite(str(image_dir / f"{split}.jpg"), image)
        (label_dir / f"{split}.txt").write_text(
            "2 0.5 0.5 0.5 0.4\n", encoding="utf-8"
        )

    data = tmp_path / "data.yaml"
    data.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: [can, pet, paper, plastic, styrofoam, vinyl, glass, battery, fluorescent]\n",
        encoding="utf-8",
    )

    def fake_predictions(sources):
        for source in sources:
            # GT is x=30..90, y=30..70.  The detector class is intentionally
            # wrong: the verifier label must still come from matched GT paper.
            yield PredictedFrame(
                source=source,
                width=120,
                height=100,
                proposals=(Proposal(4, 0.91, (30.0, 30.0, 90.0, 70.0), "styrofoam"),),
            )

    def cuda_must_not_initialize(_device):
        raise AssertionError("injected prediction provider must bypass CUDA init")

    monkeypatch.setattr(
        proposal_dataset,
        "eager_initialize_cuda_context",
        cuda_must_not_initialize,
    )
    output = tmp_path / "proposal-crops"
    summary = build_proposal_verifier_dataset(
        model_path=tmp_path / "not-loaded.pt",
        data_path=data,
        dataset_dir=dataset,
        output_dir=output,
        device="0",
        batch=1,
        imgsz=64,
        conf=0.25,
        positive_iou=0.5,
        negative_iou=0.1,
        crop_size=32,
        padding=0.0,
        max_per_class=2,
        val_max_per_class=2,
        max_background=2,
        val_max_background=2,
        seed=3,
        min_free_gb=0,
        max_output_gb=0,
        jpeg_quality=90,
        prediction_provider=fake_predictions,
    )

    with (output / "manifest.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert summary["written_crops"] == 2
    assert {row["split"] for row in rows} == {"training", "validation"}
    assert {row["material"] for row in rows} == {"2"}
    assert {row["category"] for row in rows} == {"paper"}
    assert {row["predicted_class_id"] for row in rows} == {"4"}
    assert {row["predicted_class_name"] for row in rows} == {"styrofoam"}
    assert {row["matched_iou"] for row in rows} == {"1.00000000"}
    assert {row["source_object_count"] for row in rows} == {"1"}
    assert {row["crop_object_count"] for row in rows} == {"1"}
    for row in rows:
        crop = cv2.imread(str(output / row["filepath"]))
        assert crop.shape == (32, 32, 3)


def test_real_gpu0_eager_init_runs_after_preflight_before_source_collection(
    tmp_path, monkeypatch
):
    events = []

    class StopAfterCollection(RuntimeError):
        pass

    monkeypatch.setattr(
        proposal_dataset,
        "check_storage_limits",
        lambda *args, **kwargs: events.append("storage"),
    )
    monkeypatch.setattr(
        proposal_dataset,
        "eager_initialize_cuda_context",
        lambda device: events.append(("eager", device)) or object(),
    )
    monkeypatch.setattr(
        proposal_dataset,
        "resolve_split_images",
        lambda *args, **kwargs: events.append("resolve")
        or {"training": [], "validation": []},
    )

    def stop_collection(*args, **kwargs):
        events.append("collect")
        raise StopAfterCollection

    monkeypatch.setattr(proposal_dataset, "collect_sources", stop_collection)

    with pytest.raises(StopAfterCollection):
        build_proposal_verifier_dataset(
            model_path=tmp_path / "model.pt",
            data_path=tmp_path / "data.yaml",
            dataset_dir=tmp_path / "dataset",
            output_dir=tmp_path / "output",
            device="0",
            batch=1,
            imgsz=32,
            conf=0.10,
            positive_iou=0.50,
            negative_iou=0.10,
            crop_size=32,
            padding=0.08,
            max_per_class=1,
            val_max_per_class=1,
            max_background=1,
            val_max_background=1,
            seed=1,
            min_free_gb=0,
            max_output_gb=0,
            jpeg_quality=90,
        )

    assert events == ["storage", ("eager", "0"), "resolve", "collect"]


def test_runtime_top1_with_no_ground_truth_only_background_policy(tmp_path):
    dataset = tmp_path / "dataset"
    for split_index, split in enumerate(("train", "val")):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for stem_index, stem in enumerate(("positive", "contaminated", "empty")):
            image = np.full(
                (100, 120, 3), 190 + split_index * 10 + stem_index, dtype=np.uint8
            )
            assert cv2.imwrite(str(image_dir / f"{stem}.jpg"), image)
            label_text = "" if stem == "empty" else "2 0.5 0.5 0.5 0.4\n"
            (label_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")

    data = tmp_path / "data.yaml"
    data.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: [can, pet, paper, plastic, styrofoam, vinyl, glass, battery, fluorescent]\n",
        encoding="utf-8",
    )

    def fake_predictions(sources):
        for source in sources:
            if source.path.stem == "positive":
                proposals = (
                    Proposal(3, 0.40, (0.0, 0.0, 10.0, 10.0), "plastic"),
                    Proposal(4, 0.90, (30.0, 30.0, 90.0, 70.0), "styrofoam"),
                    Proposal(2, 0.89, (30.0, 30.0, 90.0, 70.0), "paper"),
                )
            elif source.path.stem == "contaminated":
                proposals = (
                    Proposal(2, 0.80, (30.0, 30.0, 90.0, 70.0), "paper"),
                    Proposal(3, 0.99, (0.0, 0.0, 10.0, 10.0), "plastic"),
                )
            else:
                proposals = (
                    Proposal(3, 0.20, (1.0, 1.0, 20.0, 20.0), "plastic"),
                    Proposal(5, 0.85, (20.0, 20.0, 80.0, 80.0), "vinyl"),
                    Proposal(2, 0.80, (30.0, 30.0, 70.0, 70.0), "paper"),
                )
            yield PredictedFrame(source, 120, 100, proposals)

    output = tmp_path / "runtime-proposal-crops"
    summary = build_proposal_verifier_dataset(
        model_path=tmp_path / "not-loaded.pt",
        data_path=data,
        dataset_dir=dataset,
        output_dir=output,
        device="cpu",
        batch=1,
        imgsz=64,
        conf=0.25,
        positive_iou=0.5,
        negative_iou=0.1,
        crop_size=32,
        padding=0.0,
        max_per_class=10,
        val_max_per_class=10,
        max_background=10,
        val_max_background=10,
        seed=3,
        min_free_gb=0,
        max_output_gb=0,
        jpeg_quality=90,
        proposal_selection="runtime-top1",
        background_policy="no-ground-truth-only",
        prediction_provider=fake_predictions,
    )

    with (output / "manifest.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert summary["written_crops"] == 4
    assert summary["proposal_policy"] == {
        "selection_mode": "runtime-top1",
        "minimum_confidence": 0.25,
        "background_policy": "no-ground-truth-only",
    }
    assert summary["proposal_assignments"] == {
        "positive_iou": 2,
        "low_iou": 2,
        "no_ground_truth": 2,
    }
    assert summary["proposal_policy_stats"] == {
        "frames_seen": 6,
        "proposals_seen": 16,
        "discarded_by_runtime_top1": 8,
        "proposals_selected": 6,
        "background_rejected_gt_present": 2,
        "below_min_confidence": 2,
    }
    assert {(row["split"], row["category"]) for row in rows} == {
        ("training", "paper"),
        ("training", "background"),
        ("validation", "paper"),
        ("validation", "background"),
    }
    assert {row["proposal_index"] for row in rows} == {"1"}
    background_rows = [row for row in rows if row["category"] == "background"]
    assert len(background_rows) == 2
    assert {row["assignment"] for row in background_rows} == {"no_ground_truth"}
    assert {row["source_object_count"] for row in background_rows} == {"0"}
    assert {row["crop_object_count"] for row in background_rows} == {"0"}
    assert {row["predicted_confidence"] for row in rows} == {
        "0.90000000",
        "0.85000000",
    }


def test_cross_split_source_path_is_quarantined_before_prediction(tmp_path):
    dataset = tmp_path / "dataset"
    image_dir = dataset / "images" / "shared"
    label_dir = dataset / "labels" / "shared"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    assert cv2.imwrite(
        str(image_dir / "same.jpg"), np.full((32, 32, 3), 200, dtype=np.uint8)
    )
    (label_dir / "same.txt").write_text(
        "2 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )
    data = tmp_path / "data.yaml"
    data.write_text(
        "path: .\n"
        "train: images/shared\n"
        "val: images/shared\n"
        "names: [can, pet, paper, plastic, styrofoam, vinyl, glass, battery, fluorescent]\n",
        encoding="utf-8",
    )

    def prediction_must_not_run(_sources):
        raise AssertionError("prediction must not run after split leakage")

    with pytest.raises(RuntimeError, match="no valid zero-or-one-object"):
        build_proposal_verifier_dataset(
            model_path=tmp_path / "not-loaded.pt",
            data_path=data,
            dataset_dir=dataset,
            output_dir=tmp_path / "output",
            device="cpu",
            batch=1,
            imgsz=32,
            conf=0.25,
            positive_iou=0.5,
            negative_iou=0.1,
            crop_size=32,
            padding=0.0,
            max_per_class=1,
            val_max_per_class=1,
            max_background=1,
            val_max_background=1,
            seed=1,
            min_free_gb=0,
            max_output_gb=0,
            jpeg_quality=90,
            proposal_selection="runtime-top1",
            background_policy="no-ground-truth-only",
            prediction_provider=prediction_must_not_run,
        )


def test_copied_source_bytes_across_splits_are_quarantined(tmp_path):
    dataset = tmp_path / "dataset"
    train_images = dataset / "images" / "train"
    val_images = dataset / "images" / "val"
    train_labels = dataset / "labels" / "train"
    val_labels = dataset / "labels" / "val"
    for directory in (train_images, val_images, train_labels, val_labels):
        directory.mkdir(parents=True)
    source = train_images / "original.jpg"
    assert cv2.imwrite(str(source), np.full((32, 32, 3), 160, dtype=np.uint8))
    (val_images / "renamed-copy.jpg").write_bytes(source.read_bytes())
    for label in (train_labels / "original.txt", val_labels / "renamed-copy.txt"):
        label.write_text("2 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: [can, pet, paper, plastic, styrofoam, vinyl, glass, battery, fluorescent]\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no valid zero-or-one-object"):
        build_proposal_verifier_dataset(
            model_path=tmp_path / "not-loaded.pt",
            data_path=data,
            dataset_dir=dataset,
            output_dir=tmp_path / "output",
            device="cpu",
            batch=1,
            imgsz=32,
            conf=0.25,
            positive_iou=0.5,
            negative_iou=0.1,
            crop_size=32,
            padding=0.0,
            max_per_class=1,
            val_max_per_class=1,
            max_background=1,
            val_max_background=1,
            seed=1,
            min_free_gb=0,
            max_output_gb=0,
            jpeg_quality=90,
            proposal_selection="runtime-top1",
            background_policy="no-ground-truth-only",
            prediction_provider=lambda _sources: (),
        )


def test_copied_source_bytes_within_split_are_inferred_once(tmp_path):
    dataset = tmp_path / "dataset"
    train_images = dataset / "images" / "train"
    train_labels = dataset / "labels" / "train"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    original = train_images / "original.jpg"
    duplicate = train_images / "renamed-copy.jpg"
    assert cv2.imwrite(str(original), np.full((32, 32, 3), 80, dtype=np.uint8))
    duplicate.write_bytes(original.read_bytes())
    for label in (train_labels / "original.txt", train_labels / "renamed-copy.txt"):
        label.write_text("2 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    records, rejected = collect_sources(
        {"training": [duplicate, original], "validation": []}, dataset
    )

    assert len(records) == 1
    assert records[0].source_id == __import__("hashlib").sha256(
        original.read_bytes()
    ).hexdigest()
    assert rejected == {"duplicate_source_content_same_split": 1}


def test_copied_source_bytes_with_conflicting_labels_are_quarantined(tmp_path):
    dataset = tmp_path / "dataset"
    train_images = dataset / "images" / "train"
    train_labels = dataset / "labels" / "train"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    original = train_images / "original.jpg"
    duplicate = train_images / "renamed-copy.jpg"
    assert cv2.imwrite(str(original), np.full((32, 32, 3), 80, dtype=np.uint8))
    duplicate.write_bytes(original.read_bytes())
    (train_labels / "original.txt").write_text(
        "2 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )
    (train_labels / "renamed-copy.txt").write_text(
        "3 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )

    records, rejected = collect_sources(
        {"training": [original, duplicate], "validation": []}, dataset
    )

    assert records == []
    assert rejected == {"duplicate_source_content_conflicting_ground_truth": 2}
