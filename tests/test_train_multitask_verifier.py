import csv
import hashlib
import io
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

import scripts.train_multitask_verifier as trainer


STRICT_FIELDS = [
    "filepath",
    "split",
    "source_id",
    "material",
    "category",
    "dent",
    "label",
    "foreign_material",
    "source_object_count",
    "sample_id",
    "source_sha256",
    "image_sha256",
    "object_group",
    "capture_session",
    "role",
    "fold",
    "origin",
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(
        "RGB",
        (16, 16),
        color=((seed * 37) % 256, (seed * 67) % 256, (seed * 97) % 256),
    ).save(path)


def _strict_rows(root: Path, manifest: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed = 0
    for role in trainer.TRAINING_ROLES:
        for material in range(10):
            image_path = root / "images" / role / f"{material}.png"
            _write_image(image_path, seed)
            category = (
                trainer.BACKGROUND_CLASS_NAME
                if material == trainer.BACKGROUND_MATERIAL_ID
                else trainer.MATERIAL_CLASS_NAMES[material]
            )
            rows.append(
                {
                    "filepath": image_path.relative_to(manifest.parent).as_posix(),
                    "split": trainer.ROLE_TO_SPLIT[role],
                    "source_id": f"source-id-{role}-{material}",
                    "material": material,
                    "category": category,
                    "dent": -1,
                    "label": -1,
                    "foreign_material": -1,
                    "source_object_count": 0 if material == 9 else 1,
                    "sample_id": f"sample-{role}-{material}",
                    "source_sha256": _sha(f"source-{role}-{material}"),
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "object_group": f"object-{role}-{material}",
                    "capture_session": f"session-{role}-{material}",
                    "role": role,
                    "fold": f"fold-{role}",
                    "origin": "test-fixture",
                }
            )
            seed += 1
    return rows


def _write_manifest(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str] | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields or STRICT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _complete_manifest(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    manifest = tmp_path / "strict.csv"
    rows = _strict_rows(tmp_path, manifest)
    _write_manifest(manifest, rows)
    return manifest, rows


def test_model_has_binary_objectness_and_nine_class_material_contract():
    model = trainer.MultitaskCropVerifier(
        "tiny_cnn", pretrained=False, condition_heads=("label", "dent")
    )

    with torch.inference_mode():
        outputs = model(torch.randn(3, 3, 16, 16))

    assert list(outputs) == ["objectness", "material", "dent", "label"]
    assert outputs["objectness"].shape == (3, 2)
    assert outputs["material"].shape == (3, 9)
    assert outputs["dent"].shape == outputs["label"].shape == (3, 2)
    contract = trainer.build_output_contract(("label", "dent"))
    assert contract["output_order"] == list(outputs)
    assert contract["material_background_class_id"] is None
    assert contract["outputs"][1]["class_names"] == list(trainer.MATERIAL_CLASS_NAMES)
    assert "background" not in contract["outputs"][1]["class_names"]


def test_background_rows_have_zero_material_and_condition_gradient():
    outputs = {
        "objectness": torch.randn(3, 2, requires_grad=True),
        "material": torch.randn(3, 9, requires_grad=True),
        "dent": torch.randn(3, 2, requires_grad=True),
    }
    batch = {
        "objectness": torch.tensor([1, 0, 1]),
        "material": torch.tensor([2, trainer.BACKGROUND_MATERIAL_ID, 4]),
        # The background row deliberately has a label. It must still be masked.
        "dent": torch.tensor([0, 1, -1]),
    }
    criteria = {
        name: nn.CrossEntropyLoss(reduction="none")
        for name in ("objectness", "material", "dent")
    }
    weights = {name: 1.0 for name in criteria}

    total, per_task, counts = trainer.compute_multitask_loss(
        outputs, batch, criteria, weights
    )
    expected_material = nn.functional.cross_entropy(
        outputs["material"][[0, 2]], torch.tensor([2, 4])
    )
    torch.testing.assert_close(per_task["material"], expected_material)
    assert counts == {"objectness": 3, "material": 2, "dent": 1}

    total.backward()
    torch.testing.assert_close(
        outputs["material"].grad[1], torch.zeros_like(outputs["material"].grad[1])
    )
    torch.testing.assert_close(
        outputs["dent"].grad[1], torch.zeros_like(outputs["dent"].grad[1])
    )


def test_strict_manifest_preserves_lineage_and_rejects_group_leakage(tmp_path):
    manifest, raw_rows = _complete_manifest(tmp_path)
    original_bytes = manifest.read_bytes()

    rows = trainer.read_manifests([manifest])

    first = rows[0]
    assert first.lineage_record() == {
        name: str(raw_rows[0][name]) for name in trainer.LINEAGE_FIELDS
    }
    assert first.raw["sample_id"] == raw_rows[0]["sample_id"]
    assert {row.role for row in rows} == set(trainer.TRAINING_ROLES)
    assert manifest.read_bytes() == original_bytes

    validation = next(row for row in raw_rows if row["role"] == trainer.VALIDATION_ROLE)
    validation["object_group"] = raw_rows[0]["object_group"]
    _write_manifest(manifest, raw_rows)
    with pytest.raises(ValueError, match=r"object_group.*crosses train/validation"):
        trainer.read_manifests([manifest])


def test_dataset_decodes_only_the_single_descriptor_verified_bytes(tmp_path, monkeypatch):
    manifest, _ = _complete_manifest(tmp_path)
    row = trainer.read_manifests([manifest])[0]
    opened_inputs = []
    original_open = trainer.Image.open

    def tracked_open(source, *args, **kwargs):
        opened_inputs.append(source)
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(trainer.Image, "open", tracked_open)
    result = trainer.VerifierDataset(
        [row], transform=lambda _image: torch.zeros(3, 16, 16)
    )[0]

    assert result["sample_id"] == row.sample_id
    assert len(opened_inputs) == 1
    assert isinstance(opened_inputs[0], io.BytesIO)


def test_dataset_rejects_mutated_or_identical_content_replaced_image(tmp_path):
    manifest, _ = _complete_manifest(tmp_path)
    parsed = trainer.read_manifests([manifest])
    transform = lambda _image: torch.zeros(3, 16, 16)

    mutated = parsed[0]
    payload = bytearray(mutated.path.read_bytes())
    payload[len(payload) // 2] ^= 1
    mutated.path.write_bytes(payload)
    with pytest.raises(ValueError, match="identity changed|image_sha256"):
        trainer.VerifierDataset([mutated], transform=transform)[0]

    replaced = parsed[1]
    replacement = replaced.path.with_suffix(".replacement")
    replacement.write_bytes(replaced.path.read_bytes())
    replacement.replace(replaced.path)
    with pytest.raises(ValueError, match="identity changed"):
        trainer.VerifierDataset([replaced], transform=transform)[0]


def test_manifest_rejects_hardlinked_or_symlinked_image_payload(tmp_path):
    manifest, raw_rows = _complete_manifest(tmp_path)
    first_path = manifest.parent / str(raw_rows[0]["filepath"])
    hardlink = tmp_path / "hardlink.png"
    os.link(first_path, hardlink)
    with pytest.raises(ValueError, match="exactly one hard link"):
        trainer.read_manifests([manifest])

    hardlink.unlink()
    target = first_path.with_suffix(".target.png")
    first_path.replace(target)
    try:
        os.symlink(target, first_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(ValueError, match="contains a symlink"):
        trainer.read_manifests([manifest])


def test_image_consumption_rejects_byte_and_pixel_caps(tmp_path, monkeypatch):
    manifest, _ = _complete_manifest(tmp_path)
    first_size = next((tmp_path / "images").rglob("*.png")).stat().st_size
    monkeypatch.setattr(trainer, "IMAGE_CONSUMPTION_MAX_BYTES", first_size - 1)
    with pytest.raises(ValueError, match="payload size"):
        trainer.read_manifests([manifest])

    monkeypatch.setattr(trainer, "IMAGE_CONSUMPTION_MAX_BYTES", 64 * 1024 * 1024)
    row = trainer.read_manifests([manifest])[0]
    monkeypatch.setattr(trainer, "IMAGE_CONSUMPTION_MAX_PIXELS", 255)
    with pytest.raises(ValueError, match="pixel cap"):
        trainer.VerifierDataset(
            [row], transform=lambda _image: torch.zeros(3, 16, 16)
        )[0]


def test_image_consumption_rejects_partial_read_and_path_swap(tmp_path, monkeypatch):
    image_path = tmp_path / "image.png"
    _write_image(image_path, 7)
    expected_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    real_read = os.read
    calls = 0

    def partial_read(descriptor, count):
        nonlocal calls
        calls += 1
        if calls == 2:
            return b""
        return real_read(descriptor, min(count, 8))

    monkeypatch.setattr(trainer, "IMAGE_CONSUMPTION_READ_BYTES", 8)
    monkeypatch.setattr(trainer.os, "read", partial_read)
    with pytest.raises(ValueError, match="partial"):
        trainer._verified_image_stream(
            image_path,
            manifest_root=tmp_path,
            expected_sha256=expected_sha,
        )

    monkeypatch.setattr(trainer.os, "read", real_read)
    real_open = os.open
    parked = tmp_path / "parked.png"
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(image_path.read_bytes())

    def swap_after_open(path, flags):
        descriptor = real_open(path, flags)
        image_path.replace(parked)
        replacement.replace(image_path)
        return descriptor

    monkeypatch.setattr(trainer.os, "open", swap_after_open)
    with pytest.raises(ValueError, match="secure open failed|path chain changed"):
        trainer._verified_image_stream(
            image_path,
            manifest_root=tmp_path,
            expected_sha256=expected_sha,
        )


def test_image_consumption_accepts_normal_short_reads(tmp_path, monkeypatch):
    image_path = tmp_path / "short-read.bin"
    payload = b"approved-image-payload" * 4
    image_path.write_bytes(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()
    real_read = os.read

    monkeypatch.setattr(
        trainer.os,
        "read",
        lambda descriptor, count: real_read(descriptor, min(count, 3)),
    )
    stream, size, _identity = trainer._verified_image_stream(
        image_path,
        manifest_root=tmp_path,
        expected_sha256=expected_sha,
    )
    with stream:
        assert size == len(payload)
        assert stream.read() == payload


@pytest.mark.skipif(os.name != "posix", reason="QNAP/POSIX race semantics")
def test_posix_consumption_rejects_poison_then_restore(tmp_path, monkeypatch):
    image_path = tmp_path / "poison-restore.bin"
    approved = b"A" * 64
    poisoned = b"B" * 64
    image_path.write_bytes(approved)
    expected_sha = hashlib.sha256(approved).hexdigest()
    real_read = os.read
    calls = 0

    def poison_then_restore(descriptor, count):
        nonlocal calls
        calls += 1
        chunk = real_read(descriptor, min(count, 8))
        if calls == 1:
            image_path.write_bytes(poisoned)
        elif calls == 2:
            image_path.write_bytes(approved)
        return chunk

    monkeypatch.setattr(trainer, "IMAGE_CONSUMPTION_READ_BYTES", 8)
    monkeypatch.setattr(trainer.os, "read", poison_then_restore)
    with pytest.raises(ValueError, match="identity changed|image_sha256"):
        trainer._verified_image_stream(
            image_path,
            manifest_root=tmp_path,
            expected_sha256=expected_sha,
        )


@pytest.mark.skipif(os.name != "posix", reason="QNAP/POSIX special-file semantics")
def test_posix_consumption_rejects_ancestor_symlink_and_fifo(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    image_path = actual / "image.bin"
    image_path.write_bytes(b"approved")
    linked = tmp_path / "linked"
    os.symlink(actual, linked, target_is_directory=True)
    expected_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="contains a symlink"):
        trainer._verified_image_stream(
            linked / image_path.name,
            manifest_root=tmp_path,
            expected_sha256=expected_sha,
        )

    fifo = tmp_path / "payload.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        trainer._verified_image_stream(
            fifo,
            manifest_root=tmp_path,
            expected_sha256=expected_sha,
        )


@pytest.mark.parametrize(
    ("material", "category", "source_object_count"),
    [
        (trainer.BACKGROUND_MATERIAL_ID, trainer.BACKGROUND_CLASS_NAME, 1),
        (0, trainer.MATERIAL_CLASS_NAMES[0], 0),
        (0, trainer.MATERIAL_CLASS_NAMES[0], 2),
    ],
)
def test_strict_manifest_rejects_objectness_count_conflicts(
    tmp_path, material, category, source_object_count
):
    manifest, rows = _complete_manifest(tmp_path)
    rows[0]["material"] = material
    rows[0]["category"] = category
    rows[0]["source_object_count"] = source_object_count
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="single-object verifier contract"):
        trainer.read_manifests([manifest])


def test_strict_manifest_accepts_legacy_empty_and_explicit_hard_negative_background(
    tmp_path,
):
    manifest, rows = _complete_manifest(tmp_path)
    train_background = next(
        row
        for row in rows
        if row["role"] == trainer.TRAIN_ROLE
        and row["material"] == trainer.BACKGROUND_MATERIAL_ID
    )
    validation_background = next(
        row
        for row in rows
        if row["role"] == trainer.VALIDATION_ROLE
        and row["material"] == trainer.BACKGROUND_MATERIAL_ID
    )
    validation_background["source_object_count"] = 1
    validation_background["crop_object_count"] = 0
    _write_manifest(manifest, rows, [*STRICT_FIELDS, "crop_object_count"])

    parsed = trainer.read_manifests([manifest])
    parsed_train = next(row for row in parsed if row.sample_id == train_background["sample_id"])
    parsed_validation = next(
        row for row in parsed if row.sample_id == validation_background["sample_id"]
    )

    assert (parsed_train.source_object_count, parsed_train.crop_object_count) == (0, 0)
    assert (
        parsed_validation.source_object_count,
        parsed_validation.crop_object_count,
    ) == (1, 0)


@pytest.mark.parametrize(
    ("material", "source_count", "crop_count"),
    [
        (trainer.BACKGROUND_MATERIAL_ID, 0, 1),
        (trainer.BACKGROUND_MATERIAL_ID, 1, 1),
        (0, 1, 0),
    ],
)
def test_strict_manifest_rejects_crop_object_count_conflicts(
    tmp_path, material, source_count, crop_count
):
    manifest, rows = _complete_manifest(tmp_path)
    rows[0]["material"] = material
    rows[0]["category"] = (
        trainer.BACKGROUND_CLASS_NAME
        if material == trainer.BACKGROUND_MATERIAL_ID
        else trainer.MATERIAL_CLASS_NAMES[material]
    )
    rows[0]["source_object_count"] = source_count
    rows[0]["crop_object_count"] = crop_count
    _write_manifest(manifest, rows, [*STRICT_FIELDS, "crop_object_count"])

    with pytest.raises(ValueError, match="crop_object_count"):
        trainer.read_manifests([manifest])


def test_strict_manifest_requires_declared_provenance_and_preserves_excluded_roles(
    tmp_path,
):
    manifest, rows = _complete_manifest(tmp_path)
    with manifest.open("w", encoding="utf-8", newline="") as file:
        fields = [name for name in STRICT_FIELDS if name != "origin"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=r"missing strict manifest fields.*origin"):
        trainer.read_manifests([manifest])

    for index, role in enumerate(
        (trainer.CALIBRATION_ROLE, trainer.BLIND_TEST_ROLE), start=100
    ):
        image_path = tmp_path / "images" / role / "0.png"
        _write_image(image_path, index)
        rows.append(
            {
                "filepath": image_path.relative_to(manifest.parent).as_posix(),
                "split": role,
                "source_id": f"source-id-{role}",
                "material": 0,
                "category": trainer.MATERIAL_CLASS_NAMES[0],
                "dent": -1,
                "label": -1,
                "foreign_material": -1,
                "source_object_count": 1,
                "sample_id": f"sample-{role}",
                "source_sha256": _sha(f"source-{role}"),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "object_group": f"object-{role}",
                "capture_session": f"session-{role}",
                "role": role,
                "fold": f"fold-{role}",
                "origin": "test-fixture",
            }
        )
    _write_manifest(manifest, rows)

    parsed = trainer.read_manifests([manifest])
    summary = trainer._manifest_summary(parsed, [manifest])

    assert len(parsed) == 22
    assert summary["excluded_from_training_role_counts"] == {
        trainer.CALIBRATION_ROLE: 1,
        trainer.BLIND_TEST_ROLE: 1,
    }
    assert len([row for row in parsed if row.role == trainer.TRAIN_ROLE]) == 10


def test_balanced_weights_count_material_only_on_positive_rows(tmp_path):
    manifest, _ = _complete_manifest(tmp_path)
    rows = trainer.read_manifests([manifest])

    weights = trainer.build_class_weight_values(rows, (), mode="inverse")

    assert set(weights) == {"objectness", "material"}
    assert weights["objectness"].shape == (2,)
    assert weights["material"].shape == (9,)
    torch.testing.assert_close(weights["material"], torch.ones(9))
    # train has nine positive rows and one background row.
    torch.testing.assert_close(
        weights["objectness"], torch.tensor([5.0, 5.0 / 9.0])
    )


def test_origin_weighting_increases_hardware_sampling_without_duplicate_rows(tmp_path):
    manifest, raw_rows = _complete_manifest(tmp_path)
    train_rows = [row for row in raw_rows if row["role"] == trainer.TRAIN_ROLE]
    train_rows[0]["origin"] = "hardware_runtime_v3"
    train_rows[1]["origin"] = "operational_teacher_v3"
    _write_manifest(manifest, raw_rows)
    parsed = trainer.read_manifests([manifest])

    configured = trainer.parse_origin_weights(
        ["hardware_runtime_v3=100", "operational_teacher_v3=50"]
    )
    sample_weights, plan = trainer.build_origin_sampling_plan(parsed, configured)

    assert sample_weights is not None
    assert len(sample_weights) == len(train_rows)
    assert sorted(sample_weights) == [1.0] * 8 + [50.0, 100.0]
    assert plan["mode"] == "weighted_replacement"
    assert plan["samples_per_epoch"] == 10
    assert plan["manifest_rows_remain_unique"] is True
    assert plan["weighted_mass_by_origin"] == {
        "hardware_runtime_v3": 100.0,
        "operational_teacher_v3": 50.0,
        "test-fixture": 8.0,
    }
    assert sum(plan["expected_fraction_by_origin"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "raw, message",
    [
        (["missing-separator"], "expected nonempty"),
        (["hardware=0"], "finite and positive"),
        (["hardware=2", "hardware=3"], "repeated"),
    ],
)
def test_origin_weight_parser_rejects_ambiguous_or_unsafe_values(raw, message):
    with pytest.raises(ValueError, match=message):
        trainer.parse_origin_weights(raw)


def test_origin_weight_must_reference_a_training_origin(tmp_path):
    manifest, _ = _complete_manifest(tmp_path)
    parsed = trainer.read_manifests([manifest])

    with pytest.raises(ValueError, match="absent training origins"):
        trainer.build_origin_sampling_plan(parsed, {"typo-origin": 2.0})


def test_uniform_origin_weight_scale_does_not_enable_replacement_sampling(tmp_path):
    manifest, _ = _complete_manifest(tmp_path)
    parsed = trainer.read_manifests([manifest])

    sample_weights, plan = trainer.build_origin_sampling_plan(
        parsed, {"test-fixture": 100.0}
    )

    assert sample_weights is None
    assert plan["mode"] == "shuffle_without_replacement"
    assert plan["expected_fraction_by_origin"] == {"test-fixture": 1.0}


def test_weighted_loader_is_seeded_and_validation_loader_is_unweighted():
    rows = [None] * 10
    sample_weights = [100.0, 50.0, *([1.0] * 8)]

    def weighted_indices(seed):
        loader = trainer._make_loader(
            rows,
            transform=None,
            batch_size=4,
            workers=0,
            shuffle=True,
            seed=seed,
            pin_memory=False,
            sample_weights=sample_weights,
        )
        assert isinstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
        return list(loader.sampler)

    assert weighted_indices(17) == weighted_indices(17)
    assert weighted_indices(17) != weighted_indices(18)

    validation_loader = trainer._make_loader(
        rows,
        transform=None,
        batch_size=4,
        workers=0,
        shuffle=False,
        seed=19,
        pin_memory=False,
    )
    assert isinstance(validation_loader.sampler, torch.utils.data.SequentialSampler)
    assert list(validation_loader.sampler) == list(range(len(rows)))


def test_multiworker_loader_uses_spawn_and_consumes_verified_rows(tmp_path):
    manifest, _ = _complete_manifest(tmp_path)
    rows = [
        row
        for row in trainer.read_manifests([manifest])
        if row.role == trainer.TRAIN_ROLE
    ][:4]
    _, transform = trainer._build_transforms(16, smoke=True)
    loader = trainer._make_loader(
        rows,
        transform,
        batch_size=2,
        workers=2,
        shuffle=False,
        seed=23,
        pin_memory=False,
    )

    assert loader.multiprocessing_context.get_start_method() == "spawn"
    observed = [
        sample_id
        for batch in loader
        for sample_id in batch["sample_id"]
    ]
    assert observed == [row.sample_id for row in rows]


def test_condition_heads_use_current_minus_one_zero_one_labels(tmp_path):
    manifest, rows = _complete_manifest(tmp_path)
    for row in rows:
        if row["material"] != trainer.BACKGROUND_MATERIAL_ID:
            row["dent"] = int(row["material"]) % 2
            row["label"] = (int(row["material"]) + 1) % 2
    _write_manifest(manifest, rows)
    parsed = trainer.read_manifests([manifest])

    assert trainer.resolve_condition_heads(parsed, None) == ("dent", "label")
    assert trainer.resolve_condition_heads(parsed, ("dent",)) == ("dent",)
    assert trainer.resolve_condition_heads(parsed, ("label", "dent")) == (
        "dent",
        "label",
    )
    with pytest.raises(ValueError, match=r"foreign_material.*requires labels 0 and 1"):
        trainer.resolve_condition_heads(parsed, ("foreign_material",))
    weights = trainer.build_class_weight_values(parsed, ("dent",))
    assert weights["dent"].shape == (2,)


def test_seed_reproduces_model_initialization():
    trainer.seed_everything(31415)
    first = trainer.MultitaskCropVerifier(
        "tiny_cnn", pretrained=False, condition_heads=()
    )
    trainer.seed_everything(31415)
    second = trainer.MultitaskCropVerifier(
        "tiny_cnn", pretrained=False, condition_heads=()
    )
    trainer.seed_everything(27182)
    third = trainer.MultitaskCropVerifier(
        "tiny_cnn", pretrained=False, condition_heads=()
    )

    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        torch.testing.assert_close(left, right)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.state_dict().values(), third.state_dict().values())
    )


def test_onnx_output_names_follow_v3_contract(tmp_path, monkeypatch):
    captured = {}

    def fake_export(_model, _dummy, _path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    model = trainer.MultitaskCropVerifier(
        "tiny_cnn", pretrained=False, condition_heads=("label",)
    )

    trainer.export_onnx(model, tmp_path / trainer.ONNX_NAME, 16)

    assert captured["input_names"] == ["img"]
    assert captured["output_names"] == ["objectness", "material", "label"]
    assert set(captured["dynamic_axes"]) == {
        "img", "objectness", "material", "label"
    }


def test_checkpoint_selection_scores_all_heads_and_requires_full_support():
    metrics = {
        "objectness": {"balanced_accuracy": 0.8, "support": [4, 4]},
        "material": {"balanced_accuracy": 0.5, "support": [1] * 9},
        "dent": {"balanced_accuracy": 0.6, "support": [2, 2]},
    }

    assert trainer._selection_score(metrics, require_complete_support=True) == pytest.approx(
        (0.8 + 0.5 + 0.6) / 3
    )
    metrics["material"]["support"][8] = 0
    with pytest.raises(RuntimeError, match=r"material.*missing class support"):
        trainer._selection_score(metrics, require_complete_support=True)


def test_training_eager_cuda_context_creates_tensor_and_synchronizes(monkeypatch):
    events = []

    class FakeTensor:
        def __add__(self, value):
            events.append(("add", value))
            return self

        def item(self):
            return 2

    monkeypatch.setattr(
        trainer.torch.cuda,
        "is_available",
        lambda: events.append("available") or True,
    )
    monkeypatch.setattr(trainer.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        trainer.torch.cuda,
        "synchronize",
        lambda device: events.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        trainer.torch.cuda,
        "get_device_name",
        lambda device: events.append(("name", device)) or "fake-gpu",
    )
    monkeypatch.setattr(
        trainer.torch,
        "ones",
        lambda size, *, device: events.append(("ones", size, device)) or FakeTensor(),
    )

    guard = trainer.eager_initialize_cuda_context()

    assert isinstance(guard, FakeTensor)
    assert events == [
        "available",
        ("ones", 1, "cuda:0"),
        ("add", 1),
        ("synchronize", 0),
        ("name", 0),
    ]


def test_explicit_cuda_context_is_held_before_manifest_hashing(tmp_path, monkeypatch):
    events = []

    class StopBeforeManifestHashing(RuntimeError):
        pass

    guard = object()
    monkeypatch.setattr(
        trainer,
        "eager_initialize_cuda_context",
        lambda: events.append("eager") or guard,
    )

    def stop_reading(_paths):
        events.append("manifest")
        raise StopBeforeManifestHashing

    monkeypatch.setattr(trainer, "read_manifests", stop_reading)

    with pytest.raises(StopBeforeManifestHashing):
        trainer.main(
            [
                "--manifest",
                str(tmp_path / "strict.csv"),
                "--output-dir",
                str(tmp_path / "output"),
                "--no-condition-heads",
                "--device",
                "cuda",
            ]
        )

    assert events == ["eager", "manifest"]


def test_explicit_cpu_skips_eager_cuda_before_manifest_reading(tmp_path, monkeypatch):
    class StopBeforeManifestHashing(RuntimeError):
        pass

    monkeypatch.setattr(
        trainer,
        "eager_initialize_cuda_context",
        lambda: (_ for _ in ()).throw(
            AssertionError("explicit CPU training must not initialize CUDA")
        ),
    )
    monkeypatch.setattr(
        trainer,
        "read_manifests",
        lambda _paths: (_ for _ in ()).throw(StopBeforeManifestHashing),
    )

    with pytest.raises(StopBeforeManifestHashing):
        trainer.main(
            [
                "--manifest",
                str(tmp_path / "strict.csv"),
                "--output-dir",
                str(tmp_path / "output"),
                "--no-condition-heads",
                "--device",
                "cpu",
            ]
        )


def test_training_refuses_a_nonempty_output_directory(tmp_path, monkeypatch):
    manifest, _ = _complete_manifest(tmp_path)
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    monkeypatch.setattr(
        trainer,
        "eager_initialize_cuda_context",
        lambda: (_ for _ in ()).throw(
            AssertionError("output guard must run before CUDA initialization")
        ),
    )
    monkeypatch.setattr(
        trainer,
        "read_manifests",
        lambda _paths: (_ for _ in ()).throw(
            AssertionError("output guard must run before manifest hashing")
        ),
    )

    with pytest.raises(SystemExit):
        trainer.main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--no-condition-heads",
                "--smoke",
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "do not replace"
    assert not (output_dir / trainer.CHECKPOINT_NAME).exists()


def test_dry_run_validates_without_creating_training_artifacts(
    tmp_path, capsys, monkeypatch
):
    manifest, _ = _complete_manifest(tmp_path)
    absent_output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        trainer,
        "eager_initialize_cuda_context",
        lambda: (_ for _ in ()).throw(
            AssertionError("dry-run must not initialize CUDA")
        ),
    )

    result = trainer.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(absent_output),
            "--no-condition-heads",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["ok"] is True
    assert payload["mode"] == "dry-run"
    assert payload["manifest"]["rows"] == 20
    assert payload["output_contract"]["output_order"] == ["objectness", "material"]
    contract = payload["dataset_consumption_contract"]
    assert contract["schema"] == "v4_candidate_dataset_consumption.v1"
    assert contract["authority_platform"] == "unbound_local_validation"
    assert contract["manifest_payload_set_sha256"] == payload["manifest"][
        "payload_set_sha256"
    ]
    assert contract["complete_access_receipt"] is False
    assert contract["trainer_sha256"] == hashlib.sha256(
        Path(trainer.__file__).read_bytes()
    ).hexdigest()
    assert not absent_output.exists()


def test_cpu_smoke_writes_loadable_v3_checkpoint_and_metadata(tmp_path, monkeypatch):
    manifest, _ = _complete_manifest(tmp_path)
    output_dir = tmp_path / "smoke-output"
    monkeypatch.setattr(
        trainer,
        "eager_initialize_cuda_context",
        lambda: (_ for _ in ()).throw(
            AssertionError("CPU smoke must not initialize CUDA")
        ),
    )

    result = trainer.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--no-condition-heads",
            "--smoke",
            "--size",
            "16",
            "--batch",
            "20",
            "--seed",
            "17",
        ]
    )

    checkpoint = torch.load(
        output_dir / trainer.CHECKPOINT_NAME,
        map_location="cpu",
        weights_only=False,
    )
    metadata = json.loads(
        (output_dir / trainer.METADATA_NAME).read_text(encoding="utf-8")
    )
    assert result == 0
    assert checkpoint["format_version"] == 3
    assert checkpoint["model_config"] == {
        "backbone": "tiny_cnn",
        "input_size": 16,
        "condition_heads": [],
    }
    assert checkpoint["classes"] == list(trainer.MATERIAL_CLASS_NAMES)
    assert checkpoint["objectness_classes"] == list(trainer.OBJECTNESS_CLASS_NAMES)
    assert checkpoint["output_contract"]["material_background_class_id"] is None
    assert checkpoint["training_config"]["seed"] == 17
    assert checkpoint["training_config"]["deterministic_algorithms"] is True
    assert metadata["candidate_only"] is True
    assert metadata["production_runtime_modified"] is False
    assert metadata["output_contract"]["output_order"] == ["objectness", "material"]
    assert checkpoint["dataset_consumption_contract"] == metadata[
        "dataset_consumption_contract"
    ]
    assert metadata["dataset_consumption_contract"]["manifest_payload_set_sha256"] == (
        metadata["manifest_summary"]["payload_set_sha256"]
    )
    assert metadata["dataset_consumption_contract"]["complete_access_receipt"] is False
    assert metadata["onnx"] is None
    assert not (output_dir / trainer.ONNX_NAME).exists()
