import csv
import json
import sys

import pytest
import torch
import torch.nn as nn
from PIL import Image

import scripts.train_verifier as trainer


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(3, 4)

    def forward(self, image):
        return self.projection(self.pool(image).flatten(1))


@pytest.fixture
def tiny_backbone(monkeypatch):
    monkeypatch.setattr(
        trainer,
        "_build_backbone",
        lambda _name, _pretrained: (_TinyBackbone(), 4),
    )


def _row(split, material):
    return {
        "split": split,
        "material": material,
        "dent": -1,
        "label": -1,
        "foreign_material": -1,
    }


def test_background_is_opt_in_and_material_task_validation_is_dynamic(tiny_backbone):
    default_model = trainer.CropVerifier("tiny", pretrained=False)
    background_classes = trainer.material_class_names(include_background=True)
    background_model = trainer.CropVerifier(
        "tiny", pretrained=False, material_classes=background_classes,
    )

    assert trainer.material_class_names() == trainer.CLASS_NAMES
    assert background_classes == [*trainer.CLASS_NAMES, "background"]
    assert default_model.material_head[-1].out_features == 9
    assert background_model.material_head[-1].out_features == 10

    rows = [
        _row(split, material)
        for split in ("training", "validation")
        for material in range(10)
    ]
    trainer.validate_task_labels(rows, background_classes)
    assert trainer.enabled_tasks_for(rows, background_classes) == ["material"]
    with pytest.raises(ValueError, match=r"material=9"):
        trainer.validate_task_labels(rows, trainer.CLASS_NAMES)


def test_nine_class_checkpoint_expands_only_material_background_row(tiny_backbone):
    source = trainer.CropVerifier("tiny", pretrained=False)
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(float(index))

    background_classes = trainer.material_class_names(include_background=True)
    target = trainer.CropVerifier(
        "tiny", pretrained=False, material_classes=background_classes,
    )
    initial_background_weight = target.material_head[-1].weight[9].detach().clone()
    initial_background_bias = target.material_head[-1].bias[9].detach().clone()
    checkpoint = {
        "classes": list(trainer.CLASS_NAMES),
        "state_dict": source.state_dict(),
    }

    transfer = trainer.load_initial_checkpoint_state(
        target, checkpoint, background_classes,
    )

    assert transfer == {
        "mode": "expanded_background",
        "source_classes": trainer.CLASS_NAMES,
    }
    torch.testing.assert_close(
        target.material_head[-1].weight[:9], source.material_head[-1].weight,
    )
    torch.testing.assert_close(
        target.material_head[-1].bias[:9], source.material_head[-1].bias,
    )
    torch.testing.assert_close(
        target.material_head[-1].weight[9], initial_background_weight,
    )
    torch.testing.assert_close(
        target.material_head[-1].bias[9], initial_background_bias,
    )
    for target_head, source_head in (
        (target.dent_head, source.dent_head),
        (target.label_head, source.label_head),
        (target.foreign_head, source.foreign_head),
    ):
        for target_value, source_value in zip(
            target_head.state_dict().values(), source_head.state_dict().values(),
        ):
            torch.testing.assert_close(target_value, source_value)
    for target_value, source_value in zip(
        target.backbone.state_dict().values(), source.backbone.state_dict().values(),
    ):
        torch.testing.assert_close(target_value, source_value)


def test_legacy_nine_class_checkpoint_still_loads_exactly(tiny_backbone):
    source = trainer.CropVerifier("tiny", pretrained=False)
    target = trainer.CropVerifier("tiny", pretrained=False)

    transfer = trainer.load_initial_checkpoint_state(
        target,
        {"state_dict": source.state_dict()},
        trainer.material_class_names(),
    )

    assert transfer == {"mode": "exact", "source_classes": trainer.CLASS_NAMES}
    for target_value, source_value in zip(
        target.state_dict().values(), source.state_dict().values(),
    ):
        torch.testing.assert_close(target_value, source_value)


def test_background_training_writes_dynamic_checkpoint_and_metadata(
    tmp_path, monkeypatch, tiny_backbone,
):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filepath", "split", "material", "dent", "label",
                "foreign_material",
            ],
        )
        writer.writeheader()
        for split in ("training", "validation"):
            for material in range(10):
                image_path = tmp_path / f"{split}-{material}.png"
                Image.new("RGB", (12, 12), color=(material * 20, 30, 40)).save(image_path)
                writer.writerow({
                    "filepath": image_path.name,
                    "split": split,
                    "material": material,
                    "dent": -1,
                    "label": -1,
                    "foreign_material": -1,
                })

    legacy_model = trainer.CropVerifier("tiny", pretrained=False)
    legacy_checkpoint = tmp_path / "legacy.pt"
    torch.save({
        "state_dict": legacy_model.state_dict(),
        "backbone": "tiny",
        "input_size": 16,
        "classes": list(trainer.CLASS_NAMES),
    }, legacy_checkpoint)

    exported = {}

    def fake_export(model, path, size):
        with torch.inference_mode():
            outputs = model(torch.randn(1, 3, size, size))
        exported["shapes"] = [tuple(output.shape) for output in outputs]
        path.write_bytes(b"fake-onnx")

    output_dir = tmp_path / "output"
    monkeypatch.setattr(trainer, "export_onnx", fake_export)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", [
        "train_verifier.py",
        "--manifest", str(manifest),
        "--output-dir", str(output_dir),
        "--backbone", "tiny",
        "--size", "16",
        "--epochs", "1",
        "--patience", "1",
        "--batch", "20",
        "--workers", "0",
        "--init-checkpoint", str(legacy_checkpoint),
        "--include-background",
    ])

    trainer.main()

    checkpoint = torch.load(
        output_dir / "best_verifier.pt", map_location="cpu", weights_only=False,
    )
    metadata = json.loads(
        (output_dir / "verifier_metadata.json").read_text(encoding="utf-8")
    )
    assert checkpoint["classes"] == [*trainer.CLASS_NAMES, "background"]
    assert checkpoint["include_background"] is True
    assert checkpoint["background_class_id"] == 9
    assert checkpoint["state_dict"]["material_head.1.weight"].shape[0] == 10
    assert metadata["classes"] == [*trainer.CLASS_NAMES, "background"]
    assert metadata["material_class_count"] == 10
    assert metadata["include_background"] is True
    assert metadata["background_class_id"] == 9
    assert metadata["enabled_outputs"] == ["material"]
    assert metadata["initial_checkpoint_transfer"] == {
        "mode": "expanded_background",
        "source_classes": trainer.CLASS_NAMES,
    }
    assert metadata["training_config"] == {
        "label_smoothing": 0.0,
        "learning_rates": {
            "base": 1e-3,
            "backbone": 1e-3,
            "heads": 1e-3,
        },
        "class_weights": {
            "mode": "inverse",
            "beta": trainer.DEFAULT_CLASS_WEIGHT_BETA,
            "values": {
                "material": [1.0] * 10,
                "dent": None,
                "label": None,
                "foreign_material": None,
            },
        },
    }
    assert exported["shapes"] == [(1, 10), (1, 2), (1, 2), (1, 2)]


def test_background_model_keeps_four_onnx_output_names(
    tmp_path, monkeypatch, tiny_backbone,
):
    captured = {}

    def fake_onnx_export(_model, _dummy, _path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(torch.onnx, "export", fake_onnx_export)
    model = trainer.CropVerifier(
        "tiny",
        pretrained=False,
        material_classes=trainer.material_class_names(include_background=True),
    )

    trainer.export_onnx(model, tmp_path / "verifier.onnx", 16)

    assert captured["output_names"] == [
        "material", "dent", "label", "foreign_material",
    ]
