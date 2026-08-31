import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.replay_v4_candidate_metrics import replay_validation


MATERIALS = [
    "can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass",
    "battery", "fluorescent",
]
FIELDS = [
    "filepath", "split", "source_id", "material", "category", "dent",
    "label", "foreign_material", "source_object_count", "crop_object_count",
    "sample_id", "source_sha256", "image_sha256", "object_group",
    "capture_session", "role", "fold", "origin",
]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _Info:
    def __init__(self, name, shape, type_="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_


class _Session:
    def __init__(self, predictions, *, malformed=False):
        self.predictions = list(predictions)
        self.offset = 0
        self.malformed = malformed

    def get_inputs(self):
        return [_Info("img", ["batch", 3, 320, 320])]

    def get_outputs(self):
        return [
            _Info("objectness", ["batch", 2]),
            _Info("material", ["batch", 9]),
        ]

    def run(self, output_names, inputs):
        assert output_names == ["objectness", "material"]
        batch = next(iter(inputs.values())).shape[0]
        selected = self.predictions[self.offset : self.offset + batch]
        self.offset += batch
        objectness = np.full((batch, 2), -2.0, dtype=np.float32)
        material = np.full((batch, 9), -3.0, dtype=np.float32)
        for index, (objectness_id, material_id) in enumerate(selected):
            objectness[index, objectness_id] = 2.0
            material[index, material_id] = 3.0
        if self.malformed:
            objectness[0, 0] = np.nan
        return [objectness, material]


class _MutatingSession(_Session):
    def __init__(self, predictions, *, target: Path):
        super().__init__(predictions)
        self.target = target
        self.mutated = False

    def run(self, output_names, inputs):
        values = super().run(output_names, inputs)
        if not self.mutated:
            self.target.write_bytes(b"changed-during-replay")
            self.mutated = True
        return values


def _metadata(onnx_name: str) -> dict:
    return {
        "format_version": 3,
        "architecture": "multitask_crop_verifier",
        "candidate_only": True,
        "production_runtime_modified": False,
        "onnx": onnx_name,
        "model_config": {"input_size": 320},
        "preprocessing": {
            "color_space": "RGB",
            "resize": [320, 320],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "material_classes": MATERIALS,
        "objectness_classes": ["background", "material"],
        "output_contract": {
            "version": "multitask_verifier.v3",
            "output_order": ["objectness", "material"],
            "material_background_class_id": None,
            "outputs": [
                {
                    "name": "objectness",
                    "kind": "logits",
                    "activation": "softmax",
                    "class_names": ["background", "material"],
                    "shape": ["batch", 2],
                },
                {
                    "name": "material",
                    "kind": "logits",
                    "activation": "softmax",
                    "class_names": MATERIALS,
                    "shape": ["batch", 9],
                    "valid_when": {
                        "output": "objectness",
                        "class_id": 1,
                        "class_name": "material",
                    },
                },
            ],
        },
    }


@pytest.fixture
def replay_fixture(tmp_path):
    rows = []
    for role in ("train", "model_validation"):
        for material in range(10):
            token = f"{role}-{material}"
            image = tmp_path / "images" / f"{token}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            role_offset = 80 if role == "model_validation" else 0
            Image.new(
                "RGB", (16, 16), ((material * 17 + role_offset) % 256, 20, 30)
            ).save(image)
            rows.append(
                {
                    "filepath": image.relative_to(tmp_path).as_posix(),
                    "split": "validation" if role == "model_validation" else "training",
                    "source_id": f"source-{token}",
                    "material": material,
                    "category": "background" if material == 9 else MATERIALS[material],
                    "dent": -1,
                    "label": -1,
                    "foreign_material": -1,
                    "source_object_count": 0 if material == 9 else 1,
                    "crop_object_count": 0 if material == 9 else 1,
                    "sample_id": f"sample-{token}",
                    "source_sha256": _sha(f"source-{token}"),
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "object_group": f"object-{token}",
                    "capture_session": f"session-{token}",
                    "role": role,
                    "fold": f"fold-{role}",
                    "origin": "replay-test",
                }
            )
    manifest = tmp_path / "strict.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    onnx = tmp_path / "candidate.onnx"
    onnx.write_bytes(b"candidate-onnx")
    metadata = tmp_path / "multitask_verifier_metadata.json"
    metadata.write_text(json.dumps(_metadata(onnx.name)), encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_bytes((Path(__file__).parents[1] / "configs" / "detector_inference_v3.json").read_bytes())
    validation = sorted(
        (row for row in rows if row["role"] == "model_validation"),
        key=lambda row: row["sample_id"],
    )
    predictions = [
        (0, 0) if row["material"] == 9 else (1, int(row["material"]))
        for row in validation
    ]
    return {
        "rows": rows,
        "manifest": manifest,
        "onnx": onnx,
        "metadata": metadata,
        "spec": spec,
        "predictions": predictions,
    }


def _run(fixture, output, session):
    return replay_validation(
        manifest_paths=[fixture["manifest"]],
        verifier_onnx=fixture["onnx"],
        verifier_metadata=fixture["metadata"],
        inference_spec=fixture["spec"],
        output_jsonl=output / "predictions.jsonl",
        output_attestation=output / "attestation.json",
        batch_size=4,
        session_factory=lambda _path: session,
    )


def test_replays_actual_validation_crops_and_hash_binds_metrics(replay_fixture, tmp_path):
    output = tmp_path / "out"
    session = _Session(replay_fixture["predictions"])

    attestation = _run(replay_fixture, output, session)

    assert attestation["prediction_count"] == 10
    assert attestation["custom_session_factory_used"] is True
    assert (
        attestation["artifact_snapshot_contract"]
        == "read_bytes_hash_before_use_and_hash_after.v1"
    )
    assert attestation["runtime_artifact_hashes_match_snapshots"] is True
    assert attestation["metrics"]["objectness"]["confusion"] == [[1, 0], [0, 9]]
    assert attestation["metrics"]["material"]["balanced_accuracy"] == 1.0
    assert attestation["model_sha256"] == hashlib.sha256(
        replay_fixture["onnx"].read_bytes()
    ).hexdigest()
    predictions = output / "predictions.jsonl"
    assert attestation["predictions_sha256"] == hashlib.sha256(
        predictions.read_bytes()
    ).hexdigest()
    first = json.loads(predictions.read_text().splitlines()[0])
    assert len(first["objectness_logits"]) == 2
    assert len(first["material_logits"]) == 9
    assert len(first["input_tensor_sha256"]) == 64
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(replay_fixture, output, _Session(replay_fixture["predictions"]))


def test_rejects_crop_bytes_that_no_longer_match_manifest_hash(replay_fixture, tmp_path):
    image = Path(replay_fixture["manifest"]).parent / replay_fixture["rows"][0]["filepath"]
    image.write_bytes(b"tampered")
    output = tmp_path / "tampered"

    with pytest.raises(ValueError, match="image_sha256 does not match"):
        _run(replay_fixture, output, _Session(replay_fixture["predictions"]))

    assert not output.exists()


def test_nonfinite_model_output_fails_without_publishing(replay_fixture, tmp_path):
    output = tmp_path / "nonfinite"

    with pytest.raises(ValueError, match="non-finite"):
        _run(
            replay_fixture,
            output,
            _Session(replay_fixture["predictions"], malformed=True),
        )

    assert not output.exists()


@pytest.mark.parametrize("target_kind", ["model", "image"])
def test_input_mutation_during_replay_fails_before_publication(
    replay_fixture, tmp_path, target_kind
):
    if target_kind == "model":
        target = replay_fixture["onnx"]
    else:
        validation_row = next(
            row
            for row in replay_fixture["rows"]
            if row["role"] == "model_validation"
        )
        target = replay_fixture["manifest"].parent / validation_row["filepath"]
    output = tmp_path / f"mutated-{target_kind}"

    with pytest.raises(RuntimeError, match="changed during replay"):
        _run(
            replay_fixture,
            output,
            _MutatingSession(replay_fixture["predictions"], target=target),
        )

    assert not output.exists()
