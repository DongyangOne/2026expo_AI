import json
import os
from pathlib import Path

os.environ.setdefault("API_KEY", "test-key")

import numpy as np
import pytest

from app.core.config import settings
from app.models.registry import VerifierRuntime, _load_verifier_metadata
from app.services import verifier_shadow
from app.services.inference import VERIFIER_CLASS_NAMES, run_verifier


class _Node:
    def __init__(self, name: str, shape=None):
        self.name = name
        self.shape = shape


class _MaterialOnlyBackgroundSession:
    def __init__(self):
        self.requested_outputs = None

    def get_inputs(self):
        return [_Node("img", [1, 3, 320, 320])]

    def get_outputs(self):
        # material-only export에는 상태 헤드가 물리적으로 없어도 된다.
        return [_Node("material")]

    def run(self, names, inputs):
        self.requested_outputs = names
        assert inputs["img"].shape == (1, 3, 320, 320)
        logits = np.zeros((1, 10), dtype=np.float32)
        logits[0, 9] = 8.0
        return [logits]


def _background_runtime(tmp_path: Path) -> VerifierRuntime:
    return VerifierRuntime(
        session=_MaterialOnlyBackgroundSession(),
        class_names=(*VERIFIER_CLASS_NAMES, "background"),
        enabled_outputs=frozenset({"material"}),
        metadata_path=tmp_path / "verifier_metadata.json",
    )


def test_metadata_loader_reads_dynamic_classes_and_enabled_outputs(tmp_path):
    model_path = tmp_path / "verifier.onnx"
    model_path.write_bytes(b"test")
    metadata_path = tmp_path / "verifier_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "classes": [*VERIFIER_CLASS_NAMES, "background"],
                "material_class_count": 10,
                "enabled_outputs": ["material"],
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_verifier_metadata(str(model_path))

    assert loaded == (
        (*VERIFIER_CLASS_NAMES, "background"),
        frozenset({"material"}),
        metadata_path,
    )


@pytest.mark.parametrize("payload", [None, [], "invalid"])
def test_metadata_loader_rejects_non_object_payload(tmp_path, payload):
    model_path = tmp_path / "verifier.onnx"
    model_path.write_bytes(b"test")
    (tmp_path / "verifier_metadata.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="최상위 형식"):
        _load_verifier_metadata(str(model_path))


def test_metadata_loader_rejects_reordered_class_contract(tmp_path):
    model_path = tmp_path / "verifier.onnx"
    model_path.write_bytes(b"test")
    reordered = list(VERIFIER_CLASS_NAMES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    (tmp_path / "verifier_metadata.json").write_text(
        json.dumps(
            {
                "classes": [*reordered, "background"],
                "material_class_count": 10,
                "enabled_outputs": ["material"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="순서/계약"):
        _load_verifier_metadata(str(model_path))

def test_material_only_background_prediction_does_not_consume_disabled_heads(tmp_path):
    runtime = _background_runtime(tmp_path)

    result = run_verifier(
        runtime,
        np.zeros((480, 640, 3), dtype=np.uint8),
        [100, 80, 500, 420],
    )

    assert runtime.session.requested_outputs == ["material"]
    assert result["material"]["class_id"] == 9
    assert result["material"]["class_name"] == "background"
    assert result["heads"] == {}


def test_background_top1_is_written_to_shadow_log(tmp_path, monkeypatch):
    log_path = tmp_path / "verifier_shadow.jsonl"
    monkeypatch.setattr(settings, "VERIFIER_SHADOW_LOG_PATH", str(log_path))

    verifier_shadow._run_and_log(
        _background_runtime(tmp_path),
        np.zeros((480, 640, 3), dtype=np.uint8),
        [100, 80, 500, 420],
        yolo_class_id=3,
        yolo_confidence=0.72,
        client_id="hardware-background-001",
    )

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["client_id"] == "hardware-background-001"
    assert record["verifier"]["material"]["class_id"] == 9
    assert record["verifier"]["material"]["class_name"] == "background"
    assert record["material_agreement"] is False
