import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.label_operational_captures_ollama as teacher
from scripts.build_independent_localization_consensus import (
    build_consensus,
    canonical_json,
    provider_output_core,
    sha256_bytes,
    sha256_file,
)
from scripts.build_operational_teacher_manifest import (
    ARTIFACT_NAMES,
    build_operational_teacher_manifest,
)
from scripts.prepare_operational_capture_queue import prepare_queue
from scripts.operational_teacher_contract import (
    TEACHER_LABEL_SCHEMA_VERSION, build_teacher_contract,
)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _valid_teacher_label(sha: str) -> dict:
    contract, contract_sha = build_teacher_contract("teacher-model", "a" * 64)
    decision = {
        "material": "paper", "confidence": 0.9, "single_object": True,
        "foreign_material": False, "training_usable": True,
        "quality_reason": "usable",
    }
    return {
        "schema_version": TEACHER_LABEL_SCHEMA_VERSION,
        "sha256": sha, "image_ref": "capture.jpg",
        "input_image_sha256": sha, "teacher_contract": contract,
        "teacher_contract_sha256": contract_sha, "model": "teacher-model",
        "model_digest": "a" * 64, "passes": [decision, dict(decision)],
        "errors": [], "consensus": True,
        "consensus_decision": {
            "material": "paper", "single_object": True,
            "foreign_material": False, "training_usable": True,
            "quality_reason": "usable", "votes": 2, "pass_count": 2,
        },
        "minimum_confidence": 0.9,
    }


def _provider_fixture(
    root: Path, *, name: str, source_sha: str, box: list[int], model_bytes: bytes
) -> tuple[Path, Path, Path]:
    model = root / f"{name}.model"
    spec = root / f"{name}.spec"
    manifest = root / f"{name}.jsonl"
    model.write_bytes(model_bytes)
    spec.write_bytes((name + "-spec").encode())
    core = provider_output_core(
        provider=name,
        source_sha=source_sha,
        box=[float(item) for item in box],
        model_sha=sha256_file(model),
        spec_sha=sha256_file(spec),
    )
    _jsonl(
        manifest,
        [
            dict(
                core,
                provider_output_sha256=sha256_bytes(
                    canonical_json(core).encode("utf-8")
                ),
            )
        ],
    )
    return manifest, model, spec


def _teacher_row(root: Path) -> tuple[Path, Path, str]:
    captures = root / "captures"
    day = captures / "2026-08-01"
    day.mkdir(parents=True)
    image = day / "capture.jpg"
    assert cv2.imwrite(str(image), np.full((240, 320, 3), 100, dtype=np.uint8))
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    (day / "capture.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-08-01T00:00:00+09:00",
                "image": {"path": "2026-08-01/capture.jpg", "sha256": sha},
                "request": {"client_id": "private"},
                "result": {"status": "ALLOWED", "bbox": [1, 2, 3, 4]},
            }
        ),
        encoding="utf-8",
    )
    shadow = root / "shadow.jsonl"
    shadow.write_text("", encoding="utf-8")
    known = root / "known.json"
    known.write_text("{}", encoding="utf-8")
    queue_dir = root / "queue"
    prepare_queue(
        captures_dir=captures,
        shadow_log=shadow,
        known_audit=known,
        output_dir=queue_dir,
        start_kst=datetime(2026, 8, 1, tzinfo=timezone(timedelta(hours=9))),
    )
    return captures, queue_dir, sha


def test_prepare_label_localize_build_positive_e2e(tmp_path, monkeypatch):
    captures, queue_dir, sha = _teacher_row(tmp_path)
    monkeypatch.setattr(teacher, "_observe_model_digest", lambda *_args: "a" * 64)
    monkeypatch.setattr(
        teacher,
        "_request",
        lambda *_args: {
            "material": "paper",
            "confidence": 0.95,
            "single_object": True,
            "foreign_material": False,
            "training_usable": True,
            "quality_reason": "usable",
        },
    )
    labels = tmp_path / "labels.jsonl"
    teacher.label_queue(
        queue_dir / "teacher_queue.jsonl",
        labels,
        image_root=captures,
        known_audit=tmp_path / "known.json",
        url="http://127.0.0.1:11434",
        model="teacher-model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )
    a_manifest, a_model, a_spec = _provider_fixture(
        tmp_path, name="provider-a", source_sha=sha,
        box=[10, 10, 100, 100], model_bytes=b"model-a",
    )
    b_manifest, b_model, b_spec = _provider_fixture(
        tmp_path, name="provider-b", source_sha=sha,
        box=[12, 12, 102, 102], model_bytes=b"model-b",
    )
    enriched = tmp_path / "enriched.jsonl"
    result = build_consensus(
        teacher_labels=labels,
        provider_a_manifest=a_manifest, provider_a_name="provider-a",
        provider_a_model=a_model, provider_a_spec=a_spec,
        provider_b_manifest=b_manifest, provider_b_name="provider-b",
        provider_b_model=b_model, provider_b_spec=b_spec, output=enriched,
    )
    assert result["localized"] == 1
    output = tmp_path / "manifest"
    built = build_operational_teacher_manifest(
        teacher_queue=queue_dir / "teacher_queue.jsonl",
        teacher_labels=enriched,
        image_root=captures,
        known_audit=tmp_path / "known.json",
        provider_a_manifest=a_manifest,
        provider_a_name="provider-a",
        provider_a_model=a_model,
        provider_a_spec=a_spec,
        provider_b_manifest=b_manifest,
        provider_b_name="provider-b",
        provider_b_model=b_model,
        provider_b_spec=b_spec,
        capture_inventory=queue_dir / "capture_inventory.json",
        output_dir=output,
    )
    assert built["accepted"] == 1
    rows = list(csv.DictReader((output / ARTIFACT_NAMES["csv"]).open()))
    assert rows[0]["bbox_source"] == "independent_localization_consensus"
    lineage = json.loads((output / ARTIFACT_NAMES["lineage"]).read_text())
    assert lineage["portable"] is False
    assert lineage["local_only_contains_absolute_paths"] is True


def test_producer_rejects_same_model_output_hash_and_frozen_file_tamper(tmp_path):
    labels = tmp_path / "labels.jsonl"
    sha = "9" * 64
    _jsonl(labels, [_valid_teacher_label(sha)])
    a_manifest, a_model, a_spec = _provider_fixture(
        tmp_path, name="a", source_sha=sha, box=[0, 0, 100, 100], model_bytes=b"same"
    )
    b_manifest, b_model, b_spec = _provider_fixture(
        tmp_path, name="b", source_sha=sha, box=[1, 1, 101, 101], model_bytes=b"same"
    )
    kwargs = dict(
        teacher_labels=labels,
        provider_a_manifest=a_manifest, provider_a_name="a",
        provider_a_model=a_model, provider_a_spec=a_spec,
        provider_b_manifest=b_manifest, provider_b_name="b",
        provider_b_model=b_model, provider_b_spec=b_spec,
        output=tmp_path / "out.jsonl",
    )
    with pytest.raises(ValueError, match="model SHA"):
        build_consensus(**kwargs)

    b_model.write_bytes(b"different")
    with pytest.raises(ValueError, match="frozen provider files"):
        build_consensus(**kwargs)
    b_manifest, b_model, b_spec = _provider_fixture(
        tmp_path, name="b", source_sha=sha, box=[1, 1, 101, 101], model_bytes=b"same"
    )
    kwargs.update(
        provider_b_manifest=b_manifest,
        provider_b_model=b_model,
        provider_b_spec=b_spec,
    )
    manifest_row = json.loads(b_manifest.read_text())
    manifest_row["bbox_xyxy"][0] += 1
    _jsonl(b_manifest, [manifest_row])
    with pytest.raises(ValueError, match="output hash mismatch"):
        build_consensus(**kwargs)

    b_manifest, b_model, b_spec = _provider_fixture(
        tmp_path, name="b", source_sha=sha, box=[1, 1, 101, 101], model_bytes=b"same"
    )
    kwargs.update(
        provider_b_manifest=b_manifest,
        provider_b_model=b_model,
        provider_b_spec=b_spec,
    )
    b_spec.write_bytes(b"tampered-spec")
    with pytest.raises(ValueError, match="frozen provider files"):
        build_consensus(**kwargs)

    b_manifest, b_model, b_spec = _provider_fixture(
        tmp_path, name="b", source_sha=sha, box=[1, 1, 101, 101], model_bytes=b"same"
    )
    kwargs.update(
        provider_b_manifest=b_manifest,
        provider_b_model=b_model,
        provider_b_spec=b_spec,
    )
    row = json.loads(b_manifest.read_text())
    row["provider_output_sha256"] = "0" * 64
    _jsonl(b_manifest, [row])
    with pytest.raises(ValueError, match="output hash mismatch"):
        build_consensus(**kwargs)


def test_producer_low_iou_and_deployed_evidence_fail_closed(tmp_path):
    labels = tmp_path / "labels.jsonl"
    sha = "8" * 64
    _jsonl(labels, [_valid_teacher_label(sha)])
    a_manifest, a_model, a_spec = _provider_fixture(
        tmp_path, name="a", source_sha=sha, box=[0, 0, 50, 50], model_bytes=b"a"
    )
    b_manifest, b_model, b_spec = _provider_fixture(
        tmp_path, name="b", source_sha=sha, box=[100, 100, 150, 150], model_bytes=b"b"
    )
    output = tmp_path / "out.jsonl"
    result = build_consensus(
        teacher_labels=labels,
        provider_a_manifest=a_manifest, provider_a_name="a",
        provider_a_model=a_model, provider_a_spec=a_spec,
        provider_b_manifest=b_manifest, provider_b_name="b",
        provider_b_model=b_model, provider_b_spec=b_spec, output=output,
    )
    assert result["localized"] == 0
    assert "independent_localization" not in json.loads(
        output.read_text(encoding="utf-8")
    )

    row = json.loads(a_manifest.read_text())
    row["deployed"] = {"bbox": [0, 0, 50, 50]}
    _jsonl(a_manifest, [row])
    with pytest.raises(ValueError, match="deployed prediction"):
        build_consensus(
            teacher_labels=labels,
            provider_a_manifest=a_manifest, provider_a_name="a",
            provider_a_model=a_model, provider_a_spec=a_spec,
            provider_b_manifest=b_manifest, provider_b_name="b",
            provider_b_model=b_model, provider_b_spec=b_spec,
            output=output,
        )


@pytest.mark.parametrize("extra", ["client_id", "result", "filepath", "anything"])
def test_producer_rejects_any_extra_teacher_field(tmp_path, extra):
    sha = "7" * 64
    labels = tmp_path / "labels-extra.jsonl"
    row = _valid_teacher_label(sha)
    row[extra] = "must-not-copy"
    _jsonl(labels, [row])
    a_manifest, a_model, a_spec = _provider_fixture(
        tmp_path, name="extra-a", source_sha=sha,
        box=[0, 0, 100, 100], model_bytes=b"a",
    )
    b_manifest, b_model, b_spec = _provider_fixture(
        tmp_path, name="extra-b", source_sha=sha,
        box=[1, 1, 101, 101], model_bytes=b"b",
    )
    output = tmp_path / "enriched-extra.jsonl"
    with pytest.raises(ValueError, match="row shape is not exact"):
        build_consensus(
            teacher_labels=labels,
            provider_a_manifest=a_manifest, provider_a_name="extra-a",
            provider_a_model=a_model, provider_a_spec=a_spec,
            provider_b_manifest=b_manifest, provider_b_name="extra-b",
            provider_b_model=b_model, provider_b_spec=b_spec, output=output,
        )
    assert not output.exists()
