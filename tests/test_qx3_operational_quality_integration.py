"""CPU contracts: real selector plus wrappers with explicit replay/selector doubles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def _helpers(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_integration_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_prepare_assembler_qx3_wrappers_and_candidate_share_quality_evidence(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Exercise real evidence producers/consumers, not GPU model accuracy."""

    operational = _helpers("test_build_operational_teacher_manifest")
    qx3 = _helpers("test_run_v4_repro_pilot_validation")
    candidate = _helpers("test_build_v4_candidate_training_authority")
    args, objective_image, subjective_image = (
        operational._objective_quality_assembly_fixture(tmp_path / "operational")
    )
    operational.assemble_operational_quality_exclusions(**args)
    manifest = args["output_dir"] / operational.ASSEMBLY_FILES["manifest"]
    receipt = args["output_dir"] / operational.ASSEMBLY_FILES["receipt"]
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_value["scope"]["objective_prepare_bundle_validated"] is True

    cohort = qx3.cohort_base.__wrapped__(tmp_path_factory)
    for image in (objective_image, subjective_image):
        source = (
            Path(cohort["root"]) / "sources" / "training" / "images"
            / "vinyl" / f"operational-{_sha(image)}.jpg"
        )
        source.write_bytes(image.read_bytes())
        label = (
            Path(cohort["root"]) / "sources" / "training" / "labels"
            / "vinyl" / f"operational-{_sha(image)}.txt"
        )
        label.write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    env = qx3._fixture(
        tmp_path / "qx3", cohort=cohort, quality_bundle=(manifest, receipt)
    )
    # Exercise the production selector against the actual assembler output too.
    # The expensive wrapper fixture independently uses explicit selector/replay
    # doubles; neither stage is an accuracy or cryptographic-attestation test.
    from scripts.build_v4_repro_pilot_inputs import build_pilot_inputs

    selector = _helpers("test_build_v4_repro_pilot_inputs")
    selector_data, selector_dataset, _ = selector._fixture(tmp_path / "selector-sources")
    for image in (objective_image, subjective_image):
        basename = f"operational-{_sha(image)}"
        (selector_dataset / "train" / "images" / f"{basename}.jpg").write_bytes(
            image.read_bytes()
        )
        (selector_dataset / "train" / "labels" / f"{basename}.txt").write_text(
            "5 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
        with (selector_dataset / "train.txt").open("a", encoding="utf-8") as handle:
            handle.write(
                (selector_dataset / "train" / "images" / f"{basename}.jpg").as_posix()
                + "\n"
            )
    selector_ready = build_pilot_inputs(
        data_path=selector_data,
        dataset_dir=selector_dataset,
        output_dir=tmp_path / "actual-selector",
        quality_exclusion_manifest=manifest,
        quality_exclusion_assembly_receipt=receipt,
        seed=20260901,
        training_quota=1,
        validation_quota=1,
    )
    assert selector_ready["bindings"]["quality_exclusions_sha256"] == _sha(manifest)
    assert selector_ready["full_quota_met"] is True
    assert selector_ready["selected_sources"] == 20
    assert selector_ready["bindings"]["quality_exclusion_assembly_receipt_sha256"] == (
        _sha(receipt)
    )
    actual_inventory = json.loads(
        (tmp_path / "actual-selector" / "selection_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    excluded = {_sha(objective_image), _sha(subjective_image)}
    assert actual_inventory["quality_exclusion"]["matched_resolved_sources"] == 2
    assert not excluded.intersection(
        row["source_sha256"] for row in actual_inventory["selected_sources"]
    )
    result = subprocess.run(
        [qx3._integration_bash(tmp_path), qx3.SCRIPT.as_posix()],
        env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    control = Path(env["VALIDATION_DIR"]) / "control"
    qx3_ready = control / "diagnostic_ready.json"
    qx3_report = control / "reproducibility_comparison.json"
    for path in (qx3_ready, qx3_report):
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["bindings"]["quality_exclusions_sha256"] == _sha(manifest)
        assert value["bindings"]["quality_exclusion_assembly_receipt_sha256"] == (
            _sha(receipt)
        )

    fixture = candidate._fixture(tmp_path_factory.mktemp("candidate-quality"))
    bad_row = fixture["bad_row"]
    Path(bad_row["source_filepath"]).write_bytes(objective_image.read_bytes())
    bad_row.update({
        "source_sha256": _sha(objective_image),
        "origin": "ops",
        "captured_at": "2026-08-01T00:00:00+09:00",
        "auditor_sha256": candidate._fake_sha("objective-auditor"),
        "teacher_output_sha256": candidate._fake_sha("objective-teacher"),
        "localizer_output_sha256": candidate._fake_sha("objective-localizer"),
    })
    subjective_source = fixture["global_root"] / "data" / "subjective-source.jpg"
    subjective_crop = fixture["global_root"] / "data" / "subjective-crop.jpg"
    subjective_source.write_bytes(subjective_image.read_bytes())
    subjective_crop.write_bytes(b"subjective-crop-fixture")
    subjective_row = dict(bad_row)
    subjective_row.update({
        "filepath": str(subjective_crop.resolve()),
        "source_id": "subjective-source-id",
        "sample_id": "subjective-sample-id",
        "source_sha256": _sha(subjective_source),
        "image_sha256": _sha(subjective_crop),
        "object_group": "subjective-object-group",
        "capture_session": "subjective-capture-session",
        "source_filepath": str(subjective_source.resolve()),
    })
    fixture["rows"].append(subjective_row)
    fixture["quality"] = manifest
    fixture["quality_assembly_receipt"] = receipt
    candidate._refresh(fixture)
    fixture["qx3_ready"] = qx3_ready
    fixture["qx3_report"] = qx3_report
    policy = fixture["policy"]
    policy_value = json.loads(policy.read_text(encoding="utf-8"))
    policy_value["qx3_diagnostic_ready_sha256"] = _sha(qx3_ready)
    policy_value["qx3_diagnostic_report_sha256"] = _sha(qx3_report)
    candidate._dump(policy, policy_value)

    authority = candidate._run(fixture)
    assert authority["bindings"]["quality_exclusions_sha256"] == _sha(manifest)
    assert authority["bindings"]["quality_exclusion_assembly_receipt_sha256"] == (
        _sha(receipt)
    )
    assert authority["counts"]["excluded"] == {
        "operational/before_2026_08_01_kst": 1,
        "quality/too_low_resolution": 1,
        "quality/unreadable_boundary": 1,
    }
    assert authority["production_deployment_authorized"] is False
    assert (fixture["global_root"] / "authority" / "training_authority.sha256").is_file()


def test_real_zero_exclusion_assembly_reaches_selector_wrappers_and_preaudit(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Real empty assembly is evidence, not an empty-file skip or model accuracy."""
    from scripts.build_v4_repro_pilot_inputs import build_pilot_inputs
    from scripts.operational_quality_assembly_contract import _validate_quality_manifest

    operational = _helpers("test_build_operational_teacher_manifest")
    qx3 = _helpers("test_run_v4_repro_pilot_validation")
    selector = _helpers("test_build_v4_repro_pilot_inputs")
    candidate = _helpers("test_build_v4_candidate_training_authority")
    args, _, _ = operational._objective_quality_assembly_fixture(
        tmp_path / "operational-empty", no_quality_exclusions=True,
    )
    assembled = operational.assemble_operational_quality_exclusions(**args)
    assert assembled["selected_source_count"] == 0
    assert assembled["scope"]["objective_prepare_bundle_validated"] is True
    assert not any(assembled["authority"].values())
    manifest = args["output_dir"] / operational.ASSEMBLY_FILES["manifest"]
    receipt = args["output_dir"] / operational.ASSEMBLY_FILES["receipt"]
    empty_value = json.loads(manifest.read_text(encoding="utf-8"))
    assert empty_value["entries"] == []
    with pytest.raises(ValueError, match="validated full assembly"):
        _validate_quality_manifest(empty_value)

    data, dataset, _ = selector._fixture(tmp_path / "selector-sources")
    ready = build_pilot_inputs(
        data_path=data, dataset_dir=dataset, output_dir=tmp_path / "actual-selector",
        quality_exclusion_manifest=manifest,
        quality_exclusion_assembly_receipt=receipt,
        seed=20260901, training_quota=1, validation_quota=1,
    )
    assert ready["full_quota_met"] is True
    assert ready["bindings"]["quality_exclusions_sha256"] == _sha(manifest)
    assert ready["bindings"]["quality_exclusion_assembly_receipt_sha256"] == _sha(receipt)
    inventory = json.loads(
        (tmp_path / "actual-selector" / "selection_inventory.json").read_text(encoding="utf-8")
    )
    assert inventory["quality_exclusion"]["matched_resolved_sources"] == 0

    # Wrappers use explicit replay/selector doubles; the quality bundle is real.
    cohort = qx3.cohort_base.__wrapped__(tmp_path_factory)
    env = qx3._fixture(tmp_path / "qx3-empty", cohort=cohort, quality_bundle=(manifest, receipt))
    result = subprocess.run(
        [qx3._integration_bash(tmp_path), qx3.SCRIPT.as_posix()],
        env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    diagnostic = json.loads(
        (Path(env["VALIDATION_DIR"]) / "control" / "diagnostic_ready.json").read_text(encoding="utf-8")
    )
    assert diagnostic["bindings"]["quality_exclusions_sha256"] == _sha(manifest)
    assert diagnostic["training_authority"] is False

    fixture = candidate._fixture(tmp_path_factory.mktemp("candidate-empty-quality"))
    fixture["quality"] = manifest
    fixture["quality_assembly_receipt"] = receipt
    candidate._refresh(fixture)
    proposal = candidate._build_preaudit_proposal(fixture)
    assert proposal["bindings"]["quality_exclusions_sha256"] == _sha(manifest)
    assert proposal["bindings"]["quality_exclusion_assembly_receipt_sha256"] == _sha(receipt)
    assert proposal["counts"]["excluded"] == {"operational/before_2026_08_01_kst": 1}
    assert not (fixture["global_root"] / "authority" / "training_authority.sha256").exists()
