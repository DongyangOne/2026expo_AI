"""Real artifact handoff integration; detector is an explicit CPU test double."""
import csv
import hashlib
import importlib.util
from pathlib import Path

from scripts import prepare_proposal_verifier_dataset as prepare
from scripts import validate_v4_background_candidates as validate


def _load(filename):
    spec = importlib.util.spec_from_file_location("_integration_" + filename, Path(__file__).with_name(filename + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_canonical_manifest_passes_real_validator_without_lineage_rewrite(tmp_path):
    helpers = _load("test_prepare_operational_proposals")
    args = helpers.mixed.__wrapped__(tmp_path)
    args["model_path"] = tmp_path / "best.pt"
    args["model_path"].write_bytes(b"explicit CPU test model, not a production checkpoint")
    prepare.build_proposal_verifier_dataset(**args, prediction_provider=helpers._predictions)
    input_manifest = args["output_dir"] / "manifest.csv"
    initial_bytes = input_manifest.read_bytes()
    checked = input_manifest.with_name("manifest_validated.csv")
    report = validate.validate_manifest(
        input_manifest=input_manifest, dataset_info=args["output_dir"] / "dataset_info.json",
        detector_model=args["model_path"],
        inference_spec=Path(__file__).resolve().parents[1] / "configs" / "detector_inference_v3.json",
        output_manifest=checked, output_report=checked.with_suffix(".report.json"),
        operational_source_evidence_dir=args["operational_source_evidence_dir"],
        prediction_provider=helpers._predictions,
    )
    assert input_manifest.read_bytes() == initial_bytes
    rows = list(csv.DictReader(checked.open(encoding="utf-8", newline="")))
    assert len(rows) == 5
    assert {row["role"] for row in rows} == {"train", "model_validation"}
    assert len([row for row in rows if row["ground_truth_authority"] == "vlm_teacher_pseudo_label_train_only"]) == 3
    assert all(hashlib.sha256(Path(row["filepath"]).read_bytes()).hexdigest() == row["image_sha256"] for row in rows)
    assert report["contract"]["proposal_provenance"]["provider_kind"] == "custom_non_authoritative"
    assert not report["ready_for_lineage_upgrade"]
    assert not report["production_deployment_authorized"]
