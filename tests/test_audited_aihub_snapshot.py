"""Actual tiny original/materializer CPU fixtures, never training approval."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import audited_aihub_snapshot as reader
from scripts import materialize_audited_aihub_sources as materializer
from test_materialize_audited_aihub_sources import fixture as materializer_inputs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def snapshot(materializer_inputs):
    data, write_inputs, root, originals, annotations, metadata = materializer_inputs
    data["full_cohort"] = True
    data["pending_checks"] = ["protected_identity_not_training_approval"]
    args = write_inputs()
    materializer.materialize_sources(**args)
    report = root / "report.json"
    lineage = [json.loads(line) for line in (root / "lineage.jsonl").read_text().splitlines()]
    images = {"training": [], "validation": []}
    for row in lineage:
        images[row["split"]].append(root / row["image_ref"])
    return {"report_path": report, "report_sha256": sha(report), "cohort_path": args["cohort"]}, images, originals, annotations, metadata


def repin(args):
    args = dict(args)
    root = args["report_path"].parent
    report = load(args["report_path"])
    report["lineage_sha256"] = sha(root / "lineage.jsonl")
    report["excluded_sha256"] = sha(root / "excluded.jsonl")
    save(args["report_path"], report)
    ready = load(root / "snapshot_ready.json")
    ready["report_sha256"] = sha(args["report_path"])
    save(root / "snapshot_ready.json", ready)
    args["report_sha256"] = sha(args["report_path"])
    return args


def test_full_snapshot_links_original_and_resized_without_mixing_identities(snapshot):
    args, images, originals, annotations, _ = snapshot
    result = reader.load_audited_aihub_snapshot(**args)
    result.assert_source_membership(images)
    values = result.metadata_for(images["training"][0])
    assert set(values) == set(reader.METADATA_FIELDS)
    assert values["original_source_sha256"] == sha(originals[0])
    assert values["original_source_sha256"] != sha(images["training"][0])
    assert values["original_annotation_sha256"] == sha(annotations[0])
    assert values["materializer_report_sha256"] == args["report_sha256"]
    assert len(values["original_source_id"]) == 20
    assert not {"source_sha256", "source_id", "object_group", "capture_session", "training_authorized"} & values.keys()
    values["original_source_sha256"] = "changed return value"
    assert result.metadata_for(images["training"][0])["original_source_sha256"] == sha(originals[0])
    assert result.binding() == {
        "report_path": args["report_path"].as_posix(), "report_sha256": args["report_sha256"],
        "cohort_path": args["cohort_path"].as_posix(), "cohort_sha256": sha(args["cohort_path"]),
        "require_full_cohort": True,
    }
    result.recheck()


def test_split_for_resolves_only_verified_official_members(snapshot):
    args, images, originals, *_ = snapshot
    result = reader.load_audited_aihub_snapshot(**args)
    assert result.split_for(images["training"][0]) == "training"
    assert result.split_for(images["validation"][0]) == "validation"
    with pytest.raises(ValueError, match="unique audited official split"):
        result.split_for(originals[0])


def test_partial_snapshot_needs_explicit_diagnostic_opt_in(materializer_inputs):
    data, save_inputs, root, *_ = materializer_inputs
    args = save_inputs()
    materializer.materialize_sources(**args, max_sources=1)
    report = root / "report.json"
    inputs = {"report_path": report, "report_sha256": sha(report), "cohort_path": args["cohort"]}
    with pytest.raises(ValueError, match="diagnostic"):
        reader.load_audited_aihub_snapshot(**inputs)
    result = reader.load_audited_aihub_snapshot(**inputs, require_full_cohort=False)
    assert result.binding()["require_full_cohort"] is False
    assert load(report)["training_authorized"] is False


@pytest.mark.parametrize("fault", ["report", "cohort", "ready", "metadata", "original", "annotation", "derived", "sidecar"])
def test_input_hash_mismatches_are_rejected(snapshot, fault):
    args, images, originals, annotations, metadata = snapshot
    root = args["report_path"].parent
    paths = {"report": args["report_path"], "cohort": args["cohort_path"], "ready": root / "snapshot_ready.json",
             "metadata": metadata, "original": originals[0], "annotation": annotations[0],
             "derived": images["training"][0],
             "sidecar": root / load(args["report_path"]).get("unused", "lineage.jsonl")}
    if fault == "sidecar":
        row = json.loads((root / "lineage.jsonl").read_text().splitlines()[0])
        paths[fault] = root / row["label_ref"]
    path = paths[fault]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        reader.load_audited_aihub_snapshot(**args)


@pytest.mark.parametrize("fault", ["class", "bbox", "state", "split", "derived_rehash"])
def test_resealed_lineage_does_not_replace_actual_original_truth(snapshot, fault):
    args, images, *_ = snapshot
    root = args["report_path"].parent
    rows = [json.loads(line) for line in (root / "lineage.jsonl").read_text().splitlines()]
    row = rows[0]
    if fault == "class":
        row.update(class_id=3, class_name="plastic")
    elif fault == "bbox":
        row["bbox_xywh"][0] += 1
    elif fault == "state":
        row["conditions"]["label"] = 1
    elif fault == "split":
        row["split"] = "validation"
    else:
        image = root / row["image_ref"]
        image.write_bytes(image.read_bytes() + b"different but repinned JPEG")
        row["image_sha256"] = sha(image)
    (root / "lineage.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        reader.load_audited_aihub_snapshot(**repin(args))


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "wrong_split"])
def test_exact_source_membership_and_official_splits(snapshot, fault):
    args, images, *_ = snapshot
    result = reader.load_audited_aihub_snapshot(**args)
    values = {k: list(v) for k, v in images.items()}
    if fault == "missing":
        values["training"] = []
    elif fault == "extra":
        values["training"].append(args["cohort_path"])
    elif fault == "duplicate":
        values["training"].append(values["training"][0])
    else:
        values["training"], values["validation"] = values["validation"], values["training"]
    with pytest.raises(ValueError):
        result.assert_source_membership(values)


@pytest.mark.parametrize("fault", ["original", "extra_image", "failure_marker"])
def test_late_changes_fail_recheck(snapshot, fault):
    args, images, originals, *_ = snapshot
    result = reader.load_audited_aihub_snapshot(**args)
    if fault == "original":
        originals[0].write_bytes(originals[0].read_bytes() + b"late")
    elif fault == "extra_image":
        images["training"][0].with_name("extra.jpg").write_bytes(b"extra")
    else:
        (args["report_path"].parent / "failed.json").write_text("{}")
    with pytest.raises(ValueError):
        result.recheck()


def test_existing_failure_marker_or_input_symlink_never_accepted(snapshot):
    args, images, originals, *_ = snapshot
    alias = originals[0].with_name("alias.png")
    alias.write_bytes(originals[0].read_bytes())
    originals[0].unlink()
    try:
        originals[0].symlink_to(alias)
    except OSError:
        pytest.skip("OS symlink creation not available")
    with pytest.raises(ValueError, match="symlink"):
        reader.load_audited_aihub_snapshot(**args)


def test_regenerated_quality_exclusion_remains_audited_but_is_not_source(materializer_inputs):
    import cv2
    import numpy as np
    data, save_inputs, root, originals, *_ = materializer_inputs
    data["full_cohort"] = True
    ok, encoded = cv2.imencode(".png", np.full((480, 800, 3), 128, dtype=np.uint8))
    assert ok
    originals[0].write_bytes(encoded.tobytes())
    data["records"][0]["source_sha256"] = sha(originals[0])
    args = save_inputs()
    materializer.materialize_sources(**args)
    result = reader.load_audited_aihub_snapshot(root / "report.json", sha(root / "report.json"), cohort_path=args["cohort"])
    with pytest.raises(ValueError, match="not in"):
        result.metadata_for(originals[0])
    result.recheck()


def test_resealed_cohort_and_lineage_cannot_hide_unknown_original_json_class(snapshot):
    args, _, _, annotations, _ = snapshot
    root = args["report_path"].parent
    payload = load(annotations[0])
    payload["ANNOTATION_INFO"][0].update(CLASS="unknown", DETAILS="unknown")
    save(annotations[0], payload)
    cohort = load(args["cohort_path"])
    cohort["records"][0]["label_sha256"] = sha(annotations[0])
    save(args["cohort_path"], cohort)
    rows = [json.loads(line) for line in (root / "lineage.jsonl").read_text().splitlines()]
    rows[0]["label_sha256"] = sha(annotations[0])
    (root / "lineage.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = load(args["report_path"])
    report["cohort_sha256"] = sha(args["cohort_path"])
    save(args["report_path"], report)
    with pytest.raises(ValueError, match="material"):
        reader.load_audited_aihub_snapshot(**repin(args))


@pytest.mark.parametrize("fault", ["yaml", "boolean_count", "claimed_authority"])
def test_snapshot_schema_and_yaml_cannot_override_evidence(snapshot, fault):
    args, *_ = snapshot
    root = args["report_path"].parent
    if fault == "yaml":
        (root / "dataset.yaml").write_text("path: /different/source\n")
    else:
        report = load(args["report_path"])
        if fault == "boolean_count":
            report["unprocessed_sources"] = False
        else:
            report["training_authorized"] = True
        save(args["report_path"], report)
        args = repin(args)
    with pytest.raises(ValueError):
        reader.load_audited_aihub_snapshot(**args)
