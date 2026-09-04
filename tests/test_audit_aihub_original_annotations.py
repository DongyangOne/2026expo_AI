"""Small original-image/annotation checks; no detector or training authority."""

import base64
import csv
import hashlib
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

from scripts import audit_aihub_original_annotations as audit
from scripts import audit_verifier_dataset as dataset_audit


@pytest.fixture
def pair(tmp_path):
    root = tmp_path / "ai_dataset" / "학습용_데이터" / "01-1.정식개방데이터"
    label_dir = "TL_2.직접촬영_01.금속캔_001.철캔"
    source = root / "Training" / "01.원천데이터" / "TS_2.직접촬영_01.금속캔_001.철캔_1" / "a.jpg"
    label = root / "Training" / "02.라벨링데이터" / label_dir / "a.json"
    source.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48), (16, 80, 144)).save(source, format="JPEG")
    payload = {
        "IMAGE_INFO": {"FILE_NAME": "a.jpg", "IMAGE_WIDTH": 64, "IMAGE_HEIGHT": 48},
        "ANNOTATION_INFO": [{
            "CLASS": "금속캔", "DETAILS": "철캔", "DAMAGE": "원형",
            "DIRTINESS": "이물질(외부)", "POINTS": [[8, 6, 24, 18]],
        }],
    }
    label.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    row = {
        "source_id": hashlib.sha1(f"Training/{label_dir}/a.json".encode()).hexdigest()[:20],
        "category": "can", "material": "0", "source_object_count": "1",
        "source_width": "64", "source_height": "48", "split": "training",
        "source_bbox_x": "8", "source_bbox_y": "6", "source_bbox_w": "24", "source_bbox_h": "18",
    }
    return row, payload, (48, 64), source, label


@pytest.mark.parametrize("name,class_id", [
    ("철캔", 0), ("페트병", 1), ("종이", 2), ("플라스틱", 3),
    ("스티로폼", 4), ("비닐", 5), ("유리병", 6), ("건전지", 7), ("형광등", 8),
])
def test_resolve_nine_materials_without_detector_predictions(name, class_id):
    assert audit.resolve_material(name) == class_id


@pytest.mark.parametrize("values", [("정체불명",), ("",), ("비닐", "플라스틱")])
def test_unknown_and_conflicting_materials_are_not_defaulted_to_plastic(values):
    with pytest.raises(audit.AnnotationError):
        audit.resolve_material(*values)


def test_bbox_retains_xywh_and_does_not_normalize_or_expand():
    assert audit.strict_bbox([[8, 6, 24, 18]], 64, 48) == [8.0, 6.0, 24.0, 18.0]


def test_polygon_returns_tight_xywh_of_actual_vertices():
    assert audit.strict_bbox([[8, 6], [32, 6], [8, 24]], 64, 48) == [8.0, 6.0, 24.0, 18.0]


def test_zero_area_polygon_is_not_accepted_as_its_nonzero_bounding_rectangle():
    with pytest.raises(audit.AnnotationError):
        audit.strict_bbox([[1, 1], [2, 2], [3, 3]], 64, 48)


@pytest.mark.parametrize("points", [
    [], [[True, 6, 24, 18]], [[float("nan"), 6, 24, 18]],
    [[8, 6, float("inf"), 18]], [[-1, 6, 24, 18]], [[8, 6, 0, 18]],
    [[8, 6, 57, 18]], [[8, 6, 24, 18], [9, 7, 20, 15]],
])
def test_bbox_rejects_invalid_or_multiple_rectangles_without_clamping(points):
    with pytest.raises(audit.AnnotationError):
        audit.strict_bbox(points, 64, 48)


@pytest.mark.parametrize("damage,expected", [("원형", 0), ("찌그러짐", 1), ("완전압착", 1)])
def test_original_damage_is_evidence_only_and_all_training_conditions_stay_unknown(pair, damage, expected):
    row, payload, image_shape, source, label = pair
    payload["ANNOTATION_INFO"][0]["DAMAGE"] = damage
    result = audit.validate_pair(row, payload, image_shape, source, label)
    assert result["class_id"] == 0
    assert result["class_name"] == "can"
    assert result["bbox_xywh"] == [8.0, 6.0, 24.0, 18.0]
    assert result["annotation_dent"] == expected
    # In particular, DIRTINESS must not silently become a label/foreign target.
    assert result["conditions"] == {"dent": -1, "label": -1, "foreign_material": -1}


@pytest.mark.parametrize("fault", [
    "multiple_objects", "empty_objects", "unknown_class", "row_class_mismatch",
    "header_dimensions", "filename_mismatch", "source_id_mismatch", "bbox_mismatch",
])
def test_validate_pair_rejects_annotation_and_manifest_disagreement(pair, fault):
    row, payload, image_shape, source, label = pair
    if fault == "multiple_objects":
        payload["ANNOTATION_INFO"].append(dict(payload["ANNOTATION_INFO"][0]))
    elif fault == "empty_objects":
        payload["ANNOTATION_INFO"] = []
    elif fault == "unknown_class":
        payload["ANNOTATION_INFO"][0].update(CLASS="미상", DETAILS="미상")
    elif fault == "row_class_mismatch":
        row.update(category="plastic", material="3")
    elif fault == "header_dimensions":
        payload["IMAGE_INFO"]["IMAGE_WIDTH"] = 65
    elif fault == "filename_mismatch":
        payload["IMAGE_INFO"]["FILE_NAME"] = "another.jpg"
    elif fault == "source_id_mismatch":
        row["source_id"] = "0" * 20
    elif fault == "bbox_mismatch":
        row["source_bbox_w"] = "25"
    with pytest.raises(audit.AnnotationError):
        audit.validate_pair(row, payload, image_shape, source, label)


def test_source_folder_material_is_independently_checked(pair):
    row, payload, image_shape, source, label = pair
    # A self-consistent JSON+manifest plastic claim still contradicts the can folder.
    row.update(category="plastic", material="3")
    payload["ANNOTATION_INFO"][0].update(CLASS="플라스틱", DETAILS="플라스틱")
    with pytest.raises(audit.AnnotationError):
        audit.validate_pair(row, payload, image_shape, source, label)


def test_annotation_location_maps_sharded_source_to_direct_label_directory(pair):
    row, _, _, source, label = pair
    assert audit.annotation_location(source, row["split"]) == label


def test_annotation_location_preserves_official_validation_partition(pair):
    _, _, _, source, label = pair
    source = type(source)(str(source).replace("Training", "Validation").replace("TS_", "VS_"))
    label = type(label)(str(label).replace("Training", "Validation").replace("TL_", "VL_"))
    assert audit.annotation_location(source, "validation") == label


def test_annotation_location_rejects_split_disagreement(pair):
    _, _, _, source, _ = pair
    with pytest.raises(audit.AnnotationError):
        audit.annotation_location(source, "validation")


def test_read_image_checks_real_pixels_and_rejects_corrupt_source(pair):
    _, _, _, source, _ = pair
    content = source.read_bytes()
    assert audit.read_image(source) == ((48, 64), hashlib.sha256(content).hexdigest(), len(content))
    corrupt = source.with_name("corrupt.jpg")
    corrupt.write_bytes(b"not an image, despite the JPEG suffix")
    with pytest.raises(audit.AnnotationError):
        audit.read_image(corrupt)


@pytest.mark.parametrize("raw_json", [
    '{"IMAGE_INFO": {}, "IMAGE_INFO": {}}',
    '{"IMAGE_INFO": {"IMAGE_WIDTH": 64, "IMAGE_WIDTH": 65}}',
])
def test_unique_object_rejects_duplicate_keys_even_when_nested_or_equal(raw_json):
    with pytest.raises(audit.AnnotationError, match="duplicate JSON key"):
        json.loads(raw_json, object_pairs_hook=audit.unique_object)


def test_image_evidence_matches_existing_direct_grayscale_dct_phash(tmp_path):
    # A nonuniform color JPEG exercises grayscale decode and all 64 DCT bits.
    pixels = np.random.default_rng(42).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    source = tmp_path / "color_pattern.jpg"
    Image.fromarray(pixels).save(source, format="JPEG", quality=93)
    content = source.read_bytes()
    expected = f"{dataset_audit._perceptual_hash(source):016x}"
    assert audit.image_evidence(source) == (
        (48, 64), hashlib.sha256(content).hexdigest(), len(content), expected,
    )
    assert audit.image_evidence(source, perceptual=False) == (
        (48, 64), hashlib.sha256(content).hexdigest(), len(content), None,
    )


def test_cli_reads_actual_source_path_b64_header_and_publishes_snapshot_only(pair, tmp_path, monkeypatch):
    row, _, _, source, label = pair
    row["source_path_b64"] = base64.urlsafe_b64encode(os.fsencode(source)).decode("ascii")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = tmp_path / "audit_output"

    # Only the 18-group sampler is doubled; original image/JSON reads, binding,
    # validation, command-line parsing, hash checks and report publication are real.
    def single_fixture_selection(path, per_class_split):
        assert path == manifest and per_class_split == 1
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert "source_path" not in rows[0]
        return rows, {"training/can": 1}

    monkeypatch.setattr(audit, "select_rows", single_fixture_selection)
    monkeypatch.setattr(sys, "argv", [
        "audit_aihub_original_annotations", "--manifest", str(manifest),
        "--manifest-sha256", manifest_sha, "--dataset-root", str(source.parents[3]),
        "--output", str(output), "--per-class-split", "1", "--workers", "2",
    ])
    audit.main()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert (report["selected"], report["verified"], report["quarantined"]) == (1, 1, 0)
    assert report["manifest_sha256"] == manifest_sha
    assert report["workers"] == 2
    assert report["snapshot_only"] is True
    assert report["consumer_must_rehash_source_and_annotation"] is True
    assert report["training_authorized"] is False
    assert report["deployment_authorized"] is False
    record = report["records"][0]
    assert record["status"] == "verified_pair"
    assert record["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["label_sha256"] == hashlib.sha256(label.read_bytes()).hexdigest()
    assert record["source_path_b64"] == row["source_path_b64"]
    assert record["source_phash64"] == f"{dataset_audit._perceptual_hash(source):016x}"
    assert record["conditions"] == {"dent": -1, "label": -1, "foreign_material": -1}


def test_cli_wrong_manifest_sha_creates_no_output(pair, tmp_path, monkeypatch):
    _, _, _, source, _ = pair
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("source_path_b64\n", encoding="utf-8")
    output = tmp_path / "audit_output"

    def should_not_select(*args):
        pytest.fail("manifest selection ran before the SHA mismatch was rejected")

    monkeypatch.setattr(audit, "select_rows", should_not_select)
    monkeypatch.setattr(sys, "argv", [
        "audit_aihub_original_annotations", "--manifest", str(manifest),
        "--manifest-sha256", "0" * 64, "--dataset-root", str(source.parents[3]),
        "--output", str(output),
    ])
    with pytest.raises(audit.AnnotationError, match="manifest SHA256 mismatch"):
        audit.main()
    assert not output.exists()
