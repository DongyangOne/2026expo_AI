"""Prevent legacy accessory-only evidence from being reused as foreign labels."""

import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import pseudo_label_status_qwen as teacher


ERROR = "same_material_accessory_only conflicts"


def decision(*, foreign=True, accessory=True, label=False, reason="different_material"):
    return {
        "decision": {(False, False): "neither", (True, False): "label_only",
                     (False, True): "foreign_only", (True, True): "both"}[(label, foreign)],
        "has_removable_label": label, "has_true_foreign_material": foreign,
        "same_material_accessory_only": accessory, "is_single_primary_item": True,
        "confidence": 0.99, "reason": reason,
    }


def record(result=None):
    result = decision() if result is None else result
    return {
        "source_id": "synthetic", "filepath": "synthetic.jpg", "split": "training",
        "category": "plastic", "model": "fixture-model",
        "teacher": copy.deepcopy(result), "teacher_passes": [copy.deepcopy(result), copy.deepcopy(result)],
        "raw_outputs": [json.dumps(result), json.dumps(result)],
        "accepted": {"label": int(result["has_removable_label"]),
                     "foreign_material": int(result["has_true_foreign_material"]), "status_eligible": 1},
    }


@pytest.mark.parametrize("label", [False, True])
@pytest.mark.parametrize("format", ["full", "compact4", "compact7"])
def test_parser_rejects_accessory_only_foreign_positive(label, format):
    value = decision(label=label)
    if format == "compact4":
        value = {"d": value["decision"], "s": True, "c": 0.99, "e": "same_material_accessory"}
    elif format == "compact7":
        value = {"d": value["decision"], "l": label, "f": True, "a": True,
                 "s": True, "c": 0.99, "e": "different_material"}
    with pytest.raises(ValueError, match=ERROR):
        teacher.parse_teacher_output(json.dumps(value))


def test_compact7_accessory_evidence_cannot_hide_behind_false_auxiliary_flag():
    value = {"d": "neither", "l": False, "f": True, "a": False,
             "s": True, "c": 0.99, "e": "same_material_accessory"}
    with pytest.raises(ValueError, match=ERROR):
        teacher.parse_teacher_output(json.dumps(value))


@pytest.mark.parametrize("reason", ["same_material_accessory", "different_material | same_material_accessory"])
def test_normalized_evidence_code_also_blocks_conflicting_full_response(reason):
    with pytest.raises(ValueError, match=ERROR):
        teacher.parse_teacher_output(json.dumps(decision(accessory=False, reason=reason)))


@pytest.mark.parametrize("label", [False, True])
def test_valid_same_material_accessory_neither_or_label_only_still_allowed(label):
    value = {"d": "label_only" if label else "neither", "s": True,
             "c": 0.99, "e": "same_material_accessory"}
    parsed = teacher.parse_teacher_output(json.dumps(value))
    combined = teacher.consensus_teacher([parsed, {**parsed, "same_material_accessory_only": False}])
    assert teacher.accepted_status(combined, "plastic", 0.90) == {
        "label": int(label), "foreign_material": 0, "status_eligible": 1,
    }


@pytest.mark.parametrize("label", [False, True])
def test_valid_true_foreign_and_label_combination_preserved(label):
    value = teacher.parse_teacher_output(json.dumps(decision(accessory=False, label=label)))
    assert teacher.accepted_status(teacher.consensus_teacher([value, value]), "plastic", 0.90) == {
        "label": int(label), "foreign_material": 1, "status_eligible": 1,
    }


@pytest.mark.parametrize("values", [
    [decision()], [decision(), decision()], [decision(accessory=False), decision()],
    [decision(), decision(foreign=False)],
])
def test_direct_consensus_cannot_skip_guard_for_one_or_disagreeing_pass(values):
    before = copy.deepcopy(values)
    with pytest.raises(ValueError, match=ERROR):
        teacher.consensus_teacher(values)
    assert values == before


@pytest.mark.parametrize("confidence", [0.20, 0.99])
def test_direct_accepted_status_rejects_without_relabeling_as_zero(confidence):
    value = {**decision(), "confidence": confidence}
    before = copy.deepcopy(value)
    with pytest.raises(ValueError, match=ERROR):
        teacher.accepted_status(value, "plastic", 0.90)
    assert value == before


@pytest.mark.parametrize("where", ["teacher", "pass", "raw", "accepted_vs_negative_teacher"])
def test_legacy_cache_winner_rejected_without_changing_cache(tmp_path, where):
    value = record(decision(accessory=False))
    if where == "teacher":
        value["teacher"]["same_material_accessory_only"] = True
    elif where == "pass":
        value["teacher_passes"][1]["same_material_accessory_only"] = True
    elif where == "raw":
        value["raw_outputs"][1] = json.dumps({"d": "foreign_only", "s": True,
                                              "c": 0.99, "e": "same_material_accessory"})
    else:
        value["teacher"] = decision(foreign=False)
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(value) + "\n", encoding="utf-8")
    before = cache.read_bytes()
    with pytest.raises(ValueError, match=ERROR):
        teacher._load_results(cache)
    assert cache.read_bytes() == before


def test_raw_negative_accessory_evidence_cannot_support_cached_positive(tmp_path):
    value = record(decision(accessory=False))
    value["raw_outputs"] = [json.dumps({"d": "neither", "s": True, "c": 0.99, "e": "same_material_accessory"})]
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=ERROR):
        teacher._load_results(cache)


@pytest.mark.parametrize("label", [False, True])
def test_cached_negative_cannot_relabel_contradictory_raw_as_zero(tmp_path, label):
    value = record(decision(foreign=False, label=label))
    value["raw_outputs"][1] = json.dumps(decision(foreign=True, label=label))
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(value) + "\n", encoding="utf-8")
    before = cache.read_bytes()
    with pytest.raises(ValueError, match=ERROR):
        teacher._load_results(cache)
    assert cache.read_bytes() == before


@pytest.mark.parametrize("label", [False, True])
def test_valid_accessory_clean_and_label_only_cache_still_load_and_merge(tmp_path, label):
    value = record(decision(foreign=False, label=label))
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(value) + "\n", encoding="utf-8")
    before = cache.read_bytes()
    values = teacher._load_results(cache)
    output = tmp_path / "merged.csv"
    base = {"source_id": "synthetic", "filepath": "synthetic.jpg", "category": "plastic",
            "label": "-1", "foreign_material": "-1"}
    teacher.merge_manifest([base], values, output)
    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["label"] == str(int(label)) and row["foreign_material"] == "0"
    assert row["status_eligible"] == "1" and row["teacher_rejected"] == "0"
    assert cache.read_bytes() == before


@pytest.mark.parametrize("field,value", [
    ("status_eligible", "1"), ("status_eligible", True), ("status_eligible", 1.0),
    ("foreign_material", "1"), ("foreign_material", True), ("foreign_material", 1.0),
    ("label", "0"), ("label", False), ("label", 0.0),
    ("foreign_material", -1), ("status_eligible", 0),
])
def test_malformed_cached_accepted_labels_cannot_bypass_guard(tmp_path, field, value):
    cached = record(decision(accessory=False))
    cached["accepted"][field] = value
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(cached) + "\n", encoding="utf-8")
    before = cache.read_bytes()
    with pytest.raises(ValueError, match="exact integer labels"):
        teacher._load_results(cache)
    assert cache.read_bytes() == before


def test_string_accepted_flags_cannot_hide_raw_conflict_or_truncate_merge(tmp_path):
    cached = record(decision(accessory=False))
    cached["accepted"].update(status_eligible="1", foreign_material="1")
    cached["raw_outputs"] = [json.dumps(decision())]
    output = tmp_path / "merged.csv"
    output.write_bytes(b"unchanged prior output")
    base = {"source_id": "synthetic", "filepath": "synthetic.jpg", "category": "plastic"}
    with pytest.raises(ValueError, match="exact integer labels"):
        teacher.merge_manifest([base], {("synthetic", "synthetic.jpg"): cached}, output)
    assert output.read_bytes() == b"unchanged prior output"


def test_later_valid_explicit_retry_preserves_history_and_last_record_semantics(tmp_path):
    invalid = record()
    valid = record(decision(foreign=False))
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(invalid) + "\n" + json.dumps(valid) + "\n", encoding="utf-8")
    before = cache.read_bytes()
    assert teacher._load_results(cache)[("synthetic", "synthetic.jpg")] == valid
    assert cache.read_bytes() == before
    cache.write_text(json.dumps(valid) + "\n" + json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=ERROR):
        teacher._load_results(cache)


def test_direct_merge_rejects_before_opening_output_or_mutating_results(tmp_path):
    base = {"source_id": "synthetic", "filepath": "synthetic.jpg", "category": "plastic"}
    values = {("synthetic", "synthetic.jpg"): record()}
    before = copy.deepcopy(values)
    output = tmp_path / "already_exists.csv"
    output.write_bytes(b"prior output must remain unchanged")
    with pytest.raises(ValueError, match=ERROR):
        teacher.merge_manifest([base], values, output)
    assert output.read_bytes() == b"prior output must remain unchanged"
    assert values == before


def test_main_cached_reuse_stops_before_any_teacher_request_or_output_change(tmp_path, monkeypatch):
    manifest = tmp_path / "base.csv"
    base = {"source_id": "synthetic", "filepath": "synthetic.jpg", "split": "training",
            "category": "plastic", "source_path_b64": "not-read",
            "source_bbox_x": "0", "source_bbox_y": "0", "source_bbox_w": "32", "source_bbox_h": "32"}
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(base))
        writer.writeheader()
        writer.writerow(base)
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    output = tmp_path / "merged.csv"
    output.write_bytes(b"previous verified output")
    before = cache.read_bytes()
    monkeypatch.setattr(sys, "argv", ["teacher", "--manifest", str(manifest),
                                    "--output-jsonl", str(cache), "--merged-manifest", str(output)])
    monkeypatch.setattr(teacher, "_processed_records", lambda *args: pytest.fail("must not run teacher"))
    with pytest.raises(ValueError, match=ERROR):
        teacher.main()
    assert cache.read_bytes() == before
    assert output.read_bytes() == b"previous verified output"


def test_rejected_error_raw_evidence_stays_masked_and_valid_cache_still_merges(tmp_path):
    rejected = record()
    rejected.update(teacher=None, teacher_passes=[], error="contradictory raw response",
                    accepted={"label": -1, "foreign_material": -1, "status_eligible": 0})
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(rejected) + "\n", encoding="utf-8")
    values = teacher._load_results(cache)
    output = tmp_path / "merged.csv"
    base = {"source_id": "synthetic", "filepath": "synthetic.jpg", "category": "plastic",
            "label": "-1", "foreign_material": "-1"}
    teacher.merge_manifest([base], values, output)
    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["label"] == row["foreign_material"] == "-1"
    assert row["status_eligible"] == "0" and row["teacher_rejected"] == "1"


def test_process_candidate_turns_guard_failure_into_existing_unknown_error_record(monkeypatch):
    monkeypatch.setattr(teacher, "decode_source_path", lambda value: Path("synthetic.jpg"))
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(teacher, "_imread_unicode", lambda path: image)
    monkeypatch.setattr(teacher, "_crop", lambda image, row, padding: image)
    raw = '{"d":"foreign_only","s":true,"c":0.99,"e":"same_material_accessory"}'
    monkeypatch.setattr(teacher, "_infer_ollama", lambda *args: raw)
    args = SimpleNamespace(model="fixture", adaptive_consensus=False, consensus_passes=2,
                           backend="ollama", ollama_url="http://unused", image_max_side=640,
                           request_timeout=1, num_ctx=8192, request_retries=0, min_confidence=0.90)
    row = {"source_id": "synthetic", "filepath": "synthetic.jpg", "split": "training",
           "category": "plastic", "source_path_b64": "unused"}
    value = teacher._process_candidate(row, args)
    assert value["teacher"] is None and ERROR in value["error"]
    assert value["accepted"] == {"label": -1, "foreign_material": -1, "status_eligible": 0}
    assert value["raw_outputs"] == [raw]
