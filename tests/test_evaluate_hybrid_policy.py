import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_hardware_detector import build_detection_detail
from scripts.evaluate_hybrid_policy import (
    _softmax_summary,
    build_report,
    combine_predictions,
    deployment_metric_gate,
    evaluate_policy,
    external_material_name,
    load_baseline_details,
    load_manifest,
    load_verifier_predictions,
    manifest_rows_from_baseline,
    sweep_policies,
)


def _combined_row(
    name: str,
    *,
    truth: str,
    baseline: str | None,
    yolo_confidence: float | None,
    verifier: str | None,
    verifier_confidence: float | None,
    localized: bool = True,
    is_negative: bool = False,
) -> dict:
    bbox = [10.0, 10.0, 90.0, 90.0] if baseline is not None else None
    return {
        "image_key": name,
        "filepath": f"validation/{truth}/{name}",
        "source_id": name,
        "truth_external": truth,
        "is_negative": is_negative,
        "baseline_external": baseline,
        "baseline_confidence": yolo_confidence,
        "baseline_outcome": (
            "negative_false_positive"
            if is_negative and baseline is not None
            else "negative_clean"
            if is_negative
            else "positive_wrong_class"
        ),
        "localization_ok": localized,
        "selected_bbox": bbox,
        "selected_iou": 0.8 if localized and bbox else None,
        "selected_candidate": None,
        "detector_candidates": [],
        "detector_bbox_audit_complete": True,
        "verifier_internal": verifier,
        "verifier_external": None if verifier == "background" else verifier,
        "verifier_confidence": verifier_confidence,
        "verifier_runner_up": "plastic",
        "verifier_probability_gap": 0.5,
        "verifier_selected_bbox": bbox,
        "verifier_crop_source": "selected_yolo_bbox",
        "verifier_bbox_matches_selected": True,
    }


def _baseline_detail(
    name: str,
    *,
    expected: str,
    predicted: str | None,
    confidence: float | None,
) -> dict:
    bbox = [10.0, 10.0, 90.0, 90.0] if predicted is not None else None
    class_ids = {
        "can": 0,
        "pet": 1,
        "paper": 2,
        "plastic": 3,
        "styrofoam": 4,
        "vinyl": 5,
        "glass": 6,
        "battery": 7,
        "fluorescent": 8,
    }
    candidates = (
        [{
            "bbox": bbox,
            "class_id": class_ids[predicted],
            "class_name": predicted,
            "confidence": confidence,
        }]
        if predicted is not None else []
    )
    return {
        "image": name,
        "image_path": str(Path("raw") / name),
        "expected": expected,
        "predicted": predicted,
        "confidence": confidence,
        "bbox": bbox,
        "selected_bbox": bbox,
        "selected_candidate": candidates[0] if candidates else None,
        "candidates": candidates,
        "iou": 0.9 if expected != "negative" and predicted is not None else None,
        "outcome": (
            "negative_clean"
            if expected == "negative" and predicted is None
            else "negative_false_positive"
            if expected == "negative"
            else "positive_correct"
            if expected == predicted
            else "positive_wrong_class"
        ),
    }


def _verifier_prediction(
    class_name: str,
    confidence: float,
    bbox: list[float] | None,
) -> dict:
    class_id = {
        "can": 0,
        "pet": 1,
        "paper": 2,
        "plastic": 3,
        "styrofoam": 4,
        "vinyl": 5,
        "glass": 6,
        "battery": 7,
        "fluorescent": 8,
        "background": 9,
    }[class_name]
    return {
        "internal_id": class_id,
        "internal_name": class_name,
        "external_name": None if class_name == "background" else external_material_name(class_name),
        "confidence": confidence,
        "runner_up_internal_name": "plastic",
        "runner_up_confidence": 0.05,
        "probability_gap": confidence - 0.05,
        "selected_bbox": bbox,
        "crop_source": "selected_yolo_bbox",
    }


def test_detector_detail_records_selected_bbox_and_candidate_audit(tmp_path):
    truth = {
        "image": tmp_path / "capture.jpg",
        "class_id": 2,
        "class_name": "paper",
        "bbox": [10.0, 10.0, 80.0, 80.0],
    }
    candidates = [
        {
            "bbox": [9.0, 9.0, 81.0, 81.0],
            "class_id": 2,
            "class_name": "paper",
            "confidence": 0.81,
        },
        {
            "bbox": [1.0, 1.0, 20.0, 20.0],
            "class_id": 4,
            "class_name": "styrofoam",
            "confidence": 0.31,
        },
        {
            "bbox": [0.0, 0.0, 5.0, 5.0],
            "class_id": 5,
            "class_name": "vinyl",
            "confidence": 0.2,
        },
    ]

    detail = build_detection_detail(truth, candidates, threshold=0.25, min_iou=0.3)

    assert detail["predicted"] == "paper"
    assert detail["bbox"] == detail["selected_bbox"] == [9.0, 9.0, 81.0, 81.0]
    assert detail["selected_candidate"] == detail["candidates"][0]
    assert [item["class_name"] for item in detail["candidates"]] == ["paper", "styrofoam"]
    assert detail["selection_rule"] == "highest_confidence_at_or_above_threshold"
    assert detail["outcome"] == "positive_correct"


def test_pet_is_external_plastic_and_softmax_supports_optional_background():
    assert external_material_name("pet") == "plastic"
    assert external_material_name(1) == "plastic"
    assert external_material_name("plastic") == "plastic"

    nine_class = _softmax_summary(
        np.asarray([[0.0, 5.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    )
    ten_class = _softmax_summary(
        np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]])
    )

    assert nine_class["internal_name"] == "pet"
    assert nine_class["external_name"] == "plastic"
    assert ten_class["internal_name"] == "background"
    assert ten_class["external_name"] is None


def test_policy_margin_matches_runtime_verifier_minus_yolo_confidence():
    rows = [
        _combined_row(
            "vinyl.jpg",
            truth="vinyl",
            baseline="plastic",
            yolo_confidence=0.55,
            verifier="vinyl",
            verifier_confidence=0.79,
        )
    ]

    rejected = evaluate_policy(rows, 0.70, 0.25)
    accepted = evaluate_policy(rows, 0.70, 0.24)

    assert rejected["corrections"]["applied"] == 0
    assert rejected["audit"][0]["reason"] == "below_verifier_over_yolo_margin"
    assert rejected["audit"][0]["verifier_over_yolo_margin"] == pytest.approx(0.24)
    assert accepted["corrections"]["beneficial"] == 1
    assert accepted["corrections"]["harmful"] == 0


def test_background_verifier_retains_low_confidence_yolo_result():
    rows = [
        _combined_row(
            "background.jpg",
            truth="negative",
            baseline="plastic",
            yolo_confidence=0.40,
            verifier="background",
            verifier_confidence=0.99,
            localized=False,
            is_negative=True,
        )
    ]

    report = evaluate_policy(rows, 0.8, 0.3)
    audit = report["audit"][0]

    assert report["corrections"]["applied"] == 0
    assert audit["reason"] == "verifier_background_retain_yolo"
    assert audit["final"] == "plastic"
    assert audit["final_confidence"] == 0.4
    assert audit["final_allowed_like"] is False


@pytest.mark.parametrize(
    ("baseline", "confidence"),
    [("styrofoam", 0.8), ("plastic", 0.4)],
)
def test_negative_to_allowed_like_promotion_is_counted_as_harmful(baseline, confidence):
    rows = [
        _combined_row(
            "negative.jpg",
            truth="negative",
            baseline=baseline,
            yolo_confidence=confidence,
            verifier="vinyl",
            verifier_confidence=0.95,
            localized=False,
            is_negative=True,
        )
    ]

    report = evaluate_policy(rows, 0.8, 0.1)

    assert report["corrections"]["harmful"] == 1
    assert report["corrections"]["wrong_to_wrong"] == 0
    assert report["corrections"]["negative_to_allowed_like_promotions"] == 1
    assert report["audit"][0]["effect"] == "harmful_negative_promotion"


def test_sweep_requires_positive_negative_and_all_metric_gates():
    rows = [
        _combined_row(
            "benefit.jpg",
            truth="vinyl",
            baseline="plastic",
            yolo_confidence=0.40,
            verifier="vinyl",
            verifier_confidence=0.90,
        ),
        _combined_row(
            "paper.jpg",
            truth="paper",
            baseline="paper",
            yolo_confidence=0.90,
            verifier="paper",
            verifier_confidence=0.95,
        ),
        _combined_row(
            "negative.jpg",
            truth="negative",
            baseline=None,
            yolo_confidence=None,
            verifier=None,
            verifier_confidence=None,
            localized=False,
            is_negative=True,
        ),
    ]

    report = sweep_policies(rows, [0.8], [0.3])

    assert report["selection"]["enabled"] is True
    selected = report["sweep"][0]
    assert selected["metric_gate"]["passed"] is True
    assert selected["accuracy_gain"] == pytest.approx(1 / 3)
    assert selected["metric_gate"]["checks"] == {
        "has_positive_samples": True,
        "has_negative_samples": True,
        "external_accuracy_gain_at_least_5pp": True,
        "macro_f1_nondecrease": True,
        "per_class_recall_drop_within_1pp": True,
        "negative_specificity_nondecrease": True,
        "zero_harmful_corrections": True,
        "zero_negative_to_allowed_like_promotions": True,
    }

    positives_only = sweep_policies(rows[:2], [0.8], [0.3])
    assert positives_only["selection"]["enabled"] is False
    assert positives_only["sweep"][0]["metric_gate"]["checks"]["has_negative_samples"] is False


def test_manifest_and_baseline_join_include_negative_and_normalize_pet(tmp_path):
    manifest = tmp_path / "hardware_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filepath", "split", "source_id", "material", "category"],
        )
        writer.writeheader()
        writer.writerow({
            "filepath": "raw/pet.jpg",
            "split": "validation",
            "source_id": "sha-p",
            "material": "1",
            "category": "pet",
        })
        writer.writerow({
            "filepath": "raw/negative.jpg",
            "split": "validation",
            "source_id": "sha-n",
            "material": "negative",
            "category": "negative",
        })
    baseline_path = tmp_path / "baseline.json"
    baseline_payload = {
        "thresholds": {
            "0.250": {
                "details": [
                    _baseline_detail(
                        "pet.jpg", expected="pet", predicted="pet", confidence=0.8
                    ),
                    _baseline_detail(
                        "negative.jpg", expected="negative", predicted=None, confidence=None
                    ),
                ]
            }
        }
    }
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    manifest_rows = load_manifest(manifest)
    baseline_rows = load_baseline_details(baseline_path, "0.25")
    verifier = {
        "pet.jpg": _verifier_prediction("plastic", 0.9, [10.0, 10.0, 90.0, 90.0])
    }
    combined = combine_predictions(manifest_rows, baseline_rows, verifier)

    assert [row["truth_external"] for row in combined] == ["plastic", "negative"]
    assert combined[0]["baseline_external"] == "plastic"
    assert combined[0]["verifier_external"] == "plastic"
    assert combined[1]["is_negative"] is True
    assert combined[1]["verifier_internal"] is None


def test_manifest_omitting_detector_negatives_is_rejected():
    baseline = {
        "positive.jpg": _baseline_detail(
            "positive.jpg", expected="paper", predicted="paper", confidence=0.9
        ),
        "negative.jpg": _baseline_detail(
            "negative.jpg", expected="negative", predicted=None, confidence=None
        ),
    }
    manifest = manifest_rows_from_baseline({"positive.jpg": baseline["positive.jpg"]})

    with pytest.raises(ValueError, match="positive/negative"):
        combine_predictions(manifest, baseline, {})


def test_selected_bbox_report_can_pass_but_ground_truth_crop_cannot():
    baseline = {
        "vinyl.jpg": _baseline_detail(
            "vinyl.jpg", expected="vinyl", predicted="plastic", confidence=0.4
        ),
        "paper.jpg": _baseline_detail(
            "paper.jpg", expected="paper", predicted="paper", confidence=0.9
        ),
        "negative.jpg": _baseline_detail(
            "negative.jpg", expected="negative", predicted=None, confidence=None
        ),
    }
    manifest = manifest_rows_from_baseline(baseline)
    verifier = {
        "vinyl.jpg": _verifier_prediction("vinyl", 0.9, [10.0, 10.0, 90.0, 90.0]),
        "paper.jpg": _verifier_prediction("paper", 0.95, [10.0, 10.0, 90.0, 90.0]),
    }

    selected_bbox_report = build_report(
        manifest,
        baseline,
        verifier,
        [0.8],
        [0.3],
        verifier_prediction_source="selected_yolo_bbox_predictions",
    )
    gt_crop_report = build_report(
        manifest,
        baseline,
        verifier,
        [0.8],
        [0.3],
        verifier_prediction_source="ground_truth_manifest_crop",
    )

    assert selected_bbox_report["selected"]["metrics"]["accuracy"] == 1.0
    assert selected_bbox_report["deployment_gate"]["passed"] is True
    assert selected_bbox_report["evidence"]["runtime_promotion_authorized"] is True
    assert gt_crop_report["policy_search"]["selection"]["enabled"] is True
    assert gt_crop_report["deployment_gate"]["passed"] is False
    assert gt_crop_report["deployment_gate"]["evidence_checks"][
        "not_ground_truth_crop_only"
    ] is False
    assert gt_crop_report["evidence"]["runtime_promotion_authorized"] is False


def test_load_selected_bbox_predictions_supports_background_and_provenance(tmp_path):
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        json.dumps(
            {
                "prediction_source": "selected_yolo_bbox",
                "predictions": [
                    {
                        "image": "negative.jpg",
                        "selected_bbox": [1, 2, 30, 40],
                        "crop_source": "selected_yolo_bbox",
                        "material": {
                            "class_id": 9,
                            "class_name": "background",
                            "confidence": 0.97,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    predictions, source = load_verifier_predictions(
        prediction_path, include_source=True
    )

    assert source == "selected_yolo_bbox_predictions"
    assert predictions["negative.jpg"]["internal_name"] == "background"
    assert predictions["negative.jpg"]["external_name"] is None
    assert predictions["negative.jpg"]["selected_bbox"] == [1.0, 2.0, 30.0, 40.0]


def test_deployment_metric_gate_rejects_harmful_negative_promotion():
    rows = [
        _combined_row(
            "negative.jpg",
            truth="negative",
            baseline="styrofoam",
            yolo_confidence=0.8,
            verifier="plastic",
            verifier_confidence=0.95,
            localized=False,
            is_negative=True,
        ),
        _combined_row(
            "paper.jpg",
            truth="paper",
            baseline="styrofoam",
            yolo_confidence=0.4,
            verifier="paper",
            verifier_confidence=0.95,
        ),
    ]
    evaluation = evaluate_policy(rows, 0.8, 0.1)

    gate = deployment_metric_gate(evaluation)

    assert gate["checks"]["zero_harmful_corrections"] is False
    assert gate["checks"]["zero_negative_to_allowed_like_promotions"] is False
    assert gate["passed"] is False
