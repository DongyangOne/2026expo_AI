"""하드웨어 holdout에서 YOLO + crop 검증기 교정 정책을 오프라인 평가한다.

운영 코드는 변경하지 않는다. ``evaluate_hardware_detector.py``가 현재 NCNN과 원본
이미지로 만든 threshold별 ``details``를 읽고, 실제 선택 YOLO bbox를 verifier에 넣어
``검증기 신뢰도 - YOLO 신뢰도`` margin을 sweep한다. positive/negative를 모두 포함하고
PET(내부 class 1)는 외부 계약에서 plastic(class 3)으로 합쳐 평가한다.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)
VERIFIER_CLASS_NAMES = CLASS_NAMES + ("background",)
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
EXTERNAL_CLASS_NAMES = tuple(name for name in CLASS_NAMES if name != "pet")
ALLOWED_LIKE_CLASSES = frozenset({"can", "paper", "plastic", "vinyl"})
SELECTED_BBOX_EVIDENCE_SOURCES = frozenset(
    {"selected_yolo_bbox_inference", "selected_yolo_bbox_predictions"}
)
DEFAULT_TRUST_CONFIDENCE = 0.55
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

LOCALIZED_OUTCOMES = {"positive_correct", "positive_wrong_class"}


def external_material_id(class_id: int) -> int:
    """외부 응답 계약처럼 PET를 plastic으로 합친다."""
    return 3 if class_id == 1 else class_id


def _class_id(value: object, *, field: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field}: bool은 class id가 아닙니다: {value!r}")
    if isinstance(value, (int, np.integer)):
        class_id = int(value)
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field}: 잘못된 class id: {value!r}")
        class_id = int(value)
    else:
        normalized = str(value).strip().lower()
        if normalized in {"none", "negative", "not_detected", "null"}:
            return None
        if normalized in CLASS_IDS:
            return CLASS_IDS[normalized]
        try:
            class_id = int(normalized)
        except ValueError as error:
            raise ValueError(f"{field}: 알 수 없는 class: {value!r}") from error
    if not 0 <= class_id < len(CLASS_NAMES):
        raise ValueError(f"{field}: class id 범위 오류: {class_id}")
    return class_id


def external_material_name(value: object, *, field: str = "material") -> str | None:
    if isinstance(value, str) and value.strip().casefold() == "background":
        return None
    class_id = _class_id(value, field=field)
    if class_id is None:
        return None
    return CLASS_NAMES[external_material_id(class_id)]


def _verifier_class_id(value: object, *, field: str) -> int:
    """9-class와 선택적인 10번째 background verifier 출력을 모두 읽는다."""
    if isinstance(value, str) and value.strip().casefold() == "background":
        return len(CLASS_NAMES)
    if isinstance(value, (int, np.integer)) and int(value) == len(CLASS_NAMES):
        return len(CLASS_NAMES)
    if isinstance(value, float) and value.is_integer() and int(value) == len(CLASS_NAMES):
        return len(CLASS_NAMES)
    class_id = _class_id(value, field=field)
    if class_id is None:
        raise ValueError(f"{field}: verifier class가 비어 있습니다: {value!r}")
    return class_id


def _image_key(value: object) -> str:
    if value is None or not str(value).strip():
        raise ValueError("이미지 경로가 비어 있습니다.")
    return Path(str(value).replace("\\", "/")).name.casefold()


def load_manifest(manifest_path: Path, split: str = "validation") -> list[dict]:
    with manifest_path.open(encoding="utf-8", newline="") as file:
        source_rows = [
            row for row in csv.DictReader(file)
            if row.get("split", "").strip().casefold() == split.casefold()
        ]
    if not source_rows:
        raise ValueError(f"manifest에 split={split!r} 행이 없습니다.")

    rows = []
    seen = set()
    for row in source_rows:
        key = _image_key(row.get("filepath"))
        if key in seen:
            raise ValueError(f"manifest 이미지 이름 중복: {key}")
        seen.add(key)
        material = row.get("material")
        category = row.get("category")
        truth_value = material if material not in {None, ""} else category
        truth_id = _class_id(truth_value, field="manifest material")
        is_negative = truth_id is None
        if truth_value in {None, ""}:
            raise ValueError(f"manifest material이 비어 있습니다: {row.get('filepath')}")
        if category:
            category_id = _class_id(category, field="manifest category")
            if category_id != truth_id:
                raise ValueError(
                    f"manifest material/category 불일치: {row.get('filepath')} "
                    f"({truth_id} != {category_id})"
                )
        rows.append(
            {
                "image_key": key,
                "filepath": row["filepath"],
                "source_id": row.get("source_id", ""),
                "truth_internal_id": truth_id,
                "truth_external": (
                    "negative" if is_negative
                    else CLASS_NAMES[external_material_id(truth_id)]
                ),
                "is_negative": is_negative,
            }
        )
    return rows


def manifest_rows_from_baseline(baseline_details: dict[str, dict]) -> list[dict]:
    """detector report 자체에서 positive와 negative를 빠짐없이 구성한다."""
    rows = []
    for key, detail in baseline_details.items():
        expected_value = (
            detail.get("expected_class_id")
            if detail.get("expected_class_id") is not None
            else detail.get("expected")
        )
        truth_id = _class_id(expected_value, field="baseline expected")
        is_negative = truth_id is None
        rows.append(
            {
                "image_key": key,
                "filepath": (
                    detail.get("image_path")
                    or detail.get("image_relative_path")
                    or detail.get("image")
                ),
                "source_id": detail.get("source_id", key),
                "truth_internal_id": truth_id,
                "truth_external": (
                    "negative" if is_negative
                    else CLASS_NAMES[external_material_id(truth_id)]
                ),
                "is_negative": is_negative,
            }
        )
    if not rows:
        raise ValueError("baseline details가 비어 있습니다.")
    return rows


def _select_threshold_details(payload: object, threshold: str) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("baseline report는 JSON object 또는 details list여야 합니다.")
    if isinstance(payload.get("details"), list):
        return payload["details"]

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("baseline report에 thresholds/details가 없습니다.")
    selected = thresholds.get(str(threshold))
    if selected is None:
        try:
            target = float(threshold)
            matching_keys = [key for key in thresholds if float(key) == target]
        except (TypeError, ValueError):
            matching_keys = []
        if len(matching_keys) == 1:
            selected = thresholds[matching_keys[0]]
    if not isinstance(selected, dict) or not isinstance(selected.get("details"), list):
        raise ValueError(
            f"baseline threshold={threshold!r} details가 없습니다. "
            f"사용 가능: {sorted(thresholds)}"
        )
    return selected["details"]


def load_baseline_details(report_path: Path, threshold: str = "0.25") -> dict[str, dict]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    details = _select_threshold_details(payload, threshold)
    indexed: dict[str, dict] = {}
    for detail in details:
        key = _image_key(detail.get("image") or detail.get("filepath"))
        if key in indexed:
            raise ValueError(f"baseline 이미지 이름 중복: {key}")
        indexed[key] = detail
    return indexed


def _softmax_summary(logits: np.ndarray) -> dict:
    values = np.asarray(logits).reshape(-1).astype(np.float64)
    if len(values) not in {len(CLASS_NAMES), len(VERIFIER_CLASS_NAMES)}:
        raise ValueError(
            "verifier material 출력 크기 오류: "
            f"{len(values)} not in ({len(CLASS_NAMES)}, {len(VERIFIER_CLASS_NAMES)})"
        )
    class_names = CLASS_NAMES if len(values) == len(CLASS_NAMES) else VERIFIER_CLASS_NAMES
    shifted = values - values.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    order = np.argsort(probabilities)[::-1]
    best, runner_up = int(order[0]), int(order[1])
    best_name = class_names[best]
    return {
        "internal_id": best,
        "internal_name": best_name,
        "external_name": (
            None if best_name == "background"
            else CLASS_NAMES[external_material_id(best)]
        ),
        "confidence": float(probabilities[best]),
        "runner_up_internal_id": runner_up,
        "runner_up_internal_name": class_names[runner_up],
        "runner_up_confidence": float(probabilities[runner_up]),
        "probability_gap": float(probabilities[best] - probabilities[runner_up]),
    }


def _preprocess(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]


def _normalize_bbox(value: object, *, field: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field}: bbox는 좌표 4개여야 합니다: {value!r}")
    bbox = [float(coordinate) for coordinate in value]
    if not all(np.isfinite(coordinate) for coordinate in bbox):
        raise ValueError(f"{field}: 유한하지 않은 bbox: {value!r}")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{field}: 면적이 없는 bbox: {value!r}")
    return bbox


def selected_bbox(detail: dict) -> list[float] | None:
    value = detail.get("selected_bbox")
    if value is None:
        value = detail.get("bbox")
    if value is None and isinstance(detail.get("selected_candidate"), dict):
        value = detail["selected_candidate"].get("bbox")
    return _normalize_bbox(value, field="baseline selected_bbox")


def _bbox_matches(first: object, second: object, tolerance: float = 1e-3) -> bool:
    try:
        left = _normalize_bbox(first, field="first bbox")
        right = _normalize_bbox(second, field="second bbox")
    except ValueError:
        return False
    if left is None or right is None:
        return left is right
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """운영 ``run_verifier``와 같은 비율 유지 resize + 114 패딩."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized_height = max(1, int(height * scale))
    resized_width = max(1, int(width * scale))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def _preprocess_selected_bbox(
    image_path: Path,
    bbox: list[float],
    size: int,
    *,
    padding: float = 0.08,
) -> np.ndarray:
    """원본 이미지의 선택 YOLO bbox를 운영 verifier와 동일하게 전처리한다."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"원본 이미지를 읽을 수 없습니다: {image_path}")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _normalize_bbox(bbox, field="selected bbox")
    box_width, box_height = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - box_width * padding))
    y1 = max(0, int(y1 - box_height * padding))
    x2 = min(width, int(x2 + box_width * padding))
    y2 = min(height, int(y2 + box_height * padding))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"이미지 범위 내 crop 면적이 없습니다: {image_path} bbox={bbox}")
    crop = _letterbox(image[y1:y2, x1:x2], size)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]


def _resolve_raw_image_path(
    row: dict,
    detail: dict,
    raw_image_root: Path | None,
) -> Path:
    candidates = [
        detail.get("image_path"),
        detail.get("image_relative_path"),
        row.get("filepath"),
        detail.get("image"),
    ]
    for value in candidates:
        if value is None or not str(value).strip():
            continue
        path = Path(str(value))
        if path.is_absolute() and path.is_file():
            return path
        if raw_image_root is not None:
            rooted = raw_image_root / path
            if rooted.is_file():
                return rooted
            basename = raw_image_root / path.name
            if basename.is_file():
                return basename
    raise FileNotFoundError(
        f"원본 이미지 경로를 찾을 수 없습니다: {row['image_key']} "
        f"(raw_image_root={raw_image_root})"
    )


def infer_verifier(
    model_path: Path,
    manifest_rows: list[dict],
    manifest_root: Path,
) -> tuple[dict[str, dict], int]:
    """레거시 정답 crop 추론. 탐색용이며 배포 게이트 증거로 사용할 수 없다."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_size = model_input.shape[-1]
    if not isinstance(input_size, int):
        input_size = 320
    outputs = {output.name for output in session.get_outputs()}
    if "material" not in outputs:
        raise RuntimeError("검증기 출력 누락: material")

    predictions = {}
    for row in manifest_rows:
        image_path = manifest_root / row["filepath"]
        material_logits = session.run(
            ["material"], {model_input.name: _preprocess(image_path, input_size)}
        )[0]
        predictions[row["image_key"]] = _softmax_summary(material_logits)
    return predictions, input_size


def infer_verifier_from_selected_bboxes(
    model_path: Path,
    manifest_rows: list[dict],
    baseline_details: dict[str, dict],
    raw_image_root: Path | None = None,
) -> tuple[dict[str, dict], int]:
    """원본 이미지에서 실제 선택된 YOLO bbox만 crop해 verifier를 추론한다."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_size = model_input.shape[-1]
    if not isinstance(input_size, int):
        input_size = 320
    output_names = {output.name for output in session.get_outputs()}
    if "material" not in output_names:
        raise RuntimeError("검증기 출력 누락: material")

    predictions = {}
    for row in manifest_rows:
        key = row["image_key"]
        detail = baseline_details[key]
        bbox = selected_bbox(detail)
        if bbox is None:
            if detail.get("predicted") is not None:
                raise ValueError(f"선택 검출은 있지만 bbox가 없습니다: {key}")
            continue
        image_path = _resolve_raw_image_path(row, detail, raw_image_root)
        logits = session.run(
            ["material"],
            {model_input.name: _preprocess_selected_bbox(image_path, bbox, input_size)},
        )[0]
        predictions[key] = {
            **_softmax_summary(logits),
            "selected_bbox": bbox,
            "crop_source": "selected_yolo_bbox",
            "raw_image": str(image_path.resolve()),
        }
    return predictions, input_size


def combine_predictions(
    manifest_rows: list[dict],
    baseline_details: dict[str, dict],
    verifier_predictions: dict[str, dict],
    *,
    require_full_baseline_coverage: bool = True,
) -> list[dict]:
    """모든 positive/negative를 결합하고 PET를 외부 plastic으로 정규화한다."""
    combined = []
    missing_baseline = []
    manifest_keys = {row["image_key"] for row in manifest_rows}
    extra_baseline = sorted(set(baseline_details) - manifest_keys)
    if require_full_baseline_coverage and extra_baseline:
        raise ValueError(
            "manifest가 baseline의 모든 positive/negative를 포함하지 않습니다: "
            f"누락={extra_baseline[:5]}({len(extra_baseline)})"
        )
    for manifest in manifest_rows:
        key = manifest["image_key"]
        detail = baseline_details.get(key)
        if detail is None:
            missing_baseline.append(key)
            continue

        expected_value = (
            detail.get("expected_class_id")
            if detail.get("expected_class_id") is not None
            else detail.get("expected")
        )
        expected = external_material_name(expected_value, field="baseline expected")
        normalized_expected = expected or "negative"
        if normalized_expected != manifest["truth_external"]:
            raise ValueError(
                f"정답 불일치: {key} manifest={manifest['truth_external']} "
                f"baseline={normalized_expected}"
            )
        predicted_value = (
            detail.get("predicted_class_id")
            if detail.get("predicted_class_id") is not None
            else detail.get("predicted")
        )
        predicted = external_material_name(predicted_value, field="baseline predicted")
        confidence_raw = detail.get("confidence")
        if predicted is not None and confidence_raw is None:
            raise ValueError(f"baseline confidence 누락: {key}")
        baseline_confidence = float(confidence_raw) if confidence_raw is not None else None
        if baseline_confidence is not None and not 0.0 <= baseline_confidence <= 1.0:
            raise ValueError(f"baseline confidence 범위 오류: {key}={baseline_confidence}")

        outcome = str(detail.get("outcome") or "")
        if outcome:
            localization_ok = outcome in LOCALIZED_OUTCOMES
        else:
            # 분류기 전용 per-image 입력도 재사용할 수 있게 outcome이 없으면 검출을
            # 성공한 것으로 간주한다. 미감지는 predicted=None으로 남는다.
            localization_ok = predicted is not None

        bbox = selected_bbox(detail)
        candidates = detail.get("candidates")
        selected_candidate = detail.get("selected_candidate")
        bbox_audit_complete = isinstance(candidates, list)
        if predicted is not None:
            bbox_audit_complete = bbox_audit_complete and bbox is not None
            if isinstance(candidates, list) and bbox is not None:
                bbox_audit_complete = bbox_audit_complete and any(
                    isinstance(candidate, dict)
                    and _bbox_matches(candidate.get("bbox"), bbox)
                    and external_material_name(
                        candidate.get("class_id", candidate.get("class_name")),
                        field="detector candidate",
                    ) == predicted
                    for candidate in candidates
                )

        verifier = verifier_predictions.get(key)
        verifier_internal = None
        verifier_external = None
        verifier_confidence = None
        verifier_bbox = None
        verifier_crop_source = None
        verifier_bbox_matches_selected = predicted is None
        if verifier is not None:
            verifier_internal_id = _verifier_class_id(
                verifier.get("internal_id", verifier.get("internal_name")),
                field="verifier material",
            )
            verifier_internal = VERIFIER_CLASS_NAMES[verifier_internal_id]
            verifier_external = (
                None if verifier_internal == "background"
                else CLASS_NAMES[external_material_id(verifier_internal_id)]
            )
            if "external_name" in verifier and verifier.get("external_name") != verifier_external:
                raise ValueError(f"verifier external_name 불일치: {key}")
            verifier_confidence = float(verifier["confidence"])
            if not 0.0 <= verifier_confidence <= 1.0:
                raise ValueError(
                    f"verifier confidence 범위 오류: {key}={verifier_confidence}"
                )
            verifier_bbox = verifier.get("selected_bbox", verifier.get("bbox"))
            verifier_crop_source = verifier.get("crop_source")
            verifier_bbox_matches_selected = _bbox_matches(verifier_bbox, bbox)

        combined.append(
            {
                **manifest,
                "baseline_internal": detail.get("predicted"),
                "baseline_external": predicted,
                "baseline_confidence": baseline_confidence,
                "baseline_outcome": outcome or None,
                "localization_ok": localization_ok,
                "selected_bbox": bbox,
                "selected_iou": detail.get("iou"),
                "selected_candidate": selected_candidate,
                "detector_candidates": candidates,
                "detector_bbox_audit_complete": bbox_audit_complete,
                "verifier_internal": verifier_internal,
                "verifier_external": verifier_external,
                "verifier_confidence": verifier_confidence,
                "verifier_runner_up": (
                    verifier.get("runner_up_internal_name") if verifier else None
                ),
                "verifier_runner_up_confidence": (
                    verifier.get("runner_up_confidence") if verifier else None
                ),
                "verifier_probability_gap": (
                    verifier.get("probability_gap") if verifier else None
                ),
                "verifier_selected_bbox": verifier_bbox,
                "verifier_crop_source": verifier_crop_source,
                "verifier_bbox_matches_selected": verifier_bbox_matches_selected,
            }
        )
    if missing_baseline:
        raise ValueError(
            "이미지 결합 실패: "
            f"baseline 누락={missing_baseline[:5]}({len(missing_baseline)})"
        )
    return combined


def _outcome_prediction(row: dict, prediction: str | None) -> str:
    if row.get("is_negative"):
        return prediction or "negative"
    if not row["localization_ok"]:
        return "not_detected" if row["baseline_external"] is None else "localization_error"
    return prediction or "not_detected"


def _is_correct(row: dict, prediction: str | None) -> bool:
    if row.get("is_negative"):
        return prediction is None
    return row["localization_ok"] and prediction == row["truth_external"]


def classification_metrics(rows: list[dict], predictions: list[str | None]) -> dict:
    confusion: Counter[tuple[str, str]] = Counter()
    support = Counter()
    predicted_count = Counter()
    correct = Counter()
    for row, prediction in zip(rows, predictions):
        truth = row["truth_external"]
        effective_prediction = _outcome_prediction(row, prediction)
        confusion[(truth, effective_prediction)] += 1
        support[truth] += 1
        predicted_count[effective_prediction] += 1
        if truth == effective_prediction:
            correct[truth] += 1

    per_class = {}
    f1_values = []
    reported_classes = (
        ("negative",) + EXTERNAL_CLASS_NAMES
        if any(row.get("is_negative") for row in rows)
        else EXTERNAL_CLASS_NAMES
    )
    for name in reported_classes:
        true_positive = correct[name]
        precision = true_positive / predicted_count[name] if predicted_count[name] else 0.0
        recall = true_positive / support[name] if support[name] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support[name]:
            f1_values.append(f1)
        per_class[name] = {
            "support": support[name],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    total_correct = sum(
        1 for row, prediction in zip(rows, predictions)
        if _is_correct(row, prediction)
    )
    positive_support = sum(1 for row in rows if not row.get("is_negative"))
    positive_correct = sum(
        1 for row, prediction in zip(rows, predictions)
        if not row.get("is_negative") and _is_correct(row, prediction)
    )
    negative_support = len(rows) - positive_support
    negative_correct = sum(
        1 for row, prediction in zip(rows, predictions)
        if row.get("is_negative") and _is_correct(row, prediction)
    )
    return {
        "support": len(rows),
        "correct": total_correct,
        "accuracy": total_correct / len(rows) if rows else None,
        "positive_support": positive_support,
        "positive_accuracy": (
            positive_correct / positive_support if positive_support else None
        ),
        "negative_support": negative_support,
        "negative_specificity": (
            negative_correct / negative_support if negative_support else None
        ),
        "macro_f1_present_classes": sum(f1_values) / len(f1_values) if f1_values else None,
        "confusion": {
            f"{truth}->{prediction}": count
            for (truth, prediction), count in sorted(confusion.items())
        },
        "per_class": per_class,
    }


def evaluate_policy(
    rows: list[dict],
    verifier_confidence: float,
    verifier_over_yolo_margin: float,
    *,
    include_audit: bool = True,
    trust_confidence: float = DEFAULT_TRUST_CONFIDENCE,
    allow_background_veto: bool = False,
) -> dict:
    """하나의 보수적 교정 정책을 평가한다.

    margin은 운영 코드와 동일하게 verifier top-1 confidence에서 YOLO confidence를
    뺀 값이다. 검출 실패와 외부 계약상 같은 PET↔plastic은 교정하지 않는다. 실제
    운영에서는 GT localization을 알 수 없으므로 잘못된 bbox도 정책 대상에는 포함하되,
    정확도 계산에서는 교정으로 고쳐진 것으로 세지 않는다. background veto는 명시적으로
    허용한 경우에만 같은 confidence/margin 조건을 통과한 YOLO 검출을 억제한다.
    """
    baseline_predictions = [row["baseline_external"] for row in rows]
    hybrid_predictions = []
    audit = []
    corrections = Counter()

    for row in rows:
        baseline = row["baseline_external"]
        verifier = row["verifier_external"]
        baseline_confidence = row["baseline_confidence"]
        confidence = row["verifier_confidence"]
        score_margin = (
            confidence - baseline_confidence
            if confidence is not None and baseline_confidence is not None else None
        )
        background_prediction = row.get("verifier_internal") == "background"
        disagreement = baseline is not None and (
            background_prediction or (verifier is not None and verifier != baseline)
        )
        applied = False
        background_veto_applied = False
        if baseline is None:
            reason = "no_baseline_detection"
        elif row.get("selected_bbox") is None:
            reason = "missing_selected_bbox"
        elif background_prediction:
            if not allow_background_veto:
                reason = "verifier_background_retain_yolo"
            elif confidence is None or confidence < verifier_confidence:
                reason = "background_veto_below_verifier_confidence"
            elif score_margin is None or score_margin < verifier_over_yolo_margin:
                reason = "background_veto_below_verifier_over_yolo_margin"
            else:
                reason = "background_veto_applied"
                applied = True
                background_veto_applied = True
        elif verifier is None:
            reason = "no_verifier_prediction"
        elif not disagreement:
            reason = "external_agreement"
        elif confidence is None or confidence < verifier_confidence:
            reason = "below_verifier_confidence"
        elif score_margin is None or score_margin < verifier_over_yolo_margin:
            reason = "below_verifier_over_yolo_margin"
        else:
            reason = "corrected"
            applied = True

        final = verifier if applied else baseline
        final_confidence = (
            None if background_veto_applied
            else confidence if applied
            else baseline_confidence
        )
        baseline_allowed_like = bool(
            baseline in ALLOWED_LIKE_CLASSES
            and baseline_confidence is not None
            and baseline_confidence >= trust_confidence
        )
        final_allowed_like = bool(
            final in ALLOWED_LIKE_CLASSES
            and final_confidence is not None
            and final_confidence >= trust_confidence
        )
        baseline_correct = _is_correct(row, baseline)
        hybrid_correct = _is_correct(row, final)
        negative_to_allowed_like = bool(
            applied
            and row.get("is_negative")
            and final_allowed_like
            and not baseline_allowed_like
        )
        if applied:
            corrections["applied"] += 1
            if background_veto_applied:
                corrections["background_veto_applied"] += 1
            if background_veto_applied and not row.get("is_negative"):
                # positive 검출을 없애는 veto는 baseline이 이미 오답이어도 안전한
                # 개선으로 간주할 수 없다. 별도 harmful 유형으로 배포 gate에서 막는다.
                effect = "harmful_positive_suppression"
                corrections["harmful"] += 1
            elif negative_to_allowed_like:
                # 정답이 negative인 물체를 허용 계열로 올리는 전환은 baseline도
                # 오답이었더라도 운영상 악화이므로 harmful로 센다.
                effect = "harmful_negative_promotion"
                corrections["harmful"] += 1
                corrections["negative_to_allowed_like_promotions"] += 1
            elif not baseline_correct and hybrid_correct:
                effect = "beneficial"
            elif baseline_correct and not hybrid_correct:
                effect = "harmful"
            else:
                effect = "wrong_to_wrong"
            corrections[effect] += 1
        else:
            effect = "unchanged"
        hybrid_predictions.append(final)
        if include_audit:
            audit.append(
                {
                    "image": row["filepath"],
                    "source_id": row.get("source_id", ""),
                    "truth": row["truth_external"],
                    "baseline": baseline,
                    "baseline_confidence": baseline_confidence,
                    "baseline_outcome": row.get("baseline_outcome"),
                    "is_negative": bool(row.get("is_negative")),
                    "selected_bbox": row.get("selected_bbox"),
                    "selected_iou": row.get("selected_iou"),
                    "selected_candidate": row.get("selected_candidate"),
                    "detector_candidates": row.get("detector_candidates"),
                    "detector_bbox_audit_complete": row.get(
                        "detector_bbox_audit_complete", False
                    ),
                    "localization_ok": row["localization_ok"],
                    "verifier_internal": row.get("verifier_internal"),
                    "verifier": verifier,
                    "verifier_confidence": confidence,
                    "verifier_runner_up": row.get("verifier_runner_up"),
                    "verifier_probability_gap": row.get("verifier_probability_gap"),
                    "verifier_selected_bbox": row.get("verifier_selected_bbox"),
                    "verifier_crop_source": row.get("verifier_crop_source"),
                    "verifier_bbox_matches_selected": row.get(
                        "verifier_bbox_matches_selected", False
                    ),
                    "verifier_over_yolo_margin": score_margin,
                    "external_disagreement": disagreement,
                    "applied": applied,
                    "correction_type": (
                        "background_veto"
                        if background_veto_applied
                        else "class_correction" if applied
                        else None
                    ),
                    "background_veto_enabled": allow_background_veto,
                    "background_veto_applied": background_veto_applied,
                    "reason": reason,
                    "final": final,
                    "final_confidence": final_confidence,
                    "baseline_allowed_like": baseline_allowed_like,
                    "final_allowed_like": final_allowed_like,
                    "effect": effect,
                    "negative_to_allowed_like_promotion": negative_to_allowed_like,
                }
            )

    baseline_metrics = classification_metrics(rows, baseline_predictions)
    hybrid_metrics = classification_metrics(rows, hybrid_predictions)
    applied = corrections["applied"]
    beneficial = corrections["beneficial"]
    harmful = corrections["harmful"]
    wrong_to_wrong = corrections["wrong_to_wrong"]
    negative_promotions = corrections["negative_to_allowed_like_promotions"]
    background_vetoes = corrections["background_veto_applied"]
    harmful_positive_suppressions = corrections["harmful_positive_suppression"]
    return {
        "policy": {
            "verifier_confidence": verifier_confidence,
            "verifier_over_yolo_margin": verifier_over_yolo_margin,
            "trust_confidence": trust_confidence,
            "allow_background_veto": allow_background_veto,
        },
        "metrics": {
            "baseline": baseline_metrics,
            "hybrid": hybrid_metrics,
            "accuracy_gain": (
                hybrid_metrics["accuracy"] - baseline_metrics["accuracy"]
                if rows else None
            ),
            "external_accuracy_gain": (
                hybrid_metrics["positive_accuracy"]
                - baseline_metrics["positive_accuracy"]
                if hybrid_metrics["positive_accuracy"] is not None
                and baseline_metrics["positive_accuracy"] is not None
                else None
            ),
        },
        "corrections": {
            "applied": applied,
            "beneficial": beneficial,
            "harmful": harmful,
            "wrong_to_wrong": wrong_to_wrong,
            "negative_to_allowed_like_promotions": negative_promotions,
            "background_veto_applied": background_vetoes,
            "harmful_positive_suppression": harmful_positive_suppressions,
            "net_gain": beneficial - harmful,
            "beneficial_fraction": beneficial / applied if applied else None,
        },
        "audit": audit,
    }


def deployment_metric_gate(
    evaluation: dict,
    *,
    minimum_accuracy_gain: float = 0.05,
    maximum_recall_drop: float = 0.01,
) -> dict:
    """외부 8-class+negative 기준의 배포 품질 조건을 순수 계산한다."""
    baseline = evaluation["metrics"]["baseline"]
    hybrid = evaluation["metrics"]["hybrid"]
    corrections = evaluation["corrections"]
    accuracy_gain = float(
        evaluation["metrics"].get("external_accuracy_gain") or 0.0
    )
    baseline_macro_f1 = baseline["macro_f1_present_classes"]
    hybrid_macro_f1 = hybrid["macro_f1_present_classes"]

    recall_deltas = {}
    for name in EXTERNAL_CLASS_NAMES:
        baseline_class = baseline["per_class"][name]
        if not baseline_class["support"]:
            continue
        recall_deltas[name] = (
            hybrid["per_class"][name]["recall"] - baseline_class["recall"]
        )

    tolerance = 1e-12
    checks = {
        "has_positive_samples": baseline["positive_support"] > 0,
        "has_negative_samples": baseline["negative_support"] > 0,
        "external_accuracy_gain_at_least_5pp": (
            accuracy_gain + tolerance >= minimum_accuracy_gain
        ),
        "macro_f1_nondecrease": (
            baseline_macro_f1 is not None
            and hybrid_macro_f1 is not None
            and hybrid_macro_f1 + tolerance >= baseline_macro_f1
        ),
        "per_class_recall_drop_within_1pp": all(
            delta + tolerance >= -maximum_recall_drop
            for delta in recall_deltas.values()
        ),
        "negative_specificity_nondecrease": (
            baseline["negative_specificity"] is not None
            and hybrid["negative_specificity"] is not None
            and hybrid["negative_specificity"] + tolerance
            >= baseline["negative_specificity"]
        ),
        "zero_harmful_corrections": corrections["harmful"] == 0,
        "zero_negative_to_allowed_like_promotions": (
            corrections["negative_to_allowed_like_promotions"] == 0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_external_accuracy_gain": minimum_accuracy_gain,
            "maximum_per_class_recall_drop": maximum_recall_drop,
        },
        "measurements": {
            "external_accuracy_gain": accuracy_gain,
            "external_accuracy_gain_pp": accuracy_gain * 100.0,
            "macro_f1_delta": (
                hybrid_macro_f1 - baseline_macro_f1
                if hybrid_macro_f1 is not None and baseline_macro_f1 is not None
                else None
            ),
            "per_class_recall_delta": recall_deltas,
            "negative_specificity_delta": (
                hybrid["negative_specificity"] - baseline["negative_specificity"]
                if hybrid["negative_specificity"] is not None
                and baseline["negative_specificity"] is not None
                else None
            ),
            "harmful_corrections": corrections["harmful"],
            "negative_to_allowed_like_promotions": corrections[
                "negative_to_allowed_like_promotions"
            ],
        },
    }


def _validate_thresholds(values: Iterable[float], *, name: str) -> list[float]:
    result = sorted(set(float(value) for value in values))
    if not result:
        raise ValueError(f"{name} 값이 비어 있습니다.")
    if any(not 0.0 <= value <= 1.0 for value in result):
        raise ValueError(f"{name}은 0~1 범위여야 합니다: {result}")
    return result


def sweep_policies(
    rows: list[dict],
    confidence_thresholds: Iterable[float],
    margin_thresholds: Iterable[float],
    *,
    max_harmful: int = 0,
    max_wrong_to_wrong: int | None = None,
    trust_confidence: float = DEFAULT_TRUST_CONFIDENCE,
    allow_background_veto: bool = False,
) -> dict:
    """grid sweep 후 모든 배포 metric gate를 만족하는 정책만 선택한다."""
    confidences = _validate_thresholds(confidence_thresholds, name="confidence")
    margins = _validate_thresholds(margin_thresholds, name="margin")
    if max_harmful < 0 or (
        max_wrong_to_wrong is not None and max_wrong_to_wrong < 0
    ):
        raise ValueError("허용 오류 수는 0 이상이어야 합니다.")

    sweep = []
    candidates = []
    for confidence in confidences:
        for margin in margins:
            result = evaluate_policy(
                rows,
                confidence,
                margin,
                include_audit=False,
                trust_confidence=trust_confidence,
                allow_background_veto=allow_background_veto,
            )
            corrections = result["corrections"]
            metric_gate = deployment_metric_gate(result)
            conservative_safe = (
                metric_gate["passed"]
                and corrections["harmful"] <= max_harmful
                and (
                    max_wrong_to_wrong is None
                    or corrections["wrong_to_wrong"] <= max_wrong_to_wrong
                )
            )
            summary = {
                "policy": result["policy"],
                "hybrid_accuracy": result["metrics"]["hybrid"]["accuracy"],
                "hybrid_macro_f1": result["metrics"]["hybrid"]["macro_f1_present_classes"],
                "accuracy_gain": result["metrics"]["accuracy_gain"],
                "external_accuracy_gain": result["metrics"]["external_accuracy_gain"],
                "corrections": corrections,
                "metric_gate": metric_gate,
                "conservative_safe": conservative_safe,
            }
            sweep.append(summary)
            if conservative_safe:
                candidates.append(summary)

    if not candidates:
        return {
            "selection": {
                "enabled": False,
                "reason": "no_policy_passed_deployment_metrics",
                "policy": None,
                "limits": {
                    "max_harmful": max_harmful,
                    "max_wrong_to_wrong": max_wrong_to_wrong,
                },
            },
            "sweep": sweep,
        }

    # 정확도/F1을 먼저 최대화하고, 동률이면 높은 임계값을 택해 적용 범위를 좁힌다.
    selected = max(
        candidates,
        key=lambda item: (
            item["external_accuracy_gain"],
            item["hybrid_macro_f1"],
            item["hybrid_accuracy"],
            item["policy"]["verifier_confidence"],
            item["policy"]["verifier_over_yolo_margin"],
        ),
    )
    return {
        "selection": {
            "enabled": True,
            "reason": "deployment_metrics_passed",
            "policy": selected["policy"],
            "limits": {
                "max_harmful": max_harmful,
                "max_wrong_to_wrong": max_wrong_to_wrong,
            },
        },
        "sweep": sweep,
    }


def load_verifier_predictions(
    prediction_path: Path,
    *,
    include_source: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], str]:
    """실제 선택된 YOLO bbox에서 미리 계산한 verifier JSON을 읽는다.

    지원 형식은 ``{"predictions": [...]}``, ``{"predictions": {image: ...}}``,
    최상위 image-keyed object, 또는 prediction list다. 각 prediction은
    ``material`` nested object(run_verifier 형식) 또는 평평한 material 필드를 쓸 수 있다.
    이 함수는 입력이 실제 bbox에서 생성됐는지 기술적으로 증명하지 않으므로 보고서에도
    여전히 독립 검증 전 배포 금지라고 표시한다.
    """
    root_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    declared_source = None
    if isinstance(root_payload, dict):
        evidence = root_payload.get("evidence")
        declared_source = root_payload.get("prediction_source")
        if declared_source is None and isinstance(evidence, dict):
            declared_source = evidence.get("prediction_source") or evidence.get("crop_source")
    payload = root_payload
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]

    if isinstance(payload, dict):
        raw_entries = []
        for image, prediction in payload.items():
            if not isinstance(prediction, dict):
                raise ValueError(f"verifier prediction object 오류: {image}")
            raw_entries.append((image, prediction))
    elif isinstance(payload, list):
        raw_entries = []
        for prediction in payload:
            if not isinstance(prediction, dict):
                raise ValueError("verifier prediction list 원소는 object여야 합니다.")
            image = prediction.get("image") or prediction.get("filepath")
            raw_entries.append((image, prediction))
    else:
        raise ValueError("verifier predictions는 JSON object/list여야 합니다.")

    indexed = {}
    for image, raw in raw_entries:
        key = _image_key(image)
        if key in indexed:
            raise ValueError(f"verifier prediction 이미지 이름 중복: {key}")
        material = raw.get("material") if isinstance(raw.get("material"), dict) else raw
        internal_value = (
            material.get("class_id")
            if material.get("class_id") is not None
            else material.get("internal_id", material.get("class_name", material.get("internal_name")))
        )
        internal_id = _verifier_class_id(
            internal_value, field="verifier prediction material"
        )
        confidence = float(material["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"verifier prediction confidence 범위 오류: {key}={confidence}")
        runner_up = material.get("runner_up")
        runner_up_name = None
        runner_up_confidence = material.get("runner_up_confidence")
        if isinstance(runner_up, dict):
            runner_up_name = runner_up.get("class_name") or runner_up.get("internal_name")
            runner_up_confidence = runner_up.get("confidence", runner_up_confidence)
        else:
            runner_up_name = material.get("runner_up_internal_name", runner_up)
        probability_gap = material.get("probability_gap")
        if probability_gap is None and runner_up_confidence is not None:
            probability_gap = confidence - float(runner_up_confidence)
        internal_name = VERIFIER_CLASS_NAMES[internal_id]
        selected = raw.get("selected_bbox", raw.get("bbox"))
        crop_source = raw.get("crop_source")
        indexed[key] = {
            "internal_id": internal_id,
            "internal_name": internal_name,
            "external_name": (
                None if internal_name == "background"
                else CLASS_NAMES[external_material_id(internal_id)]
            ),
            "confidence": confidence,
            "runner_up_internal_name": runner_up_name,
            "runner_up_confidence": (
                float(runner_up_confidence) if runner_up_confidence is not None else None
            ),
            "probability_gap": float(probability_gap) if probability_gap is not None else None,
            "selected_bbox": (
                _normalize_bbox(selected, field=f"verifier selected_bbox {key}")
                if selected is not None else None
            ),
            "crop_source": crop_source,
        }
    if not indexed:
        raise ValueError("verifier predictions가 비어 있습니다.")

    normalized_declared = str(declared_source or "").strip().casefold()
    entry_sources = {
        str(prediction.get("crop_source") or "").strip().casefold()
        for prediction in indexed.values()
    }
    if normalized_declared in {
        "selected_yolo_bbox",
        "selected_yolo_bbox_predictions",
        "selected_bbox",
    } and entry_sources <= {"", "selected_yolo_bbox", "selected_bbox"}:
        prediction_source = "selected_yolo_bbox_predictions"
    elif entry_sources and entry_sources <= {"selected_yolo_bbox", "selected_bbox"}:
        prediction_source = "selected_yolo_bbox_predictions"
    elif (
        normalized_declared in {"ground_truth_crop", "ground_truth_manifest_crop", "gt_crop"}
        or any("ground_truth" in source or source == "gt_crop" for source in entry_sources)
    ):
        prediction_source = "ground_truth_manifest_crop"
    else:
        prediction_source = "unverified_verifier_predictions"
    if include_source:
        return indexed, prediction_source
    return indexed


def build_report(
    manifest_rows: list[dict],
    baseline_details: dict[str, dict],
    verifier_predictions: dict[str, dict],
    confidence_thresholds: Iterable[float],
    margin_thresholds: Iterable[float],
    *,
    max_harmful: int = 0,
    max_wrong_to_wrong: int | None = None,
    verifier_prediction_source: str = "ground_truth_manifest_crop",
    trust_confidence: float = DEFAULT_TRUST_CONFIDENCE,
    allow_background_veto: bool = False,
) -> dict:
    rows = combine_predictions(manifest_rows, baseline_details, verifier_predictions)
    policy_search = sweep_policies(
        rows,
        confidence_thresholds,
        margin_thresholds,
        max_harmful=max_harmful,
        max_wrong_to_wrong=max_wrong_to_wrong,
        trust_confidence=trust_confidence,
        allow_background_veto=allow_background_veto,
    )
    selected_policy = policy_search["selection"]["policy"]
    if selected_policy is None:
        selected = evaluate_policy(
            rows,
            1.000001,
            1.000001,
            include_audit=True,
            trust_confidence=trust_confidence,
            allow_background_veto=allow_background_veto,
        )
        # confidence=1/margin=1은 선택된 정책이 아니므로 오해하지 않게 명시한다.
        selected["policy"] = None
    else:
        selected = evaluate_policy(
            rows,
            selected_policy["verifier_confidence"],
            selected_policy["verifier_over_yolo_margin"],
            include_audit=True,
            trust_confidence=trust_confidence,
            allow_background_veto=allow_background_veto,
        )
    metric_gate = deployment_metric_gate(selected)
    ground_truth_crop = verifier_prediction_source == "ground_truth_manifest_crop"
    detected_rows = [row for row in rows if row["baseline_external"] is not None]
    evidence_checks = {
        "selected_yolo_bbox_evidence": (
            verifier_prediction_source in SELECTED_BBOX_EVIDENCE_SOURCES
        ),
        "not_ground_truth_crop_only": not ground_truth_crop,
        "all_detector_rows_included": len(rows) == len(baseline_details),
        "positive_and_negative_holdout_present": (
            selected["metrics"]["baseline"]["positive_support"] > 0
            and selected["metrics"]["baseline"]["negative_support"] > 0
        ),
        "selected_bbox_and_candidates_audited": all(
            row.get("detector_bbox_audit_complete", False) for row in rows
        ),
        "verifier_prediction_for_every_selected_bbox": all(
            row.get("verifier_internal") is not None for row in detected_rows
        ),
        "verifier_bbox_matches_yolo_selection": all(
            row.get("verifier_bbox_matches_selected", False) for row in detected_rows
        ),
    }
    deployment_passed = (
        selected_policy is not None
        and metric_gate["passed"]
        and all(evidence_checks.values())
    )
    failed_checks = [
        name
        for name, passed in {**metric_gate["checks"], **evidence_checks}.items()
        if not passed
    ]
    return {
        "contract": {
            "internal_classes": list(CLASS_NAMES),
            "external_classes": list(EXTERNAL_CLASS_NAMES),
            "optional_verifier_only_class": "background",
            "allowed_like_external_classes": sorted(ALLOWED_LIKE_CLASSES),
            "normalization": {"pet": "plastic"},
            "margin_definition": "verifier_confidence - yolo_confidence",
            "trust_confidence": trust_confidence,
            "background_policy": (
                "veto_at_policy_thresholds"
                if allow_background_veto
                else "retain_yolo"
            ),
            "localization_rule": sorted(LOCALIZED_OUTCOMES),
        },
        "joined_rows": len(rows),
        "evidence": {
            "verifier_prediction_source": verifier_prediction_source,
            "uses_ground_truth_manifest_crop": ground_truth_crop,
            "limitation": (
                "정답 crop 증거는 실제 YOLO 선택 bbox보다 낙관적이므로 배포 게이트에서 거부됨"
                if ground_truth_crop
                else (
                    "선택 YOLO bbox와 verifier bbox를 이미지별로 대조함"
                    if verifier_prediction_source in SELECTED_BBOX_EVIDENCE_SOURCES
                    else "verifier prediction의 선택 bbox 출처가 증명되지 않음"
                )
            ),
            "policy_selected_on_same_holdout": True,
            "runtime_promotion_authorized": deployment_passed,
            "checks": evidence_checks,
            "required_next_gate": (
                "원본 이미지의 실제 YOLO 선택 bbox로 verifier를 다시 평가"
                if ground_truth_crop
                else (
                    None if deployment_passed
                    else "실패한 배포 gate 조건을 해결한 뒤 동일 고정 holdout에서 재평가"
                )
            ),
        },
        "baseline_metrics": selected["metrics"]["baseline"],
        "policy_search": policy_search,
        "selected": {
            "policy": selected["policy"],
            "metrics": selected["metrics"]["hybrid"],
            "accuracy_gain": selected["metrics"]["accuracy_gain"],
            "external_accuracy_gain": selected["metrics"]["external_accuracy_gain"],
            "corrections": selected["corrections"],
        },
        "deployment_gate": {
            "passed": deployment_passed,
            "failed_checks": failed_checks,
            "metric_gate": metric_gate,
            "evidence_checks": evidence_checks,
        },
        "correction_audit": selected["audit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        help="선택 사항. 생략하면 detector report의 positive+negative 전체를 사용",
    )
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--baseline-threshold", default="0.25")
    parser.add_argument(
        "--raw-image-root", type=Path,
        help="원본 images/val 경로. detector report의 image_path가 유효하면 생략 가능",
    )
    verifier_source = parser.add_mutually_exclusive_group(required=True)
    verifier_source.add_argument(
        "--verifier-model", type=Path,
        help="원본 이미지의 실제 선택 YOLO bbox를 ONNX verifier로 추론",
    )
    verifier_source.add_argument(
        "--verifier-predictions", type=Path,
        help="prediction_source=selected_yolo_bbox인 per-image verifier JSON",
    )
    verifier_source.add_argument(
        "--ground-truth-crop-model", type=Path,
        help="레거시 정답 crop 탐색용. 결과는 배포 게이트에서 항상 거부",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--confidence-thresholds", nargs="+", type=float,
        default=[0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    )
    parser.add_argument(
        "--margin-thresholds", nargs="+", type=float,
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    )
    parser.add_argument("--max-harmful", type=int, default=0)
    parser.add_argument("--max-wrong-to-wrong", type=int)
    parser.add_argument("--trust-confidence", type=float, default=DEFAULT_TRUST_CONFIDENCE)
    parser.add_argument(
        "--allow-background-veto",
        action="store_true",
        help=(
            "10-class verifier가 background를 confidence/margin 임계값 이상으로 "
            "예측하면 선택된 YOLO 검출을 NOT_DETECTED로 억제"
        ),
    )
    args = parser.parse_args()

    baseline_payload = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    baseline_details = load_baseline_details(
        args.baseline_report, args.baseline_threshold
    )
    manifest_rows = (
        load_manifest(args.manifest, args.split)
        if args.manifest is not None
        else manifest_rows_from_baseline(baseline_details)
    )
    raw_image_root = args.raw_image_root
    if raw_image_root is None and isinstance(baseline_payload, dict):
        reported_root = baseline_payload.get("raw_image_root")
        if reported_root:
            raw_image_root = Path(reported_root)
        elif baseline_payload.get("dataset"):
            raw_image_root = Path(baseline_payload["dataset"]) / "images" / "val"

    if args.verifier_predictions is not None:
        verifier_predictions, prediction_source = load_verifier_predictions(
            args.verifier_predictions, include_source=True
        )
        input_size = None
    elif args.verifier_model is not None:
        verifier_predictions, input_size = infer_verifier_from_selected_bboxes(
            args.verifier_model,
            manifest_rows,
            baseline_details,
            raw_image_root,
        )
        prediction_source = "selected_yolo_bbox_inference"
    else:
        if args.manifest is None:
            parser.error("--ground-truth-crop-model에는 --manifest가 필요합니다.")
        verifier_predictions, input_size = infer_verifier(
            args.ground_truth_crop_model, manifest_rows, args.manifest.parent
        )
        prediction_source = "ground_truth_manifest_crop"
    report = build_report(
        manifest_rows,
        baseline_details,
        verifier_predictions,
        args.confidence_thresholds,
        args.margin_thresholds,
        max_harmful=args.max_harmful,
        max_wrong_to_wrong=args.max_wrong_to_wrong,
        verifier_prediction_source=prediction_source,
        trust_confidence=args.trust_confidence,
        allow_background_veto=args.allow_background_veto,
    )
    report["inputs"] = {
        "manifest": str(args.manifest.resolve()) if args.manifest is not None else None,
        "split": args.split,
        "baseline_report": str(args.baseline_report.resolve()),
        "baseline_threshold": float(args.baseline_threshold),
        "verifier_model": (
            str(args.verifier_model.resolve()) if args.verifier_model is not None else None
        ),
        "ground_truth_crop_model": (
            str(args.ground_truth_crop_model.resolve())
            if args.ground_truth_crop_model is not None else None
        ),
        "verifier_predictions": (
            str(args.verifier_predictions.resolve())
            if args.verifier_predictions is not None else None
        ),
        "verifier_input_size": input_size,
        "raw_image_root": str(raw_image_root.resolve()) if raw_image_root else None,
        "allow_background_veto": args.allow_background_veto,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "joined_rows": report["joined_rows"],
                "baseline_accuracy": report["baseline_metrics"]["accuracy"],
                "selection": report["policy_search"]["selection"],
                "selected": report["selected"],
                "deployment_gate": report["deployment_gate"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
