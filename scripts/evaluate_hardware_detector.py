"""고정 하드웨어 holdout에서 YOLO 후보를 운영 선택 방식으로 비교한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_ground_truth(dataset_dir: Path) -> list[dict]:
    names = {}
    yaml_path = dataset_dir / "dataset.yaml"
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and ":" in stripped:
            raw_id, name = stripped.split(":", 1)
            names[int(raw_id)] = name.strip()

    rows = []
    for image_path in sorted((dataset_dir / "images" / "val").glob("*")):
        label_path = dataset_dir / "labels" / "val" / f"{image_path.stem}.txt"
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line]
        if not lines:
            rows.append({"image": image_path, "class_id": None, "class_name": "negative", "bbox": None})
            continue
        class_id, cx, cy, width, height = (float(value) for value in lines[0].split())
        import cv2

        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        x1, y1 = (cx - width / 2) * w, (cy - height / 2) * h
        x2, y2 = (cx + width / 2) * w, (cy + height / 2) * h
        rows.append(
            {
                "image": image_path,
                "class_id": int(class_id),
                "class_name": names[int(class_id)],
                "bbox": [x1, y1, x2, y2],
            }
        )
    return rows


def _iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _rounded_bbox(value: list[float]) -> list[float]:
    """보고서 용량을 줄이되 crop을 재현하기에 충분한 좌표 정밀도를 보존한다."""
    return [round(float(coordinate), 5) for coordinate in value]


def select_candidate(candidates: list[dict], threshold: float) -> tuple[dict | None, list[dict]]:
    """운영 ``run_main``과 동일하게 임계값 이상 최고 신뢰 후보를 선택한다."""
    eligible = [
        candidate for candidate in candidates
        if float(candidate["confidence"]) >= threshold
    ]
    selected = max(eligible, key=lambda item: float(item["confidence"]), default=None)
    return selected, eligible


def build_detection_detail(
    truth: dict,
    candidates: list[dict],
    threshold: float,
    min_iou: float,
) -> dict:
    """단일 원본 이미지의 운영 선택 결과와 재현 가능한 bbox 감사를 만든다."""
    selected, eligible = select_candidate(candidates, threshold)
    expected = truth["class_name"]
    selected_iou = (
        _iou(truth["bbox"], selected["bbox"])
        if selected is not None and truth["bbox"] is not None else None
    )
    if truth["class_id"] is None:
        outcome = "negative_clean" if selected is None else "negative_false_positive"
    elif selected is None:
        outcome = "positive_missed"
    elif selected_iou is None or selected_iou < min_iou:
        outcome = "positive_background_false_positive"
    elif int(selected["class_id"]) != int(truth["class_id"]):
        outcome = "positive_wrong_class"
    else:
        outcome = "positive_correct"

    def audit_candidate(candidate: dict) -> dict:
        return {
            "bbox": _rounded_bbox(candidate["bbox"]),
            "class_id": int(candidate["class_id"]),
            "class_name": str(candidate["class_name"]),
            "confidence": round(float(candidate["confidence"]), 6),
        }

    selected_audit = audit_candidate(selected) if selected is not None else None
    image_path = Path(truth["image"])
    return {
        # 기존 소비자가 쓰는 필드는 그대로 유지한다.
        "image": image_path.name,
        "expected": expected,
        "predicted": selected["class_name"] if selected else None,
        "confidence": round(float(selected["confidence"]), 5) if selected else None,
        "iou": round(float(selected_iou), 5) if selected_iou is not None else None,
        "outcome": outcome,
        # 아래 필드는 원본 이미지의 실제 선택 bbox에서 verifier를 재실행하기 위한 감사 정보다.
        "image_path": str(image_path.resolve()),
        "expected_class_id": truth["class_id"],
        "predicted_class_id": int(selected["class_id"]) if selected else None,
        "bbox": selected_audit["bbox"] if selected_audit else None,
        "selected_bbox": selected_audit["bbox"] if selected_audit else None,
        "selected_candidate": selected_audit,
        "candidates": [audit_candidate(candidate) for candidate in eligible],
        "selection_rule": "highest_confidence_at_or_above_threshold",
        "selection_threshold": float(threshold),
    }


def evaluate(
    model_path: Path,
    dataset_dir: Path,
    output_path: Path,
    thresholds: list[float],
    device: str,
    batch: int,
    imgsz: int,
    min_iou: float,
    nms_iou: float = 0.70,
) -> dict:
    # 지연 import로 pure helper 테스트가 ultralytics 설치/모델 로드에 의존하지 않게 한다.
    from ultralytics import YOLO

    ground_truth = _load_ground_truth(dataset_dir)
    model = YOLO(str(model_path), task="detect")
    minimum_threshold = min(thresholds)
    sources = [str(row["image"]) for row in ground_truth]
    exported_backend = model_path.is_dir() or model_path.suffix.lower() != ".pt"
    if exported_backend:
        # Ultralytics export backends (notably NCNN) can return a single result
        # for a source list and then index it as if it were a batch.  Sequential
        # inference is slower but is required for production-faithful gating.
        predictions = [
            model.predict(
                source,
                conf=minimum_threshold,
                iou=nms_iou,
                imgsz=imgsz,
                device=device,
                batch=1,
                verbose=False,
            )[0]
            for source in sources
        ]
    else:
        predictions = model.predict(
            sources,
            conf=minimum_threshold,
            iou=nms_iou,
            imgsz=imgsz,
            device=device,
            batch=batch,
            verbose=False,
        )

    report = {
        "model": str(model_path.resolve()),
        "model_format": "ncnn" if model_path.is_dir() else model_path.suffix.lstrip(".").lower(),
        "dataset": str(dataset_dir.resolve()),
        "raw_image_root": str((dataset_dir / "images" / "val").resolve()),
        "prediction_confidence_floor": float(minimum_threshold),
        "nms_iou": float(nms_iou),
        "thresholds": {},
    }
    prediction_candidates = []
    for result in predictions:
        candidates = []
        if result.boxes is not None:
            for box, class_id, confidence in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                candidates.append(
                    {
                        "bbox": [float(value) for value in box],
                        "class_id": int(class_id),
                        "class_name": model.names[int(class_id)],
                        "confidence": float(confidence),
                    }
                )
        prediction_candidates.append(candidates)

    for threshold in thresholds:
        outcomes = Counter()
        per_class = defaultdict(Counter)
        confusion = Counter()
        details = []
        for truth, candidates in zip(ground_truth, prediction_candidates):
            detail = build_detection_detail(truth, candidates, threshold, min_iou)
            expected = detail["expected"]
            predicted = detail["predicted"]
            outcome = detail["outcome"]
            per_class[expected]["total"] += 1
            outcomes[outcome] += 1
            per_class[expected][outcome] += 1
            confusion[(expected, predicted if predicted else "none")] += 1
            details.append(detail)

        positives = sum(1 for row in ground_truth if row["class_id"] is not None)
        negatives = len(ground_truth) - positives
        correct = outcomes["positive_correct"] + outcomes["negative_clean"]
        report["thresholds"][str(threshold)] = {
            "images": len(ground_truth),
            "positives": positives,
            "negatives": negatives,
            "overall_correct": correct,
            "overall_accuracy": round(correct / len(ground_truth), 4),
            "positive_correct": outcomes["positive_correct"],
            "positive_accuracy": round(outcomes["positive_correct"] / positives, 4) if positives else None,
            "negative_clean": outcomes["negative_clean"],
            "negative_specificity": round(outcomes["negative_clean"] / negatives, 4) if negatives else None,
            "outcomes": dict(outcomes),
            "per_class": {name: dict(values) for name, values in sorted(per_class.items())},
            "confusion": {
                f"{expected}->{predicted}": count
                for (expected, predicted), count in sorted(confusion.items())
            },
            "details": details,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.55])
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-iou", type=float, default=0.3)
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.70,
        help="운영 run_main과 동일한 YOLO NMS IoU",
    )
    args = parser.parse_args()
    if not 0 <= args.nms_iou <= 1:
        parser.error("nms-iou must be in [0, 1]")
    report = evaluate(
        args.model,
        args.dataset_dir,
        args.output,
        args.thresholds,
        args.device,
        args.batch,
        args.imgsz,
        args.min_iou,
        args.nms_iou,
    )
    summary = {
        threshold: {
            key: value
            for key, value in metrics.items()
            if key in {"overall_accuracy", "positive_accuracy", "negative_specificity", "outcomes"}
        }
        for threshold, metrics in report["thresholds"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
