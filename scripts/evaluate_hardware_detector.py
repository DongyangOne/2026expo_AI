"""고정 하드웨어 holdout에서 YOLO 후보를 운영 선택 방식으로 비교한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO


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


def evaluate(
    model_path: Path,
    dataset_dir: Path,
    output_path: Path,
    thresholds: list[float],
    device: str,
    batch: int,
    imgsz: int,
    min_iou: float,
) -> dict:
    ground_truth = _load_ground_truth(dataset_dir)
    model = YOLO(str(model_path))
    minimum_threshold = min(thresholds)
    predictions = model.predict(
        [str(row["image"]) for row in ground_truth],
        conf=minimum_threshold,
        iou=0.6,
        imgsz=imgsz,
        device=device,
        batch=batch,
        verbose=False,
    )

    report = {"model": str(model_path.resolve()), "dataset": str(dataset_dir.resolve()), "thresholds": {}}
    for threshold in thresholds:
        outcomes = Counter()
        per_class = defaultdict(Counter)
        confusion = Counter()
        details = []
        for truth, result in zip(ground_truth, predictions):
            candidates = []
            if result.boxes is not None:
                for box, class_id, confidence in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                ):
                    if confidence >= threshold:
                        candidates.append(
                            {
                                "bbox": box,
                                "class_id": int(class_id),
                                "class_name": model.names[int(class_id)],
                                "confidence": float(confidence),
                            }
                        )
            best = max(candidates, key=lambda item: item["confidence"], default=None)
            expected = truth["class_name"]
            per_class[expected]["total"] += 1
            if truth["class_id"] is None:
                outcome = "negative_clean" if best is None else "negative_false_positive"
            elif best is None:
                outcome = "positive_missed"
            else:
                overlap = _iou(truth["bbox"], best["bbox"])
                if overlap < min_iou:
                    outcome = "positive_background_false_positive"
                elif best["class_id"] != truth["class_id"]:
                    outcome = "positive_wrong_class"
                else:
                    outcome = "positive_correct"
            outcomes[outcome] += 1
            per_class[expected][outcome] += 1
            confusion[(expected, best["class_name"] if best else "none")] += 1
            details.append(
                {
                    "image": truth["image"].name,
                    "expected": expected,
                    "predicted": best["class_name"] if best else None,
                    "confidence": round(best["confidence"], 5) if best else None,
                    "iou": round(_iou(truth["bbox"], best["bbox"]), 5) if best and truth["bbox"] else None,
                    "outcome": outcome,
                }
            )

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
    args = parser.parse_args()
    report = evaluate(
        args.model,
        args.dataset_dir,
        args.output,
        args.thresholds,
        args.device,
        args.batch,
        args.imgsz,
        args.min_iou,
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
