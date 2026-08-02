"""고정 crop manifest에서 하나 이상의 ONNX 검증기를 같은 조건으로 비교한다."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = [
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
]
TASK_NAMES = ("material", "dent", "label", "foreign_material")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model은 name=path 형식이어야 합니다.")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--model은 name=path 형식이어야 합니다.")
    return name.strip(), Path(raw_path)


def _softmax_prediction(logits: np.ndarray) -> tuple[int, float]:
    values = np.asarray(logits).reshape(-1).astype(np.float64)
    shifted = values - values.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    predicted = int(probabilities.argmax())
    return predicted, float(probabilities[predicted])


def _preprocess(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]


def classification_metrics(
    expected: list[int], predicted: list[int], class_count: int,
) -> dict:
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for truth, guess in zip(expected, predicted):
        confusion[truth][guess] += 1

    per_class = {}
    f1_values = []
    for class_id in range(class_count):
        true_positive = confusion[class_id][class_id]
        support = sum(confusion[class_id])
        predicted_count = sum(row[class_id] for row in confusion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            f1_values.append(f1)
        per_class[str(class_id)] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    correct = sum(confusion[index][index] for index in range(class_count))
    return {
        "support": len(expected),
        "accuracy": correct / len(expected) if expected else None,
        "macro_f1_present_classes": sum(f1_values) / len(f1_values) if f1_values else None,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def evaluate_model(model_path: Path, rows: list[dict], manifest_root: Path) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_size = model_input.shape[-1]
    if not isinstance(input_size, int):
        input_size = 320
    available = {output.name for output in session.get_outputs()}
    missing = set(TASK_NAMES) - available
    if missing:
        raise RuntimeError(f"검증기 출력 누락: {sorted(missing)}")

    labels = {task: [] for task in TASK_NAMES}
    guesses = {task: [] for task in TASK_NAMES}
    errors = []
    confidence_sum = Counter()
    confidence_count = Counter()

    for row in rows:
        image_path = manifest_root / row["filepath"]
        values = session.run(list(TASK_NAMES), {model_input.name: _preprocess(image_path, input_size)})
        predictions = {
            task: _softmax_prediction(value)
            for task, value in zip(TASK_NAMES, values)
        }
        row_errors = {}
        for task in TASK_NAMES:
            truth = int(row.get(task, -1))
            if truth < 0:
                continue
            guess, confidence = predictions[task]
            labels[task].append(truth)
            guesses[task].append(guess)
            confidence_sum[task] += confidence
            confidence_count[task] += 1
            if guess != truth:
                row_errors[task] = {
                    "expected": truth,
                    "predicted": guess,
                    "confidence": confidence,
                }
        if row_errors:
            errors.append({
                "filepath": row["filepath"],
                "source_id": row.get("source_id", ""),
                "errors": row_errors,
            })

    metrics = {}
    for task in TASK_NAMES:
        class_count = len(CLASS_NAMES) if task == "material" else 2
        metrics[task] = classification_metrics(labels[task], guesses[task], class_count)
        metrics[task]["mean_confidence"] = (
            confidence_sum[task] / confidence_count[task]
            if confidence_count[task] else None
        )
    return {
        "model": str(model_path.resolve()),
        "input_size": input_size,
        "rows": len(rows),
        "metrics": metrics,
        "misclassified_rows": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, type=_parse_model)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8", newline="") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row.get("split", "").lower() == args.split.lower()
        ]
    if not rows:
        raise SystemExit(f"[ERROR] split={args.split!r} 행이 없습니다.")

    result = {
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "row_count": len(rows),
        "models": {},
    }
    for name, model_path in args.model:
        if name in result["models"]:
            raise SystemExit(f"[ERROR] 중복 모델 이름: {name}")
        result["models"][name] = evaluate_model(model_path, rows, args.manifest.parent)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        name: {
            task: {
                "accuracy": report["metrics"][task]["accuracy"],
                "macro_f1": report["metrics"][task]["macro_f1_present_classes"],
                "support": report["metrics"][task]["support"],
            }
            for task in TASK_NAMES
        }
        for name, report in result["models"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
