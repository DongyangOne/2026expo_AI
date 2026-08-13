"""배포 검증기가 어려워하는 실사용 단일 품목 crop을 자동 선별한다.

정답 재료에 대한 확률이 낮거나 다른 재료로 예측된 샘플을 우선하되, 특정
크기 구간에 쏠리지 않도록 bbox 면적 구간별로 번갈아 선택한다. 원본 이미지는
복제하지 않고 입력 manifest와 같은 폴더에 작은 focus manifest만 만든다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _area_ratio(row: dict[str, str]) -> float:
    width = max(1.0, float(row.get("source_width", 0)))
    height = max(1.0, float(row.get("source_height", 0)))
    return (
        float(row.get("source_bbox_w", 0))
        * float(row.get("source_bbox_h", 0))
        / (width * height)
    )


def _area_bin(row: dict[str, str]) -> str:
    ratio = _area_ratio(row)
    if ratio < 0.20:
        return "small"
    if ratio < 0.45:
        return "medium"
    return "large"


def eligible(
    row: dict[str, str],
    targets: set[str],
    min_crop_bytes: int,
    min_area_ratio: float,
    max_area_ratio: float,
    clean_only: bool,
) -> bool:
    try:
        if row.get("split", "").lower() != "training":
            return False
        if row.get("category") not in targets:
            return False
        if int(row.get("source_object_count", 0)) != 1:
            return False
        if int(row.get("crop_bytes", 0)) < min_crop_bytes:
            return False
        if clean_only and row.get("raw_dirtiness") != "오염없음":
            return False
        ratio = _area_ratio(row)
        return min_area_ratio <= ratio <= max_area_ratio
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def _preprocess(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def _material_scores(
    rows: list[dict[str, str]], manifest_root: Path, model_path: Path, batch_size: int,
) -> tuple[list[dict[str, str]], int]:
    if model_path.suffix.lower() == ".pt":
        return _torch_material_scores(rows, manifest_root, model_path, batch_size)
    return _onnx_material_scores(rows, manifest_root, model_path, batch_size)


def _append_scores(
    scored: list[dict[str, str]],
    rows: list[dict[str, str]],
    logits: np.ndarray,
) -> None:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    for row, prediction, probability in zip(rows, predictions, probabilities):
        item = dict(row)
        truth = int(row["material"])
        item["_predicted"] = str(int(prediction))
        item["_truth_confidence"] = str(float(probability[truth]))
        item["_misclassified"] = "1" if int(prediction) != truth else "0"
        scored.append(item)


def _onnx_material_scores(
    rows: list[dict[str, str]], manifest_root: Path, model_path: Path, batch_size: int,
) -> tuple[list[dict[str, str]], int]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_size = model_input.shape[-1]
    if not isinstance(input_size, int):
        input_size = 320
    output_names = {output.name for output in session.get_outputs()}
    if "material" not in output_names:
        raise RuntimeError("ONNX 모델에 material 출력이 없습니다.")

    scored: list[dict[str, str]] = []
    unreadable = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        tensors = []
        readable_rows = []
        for row in batch_rows:
            try:
                tensors.append(_preprocess(manifest_root / row["filepath"], input_size))
                readable_rows.append(row)
            except FileNotFoundError:
                unreadable += 1
        if not tensors:
            continue
        logits = session.run(
            ["material"], {model_input.name: np.stack(tensors).astype(np.float32)},
        )[0]
        _append_scores(scored, readable_rows, logits)
    return scored, unreadable


def _torch_material_scores(
    rows: list[dict[str, str]], manifest_root: Path, model_path: Path, batch_size: int,
) -> tuple[list[dict[str, str]], int]:
    import torch

    from scripts.train_verifier import CropVerifier

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    input_size = int(checkpoint.get("input_size", 320))
    model = CropVerifier(checkpoint.get("backbone", "mobilenet_v3_small"), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"scoring device={device} candidates={len(rows):,}", flush=True)

    scored: list[dict[str, str]] = []
    unreadable = 0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            tensors = []
            readable_rows = []
            for row in batch_rows:
                try:
                    tensors.append(_preprocess(
                        manifest_root / row["filepath"], input_size,
                    ))
                    readable_rows.append(row)
                except FileNotFoundError:
                    unreadable += 1
            if not tensors:
                continue
            tensor = torch.from_numpy(np.stack(tensors)).to(device=device, dtype=torch.float32)
            logits = model(tensor)[0].detach().cpu().numpy()
            _append_scores(scored, readable_rows, logits)
    return scored, unreadable


def _stable_hash(row: dict[str, str], seed: int) -> str:
    value = f"{seed}|{row.get('source_id')}|{row.get('filepath')}"
    return hashlib.sha256(value.encode()).hexdigest()


def _rank(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            -int(row["_misclassified"]),
            float(row["_truth_confidence"]),
            _stable_hash(row, seed),
        ),
    )


def _round_robin_area(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    groups = defaultdict(list)
    for row in rows:
        groups[_area_bin(row)].append(row)
    queues = {
        key: deque(_rank(values, seed))
        for key, values in sorted(groups.items())
    }
    selected = []
    while queues and len(selected) < limit:
        next_queues = {}
        for key, queue in queues.items():
            if queue and len(selected) < limit:
                selected.append(queue.popleft())
            if queue:
                next_queues[key] = queue
        queues = next_queues
    return selected


def select_scored(
    rows: list[dict[str, str]], targets: list[str], per_class: int, seed: int,
) -> list[dict[str, str]]:
    selected = []
    for category in targets:
        candidates = [row for row in rows if row["category"] == category]
        chosen = _round_robin_area(candidates, min(per_class, len(candidates)), seed)
        if not chosen:
            raise RuntimeError(f"선별 가능한 {category} training 샘플이 없습니다.")
        selected.extend(chosen)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", action="append", choices=CLASS_NAMES)
    parser.add_argument("--per-class", type=int, default=4_000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--min-crop-bytes", type=int, default=12_000)
    parser.add_argument("--min-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-area-ratio", type=float, default=0.90)
    parser.add_argument("--include-dirty", action="store_true")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    targets = args.target or ["paper", "styrofoam"]
    if args.per_class < 1 or args.batch < 1:
        parser.error("per-class와 batch는 양수여야 합니다.")
    if args.output.parent.resolve() != args.manifest.parent.resolve():
        parser.error("filepath 보존을 위해 output은 입력 manifest와 같은 폴더여야 합니다.")

    with args.manifest.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = [
            row for row in reader
            if eligible(
                row, set(targets), args.min_crop_bytes,
                args.min_area_ratio, args.max_area_ratio, not args.include_dirty,
            )
        ]
    if not fieldnames:
        raise SystemExit("[ERROR] manifest header가 없습니다.")

    scored, unreadable = _material_scores(rows, args.manifest.parent, args.model, args.batch)
    selected = select_scored(scored, targets, args.per_class, args.seed)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    temp.replace(args.output)

    summary = {
        "manifest": str(args.manifest),
        "model": str(args.model),
        "eligible": dict(Counter(row["category"] for row in scored)),
        "selected": dict(Counter(row["category"] for row in selected)),
        "selected_misclassified": dict(Counter(
            row["category"] for row in selected if row["_misclassified"] == "1"
        )),
        "selected_area_bins": dict(Counter(
            f"{row['category']}/{_area_bin(row)}" for row in selected
        )),
        "unreadable": unreadable,
        "clean_only": not args.include_dirty,
        "min_crop_bytes": args.min_crop_bytes,
        "area_ratio": [args.min_area_ratio, args.max_area_ratio],
        "per_class": args.per_class,
        "seed": args.seed,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
