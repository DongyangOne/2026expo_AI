"""최대 crop manifest에서 균형 잡힌 9종 최종 학습 manifest를 선별한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)
RARE_CLASSES = {"battery", "fluorescent"}


def _stable_score(row: dict[str, str], seed: int) -> str:
    return hashlib.sha256(
        f"{seed}|{row.get('split')}|{row.get('source_id')}|{row.get('filepath')}".encode()
    ).hexdigest()


def _area_bin(row: dict[str, str]) -> str:
    width = max(1.0, float(row.get("source_width", 1)))
    height = max(1.0, float(row.get("source_height", 1)))
    area = float(row.get("source_bbox_w", 0)) * float(row.get("source_bbox_h", 0))
    ratio = area / (width * height)
    if ratio < 0.15:
        return "small"
    if ratio < 0.45:
        return "medium"
    return "large"


def _status_priority(row: dict[str, str]) -> int:
    label = int(row.get("label", -1) or -1)
    foreign = int(row.get("foreign_material", -1) or -1)
    if label == 1 or foreign == 1:
        return 0
    if label >= 0 or foreign >= 0:
        return 1
    if int(row.get("dent", -1) or -1) == 1:
        return 2
    return 3


def _valid(row: dict[str, str], min_crop_bytes: int) -> bool:
    try:
        if int(row.get("source_object_count", 0)) != 1:
            return False
        if int(row.get("crop_bytes", 0)) < min_crop_bytes:
            return False
        width = float(row.get("source_width", 0))
        height = float(row.get("source_height", 0))
        x = float(row.get("source_bbox_x", -1))
        y = float(row.get("source_bbox_y", -1))
        box_w = float(row.get("source_bbox_w", 0))
        box_h = float(row.get("source_bbox_h", 0))
        if min(width, height, box_w, box_h) <= 1 or min(x, y) < 0:
            return False
        if x + box_w > width + 2 or y + box_h > height + 2:
            return False
        ratio = box_w * box_h / (width * height)
        return 0.02 <= ratio <= 0.95
    except (TypeError, ValueError):
        return False


def _diverse(
    rows: list[dict[str, str]], seed: int, *, balance_status: bool = False,
) -> list[dict[str, str]]:
    groups = defaultdict(list)
    for row in rows:
        if balance_status:
            key = (
                int(row.get("label", -1) or -1),
                int(row.get("foreign_material", -1) or -1),
                _area_bin(row), row.get("dent", "-1"),
                row.get("raw_dirtiness", "unknown"),
            )
        else:
            key = (
                _status_priority(row), _area_bin(row), row.get("dent", "-1"),
                row.get("raw_dirtiness", "unknown"),
            )
        groups[key].append(row)
    queues = {}
    for key, values in groups.items():
        values.sort(key=lambda row: _stable_score(row, seed))
        queues[key] = deque(values)

    ordered = []
    if balance_status:
        keys = sorted(
            queues,
            key=lambda key: hashlib.sha256(f"{seed}|{key}".encode()).hexdigest(),
        )
        while keys:
            next_keys = []
            for key in keys:
                queue = queues[key]
                if queue:
                    ordered.append(queue.popleft())
                if queue:
                    next_keys.append(key)
            keys = next_keys
        return ordered

    priorities = sorted({_status_priority(row) for row in rows})
    for priority in priorities:
        keys = sorted(
            [key for key in queues if key[0] == priority],
            key=lambda key: hashlib.sha256(f"{seed}|{key}".encode()).hexdigest(),
        )
        while keys:
            next_keys = []
            for key in keys:
                queue = queues[key]
                if queue:
                    ordered.append(queue.popleft())
                if queue:
                    next_keys.append(key)
            keys = next_keys
    return ordered


def select_manifest(
    input_manifest: Path,
    output_manifest: Path,
    train_per_class: int,
    val_per_class: int,
    min_crop_bytes: int,
    seed: int,
) -> dict:
    grouped = defaultdict(list)
    invalid = 0
    fieldnames = None
    seen = set()
    with input_manifest.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        for row in reader:
            category = row.get("category", "")
            split = row.get("split", "").lower()
            identity = (split, row.get("source_id", ""), row.get("filepath", ""))
            if category not in CLASS_NAMES or split not in {"training", "validation"}:
                continue
            if identity in seen or not _valid(row, min_crop_bytes):
                invalid += 1
                continue
            seen.add(identity)
            grouped[(split, category)].append(row)
    if not fieldnames:
        raise ValueError("input manifest has no header")

    selected = []
    counts = Counter()
    status_counts = Counter()
    for split in ("training", "validation"):
        for category in CLASS_NAMES:
            rows = _diverse(
                grouped[(split, category)], seed,
                balance_status=split == "validation",
            )
            default_cap = train_per_class if split == "training" else val_per_class
            cap = len(rows) if category in RARE_CLASSES else min(default_cap, len(rows))
            chosen = rows[:cap]
            if not chosen:
                raise RuntimeError(f"no selected rows for {split}/{category}")
            selected.extend(chosen)
            counts[f"{split}/{category}"] = len(chosen)
            for row in chosen:
                for task in ("label", "foreign_material"):
                    value = int(row.get(task, -1) or -1)
                    if value >= 0:
                        status_counts[f"{task}/{split}/{value}"] += 1

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temp = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    temp.replace(output_manifest)
    return {
        "input_rows_valid": sum(len(rows) for rows in grouped.values()),
        "invalid_or_duplicate_rows": invalid,
        "selected_rows": len(selected),
        "counts": dict(counts),
        "status_counts": dict(status_counts),
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
        "rare_classes_use_all": sorted(RARE_CLASSES),
        "min_crop_bytes": min_crop_bytes,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--train-per-class", type=int, default=10_000)
    parser.add_argument("--val-per-class", type=int, default=2_000)
    parser.add_argument("--min-crop-bytes", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    summary = select_manifest(
        Path(args.input_manifest), Path(args.output_manifest),
        args.train_per_class, args.val_per_class, args.min_crop_bytes, args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
