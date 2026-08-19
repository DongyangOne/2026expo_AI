"""Build a deterministic mixed YOLO train list without copying source images.

The commercial dataset improves clean, single-object material recognition, but
fine-tuning on it alone can forget the original camera/background distribution.
This script mixes it with a class-balanced replay sampled from the original
YOLO dataset.  Only images with exactly one valid object are eligible for
replay, matching the kiosk's one-item-at-a-time operating contract.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
from collections import Counter
from pathlib import Path


CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)
RARE_CLASS_IDS = {7, 8}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _stable_score(path: Path, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}|{path.as_posix()}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _parse_single_object_label(label_path: Path) -> tuple[int | None, str | None]:
    try:
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError):
        return None, "unreadable_label"
    if not lines:
        return None, "empty_label"
    if len(lines) != 1:
        return None, "not_single_object"

    parts = lines[0].split()
    if len(parts) != 5:
        return None, "invalid_column_count"
    try:
        raw_class, cx, cy, width, height = (float(value) for value in parts)
    except ValueError:
        return None, "non_numeric_label"
    class_id = int(raw_class)
    if raw_class != class_id or not 0 <= class_id < len(CLASS_NAMES):
        return None, "invalid_class_id"
    values = (cx, cy, width, height)
    if not all(math.isfinite(value) for value in values):
        return None, "non_finite_bbox"
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return None, "invalid_bbox"
    if cx - width / 2 < -0.01 or cx + width / 2 > 1.01:
        return None, "bbox_outside_image"
    if cy - height / 2 < -0.01 or cy + height / 2 > 1.01:
        return None, "bbox_outside_image"
    return class_id, None


def _push_smallest(
    heap: list[tuple[int, str]], path: Path, score: int, target: int
) -> None:
    """Keep paths with the smallest stable scores using bounded memory."""
    item = (-score, path.as_posix())
    if len(heap) < target:
        heapq.heappush(heap, item)
        return
    if score < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _find_train_dirs(dataset_dir: Path) -> tuple[Path, Path]:
    layouts = (
        (dataset_dir / "train" / "images", dataset_dir / "train" / "labels"),
        (dataset_dir / "images" / "train", dataset_dir / "labels" / "train"),
    )
    for image_dir, label_dir in layouts:
        if image_dir.is_dir() and label_dir.is_dir():
            return image_dir, label_dir
    raise FileNotFoundError(
        "base dataset must contain train/images + train/labels or "
        "images/train + labels/train"
    )


def _load_commercial_entries(path: Path) -> list[str]:
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"commercial train list is empty: {path}")
    return entries


def _trusted_negative_entries(dataset_dir: Path | None, repeats: int) -> list[str]:
    if dataset_dir is None or repeats == 0:
        return []
    image_dir, label_dir = _find_train_dirs(dataset_dir)
    negatives = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file() and not label_path.read_text(encoding="utf-8").strip():
            negatives.extend([image_path.as_posix()] * repeats)
    if not negatives:
        raise RuntimeError(f"no trusted negative frames found in {dataset_dir}")
    return negatives


def prepare_mixed_replay_list(
    *,
    base_dataset_dir: Path,
    commercial_list: Path,
    output_dir: Path,
    validation_images: str,
    target_per_class: int,
    rare_target_per_class: int,
    seed: int,
    trusted_negative_dir: Path | None = None,
    trusted_negative_repeats: int = 0,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir, label_dir = _find_train_dirs(base_dataset_dir)

    targets = {
        class_id: rare_target_per_class if class_id in RARE_CLASS_IDS else target_per_class
        for class_id in range(len(CLASS_NAMES))
    }
    selected_heaps: dict[int, list[tuple[int, str]]] = {
        class_id: [] for class_id in range(len(CLASS_NAMES))
    }
    valid_counts = Counter()
    rejected_counts = Counter()
    scanned = 0

    for image_path in image_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        scanned += 1
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            rejected_counts["missing_label"] += 1
            continue
        class_id, reason = _parse_single_object_label(label_path)
        if reason:
            rejected_counts[reason] += 1
            continue
        assert class_id is not None
        valid_counts[CLASS_NAMES[class_id]] += 1
        _push_smallest(
            selected_heaps[class_id],
            image_path,
            _stable_score(image_path, seed),
            targets[class_id],
        )

    missing = [
        CLASS_NAMES[class_id]
        for class_id, heap in selected_heaps.items()
        if not heap
    ]
    if missing:
        raise RuntimeError("base replay has no valid images for: " + ", ".join(missing))

    rng = random.Random(seed)
    replay_entries: list[str] = []
    replay_counts = Counter()
    replay_unique_counts = Counter()
    for class_id in range(len(CLASS_NAMES)):
        class_name = CLASS_NAMES[class_id]
        unique_paths = [path for _, path in sorted(selected_heaps[class_id], reverse=True)]
        rng.shuffle(unique_paths)
        replay_unique_counts[class_name] = len(unique_paths)
        target = targets[class_id]
        entries = [unique_paths[index % len(unique_paths)] for index in range(target)]
        rng.shuffle(entries)
        replay_entries.extend(entries)
        replay_counts[class_name] = len(entries)

    commercial_entries = _load_commercial_entries(commercial_list)
    negative_entries = _trusted_negative_entries(
        trusted_negative_dir, trusted_negative_repeats
    )
    mixed_entries = replay_entries + commercial_entries + negative_entries
    rng.shuffle(mixed_entries)

    train_list = output_dir / "train_mixed.txt"
    train_list.write_text("\n".join(mixed_entries) + "\n", encoding="utf-8")
    dataset_yaml = output_dir / "dataset_mixed.yaml"
    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        f"train: {train_list.as_posix()}",
        f"val: {validation_images}",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    dataset_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    summary = {
        "base_dataset_dir": str(base_dataset_dir),
        "commercial_list": str(commercial_list),
        "single_object_only": True,
        "base_images_scanned": scanned,
        "base_valid_by_class": dict(valid_counts),
        "base_rejected": dict(rejected_counts),
        "replay_unique_by_class": dict(replay_unique_counts),
        "replay_entries_by_class": dict(replay_counts),
        "replay_entries": len(replay_entries),
        "commercial_entries": len(commercial_entries),
        "trusted_negative_entries": len(negative_entries),
        "mixed_entries": len(mixed_entries),
        "target_per_class": target_per_class,
        "rare_target_per_class": rare_target_per_class,
        "trusted_negative_repeats": trusted_negative_repeats,
        "seed": seed,
        "train_list": str(train_list),
        "dataset_yaml": str(dataset_yaml),
    }
    (output_dir / "mixed_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset-dir", required=True, type=Path)
    parser.add_argument("--commercial-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-images", required=True)
    parser.add_argument("--target-per-class", type=int, default=20_000)
    parser.add_argument("--rare-target-per-class", type=int, default=5_000)
    parser.add_argument("--trusted-negative-dir", type=Path)
    parser.add_argument("--trusted-negative-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if min(args.target_per_class, args.rare_target_per_class) < 1:
        parser.error("class targets must be positive")
    if args.trusted_negative_repeats < 0:
        parser.error("trusted-negative-repeats must be non-negative")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    prepare_mixed_replay_list(
        base_dataset_dir=arguments.base_dataset_dir,
        commercial_list=arguments.commercial_list,
        output_dir=arguments.output_dir,
        validation_images=arguments.validation_images,
        target_per_class=arguments.target_per_class,
        rare_target_per_class=arguments.rare_target_per_class,
        seed=arguments.seed,
        trusted_negative_dir=arguments.trusted_negative_dir,
        trusted_negative_repeats=arguments.trusted_negative_repeats,
    )
