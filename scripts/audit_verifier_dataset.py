"""검증기 crop manifest를 학습 전에 점검한다."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = [
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
]


def audit_manifest(manifest_path: Path, require_masked_status: bool = False) -> dict:
    with open(manifest_path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    root = manifest_path.parent
    splits = {row["split"].lower() for row in rows}
    materials = {int(row["material"]) for row in rows}
    source_splits: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    missing_images = 0

    for row in rows:
        split = row["split"].lower()
        material = int(row["material"])
        source_splits[row["source_id"]].add(split)
        counts[(split, CLASS_NAMES[material])] += 1
        missing_images += not (root / row["filepath"]).is_file()

    label_values = sorted({int(row.get("label", -1)) for row in rows})
    foreign_values = sorted({int(row.get("foreign_material", -1)) for row in rows})
    problems = []
    if not rows:
        problems.append("manifest is empty")
    if splits != {"training", "validation"}:
        problems.append(f"unexpected splits: {sorted(splits)}")
    if materials != set(range(len(CLASS_NAMES))):
        problems.append(f"missing material ids: {sorted(set(range(9)) - materials)}")
    if missing_images:
        problems.append(f"missing images: {missing_images}")
    overlap = sum(len(value) > 1 for value in source_splits.values())
    if overlap:
        problems.append(f"source ids crossing splits: {overlap}")
    if require_masked_status and (label_values != [-1] or foreign_values != [-1]):
        problems.append(
            f"unreviewed status labels detected: label={label_values}, foreign={foreign_values}"
        )

    return {
        "ok": not problems,
        "rows": len(rows),
        "missing_images": missing_images,
        "split_overlap_sources": overlap,
        "label_values": label_values,
        "foreign_material_values": foreign_values,
        "counts": {
            f"{split}/{class_name}": count
            for (split, class_name), count in sorted(counts.items())
        },
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-masked-status", action="store_true")
    args = parser.parse_args()

    result = audit_manifest(Path(args.manifest), args.require_masked_status)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

