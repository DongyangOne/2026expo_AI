"""검증기 crop manifest를 학습 전에 점검한다."""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = [
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
]


SOURCE_REFERENCE_FIELDS = {
    "source_path_b64", "source_bbox_x", "source_bbox_y",
    "source_bbox_w", "source_bbox_h", "source_width", "source_height",
}


def _decode_source_path(value: str) -> Path:
    return Path(os.fsdecode(base64.urlsafe_b64decode(value.encode("ascii"))))


def audit_manifest(
    manifest_path: Path,
    require_masked_status: bool = False,
    require_source_references: bool = False,
) -> dict:
    with open(manifest_path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    root = manifest_path.parent
    splits = {row["split"].lower() for row in rows}
    materials = {int(row["material"]) for row in rows}
    source_splits: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    missing_images = 0
    missing_source_images = 0
    invalid_source_references = 0
    object_counts = set()

    source_fields_missing = require_source_references and (
        not rows or not SOURCE_REFERENCE_FIELDS.issubset(rows[0])
    )

    for row in rows:
        split = row["split"].lower()
        material = int(row["material"])
        source_splits[row["source_id"]].add(split)
        counts[(split, CLASS_NAMES[material])] += 1
        missing_images += not (root / row["filepath"]).is_file()
        if row.get("source_object_count", ""):
            object_counts.add(int(row["source_object_count"]))
        if require_source_references and not source_fields_missing:
            try:
                source_path = _decode_source_path(row["source_path_b64"])
                source_values = [
                    float(row[name]) for name in (
                        "source_bbox_x", "source_bbox_y", "source_bbox_w", "source_bbox_h",
                        "source_width", "source_height",
                    )
                ]
                if source_values[2] <= 0 or source_values[3] <= 0:
                    invalid_source_references += 1
                elif source_values[4] <= 0 or source_values[5] <= 0:
                    invalid_source_references += 1
                missing_source_images += not source_path.is_file()
            except (KeyError, ValueError, TypeError, binascii.Error):
                invalid_source_references += 1

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
            f"unexpected pre-labeled status values: label={label_values}, foreign={foreign_values}"
        )
    if source_fields_missing:
        problems.append(
            f"source reference fields missing: {sorted(SOURCE_REFERENCE_FIELDS - set(rows[0] if rows else []))}"
        )
    if invalid_source_references:
        problems.append(f"invalid source references: {invalid_source_references}")
    if missing_source_images:
        problems.append(f"missing source images: {missing_source_images}")

    return {
        "ok": not problems,
        "rows": len(rows),
        "missing_images": missing_images,
        "missing_source_images": missing_source_images,
        "invalid_source_references": invalid_source_references,
        "split_overlap_sources": overlap,
        "label_values": label_values,
        "foreign_material_values": foreign_values,
        "source_object_counts": sorted(object_counts),
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
    parser.add_argument("--require-single-object", action="store_true")
    parser.add_argument("--require-source-references", action="store_true")
    args = parser.parse_args()

    result = audit_manifest(
        Path(args.manifest), args.require_masked_status, args.require_source_references
    )
    if args.require_single_object and result["source_object_counts"] != [1]:
        result["ok"] = False
        result["problems"].append(
            f"single-object requirement failed: {result['source_object_counts']}"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
