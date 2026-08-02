"""작은 teacher manifest의 상태 pseudo-label을 최대 crop manifest에 결합한다."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


KEY_FIELDS = ("split", "category", "source_id")
PSEUDO_FIELDS = (
    "label", "foreign_material", "status_eligible", "teacher_status",
    "teacher_confidence", "teacher_reason", "teacher_model", "teacher_rejected",
)
PROCESSED_DECISIONS = {
    "neither", "label_only", "foreign_only", "both", "ambiguous", "exclude", "error",
}


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row.get(field, "") for field in KEY_FIELDS)


def merge_manifests(
    base_manifest: Path,
    pseudo_manifest: Path,
    output_manifest: Path,
    require_processed: int = 0,
) -> dict:
    pseudo_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    pseudo_processed = 0
    duplicate_keys = 0
    with pseudo_manifest.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = _key(row)
            if not all(key):
                continue
            if key in pseudo_by_key:
                duplicate_keys += 1
            pseudo_by_key[key] = row
            if row.get("teacher_status", "") in PROCESSED_DECISIONS:
                pseudo_processed += 1

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temp = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    matched = matched_processed = rows = 0
    decisions = Counter()
    with base_manifest.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("base manifest has no header")
        fields = list(reader.fieldnames)
        for field in PSEUDO_FIELDS:
            if field not in fields:
                fields.append(field)
        with temp.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                rows += 1
                pseudo = pseudo_by_key.get(_key(row))
                if pseudo is not None:
                    matched += 1
                    decision = pseudo.get("teacher_status", "")
                    if decision in PROCESSED_DECISIONS:
                        matched_processed += 1
                        decisions[decision] += 1
                        for field in PSEUDO_FIELDS:
                            if field in pseudo:
                                row[field] = pseudo[field]
                writer.writerow(row)

    summary = {
        "base_rows": rows,
        "pseudo_rows": len(pseudo_by_key),
        "pseudo_processed": pseudo_processed,
        "matched_rows": matched,
        "matched_processed": matched_processed,
        "duplicate_pseudo_keys": duplicate_keys,
        "decisions": dict(decisions),
    }
    if require_processed and matched_processed != require_processed:
        temp.unlink(missing_ok=True)
        raise RuntimeError(
            f"processed pseudo labels matched {matched_processed:,}, "
            f"expected {require_processed:,}"
        )
    os.replace(temp, output_manifest)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--pseudo-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--require-processed", type=int, default=0)
    args = parser.parse_args()
    summary = merge_manifests(
        Path(args.base_manifest), Path(args.pseudo_manifest), Path(args.output_manifest),
        args.require_processed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
