"""자동 VLM 상태 pseudo-label을 학습에 넣기 전에 무결성과 분포를 감사한다."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


STATUS_CATEGORIES = {"can", "pet", "paper", "plastic", "vinyl"}
LABEL_CATEGORIES = {"pet", "plastic"}
SPLITS = {"training", "validation"}
VALID_STATUS = {"-1", "0", "1"}


def audit_pseudo_status(
    manifest_path: Path,
    min_coverage: float = 0.0,
    require_ready_heads: bool = False,
    require_complete: bool = False,
    max_teacher_error_rate: float = 0.0,
) -> dict:
    with open(manifest_path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    required = {
        "split", "category", "label", "foreign_material", "status_eligible",
        "teacher_status", "teacher_confidence", "teacher_model", "teacher_rejected",
    }
    fields = set(rows[0]) if rows else set()
    problems = []
    missing_fields = sorted(required - fields)
    if not rows:
        problems.append("manifest is empty")
    if missing_fields:
        problems.append(f"pseudo status fields missing: {missing_fields}")

    targets = [row for row in rows if row.get("category") in STATUS_CATEGORIES]
    pending = errors = invalid = accepted = 0
    decisions = Counter()
    values = Counter()
    if not missing_fields:
        for row in targets:
            eligible = row["status_eligible"]
            label = row["label"]
            foreign = row["foreign_material"]
            decision = row["teacher_status"]
            rejected = row["teacher_rejected"]
            decisions[decision or "pending"] += 1
            if eligible == "":
                pending += 1
                continue
            if decision == "error":
                errors += 1
            if eligible not in {"0", "1"} or label not in VALID_STATUS or foreign not in VALID_STATUS:
                invalid += 1
                continue

            split = row["split"].lower()
            category = row["category"]
            if split not in SPLITS:
                invalid += 1
                continue
            if eligible == "1":
                accepted += 1
                expected_label_values = {"0", "1"} if category in LABEL_CATEGORIES else {"-1"}
                if (
                    rejected != "0"
                    or not row["teacher_model"]
                    or label not in expected_label_values
                    or foreign not in {"0", "1"}
                ):
                    invalid += 1
                    continue
                if category in LABEL_CATEGORIES:
                    values[("label", split, label)] += 1
                values[("foreign_material", split, foreign)] += 1
            elif rejected != "1" or label != "-1" or foreign != "-1":
                invalid += 1

    processed = len(targets) - pending
    coverage = accepted / processed if processed else 0.0
    teacher_error_rate = errors / processed if processed else 0.0
    head_ready = {}
    for head in ("label", "foreign_material"):
        head_ready[head] = all(
            values[(head, split, value)] > 0
            for split in sorted(SPLITS)
            for value in ("0", "1")
        )

    if not targets:
        problems.append("no status-category rows")
    if require_complete and pending:
        problems.append(f"unprocessed status rows: {pending}")
    if teacher_error_rate > max_teacher_error_rate:
        problems.append(
            f"teacher error rate {teacher_error_rate:.4f} exceeds maximum "
            f"{max_teacher_error_rate:.4f} ({errors}/{processed})"
        )
    if invalid:
        problems.append(f"invalid pseudo status rows: {invalid}")
    if coverage < min_coverage:
        problems.append(
            f"accepted coverage {coverage:.4f} below minimum {min_coverage:.4f}"
        )
    if require_ready_heads:
        not_ready = sorted(head for head, ready in head_ready.items() if not ready)
        if not_ready:
            problems.append(f"status heads lack both classes in both splits: {not_ready}")

    return {
        "ok": not problems,
        "rows": len(rows),
        "status_target_rows": len(targets),
        "processed_rows": processed,
        "accepted_rows": accepted,
        "accepted_coverage": round(coverage, 6),
        "pending_rows": pending,
        "teacher_errors": errors,
        "teacher_error_rate": round(teacher_error_rate, 6),
        "invalid_rows": invalid,
        "head_ready": head_ready,
        "decisions": dict(sorted(decisions.items())),
        "head_counts": {
            f"{head}/{split}/{value}": count
            for (head, split, value), count in sorted(values.items())
        },
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--require-ready-heads", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--max-teacher-error-rate", type=float, default=0.0)
    args = parser.parse_args()
    if not 0 <= args.min_coverage <= 1:
        raise SystemExit("[ERROR] --min-coverage must be between 0 and 1")
    if not 0 <= args.max_teacher_error_rate <= 1:
        raise SystemExit("[ERROR] --max-teacher-error-rate must be between 0 and 1")

    result = audit_pseudo_status(
        Path(args.manifest), args.min_coverage, args.require_ready_heads,
        args.require_complete, args.max_teacher_error_rate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
