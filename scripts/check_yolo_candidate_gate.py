"""Apply original-validation and hardware-holdout deployment gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_gate(
    baseline_validation: dict,
    candidate_validation: dict,
    baseline_hardware: dict,
    candidate_hardware: dict,
    *,
    output_path: Path,
    threshold: str = "0.25",
    high_threshold: str = "0.55",
    max_original_drop: float = 0.01,
    min_hardware_gain: float = 0.05,
) -> dict:
    baseline_hw = baseline_hardware["thresholds"][threshold]
    candidate_hw = candidate_hardware["thresholds"][threshold]
    candidate_high = candidate_hardware["thresholds"][high_threshold]
    checks = {
        "base_map50_95_preserved": (
            candidate_validation["map50_95"]
            >= baseline_validation["map50_95"] - max_original_drop
        ),
        "base_recall_preserved": (
            candidate_validation["recall"]
            >= baseline_validation["recall"] - max_original_drop
        ),
        "hardware_positive_accuracy_gain": (
            candidate_hw["positive_accuracy"]
            >= baseline_hw["positive_accuracy"] + min_hardware_gain
        ),
        "hardware_negative_specificity_improved": (
            candidate_hw["negative_specificity"]
            > baseline_hw["negative_specificity"]
        ),
        "no_high_confidence_negative_false_positive": (
            candidate_high.get("outcomes", {}).get("negative_false_positive", 0) == 0
        ),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "limits": {
            "max_original_drop": max_original_drop,
            "min_hardware_gain": min_hardware_gain,
            "hardware_threshold": float(threshold),
            "high_confidence_threshold": float(high_threshold),
        },
        "baseline": {
            "map50_95": baseline_validation["map50_95"],
            "recall": baseline_validation["recall"],
            "hardware_positive_accuracy": baseline_hw["positive_accuracy"],
            "hardware_negative_specificity": baseline_hw["negative_specificity"],
        },
        "candidate": {
            "map50_95": candidate_validation["map50_95"],
            "recall": candidate_validation["recall"],
            "hardware_positive_accuracy": candidate_hw["positive_accuracy"],
            "hardware_negative_specificity": candidate_hw["negative_specificity"],
            "high_confidence_negative_false_positive": candidate_high.get(
                "outcomes", {}
            ).get("negative_false_positive", 0),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-validation", required=True, type=Path)
    parser.add_argument("--candidate-validation", required=True, type=Path)
    parser.add_argument("--baseline-hardware", required=True, type=Path)
    parser.add_argument("--candidate-hardware", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-original-drop", type=float, default=0.01)
    parser.add_argument("--min-hardware-gain", type=float, default=0.05)
    args = parser.parse_args()
    report = check_gate(
        json.loads(args.baseline_validation.read_text(encoding="utf-8")),
        json.loads(args.candidate_validation.read_text(encoding="utf-8")),
        json.loads(args.baseline_hardware.read_text(encoding="utf-8")),
        json.loads(args.candidate_hardware.read_text(encoding="utf-8")),
        output_path=args.output,
        max_original_drop=args.max_original_drop,
        min_hardware_gain=args.min_hardware_gain,
    )
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
