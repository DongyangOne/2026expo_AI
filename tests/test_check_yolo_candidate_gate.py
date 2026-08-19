from scripts.check_yolo_candidate_gate import check_gate


def _hardware(positive: float, negative: float, high_false_positive: int = 0):
    return {
        "thresholds": {
            "0.25": {
                "positive_accuracy": positive,
                "negative_specificity": negative,
                "outcomes": {},
            },
            "0.55": {
                "positive_accuracy": positive,
                "negative_specificity": negative,
                "outcomes": {"negative_false_positive": high_false_positive},
            },
        }
    }


def test_gate_passes_only_when_original_and_hardware_checks_pass(tmp_path):
    report = check_gate(
        {"map50_95": 0.88, "recall": 0.84},
        {"map50_95": 0.875, "recall": 0.835},
        _hardware(0.54, 0.50),
        _hardware(0.60, 0.67),
        output_path=tmp_path / "gate.json",
    )
    assert report["passed"] is True
    assert all(report["checks"].values())


def test_gate_rejects_catastrophic_forgetting(tmp_path):
    report = check_gate(
        {"map50_95": 0.88, "recall": 0.84},
        {"map50_95": 0.86, "recall": 0.81},
        _hardware(0.54, 0.50),
        _hardware(0.70, 0.83),
        output_path=tmp_path / "gate.json",
    )
    assert report["passed"] is False
    assert report["checks"]["base_map50_95_preserved"] is False
    assert report["checks"]["base_recall_preserved"] is False
