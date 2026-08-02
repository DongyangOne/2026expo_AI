import pytest

from scripts.evaluate_verifier import classification_metrics, external_material_id


def test_classification_metrics_ignores_absent_classes_in_macro_f1():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 3)

    assert metrics["support"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    assert metrics["macro_f1_present_classes"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["per_class"]["2"]["support"] == 0


def test_classification_metrics_handles_empty_labels():
    metrics = classification_metrics([], [], 2)

    assert metrics["accuracy"] is None
    assert metrics["macro_f1_present_classes"] is None


def test_external_material_contract_merges_pet_into_plastic():
    assert external_material_id(1) == 3
    assert external_material_id(3) == 3
    assert external_material_id(5) == 5
