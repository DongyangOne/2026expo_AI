import torch
import torch.nn as nn
import pytest

import scripts.train_verifier as trainer


class _SmallVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
        self.material_head = nn.Linear(4, 3)
        self.status_head = nn.Linear(4, 2)


def test_default_class_weights_preserve_legacy_inverse_behavior():
    values = [0, 0, 1, 2, 2, 2]

    weights = trainer.class_weights(values, classes=3)

    torch.testing.assert_close(
        weights,
        torch.tensor([1.0, 2.0, 2.0 / 3.0], dtype=torch.float32),
    )


def test_optimizer_defaults_to_base_lr_and_supports_backbone_head_lrs():
    model = _SmallVerifier()

    default_optimizer = trainer.build_optimizer(model, lr=1e-3)
    split_optimizer = trainer.build_optimizer(
        model, lr=1e-3, backbone_lr=1e-4, head_lr=2e-3,
    )

    assert [(group["name"], group["lr"]) for group in default_optimizer.param_groups] == [
        ("backbone", 1e-3), ("heads", 1e-3),
    ]
    assert [(group["name"], group["lr"]) for group in split_optimizer.param_groups] == [
        ("backbone", 1e-4), ("heads", 2e-3),
    ]
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    assert {id(parameter) for parameter in split_optimizer.param_groups[0]["params"]} == backbone_ids
    assert not backbone_ids.intersection(
        id(parameter) for parameter in split_optimizer.param_groups[1]["params"]
    )


def test_effective_number_weights_are_finite_and_mean_normalized():
    weights = trainer.class_weights(
        [0] * 100 + [1] * 5,
        classes=3,
        mode="effective-number",
        beta=0.9999,
    )

    assert torch.isfinite(weights).all()
    assert torch.all(weights > 0)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))


def test_label_smoothing_is_passed_to_each_cross_entropy_loss():
    criteria = trainer.build_criteria(
        {
            "material": torch.ones(3),
            "dent": None,
        },
        torch.device("cpu"),
        label_smoothing=0.12,
    )

    assert set(criteria) == {"material", "dent"}
    assert all(criterion.label_smoothing == 0.12 for criterion in criteria.values())
    assert all(criterion.reduction == "none" for criterion in criteria.values())


def test_training_option_validation_rejects_invalid_values():
    with pytest.raises(ValueError, match="lr must be finite and positive"):
        trainer.resolve_learning_rates(0.0)
    with pytest.raises(ValueError, match="label smoothing"):
        trainer.build_criteria({}, torch.device("cpu"), label_smoothing=1.0)
    with pytest.raises(ValueError, match="class weight beta"):
        trainer.class_weights(
            [0, 1], classes=2, mode="effective-number", beta=1.0,
        )
