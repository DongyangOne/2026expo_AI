import pytest

from scripts.operational_teacher_contract import (
    ADJUDICATION_PROMPT,
    JSON_ONLY_SUFFIX,
    PROMPTS,
    REQUEST_CONTRACT,
    build_teacher_contract,
    load_known_audit,
    render_prompt,
)


def test_contract_binds_full_rendered_prompts_model_digest_and_request():
    contract, digest = build_teacher_contract("teacher-model", "A" * 64)
    assert contract["model_identifier"] == "teacher-model"
    assert contract["model_digest"] == "a" * 64
    assert contract["prompts"]["initial"] == list(PROMPTS)
    assert contract["prompts"]["adjudication"] == ADJUDICATION_PROMPT
    assert contract["rendered_prompts"]["initial"] == [
        render_prompt(prompt) for prompt in PROMPTS
    ]
    assert contract["rendered_prompts"]["adjudication"].startswith("/no_think\n")
    assert contract["rendered_prompts"]["adjudication"].endswith(JSON_ONLY_SUFFIX)
    assert contract["request"] == REQUEST_CONTRACT
    assert len(digest) == 64


@pytest.mark.parametrize("digest", ["", "a" * 63, "g" * 64])
def test_contract_rejects_non_exact_model_digest(digest):
    with pytest.raises(ValueError, match="64 hexadecimal"):
        build_teacher_contract("teacher-model", digest)


@pytest.mark.parametrize(
    "payload",
    [
        [], {"bad": {"split": "train"}},
        {"A" * 64: {"split": "train"}, "a" * 64: {"split": "validation"}},
        {"a" * 64: {}}, {"a" * 64: None}, {"a" * 64: False},
        {"a" * 64: {"split": "val"}},
    ],
)
def test_known_audit_is_strict_and_rejects_ambiguous_or_malformed(
    tmp_path, payload
):
    import json

    path = tmp_path / "known.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_known_audit(path)


def test_known_audit_normalizes_uppercase_sha(tmp_path):
    import json

    path = tmp_path / "known.json"
    path.write_text(json.dumps({"A" * 64: {"split": "train"}}), encoding="utf-8")
    assert set(load_known_audit(path)) == {"a" * 64}
