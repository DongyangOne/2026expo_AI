import json
import hashlib
import io
import os
import sys
from pathlib import Path

import pytest

import scripts.label_operational_captures_ollama as teacher

_REAL_OBSERVE_MODEL_DIGEST = teacher._observe_model_digest


@pytest.fixture(autouse=True)
def _trusted_ollama_tags(monkeypatch):
    monkeypatch.setattr(
        teacher,
        "_observe_model_digest",
        lambda _url, model, _timeout: "b" * 64 if model == "model-v2" else "a" * 64,
    )


def _known(root: Path, value: dict | None = None) -> Path:
    path = root / "known_audit.json"
    path.write_text(json.dumps(value or {}), encoding="utf-8")
    return path


def _queue_row(image: Path, sha256: str | None = None) -> dict:
    actual = hashlib.sha256(image.read_bytes()).hexdigest()
    return {
        "sha256": sha256 if sha256 and len(sha256) == 64 else actual,
        "image_ref": image.name,
        "timestamp": "2026-08-01T00:00:00+09:00",
        "decision": "teacher_required",
    }


def _answer(
    material: str,
    confidence: float,
    *,
    single_object: bool = True,
    foreign_material: bool = False,
    training_usable: bool = True,
    quality_reason: str = "usable",
) -> dict:
    return {
        "material": material,
        "confidence": confidence,
        "single_object": single_object,
        "foreign_material": foreign_material,
        "training_usable": training_usable,
        "quality_reason": quality_reason,
    }


def test_label_queue_records_failed_image_and_continues(tmp_path, monkeypatch):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps(row)
            for row in (_queue_row(first, "first"), _queue_row(second, "second"))
        )
        + "\n",
        encoding="utf-8",
    )

    calls = {b"first": 0, b"second": 0}

    def fake_request(url, model, image, prompt, timeout):
        calls[image] += 1
        if image == b"first":
            raise ValueError("empty model content")
        return _answer("can", 0.9)

    monkeypatch.setattr(teacher, "_request", fake_request)
    output = tmp_path / "labels.jsonl"
    summary = teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="test-model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    by_sha = {row["sha256"]: row for row in rows}
    first_row = by_sha[hashlib.sha256(b"first").hexdigest()]
    second_row = by_sha[hashlib.sha256(b"second").hexdigest()]
    assert first_row["consensus"] is False
    assert first_row["minimum_confidence"] == 0
    assert first_row["errors"] == ["ValueError: empty model content"]
    assert second_row["consensus"] is True
    rendered = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in rendered
    assert "client_id" not in rendered
    assert "deployed" not in rendered
    assert "verifier" not in rendered
    assert all(not Path(row["image_ref"]).is_absolute() for row in rows)
    assert summary == {
        "schema_version": teacher.TEACHER_LABEL_SCHEMA_VERSION,
        "teacher_contract_sha256": summary["teacher_contract_sha256"],
        "queued": 2,
        "completed": 2,
        "reused": 0,
        "newly_labeled": 1,
        "retried": 0,
        "image_error": 0,
        "consensus": 1,
        "high_confidence_consensus": 1,
        "high_confidence_training_usable_consensus": 1,
    }


def test_label_queue_retries_error_checkpoint_without_duplicating_rows(tmp_path, monkeypatch):
    image = tmp_path / "retry.jpg"
    image.write_bytes(b"retry")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(_queue_row(image, "retry-sha")) + "\n", encoding="utf-8")
    output = tmp_path / "labels.jsonl"
    output.write_text(
        json.dumps(
            {
                "sha256": "retry-sha",
                "image_path": str(image),
                "model": "test-model",
                "passes": [],
                "errors": ["ValueError: truncated"],
                "consensus": False,
                "minimum_confidence": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls = 0

    def fake_request(url, model, image_bytes, prompt, timeout):
        nonlocal calls
        calls += 1
        return _answer("paper", 0.95)

    monkeypatch.setattr(teacher, "_request", fake_request)
    summary = teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="test-model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert calls == 2
    assert len(rows) == 1
    assert rows[0]["errors"] == []
    assert rows[0]["consensus"] is True
    assert summary["high_confidence_consensus"] == 1


def test_label_queue_uses_third_pass_only_to_adjudicate_disagreement(tmp_path, monkeypatch):
    image = tmp_path / "ambiguous.jpg"
    image.write_bytes(b"ambiguous")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(_queue_row(image, "ambiguous-sha")) + "\n", encoding="utf-8")
    answers = iter(
        [
            _answer("paper", 0.92),
            _answer("vinyl", 0.88),
            _answer("paper", 0.90),
        ]
    )

    monkeypatch.setattr(teacher, "_request", lambda *args: next(answers))
    output = tmp_path / "labels.jsonl"
    summary = teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="test-model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert len(row["passes"]) == 3
    assert row["consensus"] is True
    assert row["consensus_decision"] == {
        "material": "paper",
        "single_object": True,
        "foreign_material": False,
        "training_usable": True,
        "quality_reason": "usable",
        "votes": 2,
        "pass_count": 3,
    }
    assert row["minimum_confidence"] == 0.9
    assert summary["high_confidence_consensus"] == 1


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_request_accepts_schema_json_in_thinking_on_normal_stop(monkeypatch):
    body = {
        "model": "model",
        "done_reason": "stop",
        "message": {
            "content": "",
            "thinking": json.dumps(
                {
                    "material": "paper",
                    "confidence": 0.91,
                    "single_object": True,
                    "foreign_material": False,
                    "training_usable": True,
                    "quality_reason": "usable",
                }
            ),
        },
    }
    monkeypatch.setattr(teacher, "_open_no_redirect", lambda *_args, **_kwargs: _Response())
    monkeypatch.setattr(teacher.json, "load", lambda _response: body)

    result = teacher._request("http://teacher", "model", b"image", "prompt", 10)

    assert result["material"] == "paper"
    assert result["single_object"] is True


def test_request_rejects_incoherent_training_quality_fields(monkeypatch):
    body = {
        "model": "model",
        "done_reason": "stop",
        "message": {
            "content": json.dumps(
                _answer(
                    "paper",
                    0.99,
                    training_usable=False,
                    quality_reason="usable",
                )
            )
        },
    }
    monkeypatch.setattr(
        teacher, "_open_no_redirect", lambda *_args, **_kwargs: _Response()
    )
    monkeypatch.setattr(teacher.json, "load", lambda _response: body)

    with pytest.raises(ValueError, match="training_usable and quality_reason disagree"):
        teacher._request("http://teacher", "model", b"image", "prompt", 10)


def test_quality_decision_is_part_of_exact_tuple_consensus(tmp_path, monkeypatch):
    image = tmp_path / "quality-disagreement.jpg"
    image.write_bytes(b"quality-disagreement")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(_queue_row(image, "quality-disagreement-sha")) + "\n",
        encoding="utf-8",
    )
    answers = iter(
        [
            _answer("paper", 0.94),
            _answer(
                "paper",
                0.93,
                training_usable=False,
                quality_reason="person_occlusion_or_dominance",
            ),
            _answer("paper", 0.92),
        ]
    )
    monkeypatch.setattr(teacher, "_request", lambda *args: next(answers))

    output = tmp_path / "labels.jsonl"
    summary = teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="test-model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert len(row["passes"]) == 3
    assert row["schema_version"] == teacher.TEACHER_LABEL_SCHEMA_VERSION
    assert row["consensus_decision"]["training_usable"] is True
    assert row["consensus_decision"]["quality_reason"] == "usable"
    assert summary["high_confidence_training_usable_consensus"] == 1


def test_high_confidence_unusable_consensus_is_preserved_for_downstream_rejection(
    tmp_path, monkeypatch
):
    image = tmp_path / "occluded.jpg"
    image.write_bytes(b"occluded")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(_queue_row(image, "occluded-sha")) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        teacher,
        "_request",
        lambda *args: _answer(
            "plastic",
            0.96,
            training_usable=False,
            quality_reason="person_occlusion_or_dominance",
        ),
    )

    output = tmp_path / "labels.jsonl"
    summary = teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="test-model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["consensus"] is True
    assert row["consensus_decision"]["training_usable"] is False
    assert summary["high_confidence_consensus"] == 1
    assert summary["high_confidence_training_usable_consensus"] == 0


def test_legacy_completed_checkpoint_is_relabelled_fail_closed(tmp_path, monkeypatch):
    image = tmp_path / "legacy.jpg"
    image.write_bytes(b"legacy")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(_queue_row(image, "legacy-sha")) + "\n", encoding="utf-8"
    )
    output = tmp_path / "labels.jsonl"
    output.write_text(
        json.dumps(
            {
                "sha256": "legacy-sha",
                "image_path": str(image),
                "model": "old-model",
                "passes": [
                    {
                        "material": "paper",
                        "confidence": 0.99,
                        "single_object": True,
                        "foreign_material": False,
                    }
                ]
                * 2,
                "errors": [],
                "consensus": True,
                "consensus_decision": {
                    "material": "paper",
                    "single_object": True,
                    "foreign_material": False,
                },
                "minimum_confidence": 0.99,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = 0

    def fake_request(*_args):
        nonlocal calls
        calls += 1
        return _answer("paper", 0.95)

    monkeypatch.setattr(teacher, "_request", fake_request)
    teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="test-model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert calls == 2
    assert row["schema_version"] == teacher.TEACHER_LABEL_SCHEMA_VERSION
    assert row["consensus_decision"]["training_usable"] is True


def test_quality_prompt_keeps_normal_kiosk_and_light_hand_contact():
    rendered = "\n".join((*teacher.PROMPTS, teacher.ADJUDICATION_PROMPT))
    assert "가벼운 손끝 접촉" in rendered
    assert "사람 손/팔이 대상 경계를 가리거나 화면을 지배" in rendered
    assert "심하게 잘린" in rendered
    assert "불필요한 배경이 과도" in rendered
    assert "손/팔 자체는 쓰레기 개수에 포함하지" in rendered
    assert "대상 crop 경계를 신뢰할 수 없으면 false" in rendered
    assert "대상 자체가 구겨지거나" in rendered
    assert "중요한 hard case이며 촬영 실패가 아니므로" in rendered


def test_request_rejects_truncated_thinking(monkeypatch):
    body = {
        "model": "model",
        "done_reason": "length",
        "message": {"content": "", "thinking": '{"material":"paper"'},
    }
    monkeypatch.setattr(teacher, "_open_no_redirect", lambda *_args, **_kwargs: _Response())
    monkeypatch.setattr(teacher.json, "load", lambda _response: body)

    with pytest.raises(ValueError, match="empty model content"):
        teacher._request("http://teacher", "model", b"image", "prompt", 10)


def test_redirect_handler_refuses_teacher_url_redirect():
    handler = teacher._RejectRedirects()
    with pytest.raises(ValueError, match="redirect refused"):
        handler.redirect_request(None, None, 307, "redirect", {}, "https://evil.example")


@pytest.mark.parametrize("confidence", [True, float("nan"), float("inf")])
def test_request_rejects_bool_or_non_finite_confidence(monkeypatch, confidence):
    body = {
        "model": "model",
        "done_reason": "stop",
        "message": {"content": json.dumps(_answer("paper", confidence))},
    }
    monkeypatch.setattr(
        teacher, "_open_no_redirect", lambda *_args, **_kwargs: _Response()
    )
    monkeypatch.setattr(teacher.json, "load", lambda _response: body)

    with pytest.raises(ValueError, match="invalid confidence"):
        teacher._request("http://127.0.0.1:11434", "model", b"image", "prompt", 10)


def test_label_queue_verifies_sha_before_network_and_keeps_portable_output(
    tmp_path, monkeypatch
):
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"actual-bytes")
    queue = tmp_path / "queue.jsonl"
    row = _queue_row(image, "0" * 64)
    row.update(
        {
            "client_id": "should-never-be-accepted",
            "deployed": {"bbox": [1, 2, 3, 4]},
        }
    )
    queue.write_text(json.dumps(row) + "\n", encoding="utf-8")
    calls = 0

    def fake_request(*_args):
        nonlocal calls
        calls += 1
        return _answer("paper", 0.9)

    monkeypatch.setattr(teacher, "_request", fake_request)
    with pytest.raises(ValueError, match="shape is not exact"):
        teacher.label_queue(
            queue,
            tmp_path / "labels.jsonl",
            image_root=tmp_path,
            known_audit=_known(tmp_path),
            url="http://127.0.0.1:11434",
            model="model",
            model_digest="a" * 64,
            timeout=1,
            retries=1,
        )
    assert calls == 0

    queue.write_text(
        json.dumps({"sha256": "0" * 64, "image_ref": image.name,
                    "timestamp": "2026-08-01T00:00:00+09:00",
                    "decision": "teacher_required"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "labels.jsonl"
    teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )
    assert calls == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["errors"] == ["ValueError: image_sha256_mismatch"]
    assert result["image_ref"] == ""
    rendered = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in rendered
    assert "client_id" not in rendered
    assert "deployed" not in rendered
    assert "verifier" not in rendered


def test_label_queue_url_guard_and_explicit_external_override(tmp_path):
    queue = tmp_path / "queue.jsonl"
    queue.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="loopback/private"):
        teacher.label_queue(
            queue,
            tmp_path / "labels.jsonl",
            image_root=tmp_path,
            known_audit=_known(tmp_path),
            url="https://teacher.example.com",
            model="model",
            model_digest="a" * 64,
            timeout=1,
            retries=1,
        )
    summary = teacher.label_queue(
        queue,
        tmp_path / "labels.jsonl",
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="https://teacher.example.com",
        model="model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
        allow_external_url=True,
    )
    assert summary["queued"] == 0


@pytest.mark.parametrize("image_ref", ["../outside.jpg", "C:/outside.jpg"])
def test_label_queue_rejects_escaping_image_ref_before_network(
    tmp_path, monkeypatch, image_ref
):
    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"outside")
    sha = hashlib.sha256(outside.read_bytes()).hexdigest()
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps({"sha256": sha, "image_ref": image_ref,
                    "timestamp": "2026-08-01T00:00:00+09:00",
                    "decision": "teacher_required"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        teacher,
        "_request",
        lambda *_args: pytest.fail("network must not be called"),
    )
    output = tmp_path / "labels.jsonl"
    teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )
    assert "image_ref must stay relative" in json.loads(
        output.read_text(encoding="utf-8")
    )["errors"][0]


def test_label_queue_rejects_symlink_outside_image_root_before_network(
    tmp_path, monkeypatch
):
    outside = tmp_path.parent / "teacher-outside.jpg"
    outside.write_bytes(b"outside")
    link = tmp_path / "linked.jpg"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not available")
    sha = hashlib.sha256(outside.read_bytes()).hexdigest()
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps({"sha256": sha, "image_ref": link.name,
                    "timestamp": "2026-08-01T00:00:00+09:00",
                    "decision": "teacher_required"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        teacher,
        "_request",
        lambda *_args: pytest.fail("network must not be called"),
    )
    output = tmp_path / "labels.jsonl"
    teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )
    assert "image_ref escapes image_root" in json.loads(
        output.read_text(encoding="utf-8")
    )["errors"][0]


def test_per_sha_checkpoints_resume_after_crash_and_merge_once(tmp_path, monkeypatch):
    images = []
    for name in ("a.jpg", "b.jpg"):
        image = tmp_path / name
        image.write_bytes(name.encode("ascii"))
        images.append(image)
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "".join(json.dumps(_queue_row(image)) + "\n" for image in images),
        encoding="utf-8",
    )
    calls = 0

    def crashing_request(*_args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return _answer("paper", 0.9)

    monkeypatch.setattr(teacher, "_request", crashing_request)
    output = tmp_path / "labels.jsonl"
    with pytest.raises(KeyboardInterrupt):
        teacher.label_queue(
            queue,
            output,
            image_root=tmp_path,
            known_audit=_known(tmp_path),
            url="http://127.0.0.1:11434",
            model="model",
            model_digest="a" * 64,
            timeout=1,
            retries=1,
        )
    checkpoint_dir = tmp_path / "labels.jsonl.checkpoints"
    assert len(list(checkpoint_dir.glob("*.json"))) == 1
    assert not output.exists()

    resumed_calls = 0

    def resumed_request(*_args):
        nonlocal resumed_calls
        resumed_calls += 1
        return _answer("paper", 0.9)

    monkeypatch.setattr(teacher, "_request", resumed_request)
    teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="model",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert resumed_calls == 2
    assert len(rows) == len({row["sha256"] for row in rows}) == 2
    assert [row["sha256"] for row in rows] == sorted(row["sha256"] for row in rows)
    assert len(list(checkpoint_dir.glob("*.json"))) == 2


def test_contract_change_forces_checkpoint_relabel(tmp_path, monkeypatch):
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"capture")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(_queue_row(image)) + "\n", encoding="utf-8")
    output = tmp_path / "labels.jsonl"
    monkeypatch.setattr(teacher, "_request", lambda *_args: _answer("paper", 0.9))
    teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="model-v1",
        model_digest="a" * 64,
        timeout=1,
        retries=1,
    )
    first = json.loads(output.read_text(encoding="utf-8"))
    calls = 0

    def relabel(*_args):
        nonlocal calls
        calls += 1
        return _answer("paper", 0.9)

    monkeypatch.setattr(teacher, "_request", relabel)
    teacher.label_queue(
        queue,
        output,
        image_root=tmp_path,
        known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="model-v2",
        model_digest="b" * 64,
        timeout=1,
        retries=1,
    )
    second = json.loads(output.read_text(encoding="utf-8"))
    assert calls == 2
    assert first["teacher_contract_sha256"] != second["teacher_contract_sha256"]
    assert second["model"] == "model-v2"


@pytest.mark.parametrize("contract_drift", ["prompt", "options"])
def test_prompt_or_options_contract_change_forces_resume_relabel(
    tmp_path, monkeypatch, contract_drift
):
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"capture")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(_queue_row(image)) + "\n", encoding="utf-8")
    output = tmp_path / "labels.jsonl"
    monkeypatch.setattr(teacher, "_request", lambda *_args: _answer("paper", 0.9))
    teacher.label_queue(
        queue, output, image_root=tmp_path, known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="model", model_digest="a" * 64, timeout=1, retries=1,
    )
    original_builder = teacher.build_teacher_contract

    def drifted_builder(model, digest):
        contract, _ = original_builder(model, digest)
        if contract_drift == "prompt":
            contract["rendered_prompts"]["initial"][0] += " changed"
        else:
            contract["request"]["options"]["num_predict"] += 1
        return contract, teacher._sha256_bytes(
            teacher._canonical_json(contract).encode("utf-8")
        )

    monkeypatch.setattr(teacher, "build_teacher_contract", drifted_builder)
    calls = 0

    def relabel(*_args):
        nonlocal calls
        calls += 1
        return _answer("paper", 0.9)

    monkeypatch.setattr(teacher, "_request", relabel)
    teacher.label_queue(
        queue, output, image_root=tmp_path, known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434",
        model="model", model_digest="a" * 64, timeout=1, retries=1,
    )
    assert calls == 2


def test_observe_model_digest_normalizes_sha256_prefix(monkeypatch):
    payload = json.dumps({"models": [{
        "name": "teacher-model", "digest": "sha256:" + "A" * 64
    }]}).encode()
    monkeypatch.setattr(teacher, "_open_no_redirect", lambda *_args: io.BytesIO(payload))
    assert _REAL_OBSERVE_MODEL_DIGEST(
        "http://127.0.0.1:11434", "teacher-model", 1
    ) == "a" * 64


@pytest.mark.parametrize("models", [[], [{"name": "teacher-model", "digest": "bad"}]])
def test_observe_model_digest_rejects_missing_or_invalid(monkeypatch, models):
    payload = json.dumps({"models": models}).encode()
    monkeypatch.setattr(teacher, "_open_no_redirect", lambda *_args: io.BytesIO(payload))
    with pytest.raises(ValueError):
        _REAL_OBSERVE_MODEL_DIGEST("http://127.0.0.1:11434", "teacher-model", 1)


def test_declared_digest_mismatch_fails_before_network_inference(tmp_path, monkeypatch):
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"capture")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(_queue_row(image)) + "\n", encoding="utf-8")
    monkeypatch.setattr(teacher, "_observe_model_digest", lambda *_args: "b" * 64)
    monkeypatch.setattr(teacher, "_request", lambda *_args: pytest.fail("no inference"))
    with pytest.raises(ValueError, match="does not match"):
        teacher.label_queue(
            queue, tmp_path / "labels.jsonl", image_root=tmp_path,
            known_audit=_known(tmp_path), url="http://127.0.0.1:11434",
            model="model", model_digest="a" * 64, timeout=1, retries=1,
        )


def test_digest_drift_fails_before_final_publish(tmp_path, monkeypatch):
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"capture")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(_queue_row(image)) + "\n", encoding="utf-8")
    observed = iter(["a" * 64, "a" * 64, "b" * 64])
    monkeypatch.setattr(teacher, "_observe_model_digest", lambda *_args: next(observed))
    monkeypatch.setattr(teacher, "_request", lambda *_args: _answer("paper", 0.9))
    output = tmp_path / "labels.jsonl"
    with pytest.raises(ValueError, match="changed during"):
        teacher.label_queue(
            queue, output, image_root=tmp_path, known_audit=_known(tmp_path),
            url="http://127.0.0.1:11434", model="model",
            model_digest="a" * 64, timeout=1, retries=1,
        )
    assert not output.exists()


def test_known_audit_sha_and_wrong_decision_are_rejected(tmp_path, monkeypatch):
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"capture")
    row = _queue_row(image)
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(teacher, "_request", lambda *_args: pytest.fail("no inference"))
    with pytest.raises(ValueError, match="known train/validation"):
        teacher.label_queue(
            queue, tmp_path / "labels.jsonl", image_root=tmp_path,
            known_audit=_known(tmp_path, {row["sha256"]: {"split": "protected_validation"}}),
            url="http://127.0.0.1:11434", model="model",
            model_digest="a" * 64, timeout=1, retries=1,
        )
    row["decision"] = "ignored"
    queue.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="teacher_required"):
        teacher.label_queue(
            queue, tmp_path / "labels2.jsonl", image_root=tmp_path,
            known_audit=_known(tmp_path), url="http://127.0.0.1:11434",
            model="model", model_digest="a" * 64, timeout=1, retries=1,
        )


def test_extra_decision_field_never_becomes_success_checkpoint(tmp_path, monkeypatch):
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"capture")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(_queue_row(image)) + "\n", encoding="utf-8")
    answer = _answer("paper", 0.9)
    answer["unexpected"] = True
    monkeypatch.setattr(teacher, "_request", lambda *_args: answer)
    output = tmp_path / "labels.jsonl"
    with pytest.raises(ValueError, match="exactly match"):
        teacher.label_queue(
            queue, output, image_root=tmp_path, known_audit=_known(tmp_path),
            url="http://127.0.0.1:11434", model="model",
            model_digest="a" * 64, timeout=1, retries=1,
        )
    assert not output.exists()


def test_mid_image_digest_drift_leaves_no_checkpoint_and_rollback_relabels(
    tmp_path, monkeypatch
):
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"capture")
    queue = tmp_path / "queue.jsonl"
    row = _queue_row(image)
    queue.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "labels.jsonl"
    observed = iter(["a" * 64, "a" * 64, "b" * 64])
    monkeypatch.setattr(teacher, "_observe_model_digest", lambda *_args: next(observed))
    monkeypatch.setattr(teacher, "_request", lambda *_args: _answer("paper", 0.9))
    with pytest.raises(ValueError, match="during image"):
        teacher.label_queue(
            queue, output, image_root=tmp_path, known_audit=_known(tmp_path),
            url="http://127.0.0.1:11434", model="model",
            model_digest="a" * 64, timeout=1, retries=1,
        )
    checkpoint = tmp_path / "labels.jsonl.checkpoints" / f"{row['sha256']}.json"
    assert not checkpoint.exists()
    calls = 0

    def relabel(*_args):
        nonlocal calls
        calls += 1
        return _answer("paper", 0.9)

    monkeypatch.setattr(teacher, "_observe_model_digest", lambda *_args: "a" * 64)
    monkeypatch.setattr(teacher, "_request", relabel)
    teacher.label_queue(
        queue, output, image_root=tmp_path, known_audit=_known(tmp_path),
        url="http://127.0.0.1:11434", model="model",
        model_digest="a" * 64, timeout=1, retries=1,
    )
    assert calls == 2
    assert checkpoint.exists()


def _shared_fixture(tmp_path):
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"shared-capture")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(_queue_row(image)) + "\n", encoding="utf-8")
    _known(tmp_path)
    return image, queue, tmp_path / "teacher-cache"


def _shared_run(tmp_path, queue, cache, name, **kwargs):
    return teacher.label_queue(
        queue, tmp_path / name, image_root=tmp_path,
        known_audit=tmp_path / "known_audit.json", url="http://127.0.0.1:11434",
        model=kwargs.pop("model", "model-v1"),
        model_digest=kwargs.pop("model_digest", "a" * 64),
        timeout=1, retries=1, checkpoint_dir=cache, **kwargs,
    )


def test_shared_cache_reuses_same_image_for_new_immutable_output(tmp_path, monkeypatch):
    _, queue, cache = _shared_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(teacher, "_request", lambda *args: calls.append(args) or _answer("can", .9))
    first = _shared_run(tmp_path, queue, cache, "first.jsonl")
    assert len(calls) == 2
    assert first["newly_labeled"] == 1 and first["reused"] == 0
    existing = {path.relative_to(cache): path.read_bytes() for path in cache.rglob("*.json")}
    monkeypatch.setattr(teacher, "_request", lambda *_: pytest.fail("cached image must not call chat"))
    second = _shared_run(tmp_path, queue, cache, "second.jsonl")
    assert second["reused"] == 1 and second["newly_labeled"] == second["retried"] == 0
    assert second["image_error"] == 0
    assert (tmp_path / "first.jsonl").read_bytes() == (tmp_path / "second.jsonl").read_bytes()
    assert {path.relative_to(cache): path.read_bytes() for path in cache.rglob("*.json")} == existing
    assert set(second).isdisjoint({"client_id", "image_ref", "checkpoint_dir"})


def test_shared_cache_labels_only_new_images(tmp_path, monkeypatch):
    image, queue, cache = _shared_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(teacher, "_request", lambda *args: calls.append(args[2]) or _answer("paper", .9))
    _shared_run(tmp_path, queue, cache, "first.jsonl")
    calls.clear()
    new_image = tmp_path / "new.jpg"
    new_image.write_bytes(b"new-capture")
    queue.write_text("".join(json.dumps(_queue_row(path)) + "\n" for path in (image, new_image)), encoding="utf-8")
    summary = _shared_run(tmp_path, queue, cache, "second.jsonl")
    assert calls == [b"new-capture", b"new-capture"]
    assert summary["reused"] == summary["newly_labeled"] == 1
    assert summary["retried"] == 0


@pytest.mark.parametrize("change", ["model", "prompt"])
def test_shared_cache_contract_namespaces_preserve_prior_results(tmp_path, monkeypatch, change):
    _, queue, cache = _shared_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(teacher, "_request", lambda *args: calls.append(args) or _answer("paper", .9))
    first = _shared_run(tmp_path, queue, cache, "first.jsonl")
    original_root = cache / first["teacher_contract_sha256"]
    original = {path.relative_to(original_root): path.read_bytes() for path in original_root.rglob("*.json")}
    kwargs = {}
    if change == "model":
        kwargs = {"model": "model-v2", "model_digest": "b" * 64}
    else:
        build = teacher.build_teacher_contract
        def changed_contract(model, digest):
            contract, _ = build(model, digest)
            contract["rendered_prompts"]["initial"][0] += " changed"
            return contract, teacher._sha256_bytes(teacher._canonical_json(contract).encode())
        monkeypatch.setattr(teacher, "build_teacher_contract", changed_contract)
    calls.clear()
    second = _shared_run(tmp_path, queue, cache, "second.jsonl", **kwargs)
    assert len(calls) == 2
    assert second["reused"] == 0 and second["newly_labeled"] == 1
    assert first["teacher_contract_sha256"] != second["teacher_contract_sha256"]
    assert {path.relative_to(original_root): path.read_bytes() for path in original_root.rglob("*.json")} == original
    assert len(list(cache.iterdir())) == 2


def test_shared_cache_retries_incomplete_without_overwriting_previous_attempt(tmp_path, monkeypatch):
    _, queue, cache = _shared_fixture(tmp_path)
    def failed_request(*_):
        raise ValueError("truncated response")
    monkeypatch.setattr(teacher, "_request", failed_request)
    first = _shared_run(tmp_path, queue, cache, "first.jsonl")
    assert first["newly_labeled"] == 0 and first["reused"] == 0
    previous = {path: path.read_bytes() for path in cache.rglob("*.json")}
    calls = []
    monkeypatch.setattr(teacher, "_request", lambda *args: calls.append(args) or _answer("can", .9))
    second = _shared_run(tmp_path, queue, cache, "second.jsonl")
    assert len(calls) == 2
    assert second["retried"] == second["newly_labeled"] == 1
    assert second["reused"] == 0
    assert len(list(cache.rglob("*.json"))) == 2
    assert all(path.read_bytes() == content for path, content in previous.items())
    monkeypatch.setattr(teacher, "_request", lambda *_: pytest.fail("completed retry must be reused"))
    third = _shared_run(tmp_path, queue, cache, "third.jsonl")
    assert third["reused"] == 1 and third["retried"] == 0


def test_shared_cache_never_reuses_prediction_for_modified_source(tmp_path, monkeypatch):
    image, queue, cache = _shared_fixture(tmp_path)
    monkeypatch.setattr(teacher, "_request", lambda *_: _answer("can", .9))
    _shared_run(tmp_path, queue, cache, "first.jsonl")
    image.write_bytes(b"changed bytes")
    monkeypatch.setattr(teacher, "_request", lambda *_: pytest.fail("bad source must not call chat"))
    summary = _shared_run(tmp_path, queue, cache, "modified.jsonl")
    assert summary["reused"] == summary["newly_labeled"] == 0
    assert summary["image_error"] == 1
    row = json.loads((tmp_path / "modified.jsonl").read_text(encoding="utf-8"))
    assert row["consensus"] is False
    assert "image_sha256_mismatch" in row["errors"][0]


def test_shared_cache_source_change_during_labeling_cannot_publish_success(tmp_path, monkeypatch):
    image, queue, cache = _shared_fixture(tmp_path)
    def changing_request(*_):
        image.write_bytes(b"changed while labeling")
        return _answer("can", .9)
    monkeypatch.setattr(teacher, "_request", changing_request)
    with pytest.raises(ValueError, match="source image changed"):
        _shared_run(tmp_path, queue, cache, "first.jsonl")
    assert not (tmp_path / "first.jsonl").exists()
    assert not list(cache.rglob("*.json"))


@pytest.mark.parametrize("ancestor", [False, True])
def test_shared_cache_rejects_symlink_directory_and_ancestors(tmp_path, monkeypatch, ancestor):
    _, queue, _ = _shared_fixture(tmp_path)
    real = tmp_path / "real-cache"
    real.mkdir()
    linked = tmp_path / "linked-cache"
    try:
        os.symlink(real, linked, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    monkeypatch.setattr(teacher, "_request", lambda *_: pytest.fail("unsafe cache must not call chat"))
    with pytest.raises(ValueError, match="symlink"):
        _shared_run(tmp_path, queue, linked / "nested" if ancestor else linked, "labels.jsonl")
    assert not list(real.iterdir())


def test_shared_cache_rejects_output_overlap_and_existing_output(tmp_path, monkeypatch):
    _, queue, cache = _shared_fixture(tmp_path)
    monkeypatch.setattr(teacher, "_request", lambda *_: pytest.fail("unsafe output must not call chat"))
    with pytest.raises(ValueError, match="must not overlap"):
        _shared_run(tmp_path, queue, cache, "teacher-cache/labels.jsonl")
    output = tmp_path / "existing.jsonl"
    output.write_bytes(b"keep previous run")
    with pytest.raises(FileExistsError, match="new immutable output"):
        _shared_run(tmp_path, queue, cache, output.name)
    assert output.read_bytes() == b"keep previous run"


def test_shared_cache_rejects_corrupt_content_addressed_entry(tmp_path, monkeypatch):
    _, queue, cache = _shared_fixture(tmp_path)
    monkeypatch.setattr(teacher, "_request", lambda *_: _answer("can", .9))
    _shared_run(tmp_path, queue, cache, "first.jsonl")
    cached = next(cache.rglob("*.json"))
    cached.write_bytes(cached.read_bytes() + b" ")
    monkeypatch.setattr(teacher, "_request", lambda *_: pytest.fail("corrupt cache must fail closed"))
    with pytest.raises(ValueError, match="content digest mismatch"):
        _shared_run(tmp_path, queue, cache, "second.jsonl")
    assert not (tmp_path / "second.jsonl").exists()


def test_shared_cache_cli_passes_optional_directory(tmp_path, monkeypatch):
    seen = {}
    def fake_label(queue, output, **kwargs):
        seen.update(kwargs)
        return {"reused": 0, "newly_labeled": 0, "retried": 0, "image_error": 0}
    cache = tmp_path / "shared-cache"
    monkeypatch.setattr(teacher, "label_queue", fake_label)
    monkeypatch.setattr(sys, "argv", [
        "label_operational_captures_ollama.py", "--queue", "queue.jsonl",
        "--output", "labels.jsonl", "--image-root", str(tmp_path),
        "--known-audit", "known.json", "--model-digest", "a" * 64,
        "--checkpoint-dir", str(cache),
    ])
    teacher.main()
    assert seen["checkpoint_dir"] == cache


def test_shared_cache_final_output_race_preserves_other_run(tmp_path, monkeypatch):
    _, queue, cache = _shared_fixture(tmp_path)
    output = tmp_path / "raced.jsonl"
    def racing_request(*_):
        output.write_bytes(b"another-run")
        return _answer("can", .9)
    monkeypatch.setattr(teacher, "_request", racing_request)
    with pytest.raises(FileExistsError):
        _shared_run(tmp_path, queue, cache, output.name)
    assert output.read_bytes() == b"another-run"


def test_shared_cache_completed_entry_symlink_is_never_followed(tmp_path, monkeypatch):
    _, queue, cache = _shared_fixture(tmp_path)
    monkeypatch.setattr(teacher, "_request", lambda *_: _answer("can", .9))
    _shared_run(tmp_path, queue, cache, "first.jsonl")
    entry = next(cache.rglob("*.json"))
    original = tmp_path / "outside-cache.json"
    original.write_bytes(entry.read_bytes())
    entry.unlink()
    try:
        os.symlink(original, entry)
    except OSError:
        pytest.skip("symlink creation unavailable")
    monkeypatch.setattr(teacher, "_request", lambda *_: pytest.fail("cache symlink must fail closed"))
    with pytest.raises(ValueError, match="symlink"):
        _shared_run(tmp_path, queue, cache, "second.jsonl")
    assert not (tmp_path / "second.jsonl").exists()


def test_request_rejects_response_model_mismatch(monkeypatch):
    body = {
        "model": "different-model",
        "done_reason": "stop",
        "message": {"content": json.dumps(_answer("paper", 0.9))},
    }
    monkeypatch.setattr(teacher, "_open_no_redirect", lambda *_args, **_kwargs: _Response())
    monkeypatch.setattr(teacher.json, "load", lambda _response: body)
    with pytest.raises(ValueError, match="does not match requested"):
        teacher._request("http://teacher", "model", b"image", "prompt", 10)
