import base64
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_independent_visual_judges as judges


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _model_spec(
    root: Path,
    *,
    judge_id: str,
    family: str,
    model: str,
    manifest_bytes: bytes,
) -> Path:
    config = root / f"{judge_id}.model-config.json"
    config_bytes = (
        json.dumps(
            {
                "model_family": family,
                "model_type": family,
                "format": "gguf",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    config.write_bytes(config_bytes)
    manifest = root / f"{judge_id}.model-manifest"
    manifest_value = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {
            "mediaType": "application/vnd.docker.container.image.v1+json",
            "digest": f"sha256:{_sha(config_bytes)}",
            "size": len(config_bytes),
        },
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": f"sha256:{_sha(manifest_bytes)}",
                "size": len(manifest_bytes),
            }
        ],
    }
    manifest_payload = (
        json.dumps(manifest_value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest.write_bytes(manifest_payload)
    spec = root / f"{judge_id}.judge.json"
    _json(
        spec,
        {
            "schema_version": 1,
            "judge_id": judge_id,
            "model_family": family,
            "ollama_model": model,
            "ollama_url": f"http://{judge_id}.invalid:11434",
            "model_manifest_path": manifest.name,
            "model_manifest_sha256": _sha(manifest_payload),
            "model_config_path": config.name,
            "model_config_sha256": _sha(config_bytes),
        },
    )
    return spec


def _judge_specs(root: Path) -> list[Path]:
    return [
        _model_spec(
            root,
            judge_id="judge-a",
            family="internvl",
            model="internvl:8b",
            manifest_bytes=b"internvl-manifest-v1",
        ),
        _model_spec(
            root,
            judge_id="judge-b",
            family="minicpm-v",
            model="minicpm-v:8b",
            manifest_bytes=b"minicpm-manifest-v1",
        ),
    ]


def _rewrite_model_family(spec_path: Path, family: str) -> None:
    value = json.loads(spec_path.read_text())
    config_path = spec_path.parent / value["model_config_path"]
    config = json.loads(config_path.read_text())
    config["model_family"] = family
    config["model_type"] = family
    config_bytes = (
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    config_path.write_bytes(config_bytes)
    manifest_path = spec_path.parent / value["model_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["digest"] = f"sha256:{_sha(config_bytes)}"
    manifest["config"]["size"] = len(config_bytes)
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    value["model_family"] = family
    value["model_config_sha256"] = _sha(config_bytes)
    value["model_manifest_sha256"] = _sha(manifest_bytes)
    _json(spec_path, value)


def _replace_model_config(spec_path: Path, config_value: dict) -> None:
    value = json.loads(spec_path.read_text())
    config_path = spec_path.parent / value["model_config_path"]
    config_bytes = (
        json.dumps(config_value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    config_path.write_bytes(config_bytes)
    manifest_path = spec_path.parent / value["model_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["digest"] = f"sha256:{_sha(config_bytes)}"
    manifest["config"]["size"] = len(config_bytes)
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    value["model_config_sha256"] = _sha(config_bytes)
    value["model_manifest_sha256"] = _sha(manifest_bytes)
    _json(spec_path, value)


def _manifest(root: Path, *, rows: int = 2) -> tuple[Path, list[dict[str, str]]]:
    manifest = root / "v4-background-validation.csv"
    values = []
    for index in range(rows):
        source = root / f"source-{index}.jpg"
        crop = root / f"crop-{index}.jpg"
        source_bytes = f"source-image-{index}".encode()
        crop_bytes = f"crop-image-{index}".encode()
        source.write_bytes(source_bytes)
        crop.write_bytes(crop_bytes)
        values.append(
            {
                "sample_id": f"sample-{index}",
                "role": "model_validation",
                "split": "validation",
                "source_path_b64": base64.urlsafe_b64encode(
                    str(source.resolve()).encode()
                ).decode("ascii"),
                "filepath": crop.name,
                "source_sha256": _sha(source_bytes),
                "image_sha256": _sha(crop_bytes),
                "material": "9",
                "category": "background",
                "source_object_count": "0",
                # These sentinels must never enter the Ollama payload.
                "candidate_prediction": "SECRET_CANDIDATE_PAPER",
                "truth_external": "SECRET_TRUTH_BACKGROUND",
                "candidate_confidence": "0.999991",
            }
        )
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    return manifest, values


def _success_api_client(
    calls: list[tuple],
    verdict: str = "background",
    *,
    chat_response: dict | None = None,
):
    def client(spec, method, endpoint, payload, timeout):
        calls.append((spec, method, endpoint, payload, timeout))
        details = {
            "family": spec.model_family,
            "families": [spec.model_family],
        }
        if (method, endpoint) == ("GET", "/api/tags"):
            return {
                "models": [
                    {
                        "name": spec.ollama_model,
                        "model": spec.ollama_model,
                        "digest": spec.model_manifest_sha256,
                        "details": details,
                    }
                ]
            }
        if (method, endpoint) == ("POST", "/api/show"):
            assert payload == {"model": spec.ollama_model}
            return {
                "details": details,
                "capabilities": ["completion", "vision"],
            }
        if (method, endpoint) == ("POST", "/api/chat"):
            if chat_response is not None:
                return {"model": spec.ollama_model, **chat_response}
            response = {
                "model": spec.ollama_model,
                "message": {"content": json.dumps({"verdict": verdict})},
                "created_at": "2026-08-31T00:00:00Z",
                "done": True,
                "done_reason": "stop",
                "total_duration": 123,
                "prompt_eval_count": 2,
                "server_only_raw_sentinel": "MUST_NOT_BE_STORED",
            }
            return response
        raise AssertionError(f"unexpected API call: {method} {endpoint}")

    return client


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rebind_vote(vote: dict) -> dict:
    payload = {
        key: value for key, value in vote.items() if key != "vote_binding_sha256"
    }
    return {
        **payload,
        "vote_binding_sha256": _sha(judges._canonical_json_bytes(payload)),
    }


def test_runs_two_independent_judges_without_candidate_metadata_leakage(tmp_path):
    manifest, _ = _manifest(tmp_path)
    specs = _judge_specs(tmp_path)
    output = tmp_path / "votes.jsonl"
    report_path = tmp_path / "report.json"
    calls = []

    report = judges.run_independent_visual_judges(
        input_manifest=manifest,
        judge_spec_paths=specs,
        output_jsonl=output,
        output_report=report_path,
        api_client=_success_api_client(calls),
    )

    assert len(calls) == 16
    assert [(method, endpoint) for _, method, endpoint, _, _ in calls[:4]] == [
        ("GET", "/api/tags"),
        ("POST", "/api/show"),
        ("GET", "/api/tags"),
        ("POST", "/api/show"),
    ]
    chat_calls = [call for call in calls if call[2] == "/api/chat"]
    assert len(chat_calls) == 4
    assert [call[0].judge_id for call in chat_calls] == [
        "judge-a",
        "judge-a",
        "judge-b",
        "judge-b",
    ]
    for index, call in enumerate(calls[:-1]):
        if call[2] == "/api/chat":
            assert calls[index + 1][0].judge_id == call[0].judge_id
            assert calls[index + 1][1:3] == ("GET", "/api/tags")
    for _spec, method, endpoint, payload, _timeout in chat_calls:
        assert (method, endpoint) == ("POST", "/api/chat")
        serialized = json.dumps(payload)
        assert "SECRET_CANDIDATE_PAPER" not in serialized
        assert "SECRET_TRUTH_BACKGROUND" not in serialized
        assert "0.999991" not in serialized
        assert payload["options"] == {"temperature": 0}
        assert payload["messages"][0]["content"] == judges.PROMPT
        assert len(payload["messages"][0]["images"]) == 2
    votes = _read_jsonl(output)
    assert len(votes) == 4
    assert {
        (vote["sample_id"], vote["judge_id"]) for vote in votes
    } == {
        (f"sample-{row}", f"judge-{judge}")
        for row in range(2)
        for judge in ("a", "b")
    }
    for vote in votes:
        assert vote["verdict"] == "background"
        assert vote["evidence_schema"] == judges.EVIDENCE_SCHEMA
        assert vote["evidence_schema_version"] == judges.EVIDENCE_SCHEMA_VERSION
        assert vote["canonical_json_contract"] == judges.CANONICAL_JSON_CONTRACT
        assert vote["official_ollama_http"] is False
        assert vote["authoritative_evidence"] is False
        assert len(vote["model_weight_layer_sha256"]) == 64
        binding = {key: value for key, value in vote.items() if key != "vote_binding_sha256"}
        assert vote["vote_binding_sha256"] == _sha(
            judges._canonical_json_bytes(binding)
        )
        assert vote["prompt_sha256"] == _sha(judges.PROMPT.encode())
        assert len(vote["model_manifest_sha256"]) == 64
        assert len(vote["model_config_sha256"]) == 64
        assert len(vote["source_sha256"]) == 64
        assert len(vote["crop_sha256"]) == 64
        assert len(vote["runner_script_sha256"]) == 64
        assert len(vote["canonical_raw_response_sha256"]) == 64
        expected_raw_response = {
            "model": vote["ollama_model"],
            "message": {"content": json.dumps({"verdict": "background"})},
            "created_at": "2026-08-31T00:00:00Z",
            "done": True,
            "done_reason": "stop",
            "total_duration": 123,
            "prompt_eval_count": 2,
            "server_only_raw_sentinel": "MUST_NOT_BE_STORED",
        }
        assert vote["canonical_raw_response"] == expected_raw_response
        assert vote["canonical_raw_response_sha256"] == _sha(
            judges._canonical_json_bytes(expected_raw_response)
        )
        assert vote["server_model_digest"] == vote["model_manifest_sha256"]
        assert "vision" in vote["server_capabilities"]
        assert isinstance(vote["postchat_tags_response"], dict)
        assert vote["postchat_server_model_digest"] == vote["server_model_digest"]
        assert vote["postchat_tags_response_sha256"] == _sha(
            judges._canonical_json_bytes(vote["postchat_tags_response"])
        )
        assert vote["postflight_identity_set_sha256"] == report[
            "postflight_identity_set_sha256"
        ]
        assert vote["server_digest_contract"] == judges.SERVER_DIGEST_CONTRACT
    stored_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == stored_report
    assert report["artifact_role"] == "diagnostic_veto_only_not_promotion_authority"
    assert report["authority"] == {
        "promotion_authority": False,
        "ground_truth_authority": False,
        "may_relabel_truth": False,
        "may_tune_thresholds": False,
        "allowed_actions": ["diagnostic", "veto", "request_more_evidence"],
    }
    assert report["coverage"]["every_judge_exactly_one_vote_per_row"] is True
    assert report["results_jsonl_sha256"] == _sha(output.read_bytes())
    assert report["evidence_schema"] == judges.EVIDENCE_SCHEMA
    assert report["evidence_schema_version"] == judges.EVIDENCE_SCHEMA_VERSION
    assert report["canonical_json_contract"] == judges.CANONICAL_JSON_CONTRACT
    assert report["official_ollama_http"] is False
    assert report["authoritative_evidence"] is False
    assert report["artifact_pair"]["complete"] is True
    assert report["evidence_pair_id"] == votes[0]["evidence_pair_id"]
    assert report["evidence_pair_id"] == _sha(
        judges._canonical_json_bytes(report["evidence_pair_seed"])
    )
    assert report["postflight_identity_set_sha256"] == _sha(
        judges._canonical_json_bytes(report["postflight_identity_set"])
    )
    assert all(vote["evidence_pair_id"] == report["evidence_pair_id"] for vote in votes)
    assert report["raw_response_content_stored"] is True
    assert report["request_content_stored"] is False
    assert report["image_content_stored"] is False
    assert report["evidence_jsonl_sha256"] == _sha(output.read_bytes())
    assert report["evidence_jsonl_line_count"] == 4
    assert report["runner_script_sha256"] == _sha(
        Path(judges.__file__).resolve().read_bytes()
    )
    assert len(report["canonical_raw_response_sha256_by_vote"]) == 4
    for judge in report["judges"]:
        for phase in ("preflight_tags", "preflight_show", "postflight_tags", "postflight_show"):
            response = judge[f"{phase}_response"]
            assert judge[f"{phase}_response_sha256"] == _sha(
                judges._canonical_json_bytes(response)
            )
        assert judge["postflight_identity_matches_preflight"] is True
    assert "MUST_NOT_BE_STORED" in output.read_text(encoding="utf-8")
    assert "MUST_NOT_BE_STORED" not in report_path.read_text(encoding="utf-8")
    assert judges.PROMPT not in output.read_text(encoding="utf-8")
    assert base64.b64encode(b"source-image-0").decode() not in output.read_text(
        encoding="utf-8"
    )
    assert base64.b64encode(b"crop-image-0").decode() not in output.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "response, message",
    [
        ({}, "missing message"),
        ({"message": {}}, "missing message.content"),
        ({"message": {"content": ""}}, "missing message.content"),
        ({"message": {"content": "```json"}}, "strict JSON object"),
        (
            {"message": {"content": '{"verdict":"background","reason":"x"}'}},
            "only the verdict",
        ),
        ({"message": {"content": '{"verdict":"paper"}'}}, "verdict must be"),
    ],
)
def test_missing_or_invalid_judge_vote_fails_closed_without_output(
    tmp_path, response, message
):
    manifest, _ = _manifest(tmp_path, rows=1)
    output = tmp_path / "votes.jsonl"
    report = tmp_path / "report.json"

    with pytest.raises(ValueError, match=message):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=output,
            output_report=report,
            api_client=_success_api_client([], chat_response=response),
        )

    assert not output.exists()
    assert not report.exists()


def test_endpoint_tag_swap_is_rejected_before_chat(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    calls = []
    base_client = _success_api_client(calls)

    def client(spec, method, endpoint, payload, timeout):
        if spec.judge_id == "judge-a" and endpoint == "/api/tags":
            calls.append((spec, method, endpoint, payload, timeout))
            return {
                "models": [
                    {
                        "name": "swapped:8b",
                        "model": "swapped:8b",
                        "digest": spec.model_manifest_sha256,
                        "details": {
                            "family": spec.model_family,
                            "families": [spec.model_family],
                        },
                    }
                ]
            }
        return base_client(spec, method, endpoint, payload, timeout)

    with pytest.raises(ValueError, match="exactly one exact model tag"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=tmp_path / "votes.jsonl",
            output_report=tmp_path / "report.json",
            api_client=client,
        )
    assert all(endpoint != "/api/chat" for _, _, endpoint, _, _ in calls)


def test_endpoint_manifest_digest_swap_is_rejected_before_chat(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    calls = []
    base_client = _success_api_client(calls)

    def client(spec, method, endpoint, payload, timeout):
        response = base_client(spec, method, endpoint, payload, timeout)
        if spec.judge_id == "judge-a" and endpoint == "/api/tags":
            response["models"][0]["digest"] = "0" * 64
        return response

    with pytest.raises(ValueError, match="server model digest does not match"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=tmp_path / "votes.jsonl",
            output_report=tmp_path / "report.json",
            api_client=client,
        )
    assert all(endpoint != "/api/chat" for _, _, endpoint, _, _ in calls)


@pytest.mark.parametrize("endpoint", ["/api/tags", "/api/show"])
def test_endpoint_family_swap_is_rejected_before_chat(tmp_path, endpoint):
    manifest, _ = _manifest(tmp_path, rows=1)
    calls = []
    base_client = _success_api_client(calls)

    def client(spec, method, actual_endpoint, payload, timeout):
        response = base_client(spec, method, actual_endpoint, payload, timeout)
        if spec.judge_id == "judge-a" and actual_endpoint == endpoint:
            owner = response["models"][0] if endpoint == "/api/tags" else response
            owner["details"] = {"family": "gemma3", "families": ["gemma3"]}
        return response

    with pytest.raises(ValueError, match="family does not match"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=tmp_path / "votes.jsonl",
            output_report=tmp_path / "report.json",
            api_client=client,
        )
    assert all(actual != "/api/chat" for _, _, actual, _, _ in calls)


def test_endpoint_qwen_family_is_rejected_even_with_non_qwen_tag(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    calls = []
    base_client = _success_api_client(calls)

    def client(spec, method, endpoint, payload, timeout):
        response = base_client(spec, method, endpoint, payload, timeout)
        if spec.judge_id == "judge-a" and endpoint == "/api/show":
            response["details"] = {
                "family": "qwen2-vl",
                "families": ["qwen2-vl"],
            }
        return response

    with pytest.raises(ValueError, match="Qwen/project teacher family"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=tmp_path / "votes.jsonl",
            output_report=tmp_path / "report.json",
            api_client=client,
        )


def test_endpoint_without_vision_capability_is_rejected_before_chat(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    calls = []
    base_client = _success_api_client(calls)

    def client(spec, method, endpoint, payload, timeout):
        response = base_client(spec, method, endpoint, payload, timeout)
        if spec.judge_id == "judge-a" and endpoint == "/api/show":
            response["capabilities"] = ["completion", "tools"]
        return response

    with pytest.raises(ValueError, match="vision capability"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=tmp_path / "votes.jsonl",
            output_report=tmp_path / "report.json",
            api_client=client,
        )
    assert all(endpoint != "/api/chat" for _, _, endpoint, _, _ in calls)


def test_postchat_tag_identity_swap_fails_without_publication(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    calls = []
    base_client = _success_api_client(calls)
    tag_counts = {}

    def client(spec, method, endpoint, payload, timeout):
        response = base_client(spec, method, endpoint, payload, timeout)
        if endpoint == "/api/tags":
            tag_counts[spec.judge_id] = tag_counts.get(spec.judge_id, 0) + 1
            if spec.judge_id == "judge-a" and tag_counts[spec.judge_id] == 2:
                response["models"][0]["digest"] = "0" * 64
        return response

    output = tmp_path / "votes.jsonl"
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="server model digest does not match"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=output,
            output_report=report,
            api_client=client,
        )
    assert not output.exists()
    assert not report.exists()


def test_postflight_identity_mismatch_fails_without_publication(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    calls = []
    base_client = _success_api_client(calls)
    show_counts = {}

    def client(spec, method, endpoint, payload, timeout):
        response = base_client(spec, method, endpoint, payload, timeout)
        if endpoint == "/api/show":
            show_counts[spec.judge_id] = show_counts.get(spec.judge_id, 0) + 1
            if spec.judge_id == "judge-a" and show_counts[spec.judge_id] == 2:
                response["capabilities"].append("tools")
        return response

    output = tmp_path / "votes.jsonl"
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="identity changed during postflight"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=output,
            output_report=report,
            api_client=client,
        )
    assert not output.exists()
    assert not report.exists()


def test_default_transport_is_marked_authoritative_but_custom_client_is_not(
    tmp_path, monkeypatch
):
    manifest, _ = _manifest(tmp_path, rows=1)
    mock_transport = _success_api_client([])
    monkeypatch.setattr(judges, "_ollama_api_request", mock_transport)

    output = tmp_path / "votes.jsonl"
    report = judges.run_independent_visual_judges(
        input_manifest=manifest,
        judge_spec_paths=_judge_specs(tmp_path),
        output_jsonl=output,
        output_report=tmp_path / "report.json",
    )

    assert report["official_ollama_http"] is True
    assert report["authoritative_evidence"] is True
    assert all(vote["official_ollama_http"] is True for vote in _read_jsonl(output))
    assert all(vote["authoritative_evidence"] is True for vote in _read_jsonl(output))


def test_chat_response_model_tag_swap_is_rejected(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    response = {
        "model": "swapped:8b",
        "message": {"content": '{"verdict":"background"}'},
    }

    with pytest.raises(ValueError, match="response model does not match"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=tmp_path / "votes.jsonl",
            output_report=tmp_path / "report.json",
            api_client=_success_api_client([], chat_response=response),
        )


def test_canonical_raw_response_hash_tamper_breaks_vote_binding(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    specs_paths = _judge_specs(tmp_path)
    output = tmp_path / "votes.jsonl"
    judges.run_independent_visual_judges(
        input_manifest=manifest,
        judge_spec_paths=specs_paths,
        output_jsonl=output,
        output_report=tmp_path / "report.json",
        api_client=_success_api_client([]),
    )
    rows, _ = judges.load_background_rows(manifest)
    specs = judges.load_judge_specs(specs_paths)
    votes = _read_jsonl(output)
    votes[0]["canonical_raw_response"]["ground_truth"] = "injected-after-run"
    votes[0]["canonical_raw_response_sha256"] = _sha(
        judges._canonical_json_bytes(votes[0]["canonical_raw_response"])
    )

    with pytest.raises(ValueError, match="vote binding SHA-256 mismatch"):
        judges._validate_vote_coverage(rows, specs, votes)


@pytest.mark.parametrize(
    "injected_response",
    [
        {
            "message": {"content": '{"verdict":"background"}'},
            "ground_truth": "background",
        },
        {
            "message": {"content": '{"verdict":"background"}'},
            "request": {"messages": []},
        },
        {
            "message": {"content": '{"verdict":"background"}'},
            "prompt": "truncated request echo",
        },
        {
            "message": {"content": '{"verdict":"background"}'},
            "images": ["echo"],
        },
        {
            "message": {"content": '{"verdict":"background"}'},
            "echo": base64.b64encode(b"source-image-0").decode(),
        },
        {
            "message": {"content": '{"verdict":"background"}'},
            "candidate_prediction": "paper",
            "candidate_confidence": 0.999,
        },
        {
            "message": {"content": '{"verdict":"background"}'},
            "debug": "confidence=0.999",
        },
    ],
)
def test_forbidden_metadata_or_request_echo_fails_without_publication(
    tmp_path, injected_response
):
    manifest, _ = _manifest(tmp_path, rows=1)
    output = tmp_path / "votes.jsonl"
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="forbidden|echoes request"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=output,
            output_report=report,
            api_client=_success_api_client([], chat_response=injected_response),
        )
    assert not output.exists()
    assert not report.exists()


def test_duplicate_input_sample_id_is_rejected_before_http(tmp_path):
    manifest, rows = _manifest(tmp_path)
    rows[1]["sample_id"] = rows[0]["sample_id"]
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    called = False

    def client(*_args):
        nonlocal called
        called = True
        raise AssertionError("HTTP must not be called")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=tmp_path / "votes.jsonl",
            output_report=tmp_path / "report.json",
            api_client=client,
        )
    assert called is False


def test_vote_coverage_rejects_missing_duplicate_and_unexpected_votes(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    rows, _ = judges.load_background_rows(manifest)
    specs = judges.load_judge_specs(_judge_specs(tmp_path))
    server_binding = judges.ServerBinding(
        model_digest=specs[0].model_manifest_sha256,
        tag_model_families=specs[0].model_config_families,
        show_model_families=specs[0].model_config_families,
        model_families=specs[0].model_config_families,
        capabilities=("vision",),
        tags_response={"models": []},
        show_response={"capabilities": ["vision"]},
        tags_response_sha256="1" * 64,
        show_response_sha256="2" * 64,
    )
    postchat_tags_response = {"models": [{"name": specs[0].ollama_model}]}
    postchat_tags = judges.TagsEvidence(
        model_digest=specs[0].model_manifest_sha256,
        model_families=specs[0].model_config_families,
        response=postchat_tags_response,
        response_sha256=_sha(
            judges._canonical_json_bytes(postchat_tags_response)
        ),
    )
    raw_response = {
        "model": specs[0].ollama_model,
        "message": {"content": '{"verdict":"background"}'},
    }
    first = judges._vote(
        specs[0],
        server_binding,
        rows[0],
        "background",
        postchat_tags=postchat_tags,
        canonical_raw_response=raw_response,
        canonical_raw_response_sha256=_sha(
            judges._canonical_json_bytes(raw_response)
        ),
        runner_script_sha256="4" * 64,
        evidence_pair_id="5" * 64,
        official_ollama_http=False,
        authoritative_evidence=False,
    )

    with pytest.raises(ValueError, match="exactly one vote"):
        judges._validate_vote_coverage(rows, specs, [first])
    with pytest.raises(ValueError, match="exactly one vote"):
        judges._validate_vote_coverage(rows, specs, [first, first])
    unexpected = _rebind_vote({**first, "judge_id": "unknown-judge"})
    with pytest.raises(ValueError, match="exactly one vote"):
        judges._validate_vote_coverage(rows, specs, [first, unexpected])


def test_requires_validated_filepath_field_instead_of_a_new_crop_path(tmp_path):
    manifest, rows = _manifest(tmp_path, rows=1)
    rows[0]["crop_path"] = rows[0].pop("filepath")
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="filepath"):
        judges.load_background_rows(manifest)


def test_hard_negative_source_one_crop_zero_is_accepted(tmp_path):
    manifest, rows = _manifest(tmp_path, rows=1)
    rows[0]["source_object_count"] = "1"
    rows[0]["crop_object_count"] = "0"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    parsed, _ = judges.load_background_rows(manifest)

    assert len(parsed) == 1
    assert parsed[0].crop_path == (tmp_path / rows[0]["filepath"]).resolve()


def test_hard_negative_source_one_without_crop_count_is_rejected(tmp_path):
    manifest, rows = _manifest(tmp_path, rows=1)
    rows[0]["source_object_count"] = "1"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="crop_object_count is required"):
        judges.load_background_rows(manifest)


def test_legacy_source_zero_without_crop_count_is_accepted(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)

    parsed, _ = judges.load_background_rows(manifest)

    assert len(parsed) == 1


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda specs, root: _rewrite_model_family(specs[1], "INTERNVL"),
            "distinct model_family",
        ),
        (
            lambda specs, root: _json(
                specs[1],
                {
                    **json.loads(specs[1].read_text()),
                    "model_family": "Qwen3-VL",
                },
            ),
            "independent of the Qwen",
        ),
        (
            lambda specs, root: _json(
                specs[1],
                {
                    **json.loads(specs[1].read_text()),
                    "judge_id": "JUDGE-A",
                },
            ),
            "judge_id values must be unique",
        ),
    ],
)
def test_rejects_duplicate_identity_or_qwen_teacher_family(tmp_path, mutator, message):
    specs = _judge_specs(tmp_path)
    mutator(specs, tmp_path)
    with pytest.raises(ValueError, match=message):
        judges.load_judge_specs(specs)


def test_requires_two_judges(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        judges.load_judge_specs(_judge_specs(tmp_path)[:1])


def test_judges_cannot_share_the_same_oci_model_weight_layer(tmp_path):
    shared_weight = b"same-underlying-model-weights"
    specs = [
        _model_spec(
            tmp_path,
            judge_id="judge-a",
            family="internvl",
            model="internvl:8b",
            manifest_bytes=shared_weight,
        ),
        _model_spec(
            tmp_path,
            judge_id="judge-b",
            family="minicpm-v",
            model="minicpm-v:8b",
            manifest_bytes=shared_weight,
        ),
    ]

    with pytest.raises(ValueError, match="distinct OCI model weight layer"):
        judges.load_judge_specs(specs)


def test_model_manifest_tamper_is_rejected_before_http(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    specs = _judge_specs(tmp_path)
    first_spec = json.loads(specs[0].read_text())
    (tmp_path / first_spec["model_manifest_path"]).write_bytes(b"tampered")
    called = False

    def client(*_args):
        nonlocal called
        called = True
        raise AssertionError("HTTP must not be called")

    with pytest.raises(ValueError, match="model manifest SHA-256 mismatch"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=specs,
            output_jsonl=tmp_path / "votes.jsonl",
            output_report=tmp_path / "report.json",
            api_client=client,
        )
    assert called is False


def test_model_config_tamper_is_rejected(tmp_path):
    specs = _judge_specs(tmp_path)
    first_spec = json.loads(specs[0].read_text())
    (tmp_path / first_spec["model_config_path"]).write_text(
        '{"model_family":"internvl","model_type":"internvl","changed":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model config SHA-256 mismatch"):
        judges.load_judge_specs(specs)


def test_qwen_config_cannot_be_disguised_as_an_independent_family(tmp_path):
    specs = _judge_specs(tmp_path)
    second = json.loads(specs[1].read_text())
    config_path = tmp_path / second["model_config_path"]
    qwen_config = b'{"model_family":"qwen3-vl","model_type":"qwen3-vl"}\n'
    config_path.write_bytes(qwen_config)

    manifest_path = tmp_path / second["model_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["digest"] = f"sha256:{_sha(qwen_config)}"
    manifest["config"]["size"] = len(qwen_config)
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    second.update(
        {
            "model_family": "gemma",
            "model_manifest_sha256": _sha(manifest_bytes),
            "model_config_sha256": _sha(qwen_config),
        }
    )
    _json(specs[1], second)

    with pytest.raises(
        ValueError, match="declared model_family does not match immutable model config"
    ):
        judges.load_judge_specs(specs)


def test_qwen_model_type_cannot_hide_behind_a_gemma_family(tmp_path):
    specs = _judge_specs(tmp_path)
    second = json.loads(specs[1].read_text())
    config_path = tmp_path / second["model_config_path"]
    disguised = b'{"model_family":"minicpm-v","model_type":"qwen2-vl"}\n'
    config_path.write_bytes(disguised)
    manifest_path = tmp_path / second["model_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["digest"] = f"sha256:{_sha(disguised)}"
    manifest["config"]["size"] = len(disguised)
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    second["model_config_sha256"] = _sha(disguised)
    second["model_manifest_sha256"] = _sha(manifest_bytes)
    _json(specs[1], second)

    with pytest.raises(
        ValueError, match="immutable model config identifies a Qwen"
    ):
        judges.load_judge_specs(specs)


def test_real_gemma4_size_shaped_model_type_is_not_treated_as_family(tmp_path):
    specs = _judge_specs(tmp_path)
    _rewrite_model_family(specs[1], "gemma4")
    _replace_model_config(
        specs[1],
        {
            "model_family": "gemma4",
            "model_families": ["gemma4"],
            "model_type": "25.8B",
            "renderer": "gemma4",
            "parser": "gemma4",
            "format": "gguf",
        },
    )

    loaded = judges.load_judge_specs(specs)

    gemma = next(spec for spec in loaded if spec.judge_id == "judge-b")
    assert gemma.model_config_families == ("gemma4",)


def test_real_ollama_oci_platform_architecture_is_not_treated_as_model_family(
    tmp_path,
):
    specs = _judge_specs(tmp_path)
    _rewrite_model_family(specs[1], "gemma3")
    _replace_model_config(
        specs[1],
        {
            "model_format": "gguf",
            "model_family": "gemma3",
            "model_families": ["gemma3"],
            "model_type": "4.3B",
            "file_type": "Q4_K_M",
            "architecture": "amd64",
            "os": "linux",
            "model_info": {
                "model_family": "gemma3",
                "general.architecture": "gemma3",
            },
        },
    )

    loaded = judges.load_judge_specs(specs)

    gemma = next(spec for spec in loaded if spec.judge_id == "judge-b")
    assert gemma.model_config_families == ("gemma3",)


@pytest.mark.parametrize(
    "field, disguised_value",
    [("model_type", "qwen2-vl"), ("renderer", "qwen")],
)
def test_qwen_architecture_signal_cannot_hide_behind_gemma4_family(
    tmp_path, field, disguised_value
):
    specs = _judge_specs(tmp_path)
    _rewrite_model_family(specs[1], "gemma4")
    config = {
        "model_family": "gemma4",
        "model_families": ["gemma4"],
        "model_type": "25.8B",
        "renderer": "gemma4",
        "parser": "gemma4",
        "format": "gguf",
    }
    config[field] = disguised_value
    _replace_model_config(specs[1], config)

    with pytest.raises(
        ValueError, match="immutable model config identifies a Qwen"
    ):
        judges.load_judge_specs(specs)


def test_qwen_remains_blocked_when_an_additional_teacher_family_is_supplied(tmp_path):
    specs = _judge_specs(tmp_path)
    second = json.loads(specs[1].read_text())
    config_path = tmp_path / second["model_config_path"]
    qwen_config = b'{"model_family":"qwen3-vl","model_type":"qwen3-vl"}\n'
    config_path.write_bytes(qwen_config)
    manifest_path = tmp_path / second["model_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["digest"] = f"sha256:{_sha(qwen_config)}"
    manifest["config"]["size"] = len(qwen_config)
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    second["model_family"] = "qwen3-vl"
    second["model_config_sha256"] = _sha(qwen_config)
    second["model_manifest_sha256"] = _sha(manifest_bytes)
    _json(specs[1], second)

    with pytest.raises(ValueError, match="independent of the Qwen"):
        judges.load_judge_specs(specs, teacher_model_families=("project-custom",))


def test_arbitrary_text_is_not_accepted_as_model_manifest(tmp_path):
    specs = _judge_specs(tmp_path)
    first = json.loads(specs[0].read_text())
    manifest_path = tmp_path / first["model_manifest_path"]
    manifest_path.write_bytes(b"not-an-ollama-manifest")
    first["model_manifest_sha256"] = _sha(manifest_path.read_bytes())
    _json(specs[0], first)

    with pytest.raises(
        ValueError, match="Ollama tag manifest must be a UTF-8 JSON object"
    ):
        judges.load_judge_specs(specs)


@pytest.mark.parametrize("field", ["source_sha256", "image_sha256"])
def test_source_or_crop_hash_mismatch_is_rejected(tmp_path, field):
    manifest, rows = _manifest(tmp_path, rows=1)
    rows[0][field] = "0" * 64
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="does not match"):
        judges.load_background_rows(manifest)


def test_manifest_relative_filepath_byte_tamper_is_rejected(tmp_path):
    manifest, rows = _manifest(tmp_path, rows=1)
    crop_path = tmp_path / rows[0]["filepath"]
    crop_path.write_bytes(b"tampered-after-manifest")

    with pytest.raises(ValueError, match="crop SHA-256 does not match crop bytes"):
        judges.load_background_rows(manifest)


def test_existing_output_is_never_overwritten_and_skips_http(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    output = tmp_path / "votes.jsonl"
    report = tmp_path / "report.json"
    output.write_text("keep-me", encoding="utf-8")
    called = False

    def client(*_args):
        nonlocal called
        called = True
        raise AssertionError("HTTP must not be called")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=output,
            output_report=report,
            api_client=client,
        )
    assert output.read_text(encoding="utf-8") == "keep-me"
    assert not report.exists()
    assert called is False


def test_second_judge_failure_leaves_no_partial_outputs(tmp_path):
    manifest, _ = _manifest(tmp_path, rows=1)
    calls = 0
    preflight_calls = []
    base_client = _success_api_client(preflight_calls)

    def client(spec, method, endpoint, payload, timeout):
        nonlocal calls
        if endpoint != "/api/chat":
            return base_client(spec, method, endpoint, payload, timeout)
        calls += 1
        if calls == 2:
            return {"model": spec.ollama_model, "message": {"content": ""}}
        return {
            "model": spec.ollama_model,
            "message": {"content": '{"verdict":"background"}'},
        }

    output = tmp_path / "votes.jsonl"
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="missing message.content"):
        judges.run_independent_visual_judges(
            input_manifest=manifest,
            judge_spec_paths=_judge_specs(tmp_path),
            output_jsonl=output,
            output_report=report,
            api_client=client,
        )
    assert calls == 2
    assert not output.exists()
    assert not report.exists()
