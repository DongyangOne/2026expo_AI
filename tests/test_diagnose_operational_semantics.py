"""Bounded CPU diagnostics contracts; mocked replies do not prove model accuracy."""

import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import urllib.error

import pytest

from scripts import build_operational_source_evidence as adapter
from scripts import diagnose_operational_semantics as diagnostic


MODELS = {"qwen3-vl:8b": "a" * 64, "qwen3.5:9b-q4_K_M": "b" * 64}
SECRET = "private-token-and-response-must-not-be-published"


def _helpers():
    path = Path(__file__).with_name("test_build_operational_source_evidence.py")
    spec = importlib.util.spec_from_file_location("_semantic_diagnostic_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
            for path in root.rglob("*")}


def _published_text(root):
    return "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*") if path.is_file())


def _decision(**updates):
    return {"material": "can", "target_bbox_xyxy": [100, 100, 900, 900],
            "target_identifiable": True, "visible_cues": ["metal_rim"],
            "foreign_material": "unknown", "label": "unknown", "quality": "usable",
            "confidence": 0.95, **updates}


def _response(model, decision):
    return {"model": model, "done": True, "done_reason": "stop",
            "message": {"content": json.dumps(decision), "thinking": SECRET},
            "prompt_eval_count": 20, "eval_count": 30, "total_duration": 1_000_000,
            "load_duration": 100_000, "prompt_eval_duration": 200_000, "eval_duration": 700_000}


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    helper = _helpers()
    inputs = helper.case.__wrapped__(tmp_path)
    bundle, _ = helper._bundle(inputs)
    records = adapter.validate_source_evidence_bundle(bundle)
    hold = bundle.parent / "material_semantics_hold.json"
    hold.write_bytes(b'{"quarantined":true,"reason":"observed_material_errors"}\n')
    state = {
        "bundle": bundle, "records": records, "hold": hold,
        "sources": {row["source_sha256"]: Path(row["source_filepath"]).read_bytes() for row in records},
        "calls": [], "models": dict(MODELS), "show": {"capabilities": ["completion", "vision"]},
        "runtime": {name: {"name": name, "digest": sha, "context_length": 8192,
                           "size_vram": 6_000_000_000} for name, sha in MODELS.items()},
        "arguments": dict(source_bundle_dir=bundle,
                          source_sha256s=[row["source_sha256"] for row in records],
                          expected_models=dict(MODELS), output_dir=tmp_path / "diagnostic",
                          semantic_hold_file=hold),
    }
    def request(url, endpoint, payload=None, *, timeout):
        assert url == diagnostic.URL and timeout == 300
        state["calls"].append((endpoint, copy.deepcopy(payload)))
        if endpoint == "/api/tags":
            return {"models": [{"name": name, "digest": sha} for name, sha in state["models"].items()]}
        if endpoint == "/api/show":
            return copy.deepcopy(state["show"])
        if endpoint == "/api/generate":
            state["active_model"] = payload["model"]
            reply = {"model": payload["model"], "done": True, "done_reason": "load", "response": ""}
            if state.get("alter_preload"):
                state["alter_preload"](reply)
            return reply
        if endpoint == "/api/ps":
            return {"models": [copy.deepcopy(state["runtime"][state["active_model"]])]}
        assert endpoint == "/api/chat"
        if state.get("mutate"):
            state["mutate"](state, payload)
        if state.get("error"):
            raise state["error"]
        reply = _response(payload["model"], state.get("decision", _decision()))
        if state.get("alter_reply"):
            state["alter_reply"](reply)
        return reply
    monkeypatch.setattr(diagnostic, "_request_json", request)
    return state


def _run(state):
    return diagnostic.diagnose_semantics(**state["arguments"])


def _failed(state):
    output = state["arguments"]["output_dir"]
    if output.exists():
        failure = json.loads((output / diagnostic.FILES["failure"]).read_bytes())
        assert failure["status"] == "diagnostic_failed"
        assert all(value is False for value in failure["authority"].values())
    assert state["hold"].exists()


def test_six_original_images_no_target_answer_leakage_no_authority_hold_unchanged(setup_run, capsys):
    state = setup_run
    originals, hold = _snapshot(state["bundle"]), state["hold"].read_bytes()
    summary = _run(state)
    assert summary["requests_completed"] == 6 and summary["status"] == "diagnostic_complete"
    assert summary["gpu_runtime_verified_model_groups"] == 2
    assert summary["automatic_decision"] is None
    assert summary["semantic_hold_action"] == "unchanged"
    assert all(value is False for value in summary["authority"].values())
    assert _snapshot(state["bundle"]) == originals and state["hold"].read_bytes() == hold
    chats = [payload for endpoint, payload in state["calls"] if endpoint == "/api/chat"]
    assert len(chats) == 6
    for model in MODELS:
        sent = []
        for payload in [item for item in chats if item["model"] == model]:
            assert set(payload) == {"model", "messages", "format", "stream", "think", "keep_alive", "options"}
            assert payload["format"] == diagnostic.SCHEMA and payload["options"] == diagnostic.OPTIONS
            assert payload["stream"] is False and payload["think"] is False
            assert payload["keep_alive"] == "5m"
            assert len(payload["messages"]) == 1
            message = payload["messages"][0]
            assert set(message) == {"role", "content", "images"}
            assert message["content"] == diagnostic.PROMPT and len(message["images"]) == 1
            image = base64.b64decode(message["images"][0], validate=True)
            sha = hashlib.sha256(image).hexdigest()
            assert state["sources"][sha] == image
            sent.append(sha)
            for row in state["records"]:
                assert row["source_sha256"] not in message["content"]
                assert row["source_filepath"] not in message["content"]
        assert set(sent) == set(state["sources"]) and len(sent) == 3
    output = state["arguments"]["output_dir"]
    assert set(path.name for path in output.iterdir()) == {diagnostic.FILES[key] for key in ("contract", "requests", "summary", "warmups")}
    warmups = [json.loads(line) for line in (output / diagnostic.FILES["warmups"]).read_text().splitlines()]
    entries = [json.loads(line) for line in (output / diagnostic.FILES["requests"]).read_text().splitlines()]
    assert len(warmups) == 2 and all(row["gpu_runtime_verified"] is True for row in warmups)
    for warmup in warmups:
        assert all(value is False for value in warmup["authority"].values())
        bound = hashlib.sha256(diagnostic._json_bytes(warmup)).hexdigest()
        assert all(row["gpu_warmup_sha256"] == bound for row in entries if row["model"] == warmup["model"])
    text = _published_text(output) + capsys.readouterr().out
    assert SECRET not in text
    for image in state["sources"].values():
        assert base64.b64encode(image).decode() not in text


def test_each_model_is_preloaded_verified_and_processed_as_one_serial_group(setup_run):
    state = setup_run
    _run(state)
    sequence = [(endpoint, payload) for endpoint, payload in state["calls"]
                if endpoint in {"/api/generate", "/api/ps", "/api/chat"}]
    assert [endpoint for endpoint, _ in sequence] == ["/api/generate", "/api/ps", *(["/api/chat"] * 3)] * 2
    for offset, model in zip((0, 5), MODELS):
        preload = sequence[offset][1]
        assert preload == {"model": model, "prompt": "", "stream": False,
                           "keep_alive": "5m", "options": diagnostic.OPTIONS}
        assert all(sequence[index][1]["model"] == model for index in range(offset + 2, offset + 5))


@pytest.mark.parametrize("field,value", [
    ("size_vram", 0), ("size_vram", True), ("size_vram", "6000000000"),
    ("name", "unrequested:model"), ("digest", "f" * 64),
    ("context_length", 4096), ("context_length", "8192"),
])
def test_invalid_or_cpu_only_first_model_runtime_sends_zero_images(setup_run, field, value):
    state = setup_run
    state["runtime"][next(iter(MODELS))][field] = value
    with pytest.raises(ValueError, match="diagnostic failed"):
        _run(state)
    _failed(state)
    assert not any(endpoint == "/api/chat" for endpoint, _ in state["calls"])


def test_second_model_cpu_fallback_is_checked_again_before_any_second_group_image(setup_run):
    state = setup_run
    first, second = MODELS
    state["runtime"][second]["size_vram"] = 0
    with pytest.raises(ValueError, match="diagnostic failed"):
        _run(state)
    _failed(state)
    chats = [payload for endpoint, payload in state["calls"] if endpoint == "/api/chat"]
    assert len(chats) == 3 and all(payload["model"] == first for payload in chats)


@pytest.mark.parametrize("field,value", [("done", False), ("done_reason", "length"),
    ("model", "unrequested:model"), ("response", SECRET)])
def test_failed_or_nonempty_preload_cannot_send_images_or_echo_raw_output(setup_run, field, value):
    state = setup_run
    state["alter_preload"] = lambda reply: reply.update({field: value})
    with pytest.raises(ValueError) as error:
        _run(state)
    _failed(state)
    assert not any(endpoint == "/api/chat" for endpoint, _ in state["calls"])
    assert SECRET not in str(error.value) + _published_text(state["arguments"]["output_dir"])


@pytest.mark.parametrize("updates", [
    {"target_bbox_xyxy": [True, 100, 900, 900]}, {"target_bbox_xyxy": [100, 100, 100, 900]},
    {"target_bbox_xyxy": [-1, 100, 900, 900]}, {"target_bbox_xyxy": [100, 100, 1001, 900]},
    {"target_bbox_xyxy": [100, 100, float("nan"), 900]}, {"material": "metal"},
    {"confidence": True}, {"visible_cues": [SECRET]}, {"target_identifiable": False},
])
def test_invalid_bbox_or_enum_fails_without_untrusted_output(setup_run, updates):
    state = setup_run
    state["decision"] = _decision(**updates)
    with pytest.raises(ValueError, match="diagnostic failed"):
        _run(state)
    _failed(state)
    assert SECRET not in _published_text(state["arguments"]["output_dir"])


@pytest.mark.parametrize("field,value", [("done", False), ("done_reason", "length"),
    ("model", "unrequested:model"), ("prompt_eval_count", True), ("eval_count", -1)])
def test_nonstop_wrong_model_or_invalid_counters_fail(setup_run, field, value):
    state = setup_run
    state["alter_reply"] = lambda reply: reply.update({field: value})
    with pytest.raises(ValueError, match="diagnostic failed"):
        _run(state)
    _failed(state)


@pytest.mark.parametrize("when", ["before", "during"])
def test_actual_installed_model_digest_must_match_and_stay_fixed(setup_run, when):
    state = setup_run
    if when == "before":
        state["models"][next(iter(MODELS))] = "f" * 64
    else:
        state["mutate"] = lambda state, _: state["models"].update({next(iter(MODELS)): "f" * 64})
    with pytest.raises(ValueError, match="diagnostic failed"):
        _run(state)
    _failed(state)


@pytest.mark.parametrize("when", ["request", "publication"])
def test_source_bytes_cannot_drift_during_request_or_publication(setup_run, monkeypatch, when):
    state = setup_run
    source = Path(state["records"][0]["source_filepath"])
    def change():
        source.write_bytes(source.read_bytes() + b"changed")
    if when == "request":
        state["mutate"] = lambda *_: change()
    else:
        original = diagnostic._write_exclusive
        def publish_then_change(path, value):
            result = original(path, value)
            if path.name == diagnostic.FILES["summary"]:
                change()
            return result
        monkeypatch.setattr(diagnostic, "_write_exclusive", publish_then_change)
    with pytest.raises(ValueError, match="diagnostic failed"):
        _run(state)
    _failed(state)


def test_deleting_existing_hold_during_diagnosis_cannot_report_success(setup_run):
    state = setup_run
    state["mutate"] = lambda *_: state["hold"].unlink(missing_ok=True)
    with pytest.raises(ValueError, match="diagnostic failed"):
        _run(state)
    failure = json.loads((state["arguments"]["output_dir"] / diagnostic.FILES["failure"]).read_bytes())
    assert failure["status"] == "diagnostic_failed"


@pytest.mark.parametrize("mode", ["http", "json", "thinking_prose", "duplicate_key"])
def test_raw_errors_and_reasoning_are_not_published_or_echoed(setup_run, capsys, mode):
    state = setup_run
    if mode == "http":
        state["error"] = urllib.error.HTTPError(diagnostic.URL, 500, SECRET, {}, io.BytesIO(SECRET.encode()))
    elif mode == "json":
        state["error"] = json.JSONDecodeError(SECRET, SECRET, 0)
    else:
        def alter(reply):
            if mode == "thinking_prose":
                reply["message"] = {"content": "", "thinking": SECRET + "\n" + json.dumps(_decision())}
            else:
                reply["message"]["content"] = json.dumps(_decision())[:-1] + ',"material":"plastic"}'
        state["alter_reply"] = alter
    with pytest.raises(ValueError) as error:
        _run(state)
    _failed(state)
    assert SECRET not in str(error.value) + _published_text(state["arguments"]["output_dir"]) + capsys.readouterr().out


@pytest.mark.parametrize("target", ["bundle", "images", "teacher", "quality"])
def test_output_cannot_pollute_sealed_input_directories(setup_run, target):
    state = setup_run
    receipt = json.loads((state["bundle"] / adapter.FILES["receipt"]).read_bytes())
    if target == "bundle":
        root = state["bundle"]
    elif target == "images":
        root = Path(receipt["image_root"])
    else:
        prefix = "teacher_output_" if target == "teacher" else "quality_"
        root = Path(next(binding["path"] for name, binding in receipt["inputs"].items() if name.startswith(prefix))).parent
    before = _snapshot(root)
    state["arguments"]["output_dir"] = root / "diagnostic-output"
    with pytest.raises(ValueError, match="nested"):
        _run(state)
    assert _snapshot(root) == before and not state["calls"]


def test_existing_output_is_never_overwritten(setup_run):
    state = setup_run
    output = state["arguments"]["output_dir"]
    output.mkdir()
    (output / "keep.txt").write_bytes(b"original")
    before = _snapshot(output)
    with pytest.raises(FileExistsError):
        _run(state)
    assert _snapshot(output) == before and not state["calls"]


def test_vision_capability_is_required_before_sending_images(setup_run):
    state = setup_run
    state["show"]["capabilities"] = ["completion"]
    with pytest.raises(ValueError, match="diagnostic failed"):
        _run(state)
    _failed(state)
    assert not any(endpoint == "/api/chat" for endpoint, _ in state["calls"])


def test_unidentifiable_target_stays_unknown_without_fabricated_bbox(setup_run):
    state = setup_run
    state["decision"] = _decision(target_identifiable=False, target_bbox_xyxy=None,
                                   material="unknown", visible_cues=[], quality="boundary_unreadable")
    summary = _run(state)
    assert all(row["target_bbox_iou"] is None for row in summary["agreements"])
    assert all(value is False for value in summary["authority"].values())


@pytest.mark.parametrize("url", [
    "http://example.com:11434", "https://8.8.8.8", "file:///tmp/ollama",
    "http://user:password@127.0.0.1:11434",
])
def test_external_or_credential_url_is_rejected_before_requests(setup_run, monkeypatch, url):
    state = setup_run
    def forbidden_request(*args, **kwargs):
        pytest.fail("invalid diagnostic URL must not send source images")
    monkeypatch.setattr(diagnostic, "_request_json", forbidden_request)
    with pytest.raises(ValueError):
        diagnostic.diagnose_semantics(**state["arguments"], url=url)
    assert not state["arguments"]["output_dir"].exists()


@pytest.mark.parametrize("change", ["two", "four", "duplicate", "unknown", "malformed"])
def test_only_three_distinct_bound_sources_are_accepted(setup_run, monkeypatch, change):
    state = setup_run
    selected = state["arguments"]["source_sha256s"]
    state["arguments"]["source_sha256s"] = {
        "two": selected[:2], "four": selected + ["f" * 64],
        "duplicate": [selected[0], selected[0], selected[2]],
        "unknown": [selected[0], selected[1], "f" * 64],
        "malformed": [selected[0], selected[1], "not-a-sha"],
    }[change]
    def forbidden_request(*args, **kwargs):
        pytest.fail("unbound sources must not be sent")
    monkeypatch.setattr(diagnostic, "_request_json", forbidden_request)
    with pytest.raises(ValueError):
        diagnostic.diagnose_semantics(**state["arguments"])
    assert not state["arguments"]["output_dir"].exists()


@pytest.mark.parametrize("models", [
    {"qwen3-vl:8b": "a" * 64},
    {"qwen3-vl:8b": "a" * 64, "unknown:9b": "b" * 64},
    {"qwen3-vl:8b": "a" * 64, "qwen3.5:9b-q4_K_M": "a" * 64},
    {"qwen3-vl:8b": "not-a-sha", "qwen3.5:9b-q4_K_M": "b" * 64},
])
def test_exact_two_distinct_pinned_models_required(setup_run, monkeypatch, models):
    state = setup_run
    state["arguments"]["expected_models"] = models
    def forbidden_request(*args, **kwargs):
        pytest.fail("invalid model contract must not send images")
    monkeypatch.setattr(diagnostic, "_request_json", forbidden_request)
    with pytest.raises(ValueError):
        diagnostic.diagnose_semantics(**state["arguments"])
    assert not state["arguments"]["output_dir"].exists()
