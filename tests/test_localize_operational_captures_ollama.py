import base64
import hashlib
import io
import json
import struct
import urllib.error

import pytest

from scripts import localize_operational_captures_ollama as localizer
from scripts.build_independent_localization_consensus import _provider_rows


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _decision(status="single", box=None, confidence=0.95):
    return {"status": status, "bbox_norm": box if box is not None else [100, 200, 900, 800],
            "confidence": confidence}


def _response(decision=None):
    return json.dumps({"model": "vision:3b", "done": True, "done_reason": "stop",
                       "message": {"content": json.dumps(decision or _decision())}}).encode()


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    pixels = localizer.capture_queue.np.full((240, 320, 3), 128, dtype="uint8")
    encoded = localizer.capture_queue.cv2.imencode(".png", pixels)[1].tobytes()
    image = images / "raw.png"
    image.write_bytes(encoded)
    queue = tmp_path / "queue.jsonl"
    entry = {"sha256": _sha(encoded), "image_ref": image.name,
             "timestamp": "2026-08-01T00:00:00+09:00", "decision": "teacher_required"}
    queue.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    known = tmp_path / "known.json"
    known.write_text("{}", encoding="utf-8")
    model = tmp_path / "weights.gguf"
    model.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 1, 1) + b"fixture tensor data")
    show = {"capabilities": ["completion", "vision"], "details": {"format": "gguf"},
            "modelfile": f"FROM /root/.ollama/models/blobs/sha256-{_sha(model.read_bytes())}\n"}
    state = {"calls": [], "reply": _response(), "show": show, "entry": entry,
             "image": image, "model": model}

    def post(url, endpoint, payload, timeout):
        if endpoint == "/api/show":
            return json.dumps(state["show"]).encode()
        assert endpoint == "/api/chat"
        state["calls"].append(payload)
        if state.get("mutate"):
            state["mutate"]()
        return state["reply"]

    monkeypatch.setattr(localizer, "_post", post)
    monkeypatch.setattr(localizer.teacher, "_observe_model_digest", lambda *args: "a" * 64)
    state["args"] = dict(queue_path=queue, image_root=images, known_audit=known,
                         output_dir=tmp_path / "out", provider="vision_provider",
                         model="vision:3b", model_digest="a" * 64, model_file=model)
    return state


def test_exact_provider_contract_and_raw_image_only_request(setup_run):
    state = setup_run
    receipt = localizer.localize_queue(**state["args"])
    out = state["args"]["output_dir"]
    rows, evidence = _provider_rows(out / localizer.FILES["manifest"],
                                    provider="vision_provider", model_file=state["model"],
                                    spec_file=out / localizer.FILES["spec"])
    row = rows[state["entry"]["sha256"]]
    assert row["bbox_xyxy"] == [32.0, 48.0, 288.0, 192.0]
    assert row["model_sha256"] == _sha(state["model"].read_bytes())
    assert row["model_sha256"] != state["args"]["model_digest"]
    assert receipt["accepted"] == 1 and receipt["rejected"] == 0
    assert not any(receipt["authority"].values())
    payload = state["calls"][0]
    assert payload["messages"] == [{"role": "user", "content": localizer.PROMPT,
                                     "images": [base64.b64encode(state["image"].read_bytes()).decode()]}]
    assert set(payload) == {"model", "messages", "format", "stream", "think", "keep_alive", "options"}
    assert payload["options"]["num_ctx"] == 8192
    assert state["entry"]["sha256"] not in json.dumps(payload)
    raw = json.loads((out / localizer.FILES["raw"]).read_text())
    assert base64.b64decode(raw["raw_response_b64"]) == state["reply"]
    assert raw["raw_response_sha256"] == _sha(state["reply"])
    assert raw["inference_spec_sha256"] == evidence["inference_spec_sha256"]
    for line in (out / localizer.FILES["marker"]).read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert _sha((out / name).read_bytes()) == digest


@pytest.mark.parametrize("status", ["empty", "multiple", "ambiguous"])
def test_non_single_excluded_but_receipt_retained(setup_run, status):
    setup_run["reply"] = _response(_decision(status, []))
    receipt = localizer.localize_queue(**setup_run["args"])
    assert receipt["accepted"] == 0
    assert receipt["reason_counts"] == {status: 1}
    assert (setup_run["args"]["output_dir"] / localizer.FILES["manifest"]).read_bytes() == b""


@pytest.mark.parametrize("decision", [
    _decision(box=[True, 2, 3, 4]), _decision(box=[-1, 2, 3, 4]),
    _decision(box=[1, 2, 1001, 4]), _decision(box=[100, 200, 100, 300]),
    _decision(box=[1, 2, 3]), _decision(confidence=True),
    _decision(confidence=float("nan")), _decision(box=[0, 0, float("inf"), 100]),
    _decision("empty", [1, 2, 3, 4]), dict(_decision(), material="plastic"),
])
def test_invalid_decision_does_not_publish_provider_row(setup_run, decision):
    setup_run["reply"] = _response(decision)
    receipt = localizer.localize_queue(**setup_run["args"])
    assert receipt["accepted"] == 0
    assert receipt["reason_counts"] == {"invalid_reply:ValueError": 1}


def test_low_confidence_is_not_accepted(setup_run):
    setup_run["reply"] = _response(_decision(confidence=0.79))
    assert localizer.localize_queue(**setup_run["args"])["reason_counts"] == {"low_confidence": 1}


@pytest.mark.parametrize("reply", [
    b"not JSON", b'{"model":"vision:3b","model":"vision:3b"}',
    json.dumps({"model": "vision:3b", "done": True, "done_reason": "length", "message": {}}).encode(),
    json.dumps({"model": "another", "done": True, "done_reason": "stop", "message": {}}).encode(),
])
def test_raw_invalid_responses_are_preserved(setup_run, reply):
    setup_run["reply"] = reply
    assert localizer.localize_queue(**setup_run["args"])["accepted"] == 0
    raw = json.loads((setup_run["args"]["output_dir"] / localizer.FILES["raw"]).read_text())
    assert base64.b64decode(raw["raw_response_b64"]) == reply


@pytest.mark.parametrize("update", [
    {"timestamp": "2026-07-31T23:59:59+09:00"},
    {"timestamp": "2026-08-01T00:00:00"}, {"sha256": None},
    {"deployed": {"bbox": [1, 2, 3, 4]}}, {"decision": "approved"},
])
def test_queue_contract_rejected_before_chat(setup_run, update):
    row = dict(setup_run["entry"], **update)
    setup_run["args"]["queue_path"].write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        localizer.localize_queue(**setup_run["args"])
    assert setup_run["calls"] == []
    assert not setup_run["args"]["output_dir"].exists()


def test_known_audit_protected_capture_not_sent(setup_run):
    setup_run["args"]["known_audit"].write_text(json.dumps({setup_run["entry"]["sha256"]:
                                                           {"split": "protected_validation"}}))
    with pytest.raises(ValueError, match="protected"):
        localizer.localize_queue(**setup_run["args"])
    assert not setup_run["calls"]


@pytest.mark.parametrize("mutation", ["json_model", "from_mismatch", "not_vision", "package_digest"])
def test_model_identity_is_actual_vision_blob(setup_run, mutation, monkeypatch):
    if mutation == "json_model":
        setup_run["model"].write_bytes(b'{"fake_weight_identity":"' + b"a" * 64 + b'"}')
    elif mutation == "from_mismatch":
        setup_run["show"]["modelfile"] = "FROM /blobs/sha256-" + "b" * 64
    elif mutation == "not_vision":
        setup_run["show"]["capabilities"] = ["completion"]
    else:
        monkeypatch.setattr(localizer.teacher, "_observe_model_digest", lambda *args: "b" * 64)
    with pytest.raises(ValueError):
        localizer.localize_queue(**setup_run["args"])
    assert not setup_run["calls"]
    assert not setup_run["args"]["output_dir"].exists()


@pytest.mark.parametrize("target", ["image", "model", "queue", "audit", "show"])
def test_mutated_sources_fail_without_ready_marker(setup_run, target):
    state = setup_run
    def mutate():
        if target == "image":
            state["image"].write_bytes(b"changed")
        elif target == "model":
            state["model"].write_bytes(state["model"].read_bytes() + b"changed")
        elif target == "queue":
            state["args"]["queue_path"].write_bytes(b"changed")
        elif target == "audit":
            state["args"]["known_audit"].write_bytes(b"changed")
        else:
            state["show"]["extra"] = "changed"
    state["mutate"] = mutate
    with pytest.raises(ValueError, match="changed"):
        localizer.localize_queue(**state["args"])
    out = state["args"]["output_dir"]
    assert (out / "failed.json").exists()
    assert not (out / localizer.FILES["marker"]).exists()


def test_existing_output_is_untouched(setup_run):
    output = setup_run["args"]["output_dir"]
    output.mkdir()
    prior = output / "prior"
    prior.write_bytes(b"keep")
    with pytest.raises(ValueError, match="new"):
        localizer.localize_queue(**setup_run["args"])
    assert list(output.iterdir()) == [prior] and prior.read_bytes() == b"keep"


def test_objectively_bad_capture_is_not_sent(setup_run):
    setup_run["image"].write_bytes(b"not an image")
    receipt = localizer.localize_queue(**setup_run["args"])
    assert receipt["accepted"] == 0 and not setup_run["calls"]
    assert "image_unreadable" in receipt["decisions"][0]["reason"]


def test_source_symlink_is_not_followed(setup_run):
    source = setup_run["image"]
    target = source.with_name("target.png")
    source.rename(target)
    try:
        source.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation not available")
    assert localizer.localize_queue(**setup_run["args"])["accepted"] == 0
    assert not setup_run["calls"]


def test_http_failure_preserves_raw_body_but_not_ready(setup_run, monkeypatch):
    original = localizer._post
    def post(url, endpoint, payload, timeout):
        if endpoint == "/api/chat":
            raise urllib.error.HTTPError(url, 500, "failed", {}, io.BytesIO(b'{"error":"CUDA unavailable"}'))
        return original(url, endpoint, payload, timeout)
    monkeypatch.setattr(localizer, "_post", post)
    with pytest.raises(RuntimeError, match="transport"):
        localizer.localize_queue(**setup_run["args"])
    out = setup_run["args"]["output_dir"]
    assert (out / "failed.json").exists()
    assert not (out / localizer.FILES["marker"]).exists()
    raw = json.loads((out / localizer.FILES["raw"]).read_text())
    assert base64.b64decode(raw["raw_response_b64"]) == b'{"error":"CUDA unavailable"}'


def test_show_json_key_order_may_change_but_initial_raw_hash_is_preserved(setup_run, monkeypatch):
    original = localizer._post
    initial = json.dumps(setup_run["show"]).encode()
    count = 0
    def post(url, endpoint, payload, timeout):
        nonlocal count
        if endpoint == "/api/show":
            count += 1
            return initial if count == 1 else json.dumps(dict(reversed(list(setup_run["show"].items()))), indent=2).encode()
        return original(url, endpoint, payload, timeout)
    monkeypatch.setattr(localizer, "_post", post)
    receipt = localizer.localize_queue(**setup_run["args"])
    assert receipt["model_show_sha256"] == _sha(initial)
    assert (setup_run["args"]["output_dir"] / localizer.FILES["show"]).read_bytes() == initial


def test_progress_is_flushed_without_private_ids(setup_run, capsys):
    localizer.localize_queue(**setup_run["args"])
    output = capsys.readouterr().out
    assert json.loads(output) == {"processed": 1, "queued": 1, "accepted": 1, "reason": "accepted"}
    assert setup_run["entry"]["sha256"] not in output
    assert setup_run["image"].name not in output


def test_output_symlink_ancestor_rejected(setup_run, tmp_path):
    target = tmp_path / "target_dir"
    target.mkdir()
    link = tmp_path / "output_link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation not available")
    setup_run["args"]["output_dir"] = link / "out"
    with pytest.raises(ValueError, match="symlink"):
        localizer.localize_queue(**setup_run["args"])
    assert list(target.iterdir()) == []
