import json
from pathlib import Path

import scripts.label_operational_captures_ollama as teacher


def _queue_row(image: Path, sha256: str) -> dict:
    return {"sha256": sha256, "image_path": str(image)}


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
        return {
            "material": "can",
            "confidence": 0.9,
            "single_object": True,
            "foreign_material": False,
        }

    monkeypatch.setattr(teacher, "_request", fake_request)
    output = tmp_path / "labels.jsonl"
    summary = teacher.label_queue(
        queue,
        output,
        url="http://teacher",
        model="test-model",
        timeout=1,
        retries=1,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["consensus"] is False
    assert rows[0]["minimum_confidence"] == 0
    assert rows[0]["errors"] == ["ValueError: empty model content"]
    assert rows[1]["consensus"] is True
    assert summary == {
        "queued": 2,
        "completed": 2,
        "consensus": 1,
        "high_confidence_consensus": 1,
    }

