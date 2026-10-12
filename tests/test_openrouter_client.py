import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

os.environ.setdefault("API_KEY", "test-key")

from app.services import openrouter_client


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

    def raise_for_status(self):
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, request=self.request)
            raise httpx.HTTPStatusError("failed", request=self.request, response=response)

    def json(self):
        return self._payload


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _configure(monkeypatch, tmp_path):
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_ENABLED", True)
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_PRIMARY_PROVIDER", "deepinfra")
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_FALLBACK_PROVIDERS", "dekallm,parasail")
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_SESSION_ID", "expo-production")
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_TOTAL_TIMEOUT_SEC", 5.0)
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_ATTEMPT_TIMEOUT_SEC", 3.2)
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_KEEPALIVE_START_HOUR", 8)
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_KEEPALIVE_END_HOUR", 19)
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_MAX_TOKENS", 120)
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_USAGE_LOG_PATH", str(tmp_path / "usage.jsonl"))


def test_generate_pins_deepinfra_and_bypasses_openrouter_response_cache(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    content = '{"material":"paper"}'
    client = _Client([_Response({
        "id": "gen-1",
        "provider": "DeepInfra",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24, "cost": 0.00001},
    })])
    monkeypatch.setattr(openrouter_client, "_get_http_client", lambda: client)

    result = openrouter_client.generate("Return JSON", ["YWJj"])

    assert result == content
    _, kwargs = client.calls[0]
    assert kwargs["headers"]["X-OpenRouter-Cache"] == "false"
    assert kwargs["json"]["provider"] == {"only": ["deepinfra"], "allow_fallbacks": False}
    assert kwargs["json"]["session_id"] == "expo-production"
    assert kwargs["json"]["model"] == "qwen/qwen3.8-27b"
    assert kwargs["json"]["reasoning"] == {"effort": "none"}
    assert kwargs["json"]["max_completion_tokens"] == 120
    record = json.loads((tmp_path / "usage.jsonl").read_text(encoding="utf-8"))
    assert record["provider_requested"] == "deepinfra"
    assert record["provider_returned"] == "DeepInfra"
    assert record["status"] == "ok"


def test_generate_falls_back_to_same_model_provider_after_primary_error(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    client = _Client([
        httpx.ReadTimeout("slow", request=request),
        _Response({
            "provider": "DekaLLM",
            "choices": [{"message": {"content": '{"material":"vinyl"}'}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
        }),
    ])
    monkeypatch.setattr(openrouter_client, "_get_http_client", lambda: client)

    result = openrouter_client.generate("Return JSON", ["YWJj"])

    assert result == '{"material":"vinyl"}'
    providers = [call[1]["json"]["provider"]["only"] for call in client.calls]
    assert providers == [["deepinfra"], ["dekallm"]]
    records = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["status"] for record in records] == ["fallback", "ok"]


def test_keepalive_uses_primary_provider_and_one_output_token(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    client = _Client([_Response({
        "provider": "DeepInfra",
        "choices": [{"message": {"content": "OK"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    })])
    monkeypatch.setattr(openrouter_client, "_get_http_client", lambda: client)

    assert openrouter_client.keepalive_once(force=True) is True

    _, kwargs = client.calls[0]
    body = kwargs["json"]
    assert body["provider"] == {"only": ["deepinfra"], "allow_fallbacks": False}
    assert body["max_completion_tokens"] == 1
    assert body["session_id"] == "expo-production"
    assert body["messages"][0]["content"] == "Reply with OK."
    status = openrouter_client.health_status()
    assert status["status"] == "ok"
    assert status["last_keepalive_ok"] is True


def test_disabled_client_does_not_start_keepalive_thread(monkeypatch):
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_ENABLED", False)
    monkeypatch.setattr(openrouter_client.settings, "OPENROUTER_API_KEY", None)

    assert openrouter_client.start_keepalive() is False


def test_keepalive_skips_before_08_and_from_19_kst(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    client = _Client([])
    monkeypatch.setattr(openrouter_client, "_get_http_client", lambda: client)

    before = datetime(2026, 10, 13, 7, 59, tzinfo=ZoneInfo("Asia/Seoul"))
    at_end = datetime(2026, 10, 13, 19, 0, tzinfo=ZoneInfo("Asia/Seoul"))

    assert openrouter_client.keepalive_once(now=before) is None
    assert openrouter_client.keepalive_once(now=at_end) is None
    assert client.calls == []


def test_keepalive_runs_from_08_until_before_19_kst(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    client = _Client([
        _Response({"provider": "DeepInfra", "choices": [{"message": {"content": "OK"}}]}),
        _Response({"provider": "DeepInfra", "choices": [{"message": {"content": "OK"}}]}),
    ])
    monkeypatch.setattr(openrouter_client, "_get_http_client", lambda: client)

    at_start = datetime(2026, 10, 13, 8, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    before_end = datetime(2026, 10, 13, 18, 59, 59, tzinfo=ZoneInfo("Asia/Seoul"))

    assert openrouter_client.keepalive_once(now=at_start) is True
    assert openrouter_client.keepalive_once(now=before_end) is True
    assert len(client.calls) == 2
