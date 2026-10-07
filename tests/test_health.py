import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("API_KEY", "test-key")

from app.main import app, health_check
from app.services import local_llm


class _Registry:
    @staticmethod
    def status():
        return {"main": True, "state": True, "verifier": True}


def _request():
    state = SimpleNamespace(registry=_Registry())
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_health_exposes_healthy_primary_llm(monkeypatch):
    monkeypatch.setattr(local_llm, "health_status", lambda: {
        "enabled": True,
        "required": True,
        "status": "ok",
        "reachable": True,
        "model": "minicpm-v4.5:8b",
        "model_available": True,
        "accelerator": "gpu",
        "size_vram_bytes": 6818260582,
        "latency_ms": 12.3,
    })

    response = asyncio.run(health_check(_request()))

    assert response.status == "ok"
    assert response.llm.status == "ok"
    assert response.llm.model_available is True


def test_health_is_degraded_when_primary_llm_is_unavailable(monkeypatch):
    monkeypatch.setattr(local_llm, "health_status", lambda: {
        "enabled": True,
        "required": True,
        "status": "unavailable",
        "reachable": False,
        "model": "minicpm-v4.5:8b",
        "model_available": False,
        "accelerator": "unknown",
        "size_vram_bytes": None,
        "latency_ms": 3000.0,
    })

    response = asyncio.run(health_check(_request()))

    assert response.status == "degraded"
    assert response.llm.status == "unavailable"


def test_health_swagger_documents_llm_fields():
    operation = app.openapi()["paths"]["/health"]["get"]
    schema_ref = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    schema = app.openapi()["components"]["schemas"][schema_ref.rsplit("/", 1)[-1]]

    assert set(schema["properties"]) == {"status", "models", "llm"}
    llm_ref = schema["properties"]["llm"]["$ref"]
    llm_schema = app.openapi()["components"]["schemas"][llm_ref.rsplit("/", 1)[-1]]
    assert {"status", "reachable", "model", "model_available", "latency_ms"} <= set(
        llm_schema["properties"]
    )
    assert {"accelerator", "size_vram_bytes"} <= set(llm_schema["properties"])


def test_health_is_degraded_when_primary_llm_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(local_llm, "health_status", lambda: {
        "enabled": True,
        "required": True,
        "status": "cpu_fallback",
        "reachable": True,
        "model": "minicpm-v4.5:8b",
        "model_available": True,
        "accelerator": "cpu",
        "size_vram_bytes": 0,
        "latency_ms": 15.0,
    })

    response = asyncio.run(health_check(_request()))

    assert response.status == "degraded"
    assert response.llm.status == "cpu_fallback"
    assert response.llm.accelerator == "cpu"
