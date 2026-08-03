import asyncio
import os

import pytest
from fastapi import HTTPException

os.environ.setdefault("API_KEY", "test-key")

from app.core import security


def test_non_ascii_api_key_is_rejected_without_internal_error(monkeypatch):
    monkeypatch.setattr(security.settings, "API_KEY", "test-key")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(security.verify_api_key("잘못된-키"))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["code"] == "UNAUTHORIZED"


def test_valid_api_key_still_passes(monkeypatch):
    monkeypatch.setattr(security.settings, "API_KEY", "test-key")

    assert asyncio.run(security.verify_api_key("test-key")) is None
