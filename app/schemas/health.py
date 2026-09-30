from typing import Literal

from pydantic import BaseModel, Field


class ModelHealth(BaseModel):
    main: bool = Field(..., description="주 YOLO 모델 로드 여부")
    state: bool = Field(..., description="상태 보조 모델 로드 여부")
    verifier: bool = Field(..., description="crop 검증기 로드 여부")


class LLMHealth(BaseModel):
    enabled: bool = Field(..., description="NAS Vision LLM 기능 활성화 여부")
    required: bool = Field(..., description="현재 분류에서 LLM이 필수(primary)인지 여부")
    status: Literal["ok", "disabled", "unavailable", "model_missing"] = Field(
        ..., description="NAS LLM gateway와 운영 모델 상태"
    )
    reachable: bool = Field(..., description="NAS Ollama API 응답 가능 여부")
    model: str = Field(..., description="운영에 설정된 Vision LLM 모델명")
    model_available: bool = Field(..., description="설정한 모델이 NAS 모델 목록에 존재하는지 여부")
    latency_ms: float | None = Field(None, description="NAS 상태 확인 왕복 시간(ms)")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(
        ..., description="필수 YOLO와 primary LLM이 모두 정상이면 ok, 아니면 degraded"
    )
    models: ModelHealth
    llm: LLMHealth

