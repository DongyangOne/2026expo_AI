import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from app.api.v1 import detect
from app.core.config import settings
from app.models.registry import ModelRegistry
from app.schemas.response import ErrorResponse
from app.services import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== 서버 시작 — 모델 로드 중 ===")
    try:
        app.state.registry = ModelRegistry(
            main_path=settings.MAIN_MODEL_PATH,
            state_path=settings.STATE_MODEL_PATH,
            verifier_path=settings.VERIFIER_MODEL_PATH,
        )
        logger.info("모델 로드 완료: %s", app.state.registry.status())
    except FileNotFoundError as exc:
        logger.critical("모델 파일 없음 — 감지 요청 처리 불가: %s", exc)
        app.state.registry = None
    except Exception:
        logger.critical("모델 로드 실패", exc_info=True)
        app.state.registry = None
    yield
    pipeline.shutdown()
    logger.info("=== 서버 종료 ===")


app = FastAPI(
    title="2026 EXPO 재활용품 AI 분류 서버",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.include_router(detect.router, prefix="/api/v1", tags=["detect"])


# ── 예외 핸들러 ──────────────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        error = detail
    else:
        error = {"code": "HTTP_ERROR", "message": str(detail)}
    return JSONResponse(status_code=exc.status_code, content={"error": error})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "VALIDATION_ERROR", "message": str(first_error.get("msg", exc))}},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("처리되지 않은 예외: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "서버 내부 오류가 발생했습니다."}},
    )


# ── 시스템 엔드포인트 ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"], summary="서버 및 모델 상태 확인")
async def health_check(request: Request) -> dict:
    registry = getattr(request.app.state, "registry", None)
    return {
        "status": "ok",
        "models": registry.status() if registry else {
            "main": False, "state": False, "verifier": False,
        },
    }
