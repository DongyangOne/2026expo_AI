import hmac

from fastapi import Header, HTTPException, status

from app.core.config import settings


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    if not hmac.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "유효하지 않은 API 키입니다."},
        )
