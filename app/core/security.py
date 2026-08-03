import hmac

from fastapi import Header, HTTPException, status

from app.core.config import settings


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    # str 비교는 한쪽에 비ASCII 문자가 있으면 TypeError를 일으킨다. UTF-8 bytes로
    # 비교해 잘못된 어떤 헤더 값도 500이 아니라 일관된 401로 처리한다.
    if not hmac.compare_digest(
        x_api_key.encode("utf-8"),
        settings.API_KEY.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "유효하지 않은 API 키입니다."},
        )
