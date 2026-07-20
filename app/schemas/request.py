from typing import Optional

from fastapi import File, Form, UploadFile


class DetectFormData:
    """
    multipart/form-data 요청 파라미터.
    FastAPI Depends() 로 주입된다.
    """

    def __init__(
        self,
        image: UploadFile = File(..., description="분류할 쓰레기 이미지 (jpg/png)"),
        client_id: str = Form(
            ...,
            min_length=1,
            max_length=128,
            description="하드웨어가 전달하는 사용자/피드백 구분 ID (응답과 Spring 콜백에 그대로 반환)",
        ),
        weight_g: Optional[float] = Form(None, description="무게 센서 값 (그램, 미입력 시 이상 감지 생략)"),
    ):
        self.image = image
        self.client_id = client_id
        self.weight_g = weight_g
