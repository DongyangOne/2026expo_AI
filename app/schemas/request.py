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
        weight_g: Optional[float] = Form(None, description="무게 센서 값 (그램, 미입력 시 이상 감지 생략)"),
    ):
        self.image = image
        self.weight_g = weight_g
