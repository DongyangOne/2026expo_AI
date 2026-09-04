from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # ── 인증 ────────────────────────────────────────────────────────────────────
    API_KEY: str

    # ── 모델 경로 ────────────────────────────────────────────────────────────────
    # 메인 9-class YOLO. NCNN: 디렉터리 경로 / .pt: 파일 경로
    MAIN_MODEL_PATH: str = "weights/yolo26m_best_ncnn_model"
    # 상태 멀티헤드(dent/label) ONNX 경로. None = 미탑재 → conditions=null (무게만 검사)
    STATE_MODEL_PATH: str | None = "weights/multihead.onnx"
    # YOLO bbox를 320px crop으로 다시 확인하는 임시 9종+상태 검증기.
    VERIFIER_MODEL_PATH: str | None = "weights/verifier_qwen35_mnv3_v1.onnx"
    # 초기에는 응답/콜백을 바꾸지 않고 YOLO와의 비교 결과만 별도 JSONL에 기록한다.
    VERIFIER_SHADOW_ENABLED: bool = True
    VERIFIER_SHADOW_LOG_PATH: str = "logs/verifier_shadow.jsonl"

    # YOLO와 crop 검증기가 응답 클래스에서 불일치하면 확정하지 않고 일반쓰레기로 보낸다.
    # 학습에 쓰이지 않은 하드웨어 감사 35건에서 두 모델이 합의한 21건은 전부 정답이었고,
    # 불일치한 14건은 YOLO 4건·검증기 5건만 맞아 어느 쪽도 근거가 되지 못했다.
    # 켜면 확정 오답이 7→0건으로 줄지만 보류가 3→14건으로 늘어 처리량을 포기한다.
    VERIFIER_AGREEMENT_GATE_ENABLED: bool = False

    # 저신뢰 PET/PLASTIC과 VINYL이 같은 bbox에서 경쟁할 때만 crop 검증기로 교정한다.
    VINYL_CORRECTION_ENABLED: bool = True
    VINYL_CANDIDATE_CONF: float = 0.10
    VINYL_CANDIDATE_IOU: float = 0.70
    VINYL_CANDIDATE_RATIO: float = 0.40
    VINYL_VERIFIER_CONF: float = 0.65
    VINYL_VERIFIER_MARGIN: float = 0.25

    # ── 추론 설정 ────────────────────────────────────────────────────────────────
    # 2단계 임계값:
    #   DETECT_CONF 미만 → 박스 없음 → NOT_DETECTED
    #   DETECT_CONF 이상 ~ TRUST_CONF 미만 → LOW_CONFIDENCE (일반쓰레기)
    #   TRUST_CONF 이상 → ALLOWED / REJECTED 정상 판정
    DETECT_CONF: float = 0.25       # YOLO 박스 생성 최소 신뢰도
    DETECT_IOU: float = 0.70        # NMS IoU (Ultralytics 기본값을 명시적으로 고정)
    TRUST_CONF: float = 0.55        # 신뢰 판정 임계값
    IMG_SIZE: int = 640
    SUB_IMG_SIZE: int = 224
    DEVICE: str = "cpu"              # Pi5 CPU, GPU 환경이면 "0"

    # ── 무게 이상 감지 ────────────────────────────────────────────────────────────
    WEIGHT_ANOMALY_ENABLED: bool = True

    # ── Spring 콜백 ──────────────────────────────────────────────────────────────
    # None 이면 콜백 비활성 (Spring URL 미설정 시)
    SPRING_CALLBACK_URL: str | None = None
    SPRING_TIMEOUT_SEC: float = 3.0
    # 순간적인 타임아웃/연결 오류/5xx 응답만 재시도한다. 4xx는 계약 오류이므로 즉시 종료한다.
    SPRING_MAX_ATTEMPTS: int = 3
    SPRING_RETRY_BACKOFF_SEC: float = 0.5

    # ── 결과 로깅 ────────────────────────────────────────────────────────────────
    # True: 판정 결과를 logs/results.jsonl 에 항상 기록 (Spring URL 없어도 동작)
    LOG_RESULTS: bool = True
    LOG_DIR: str = "logs"

    # ── 재학습/오인식 검수용 요청 캡처 ───────────────────────────────────────────
    # 원본 이미지와 판정 JSON을 동일한 capture_id로 저장한다.
    CAPTURE_REQUESTS: bool = True
    CAPTURE_DIR: str = "logs/captures"
    CAPTURE_MAX_IMAGE_BYTES: int = 20 * 1024 * 1024
    CAPTURE_RETENTION_DAYS: int = 90
    CAPTURE_MAX_STORAGE_MB: int = 10 * 1024


settings = Settings()
