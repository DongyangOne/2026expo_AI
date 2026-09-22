# 2026 동양미래 EXPO 재활용품 AI 서버

카메라 이미지와 무게 센서값으로 단일 투입 쓰레기를 판정하는 FastAPI 서버입니다.

- YOLO: 객체 bbox와 `NOT_DETECTED` 판정
- NAS 로컬 Vision LLM: bbox crop의 품목·라벨·압착·외부 이물질 최종 판정
- 규칙 엔진: 무게·상태 조건을 조합해 `ALLOWED`·`REJECTED`·`GENERAL_WASTE` 결정
- Spring 콜백: 하드웨어 즉시 응답과 같은 JSON을 백그라운드 전송

## API

`POST /api/v1/detect` (`multipart/form-data`)

| 필드 | 필수 | 설명 |
|---|---:|---|
| `image` | 예 | JPG 또는 PNG 이미지 |
| `client_id` | 예 | 하드웨어·사용자·피드백 구분 ID. 응답과 Spring 콜백에 그대로 포함 |
| `weight_g` | 아니오 | 그램 단위 무게. 생략하면 무게 이상 검사를 하지 않음 |

헤더 `X-API-Key`가 필요합니다. 전체 요청·응답 스키마와 모든 분기표는 실행 중인 Swagger
`/docs` 및 [로컬 LLM 운영 규약](docs/NACO_LOCAL_LLM_ROLLOUT.md)에 있습니다.

### 핵심 응답 규약

| status | 의미 | class/코드 |
|---|---|---|
| `ALLOWED` | 조건을 충족한 캔·플라스틱(PET 포함)·종이·비닐 투입 허용 | 정상 비닐도 `5 / vinyl` |
| `REJECTED` | 재처리 또는 완전 수거 거부 | `guidance` 또는 `rejection`으로 원인 전달 |
| `GENERAL_WASTE` | LLM이 최종 품목을 확정하지 못한 보류 | `general.code=LOW_CONFIDENCE`, `classification` 생략 |
| `NOT_DETECTED` | 빈 저울 하한 또는 bbox 미감지 | `classification` 생략 |

PET는 외부 계약에서 항상 `class_id=3`, `class_name=plastic`으로 통합합니다.

`LOCAL_LLM_MODE=primary`에서 LLM이 장애·JSON 오류·저신뢰·복수 물체를 반환하면 YOLO 품목으로
대체하지 않습니다. 기본 설정 `LOCAL_LLM_PRIMARY_FALLBACK_TO_YOLO=false`에서는
`GENERAL_WASTE / LOW_CONFIDENCE`로 fail-closed 처리합니다.

## guidance / rejection 코드

| 조건 | 코드 |
|---|---|
| 플라스틱(PET 포함)·캔 무게 이상 또는 내용물 존재 추정 | `EMPTY_CONTENTS` |
| 종이·비닐 무게 이상 | `WEIGHT_ANOMALY` |
| 다른 재질의 부착물·혼합 이물질 | `FOREIGN_MATERIAL` |
| 플라스틱(PET 포함) 라벨 미제거 | `REMOVE_LABEL` |
| PET병·캔 미압착 | `COMPRESS` |
| 유리·건전지·형광등·스티로폼 | `GLASS`·`BATTERY`·`FLUORESCENT`·`STYROFOAM` rejection code |

같은 재질 부속품(예: 플라스틱 빨대)은 `FOREIGN_MATERIAL` 대상이 아닙니다.

## 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Swagger: `http://localhost:8000/docs`
OpenAPI JSON: `http://localhost:8000/openapi.json`

## 필수 환경 변수

```env
API_KEY=replace-me
LOCAL_LLM_MODE=primary
# API-key 인증이 적용된 공개 TLS gateway. Ollama 원 포트는 공개하지 않는다.
LOCAL_LLM_BASE_URL=https://llm.naco.kro.kr
LOCAL_LLM_API_KEY=replace-me
LOCAL_LLM_MODEL=qwen3.5:9b-q4_K_M
LOCAL_LLM_PRIMARY_FALLBACK_TO_YOLO=false
SPRING_CALLBACK_URL=https://oneexpo.kro.kr/api/v1/feedback-detail/result
```

전체 설정 예시는 [.env.example](.env.example)에 있습니다. 실제 API key와 gateway 인증값은
Git에 넣지 않습니다.

## 로그와 캡처

`LOG_RESULTS=true`이면 `logs/results.jsonl`에 결과를 기록합니다. `CAPTURE_REQUESTS=true`이면
`logs/captures/`에 원본 이미지와 판정 JSON을 저장합니다. 이 데이터는 재학습 후보일 뿐 자동 정답이나
자동 배포 근거가 아닙니다.

## 검증과 배포

로컬 핵심 회귀 검증:

```bash
python -m pytest -q tests/test_pipeline.py tests/test_local_llm.py tests/test_verifier_pipeline.py
```

`main` push 시 GitHub Actions가 테스트를 실행합니다. Pi 운영 checkout은 현재 원격 `main`과
분기되어 있으므로, 자동 pull 성공을 배포 증거로 보지 않습니다. 운영 반영은 컨테이너 health와
`/openapi.json`을 실제로 확인합니다.
