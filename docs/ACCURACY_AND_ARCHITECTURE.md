# EXPO AI 구성 및 평가 결과

기준일: 2026-09-30

## 운영 구성

| 위치 | 모델 | 역할 |
|---|---|---|
| Raspberry Pi 5 | `yolo26m_best_ncnn_model` | 물체 위치(`bbox`) 검출, 9종 후보와 신뢰도 생성 |
| QNAP NAS Ollama | `minicpm-v4.5:8b` | 최종 품목, 라벨, 압착, 외부 이물질 판정 |
| Raspberry Pi 5 | `multihead.onnx` | 라벨·압착 상태의 기본값 생성 |
| Raspberry Pi 5 | `verifier_qwen35_mnv3_v1.onnx` | crop 재검증, 저신뢰 보조 판정, 상태 헤드 및 비교 로그 |

운영 모드는 `LOCAL_LLM_MODE=primary`다. 최종 품목은 NAS Vision LLM이 결정하며,
YOLO는 물체 위치와 재판정 경로 선택에 사용된다. LLM 장애·형식 오류·저신뢰·복수 물체
판정 시 YOLO 품목으로 대체하지 않고 `GENERAL_WASTE / LOW_CONFIDENCE`를 반환한다.

## 처리 순서

1. 하드웨어가 이미지, `client_id`, 선택 항목인 `weight_g`를 `POST /api/v1/detect`로 보낸다.
2. Pi의 YOLO26m NCNN 모델이 9종 후보와 bbox를 만든다.
3. bbox가 있으면 전체 프레임과 10% 여백을 둔 crop을 NAS Vision LLM에 전달한다.
4. bbox가 없으면 원본 전체 프레임을 Vision LLM에 전달한다.
5. Vision LLM이 품목과 내부 압착 대상 여부 `compression_required`, `has_label`,
   `is_dented`, `has_foreign_material`, `is_single_primary_item`을 JSON으로 반환한다.
6. 필요한 경우에만 비닐/플라스틱 또는 재질 충돌 2차 판정을 실행한다.
7. 무게와 상태 규칙을 적용해 `ALLOWED`, `REJECTED`, `GENERAL_WASTE`,
   `NOT_DETECTED` 중 하나를 결정한다.
8. 같은 JSON을 하드웨어에 반환하고 Spring callback으로 백그라운드 전송한다.

## 모델 입력과 추론 설정

### YOLO26m

- 입력 크기: 640px
- bbox 생성 임계값: `DETECT_CONF=0.25`
- NMS IoU: `0.70`
- 내부 클래스: `can`, `pet`, `paper`, `plastic`, `styrofoam`, `vinyl`,
  `glass`, `battery`, `fluorescent`
- 내부 `pet / class_id=1`은 외부 응답에서 `plastic / class_id=3`으로 변환한다.

### MiniCPM-V 4.5 8B

- 전체 프레임: 최대 768px
- bbox crop: 10% 여백, 최대 640px
- 플라스틱 상세 재판정 crop: 최대 896px
- 최소 인정 신뢰도: `0.80`
- `temperature=0`, `think=false`, 최대 출력 96 tokens
- 모델 상주 시간: 24시간
- 출력 형식: 품목, 신뢰도, 내부 압착 대상 여부, 라벨, 압착, 외부 이물질,
  단일 주 물체 여부를 포함한 JSON

## 선택적 2차 판정

일반 요청은 Vision LLM을 한 번 호출한다. 다음 조건에서만 두 번째 판정을 실행한다.

| 조건 | 2차 판정 |
|---|---|
| YOLO가 `vinyl`, 1차 LLM이 `plastic` | 얇은 필름·봉투와 단단한 용기를 구분하는 전용 판정 |
| YOLO `vinyl` 신뢰도 0.50 이상, 무게 5g 이하 | 첫 호출부터 vinyl/plastic 전용 판정 사용 |
| 고신뢰 YOLO와 LLM이 지정 재질 쌍에서 충돌 | 두 후보만 비교하는 재질 판정 |
| YOLO `plastic` 신뢰도 0.95 이상, LLM이 `paper` 또는 `styrofoam` | 896px crop을 포함한 플라스틱 재질 판정 |
| LLM이 외부 이물질을 감지 | 배경과 고정 장치를 제외하기 위한 crop 전용 재확인 |

재질 충돌 판정 대상은 캔/플라스틱, 플라스틱/종이, 플라스틱/스티로폼,
비닐/종이, PET/유리, 유리/플라스틱, 형광등/유리, 형광등/플라스틱이다.

## 분류 및 상태 규칙

| 품목 | 정상 처리 | 상태 이상 |
|---|---|---|
| 캔 | `ALLOWED` | 무게 이상 `EMPTY_CONTENTS`, 미압착 `COMPRESS` |
| PET병 | 외부 `plastic/3`, 조건 충족 시 `ALLOWED` | `EMPTY_CONTENTS`, `REMOVE_LABEL`, `COMPRESS` |
| 플라스틱 | 압착 비대상이거나 압착 대상이 압착된 경우 `ALLOWED` | `EMPTY_CONTENTS`, `REMOVE_LABEL`, 압착 대상 미압착 `COMPRESS` |
| 종이 | `ALLOWED` | `WEIGHT_ANOMALY` |
| 비닐 | `ALLOWED` | `WEIGHT_ANOMALY` |
| 유리 | `REJECTED` | `rejection.code=GLASS` |
| 건전지 | `REJECTED` | `rejection.code=BATTERY` |
| 형광등·전구 | `REJECTED` | `rejection.code=FLUORESCENT` |
| 스티로폼 | `REJECTED` | `rejection.code=STYROFOAM` |

허용 품목에 다른 재질이 붙거나 섞여 있으면 `FOREIGN_MATERIAL`을 반환한다.
여러 조건이 동시에 발생하면 `guidance` 배열에 함께 포함한다.

일반 플라스틱은 손으로 안전하게 부피를 줄일 수 있는 얇고 속이 빈 병·용기에만 압착을
요구한다. 카페 테이크아웃 컵, 일반 컵, 뚜껑, 트레이, 두꺼운 밀폐용기, 작은 부품 및
단단하거나 깨질 수 있는 플라스틱은 압착 대상에서 제외한다. `compression_required`는 내부
LLM 계약에만 있고 Spring에는 보내지 않으며, 비대상 품목은 `conditions.is_dented`도 생략한다.

카페 컵은 빨대와 컵홀더를 제거한 상태에서 재질에 따라 분류한다. 단독 빨대는
`GENERAL_WASTE`, 단독 컵홀더는 `paper`다. 컵에 빨대나 컵홀더가 붙어 있으면
컵 품목을 유지하고 `FOREIGN_MATERIAL`을 반환한다.

## 무게 판정

bbox 면적 비율로 물체 크기를 S/M/L로 나누고 품목별 빈 무게에 15g 여유를 더해
무게 이상을 계산한다. `weight_g`가 없으면 무게 검사를 생략한다.

`primary` 모드에서는 1g 미만의 종이·비닐도 이미지로 판정한다. 1g 하한 가드는
Vision LLM을 사용하지 않는 기존 추론 경로에만 적용된다.

## 응답과 기록

- `X-Process-Time-Ms`: 이미지 수신부터 최종 AI 판정까지의 시간
- Pi 로그: YOLO, NAS LLM, 2차 판정, Spring callback 시간을 각각 기록
- `logs/results.jsonl`: 최종 응답
- `logs/callbacks.jsonl`: Spring 전송 결과와 재시도 기록
- `logs/captures/`: 원본 이미지와 요청·판정 JSON
- capture 기본 보존 기간 90일, 최대 용량 10GB
- Spring callback은 최대 3회 시도하며 하드웨어 응답과 별도로 실행

## 평가 결과

### AIHub 9종 고정 세트

평가 이미지는 클래스별 30장, 총 270장이다. 입력 무게는 20g으로 고정했고
Pi의 `pipeline.run`에서 NCNN YOLO와 NAS Vision LLM까지 실행했다. Spring callback은
평가에서 제외했다.

| 지표 | 결과 |
|---|---:|
| 전체 정확도 | 259/270, 95.93% |
| 위험 거부품 recall | 114/120, 95.00% |
| 미감지 | 0/270 |
| 실행 오류 | 0/270 |
| 평균 응답 시간 | 3.56초 |
| 중앙값 | 3.17초 |
| p95 | 6.19초 |
| 최대 | 7.60초 |

| 클래스 | 정답/전체 | recall |
|---|---:|---:|
| 캔 | 30/30 | 100.00% |
| PET | 30/30 | 100.00% |
| 종이 | 30/30 | 100.00% |
| 플라스틱 | 27/30 | 90.00% |
| 스티로폼 | 27/30 | 90.00% |
| 비닐 | 28/30 | 93.33% |
| 유리 | 28/30 | 93.33% |
| 건전지 | 30/30 | 100.00% |
| 형광등 | 29/30 | 96.67% |

위 표는 2026-09-30 현재 배포 코드의 플라스틱 재질 판정을 포함한 재평가 결과다.

### 신규 카메라 운영 캡처

| 평가 | 결과 |
|---|---:|
| 품목 분류 | 22/22 |
| 품목 미감지 | 0/22 |
| 빈 장면 | 1/1 `NOT_DETECTED` |
| 23요청 평균 응답 시간 | 3.73초 |
| 23요청 p95 | 4.85초 |

운영 캡처 22장은 클래스별 수량이 동일하지 않은 실제 요청 표본이다.

## 평가 자료

- [기준 270장 평가](evaluations/AIHUB_270_2832A2C.md)
- [선택적 재질 충돌 판정 평가](evaluations/AIHUB_270_AF27C3E.md)
- [연결 재사용·상세 crop 적용 평가](evaluations/AIHUB_270_C5347EE.md)
- [2026-09-30 현재 배포 코드 재평가](evaluations/AIHUB_270_CURRENT_20260930.md)
- [평가 manifest](evaluations/artifacts/aihub_270_2832a2c/manifest.csv)

다음 평가는 새 카메라로 9종을 클래스별 30장씩 촬영하고, 라벨·압착·이물질·내용물은
각 상태별 이미지 세트로 나누어 측정한다.
