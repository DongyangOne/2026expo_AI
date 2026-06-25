# 2026 동양미래 EXPO — 재활용 분류 AI 워크플로우

> 키오스크 재활용품 자동 분류 시스템. 사진 + 무게센서 입력 → 분류·상태 판정 → Spring 서버 전송.
> 최종 갱신: 2026-06-20

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | 키오스크에 투입된 재활용품을 사진+무게로 분류하고 상태(라벨/찌그러짐/내용물)를 판정 |
| 추론 환경 | Raspberry Pi 5 (CPU, NCNN) |
| 입력 | 카메라 사진 + 무게센서값 (**투입부/센서는 타팀 하드웨어**) |
| 출력 | `DetectResponse` JSON → Spring 서버 POST (**자체 DB 없음**) |
| 담당 범위 | AI 처리 + 결과 API 전송 (내가 담당) / 사진·무게 수집 (타팀) |

---

## 2. 시스템 아키텍처

```
[타팀 하드웨어]                  [내 담당: FastAPI on Pi5]                [Spring]
 카메라 사진  ─┐
              ├─→ POST /detect ─→ 메인 YOLO26m (9클래스)                    
 무게센서값   ─┘                    ├─→ 상태 멀티헤드 분류기 (페트병/캔)     
                                    ├─→ 무게 이상 로직                       
                                    └─→ DetectResponse ──────────→ Spring POST
```

---

## 3. 분류 체계

### 9클래스 (메인 모델)
`can · pet · paper · plastic · styrofoam · vinyl · glass · battery · fluorescent`

| 구분 | 클래스 | status | 후속 |
|------|--------|--------|------|
| **재활용 허용** | 페트·플라스틱·캔·종이 | ALLOWED(조건충족) / REJECTED(불충족) | 조건검사 후 재활용 함 |
| **일반쓰레기** | 비닐 · 저신뢰 · 미분류 | GENERAL_WASTE | 일반함 |
| **수거 거부** | 유리·건전지·형광등·스티로폼 | REJECTED | 분리 불가 안내 |
| **미감지** | — | NOT_DETECTED | 재시도 |

### 상태 조건 (허용 품목) — 충족=ALLOWED, 불충족=REJECTED(재처리)
| 품목 | 라벨 떼기 | 압착 | 무게정상 |
|------|:---:|:---:|:---:|
| 페트병 | ✅ | ✅ | ✅ |
| 플라스틱 | ✅ | — | ✅ |
| 캔 | — | ✅ | ✅ |
| 종이 | — | — | ✅ |

> 조건 하나라도 불충족 → `REJECTED` + guidance(`REMOVE_LABEL`/`COMPRESS`/`EMPTY_CONTENTS`) 재처리 안내 → 사용자 처리 후 재투입.
> 비닐 = 일반쓰레기(`GENERAL_WASTE`), 유리·건전지·형광등·스티로폼 = 완전 수거거부(`REJECTED`+`rejection`). 색 구분/안내 없음.

---

## 4. 런타임 추론 파이프라인

`app/services/pipeline.py` 의 `run()`:

```
1. 이미지 디코드
2. 메인 YOLO 감지 (conf=DETECT_CONF 0.25)
     └ 박스 없음 → NOT_DETECTED
3. 최고신뢰 박스 → class_id, confidence, bbox
4. 신뢰도 판정
     └ confidence < TRUST_CONF(0.55) → LOW_CONFIDENCE (일반쓰레기)
5. [허용: 페트/플라스틱/캔/종이]
     ├ 상태 멀티헤드(crop ONNX) → conditions.is_dented / has_label
     ├ 무게 → weight.anomaly
     └ build_guidance() → 불충족 안내. 비면 ALLOWED, 있으면 REJECTED(재처리)
6. [거부: 유리/건전지/형광등/스티로폼] → REJECTED + rejection
7. [비닐] → GENERAL_WASTE + general / [저신뢰] → GENERAL_WASTE
8. DetectResponse 조립 → Spring 콜백(fire-and-forget)
```

**2단계 신뢰도 게이트:**
- `DETECT_CONF=0.25`: 박스 생성 최소선 (미만 = 미감지)
- `TRUST_CONF=0.55`: 분류 신뢰선 (미만 = 일반쓰레기)

---

## 5. 모델 구성 (3-tier)

| # | 모델 | 입력 | 출력 | 백본 | 포맷 | 상태 |
|---|------|------|------|------|------|------|
| 1 | 메인 감지 | 640px | 9클래스 bbox | YOLO26m | NCNN | 변환완료, 학습 보류 |
| 2 | 상태 멀티헤드 | 224px crop | dent(2) + label(2) | MobileNetV3-Small | NCNN INT8 | crop 추출 중 |
| (3) | 색/재질 SubClass | 224px crop | 무색/유색, 철/알루미늄 | — | — | **보류** (상태 우선) |

> 모델 3은 이전 설계의 색·재질 세부분류. 이번 우선순위는 **상태(라벨/찌그러짐)**라 보류.
> 단 캔 철/알루미늄은 무게가 달라(철>알루미늄) 무게로직 정밀도에 도움 → 추후 재검토.

### 멀티헤드 설계 (모델 2)
- **공유 백본 1개** + dent 헤드 + label 헤드. 추론 1회로 두 출력.
- 페트병이면 두 헤드 다 읽고, 캔이면 dent만 읽음 (YOLO가 품목 이미 구분).
- dent는 페트+캔 공통 학습(데이터↑), label은 페트병만(캔은 loss 마스킹).

---

## 6. 학습 파이프라인

### 6-1. 메인 모델 (YOLO26m)
```
AI Hub 원본 JSON+이미지 (2TB)
  └→ scripts/convert_v2.py
       · POLYGON 라벨 복원 (v1은 20~50% 누락 버그)
       · 640px 리사이즈 (4MB→~50KB, I/O 70배↓)
       · 폴더당 15000 stride 샘플링
  └→ /share/Container/yolo_dataset_9class_v2/ (train/val + dataset.yaml)
  └→ YOLO26m 학습 → best.pt → NCNN export
```

### 6-2. 상태 멀티헤드 분류기
```
AI Hub 원본 JSON+이미지 (2TB, 직접촬영)
  └→ scripts/extract_crops.py
       · bbox crop + 12% 패딩 → 224px letterbox
       · DAMAGE → dent (원형0 / 찌그러짐·완전압착1)
       · DIRTINESS → label (오염없음0 / 이물질외부1, 내부·전체는 -1 마스킹)
       · ※ 원본 사용 필수 (640변환본은 파일명 리네임돼 JSON 매칭 불가)
  └→ /share/Container/crops_state_v1/ (pet/ + can/ + manifest.csv)
  └→ scripts/train_classifier.py
       · MobileNetV3-Small + dent/label 헤드
       · 클래스 불균형 → CrossEntropy class-weight
  └→ best.pt → multihead.onnx → onnx2ncnn
```

---

## 7. 데이터셋 (AI Hub dataSetSn=71362)

- **재활용품 분류 및 선별 데이터**, 약 100만 장, JSON 라벨 (BBOX 70% / POLYGON 30%)
- 촬영: 직접촬영 70.2% (키오스크 환경 유사) + 선별영상추출 29.8% (컨베이어)
- **객체 속성 5개**: `DAMAGE` `DIRTINESS` `COVER` `TRANSPARENCY` `SHAPE`
- **핵심 발견**: `DIRTINESS=이물질(외부)` ≈ **라벨 부착** (육안검증). `오염없음`=라벨 뗀 맨 용기.
- 상세: `memory/dataset_attribute_labels.md`, 데이터 점검: `docs/DATA_AUDIT.md`

### 실측 상태 분포 (직접촬영)
| 품목 | 찌그러짐+완전압착 | 라벨있음/없음 |
|------|------|------|
| 페트병 무색 | 29.7% (21,659) | 71.7% / 20% |
| 페트병 유색 | 19.3% (11,889) | 75% / 23% |
| 철캔 | 18.8% (10,616) | 50% / 46% |
| 플라스틱 PE | 4.5% (2,166) ⚠️데이터빈약 | 87% / 12% |

---

## 8. 배포 (Raspberry Pi 5)

- 모든 모델 **NCNN** 포맷 (ARM 최적화), 상태 분류기는 **INT8 양자화**로 2배 가속
- `app/core/config.py`: `MAIN_MODEL_PATH`, `PET_MODEL_PATH`, `CAN_MODEL_PATH`, `DEVICE="cpu"`
- 모델 미탑재 시 graceful degradation (sub_class/conditions = null)
- Spring 콜백: `SPRING_CALLBACK_URL` (fire-and-forget, 실패해도 응답 영향 없음)

---

## 9. 인프라 (NAS / 학습 환경)

| 항목 | 값 |
|------|-----|
| NAS | QNAP, 192.168.0.110, Ryzen 1700 + RTX 2000 Ada 16GB |
| Docker | `/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker` (PATH에 없음, sudo 필요) |
| 학습 이미지 | `ultralytics/ultralytics:latest` |
| SMB | `\\192.168.0.110\Container` → 컨테이너 `/app` (`/share/Container`) |
| 접속 | plink `-hostkey ssh-ed25519 255 SHA256:HLPv3nWI2ozOxgy2AAzjguq1rrON+KOuWhh3zscXpy0` |
| 한글 출력 | plink가 한글 깨뜨림 → `base64 -w0` 우회 후 로컬 디코드 |
| 원본 데이터 | `/share/Container/ai_dataset/학습용_데이터/01-1.정식개방데이터/Training/` |

> 상세 접속법: `memory/nas-dataset-workflow.md`

---

## 10. 현재 진행 상황 (2026-06-20)

| 단계 | 상태 |
|------|------|
| 데이터 변환 (convert_v2) | ✅ 완료 (train 라벨 518,728 / val 68,147) |
| └ 변환결과물 이미지 수 이상 | ⚠️ 점검 필요 (`docs/DATA_AUDIT.md`) |
| crop 추출 (extract_crops) | ⚙️ 진행 중 (23만, 디스크병목 ~26/초, ETA ~2.3h) |
| 멀티헤드 학습 | ⏳ crop 완료 후 |
| NCNN 변환 + 파이프라인 통합 | ⏳ 대기 |
| 메인 YOLO26m 학습 | ⏸️ 보류 (데이터 점검 후 재개) |

---

## 11. 코드 맵

```
app/
  core/config.py          설정 (임계값, 모델경로, Spring URL)
  models/registry.py      모델 보관소 (graceful degradation)
  schemas/
    enums.py              WasteClass, DetectionStatus, GuidanceCode, RejectionCode, GeneralWasteCode
    response.py           DetectResponse DTO (status/classification/conditions/weight/guidance/rejection/general)
    request.py
  services/
    pipeline.py           흐름 제어만 (run)
    inference.py          모델 추론 (전처리 + YOLO·멀티헤드 세션 호출)
    guidance.py           허용/거부/일반 판정 + 안내 (도메인 규칙)
    weight_check.py       무게 이상 (bbox 크기보정 + 단일임계 폴백)
    spring_client.py      Spring 콜백 (fire-and-forget)
  api/v1/detect.py        엔드포인트
  main.py                 앱 + 통합 에러 핸들러

tests/
  test_guidance.py        도메인 규칙 단위테스트
  test_weight_check.py    무게 판정 단위테스트

scripts/
  convert_v2.py           메인 학습용 변환 (폴리곤복원 + 640리사이즈)
  extract_crops.py        상태분류기 crop 추출 (bbox + DAMAGE/DIRTINESS)
  train_classifier.py     멀티헤드 학습 (MobileNetV3 + dent/label)

docs/
  WORKFLOW.md             (이 문서)
  WEIGHT_KIOSK_PARAMS.md  무게 로직 + 키오스크 파라미터 체크리스트
  DATA_AUDIT.md           변환 데이터 정합성 점검
```
