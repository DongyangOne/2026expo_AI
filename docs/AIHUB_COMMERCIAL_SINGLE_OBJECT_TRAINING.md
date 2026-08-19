# AI Hub 상용 단일 객체 학습 파이프라인

## 데이터 출처와 사용 경계

- 출처: AI Hub `재활용품 분류 및 선별 데이터`(datasetSn `71362`)
- 사용 범위: AI Hub 개방 데이터 이용정책에 따라 지능형 제품·서비스의 영리적·비영리적 연구·개발에 활용한다.
- 배포 범위: 학습된 모델과 서비스 산출물만 배포하며, AI Hub 원천 이미지·라벨·선별본은 제3자에게 제공·양도·판매하지 않는다.
- 표시: 외부 보고서와 모델 카드에는 과학기술정보통신부·한국지능정보사회진흥원 AI Hub 데이터 활용 사실과 데이터셋명을 명시한다.
- 예외 확인: 데이터셋별 별도 이용조건이 추가되면 해당 조건을 우선한다. 다운로드 계정의 동의 내역과 데이터셋 페이지 사본을 보관한다.

참조:

- https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=71362&topMenu=
- https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105

## 2026-08-13 선별 실행

원본을 훼손하지 않고 `/share/Container/yolo_commercial_single_v1_20260813`에 학습용 파생본만 만든다.

- 공식 `Training` 분할만 사용
- 라벨 객체 수가 정확히 1개인 직접 촬영 이미지 사용
- 9종 클래스별 최대 10,000개 원본 사용
- 최소 해상도 320px
- bbox 면적 비율 4~80%
- 흐림 점수 20 미만 제외
- 평균 밝기 18 미만 또는 238 초과 제외
- 클래스 내부 시각 중복 제외
- 긴 변 640px, JPEG 품질 90으로 저장
- NAS 잔여 공간이 500GB 미만이면 즉시 중단
- 하드웨어 학습 분할만 8회 반복하고 하드웨어 검증 분할은 절대 학습에 넣지 않음

건전지와 형광등처럼 원본 수가 10,000개 미만인 클래스는 원본 파일을 복제하지 않는다. 선별 완료 후 `train_balanced.txt`에서 10,000개 수준으로 반복 샘플링하고, YOLO의 매 epoch 증강을 적용한다.

학습 시작 전 고유 이미지 수를 다시 검사한다. 일반 클래스는 7,000개, 원래 원본이 적은 건전지·형광등은 1,500개 미만이면 자동 학습을 중단하고 선별 기준을 재검토한다.

## 2026-08-18 단일 데이터 미세조정 결과

상용 단일 객체 균형 목록 90,296건만 사용해 전체 층을 `optimizer=auto`,
`lr0=0.01`로 미세조정한 후보는 epoch 16에서 조기 종료되었고 승격하지 않았다.

| 모델 | original val mAP50-95 | original val recall |
|---|---:|---:|
| 기존 `trash_v2_full-2` epoch 40 | 0.88858 | 0.84564 |
| 단일 데이터 후보 best(epoch 1) | 0.86782 | 0.82577 |

원본 검증 분포에서 mAP50-95가 2.08%p, recall이 1.99%p 하락했다. 이후
epoch에서 지표가 더 크게 무너졌으므로 이 후보는 운영 모델과 NCNN 파일을 변경하지
않았다. 원인은 단일 객체 데이터만으로 전체 층을 큰 학습률로 갱신해 기존 배경·촬영
분포를 망각한 것으로 본다.

## 2026-08-19 안전 혼합 replay 재학습

새 파이프라인은 다음 원칙을 적용한다.

1. 기존 상용 단일 객체 균형 목록 약 9만 건을 유지한다.
2. `yolo_dataset_9class_v2`에서 라벨이 정확히 1개이고 bbox가 유효한 원본만
   결정론적으로 다시 선별한다. 일반 7종은 클래스별 2만 건, 건전지·형광등은
   클래스별 5천 건을 replay한다.
3. 기존 하드웨어 train의 빈 장면만 반복해 background false positive를 보정한다.
   기존 hardware val 41장은 어떤 형태로도 학습에 넣지 않는다.
4. 여러 물체를 합성하는 mosaic/mixup/copy-paste는 사용하지 않는다.
5. Stage A는 기존 epoch 40에서 시작해 `freeze=20`, 명시적 AdamW,
   `lr0=1e-4`로 5 epoch만 학습한다.
6. Stage A가 원본 분포를 보존했지만 하드웨어 게이트만 놓친 경우에만 Stage B에서
   `freeze=10`, `lr0=3e-5`로 최대 3 epoch 추가 보정한다.
7. 어떤 단계도 운영 가중치를 자동 교체하지 않는다. 모든 게이트를 통과하면
   `selected_candidate.txt`에 후보 경로만 기록한다.

실행 스크립트는 `scripts/watch_safe_mixed_yolo_pipeline.sh`이며, 데이터 목록은
원본 이미지를 복사하지 않고 `train_mixed.txt`로 구성한다.

## 운영 캡처 사용 경계

Pi 서버의 2026-08-01 00:00 KST 이후 캡처는 141건, SHA-256 중복 제거 후
133장이다.

- 기존 audit와 일치한 train 72장: 학습 재사용 가능
- 기존 고정 validation 41장: 계속 holdout으로 보호
- 신규 20장: 기존 예측을 정답으로 사용하지 않고 Qwen3-VL 8B의 서로 다른 두
  구조화 판정이 일치한 경우만 후보로 만든다.
- `client_id`는 NAS 교사 큐에 내보내지 않는다.
- teacher와 기존 YOLO 클래스가 다르거나 외부 이물질·다중 객체인 이미지는 detector
  학습에서 제외하고 hard-case 분석에만 남긴다.

관련 스크립트는 `prepare_operational_capture_queue.py`와
`label_operational_captures_ollama.py`다.

## 학습과 승격 기준

1. 기존 `trash_v2_full-2/weights/best.pt`에서 YOLO 학습을 이어 간다.
2. 위의 5+3 epoch 단계형 저학습률 보정만 수행한다.
3. AI Hub 공식 Validation 분할은 학습에 사용하지 않는다.
4. 학습 종료 후 하드웨어 고정 검증 분할을 별도로 평가한다.
5. AI Hub 지표와 하드웨어 지표가 함께 개선된 후보만 NCNN 변환 및 Pi 배포 대상으로 검토한다.
6. 평가 전에는 운영 가중치를 자동 교체하지 않는다.

자동 승격 게이트는 모두 통과해야 한다.

- original val mAP50-95 하락 1%p 이내
- original val recall 하락 1%p 이내
- hardware object accuracy가 baseline보다 5%p 이상 개선
- hardware negative specificity가 baseline보다 개선
- confidence 0.55 이상인 hardware 빈 장면 false positive 0건

내부 학습에서는 `pet=1`과 `plastic=3`을 구분해 특징 손실을 줄인다. 외부 API와 Spring 콜백에서는 PET도 `plastic/class_id=3`으로만 정규화한다.
