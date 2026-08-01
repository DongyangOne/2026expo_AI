# 하드웨어 캡처 적응 프로토타입 (2026-08-01)

## 결론

Pi5 운영 모델은 교체하지 않았다. 노트북 RTX 3080에서 만든 `freeze20` 후보가
하드웨어 holdout 성능을 개선했지만 빈 장면 오검출은 줄이지 못했기 때문이다.
이 후보와 정제 데이터는 NAS v7 teacher 종료 후 기존 9종 원본 데이터와 합치는
hard sample로 사용한다.

## 데이터 정제 결과

- Pi5 수집 요청: 126건
- SHA-256 중복 제거 후 고유 이미지: 116장
- YOLO detector 데이터: 107장 (train 66 / val 41)
- crop/verifier 상태 manifest: 103장
- detector 제외: 혼합 이물질 8장, 지원 범위가 불명확한 부직포 1장
- negative: 빈 장비, 사람, 장비 밖 물체 12장
- PET는 내부 학습 class `pet=1`을 유지하며 외부 응답에서만 `plastic=3`으로 정규화한다.
- 동일 실물의 연속 촬영본은 `object_group` 단위로 묶어 train/val 누수를 막았다.

## 고정 holdout 비교

| 모델 | 임계값 | 전체 정확도 | 물체 정분류 | 빈 장면 specificity |
|---|---:|---:|---:|---:|
| 기존 `yolo26m_best.pt` | 0.25 | 53.66% | 54.29% | 50.00% |
| `freeze20` best | 0.25 | **60.98%** | **62.86%** | 50.00% |
| `freeze10` best | 0.25 | 51.22% | 54.29% | 33.33% |
| 기존 `yolo26m_best.pt` | 0.55 | 43.90% | 42.86% | 50.00% |
| `freeze20` best | 0.55 | 46.34% | 45.71% | 50.00% |
| `freeze10` best | 0.55 | **53.66%** | **48.57%** | **83.33%** |

`freeze20`은 `DETECT_CONF=0.25` 기준으로 기존 대비 전체 +7.32%p, 물체
정분류 +8.57%p다. 기존 `_probe` 13장에서는 baseline과 최상위 클래스가 모두
같았다. 그러나 빈 장면 false positive가 3건으로 그대로라 배포 게이트를 통과하지
못했다.

`freeze10`은 신뢰 임계값 0.55에서 더 보수적이지만 0.25 구간의 오분류가 늘었고,
장시간 학습 중 노트북 GPU가 92°C에 도달해 14 epoch 이후 중단했다. 운영 후보로
사용하지 않는다.

## 로컬 산출물

Git에 이미지와 모델 파일은 넣지 않는다. 같은 노트북의 다음 경로에 보존한다.

- 원본 스냅샷: `runs/hardware_capture_prep_20260801/raw/captures`
- SHA 고정 audit spec: `runs/hardware_capture_prep_20260801/audit_spec_v1.json`
- 정제 데이터: `runs/hardware_capture_prep_20260801/dataset_v1`
- baseline 지표: `runs/hardware_capture_prep_20260801/baseline_detector_metrics.json`
- freeze20 지표: `runs/hardware_capture_prep_20260801/candidate_freeze20_metrics.json`
- probe 회귀 비교: `runs/hardware_capture_prep_20260801/probe_regression.json`
- freeze20 가중치: `runs/detect/runs/hardware_capture_prep_20260801/training/candidate_freeze20/weights/best.pt`

## NAS v7 종료 후 합치는 순서

1. 기존 9종 YOLO train/val을 그대로 보존한다.
2. `dataset_v1/yolo`의 train hard sample을 기존 train에 추가하고, 하드웨어
   표본은 sampler 또는 복제로 3~5배 가중한다.
3. 현재 hardware val 41장은 학습에 넣지 않고 별도 검증셋으로 유지한다.
4. glass/battery/fluorescent가 없는 소규모 캡처만으로 최종 학습하지 않는다.
5. 기존 원본 validation의 클래스별 AP와 hardware holdout을 함께 비교한다.
6. 다음 게이트를 모두 통과한 모델만 NCNN으로 export한다.
   - 기존 원본 mAP50-95 하락 1%p 이내
   - hardware 물체 정분류가 baseline 대비 5%p 이상 개선
   - hardware 빈 장면 specificity가 baseline보다 개선
   - 신뢰도 0.55 이상인 빈 장면 false positive 0건
7. NCNN 후보는 Pi5의 별도 경로에서 smoke test한 뒤에만 운영 모델 경로를 바꾼다.

## 재현 명령

정제, 후보 학습, 비교 명령은 [WORKFLOW.md](WORKFLOW.md)의
`하드웨어 캡처 자동 정제 및 노트북 후보 학습` 절을 따른다.
