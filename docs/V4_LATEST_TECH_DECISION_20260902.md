# V4 최신 기술 적용 결정 (2026-09-02)

## 결론

메인 detector는 YOLO를 유지한다. 현재 프로젝트의 `yolo26m_best.pt`는 이미
YOLO26m 9-class 모델이고, 고정 NAS 이미지의 Ultralytics 8.4.60도 YOLO26 및
YOLOE-26을 지원한다. 따라서 모델 이름만 바꾸는 재학습보다 데이터 품질, bbox
정확도, 독립 teacher 합의, 운영-domain hard sample을 우선 개선한다.

어떤 offline 후보도 신규 독립 blind set과 실제 하드웨어 end-to-end gate를 모두
통과하기 전에는 production/Pi 모델을 교체하지 않는다.

## 이번에 적용한 항목

1. 운영 캡처는 `2026-08-01 00:00:00 KST` 이후만 사용한다. 큐 생성기와 최종
   teacher manifest가 각각 날짜와 timezone을 확인하며, 더 이른 override와 timestamp
   없는 입력은 fail-closed로 제외한다.
2. 파일 존재, SHA-256, decode, 최소 해상도, 극단 노출을 teacher 전단에서 검사한다.
   카메라별 보정 없이 임의 blur 임계값은 사용하지 않는다.
3. 서로 다른 blind prompt pass가 재질, 단일 객체, 외부 이물질, 학습 적합성, 품질
   사유의 전체 tuple에 합의해야 한다. 불일치할 때만 제3 pass로 다수결한다. 전체
   prompt/schema/Ollama option과 모델 digest를 하나의 trusted contract로 고정한다.
   실행 시작·각 이미지 직전·모든 pass 직후·최종 게시 직전에 실제 Ollama
   `/api/tags` digest를 확인하고 `/api/chat` 응답 모델명도 정확히 대조한다. 중간
   변경이 감지되면 해당 이미지 checkpoint를 쓰지 않고 즉시 중단한다.
   동일 VLM의 반복 판정은 학습 후보 정제용일 뿐 독립 모델 검증으로 부르지 않으며,
   기존 배포 모델의 예측은 정답·선정·bbox 권한으로 사용하지 않는다.
4. 심한 frame crop, 사람 손·팔의 가림/지배, 과도한 배경이나 다중 물체 혼잡, 경계
   판독 불가 이미지는 고신뢰 재질 합의가 있어도 학습에서 제외한다. 반대로 캔·병·종이
   자체의 구김·찌그러짐은 촬영 실패가 아니므로 경계와 재질을 판독할 수 있으면 유효한
   hard case로 유지한다.
5. teacher 큐에는 capture root 상대경로와 원본 SHA만 전달한다. root 이탈, 외부 symlink,
   원본 변조, 이전 teacher 계약의 checkpoint는 거부한다. 심한 잘림·사람 지배·과도한
   배경/다중 물체·경계 판독 불가 판정은 사유와 함께 보존하되 학습과 calibration에서는
   제외한다.
6. 양성 운영 crop은 배포 YOLO bbox를 재사용하지 않는다. 입력 이미지 SHA와 서로 다른
   두 localizer의 실제 manifest·모델·추론 계약 파일 SHA 및 각 출력 SHA를 결박하고,
   bbox IoU가 고정 `0.75` 이상일 때 coordinate mean box만 만든다. 최종 manifest builder도
   이 파일들을 다시 열어 해시와 source별 출력을 재검산한 경우에만 crop-ready로 인정한다.
   관련 producer는 `scripts/build_independent_localization_consensus.py`다.
7. V4 reproducibility selector는 bounded drift anchor 다음에 같은 split에서 과거
   detector observation이 있던 current-GT source를 우선한다. 과거 category는 선택
   힌트일 뿐이며 current YOLO label과 frozen replay만 판정 권한을 가진다.
8. selector 재실행은 GPU validator 직전이 아니라 별도의 CPU-only 불변 사전 감사에서
   수행한다. 전체 eligible universe의 deterministic top-K를 inventory와 train/validation
   list 단위로 byte 비교해 봉인하고, validator는 그 증거만 확인한다. 이 순서는 QNAP
   page cache가 GPU 초기화를 방해하는 위험을 줄인다.
9. teacher labeler와 최종 builder는 trusted known-audit와 capture inventory를 다시 확인해
   `teacher_required`만 받는다. 기존 train/보호 validation SHA를 teacher pseudo-label로
   재유입시키지 않으며, 최종 manifest의 절대경로 필드는 NAS 내부 후속 단계용이므로
   `portable=false`, `local_only_contains_absolute_paths=true`로 명시한다.
10. 재현 pilot에서 이미 확인된 촬영 실패는
    `scripts/build_v4_quality_exclusion_manifest.py`로 별도 봉인한다. 출력에는 원본
    `source_sha256`과 허용된 품질 사유만 기록하고 경로·파일명·사용자 식별자는 넣지
    않는다. manifest는 비어 있을 수 없고 최대 100개 SHA로 제한하며, selection·정답·
    replay·학습·calibration·blind·배포 권한을 모두 `false`로 고정한다.
11. selector는 제외 SHA가 현재 해석된 데이터셋에 실제 존재하지 않거나 manifest의
    canonical hash·정렬·개수·사유·권한이 정확하지 않으면 중단한다. 제외 SHA와 바이트가
    같은 모든 source는 선택하지 않는다. 별도 CPU selection audit가 selector를 다시
    실행해 동일 결과를 byte 단위로 봉인하고, GPU validator는 그 증거와 최종 raw
    manifest에 제외 SHA가 없음을 확인한다. GPU 단계 직전에 전체 원본을 다시 읽어
    QNAP page cache를 채우지 않는다.

`2026-08-01` 컷오프는 운영 카메라 캡처에만 적용한다. 상용 사용 조건을 충족한 AI Hub
원본 corpus를 날짜 때문에 버리지 않는다. `severe_frame_crop`, 사람 가림·지배, 다중
물체·혼잡, 경계/대상 판독 불가, 저해상도, 극단 노출 같은 촬영 실패만 제외하며,
재활용품 자체의 구김·찌그러짐·압착은 제외 사유로 허용하지 않는다.

단일 촬영 실패는 다음처럼 새 불변 파일로 만든다. 입력·출력은 심볼릭 링크가 없는
절대 물리 경로를 사용하며 기존 출력은 덮어쓰지 않는다.

```bash
python scripts/build_v4_quality_exclusion_manifest.py \
  --source /absolute/path/to/bad-capture.jpg \
  --reason severe_frame_crop \
  --output /absolute/control/quality-exclusions.json
```

여러 장은 `path,reason` 두 컬럼의 UTF-8 CSV와 `--image-root`를 사용한다. 이후 selector,
CPU audit, validator에는 동일 파일을 `--quality-exclusion-manifest` 또는
`QUALITY_EXCLUSION_MANIFEST`로 전달한다.

## 다음 실험 순서

### 1. bbox/라벨 자동 감사

- YOLO26 detection box, YOLOE-26 segmentation mask, SAM 계열의 box-prompt mask를
  서로 독립적으로 생성한다.
- mask에서 tight box를 만들고 모델 간 mask/box 합의가 충분한 샘플만 자동 교정
  후보로 사용한다. 불일치는 학습에서 격리한다.
- 같은 이미지로 임계값을 맞춰 통과시키지 않는다. calibration과 blind evidence는
  source SHA, object group, capture session 기준으로 분리한다.

SAM 3.1은 2026-03 release의 주 개선점이 다중 객체 video tracking 효율이므로, 단일
이미지 bbox 감사에는 우선 기존 image prompt 경로를 작은 isolated ablation으로
비교한다. 무조건 3.1을 채택하지 않는다.

### 2. 합의형 pseudo-label

단일 Qwen/VLM 또는 단일 open-vocabulary detector의 결과를 바로 정답으로 쓰지 않는다.
YOLOE-26, 독립 open-vocabulary detector, segmentation 결과의 공간 합의와 confidence
합의를 모두 만족한 train-only pseudo-label만 사용한다. 이는 real-world ZeroWaste에서
weighted box fusion과 consensus-aware soft pseudo-label이 단일 zero-shot보다 나았다는
최근 waste detection 연구 방향과 일치한다.

### 3. hard-example와 OOD mining

DINOv3 dense embedding은 production detector 대체가 아니라 다음 용도로만 검토한다.

- 동일 실물·근접 중복 탐지와 split leakage 감사
- 기존 9-class prototype에서 멀리 떨어진 OOD/저신뢰 운영 샘플 군집화
- paper/styrofoam, plastic/vinyl처럼 반복되는 confusion pair의 hard-negative 선정

### 4. 외부 데이터

- AI Hub 상용 사용 가능 9종 단일 객체 원본은 계속 main replay 데이터로 유지한다.
- SortWaste는 5,261장, 87,000개 이상 box의 8-class 산업 폐기물 데이터지만 dense
  multi-object conveyor 장면이므로 전체 frame을 현재 단일 객체 학습에 직접 넣지 않는다.
  공식 저장소에서 명확한 dataset license를 확인하기 전에는 다운로드·학습도 하지 않는다.
- 사용 가능해지면 class mapping과 license를 고정하고, instance crop 중 단일 객체 및
  경계 품질 gate를 통과한 항목만 별도 ablation에 사용한다.

## 고정 비교 게이트

- 기존 AI Hub validation의 class별 precision, recall, mAP50-95 비열화 한도
- 2026-08-01 이후 운영 capture의 object-group/session 분리 holdout
- 빈 장면 false positive와 unknown/foreign-material 거부율
- plastic-vinyl, paper-styrofoam confusion matrix
- bbox IoU 및 mask boundary 품질
- Pi5 NCNN latency, peak memory, 실제 Spring callback end-to-end 계약

## 라이선스 게이트

- YOLO26/YOLOE는 AGPL-3.0 또는 Ultralytics Enterprise 조건이다. 비공개 상용 서비스와
  하드웨어 탑재 범위는 모델 승격 전에 현재 프로젝트의 라이선스 적합성을 별도로
  확인한다: https://www.ultralytics.com/license
- SAM 3.1과 DINOv3는 각각 Meta의 별도 custom license다. 자동 라벨 감사 실험과
  production 배포 가능 범위를 동일하게 간주하지 않고, 배포 전 원문 조건을 검토한다.
- BMVC 2025 구현 코드의 MIT 조건과 학습 데이터·외부 모듈의 사용 조건은 분리해서
  확인한다. SortWaste는 공식 저장소에서 명시적인 dataset license를 확인하기 전까지
  다운로드·학습 투입하지 않는다.

## 1차 자료

- Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26
- YOLO26 training recipe: https://docs.ultralytics.com/guides/yolo26-training-recipe
- Ultralytics YOLOE-26: https://docs.ultralytics.com/models/yoloe
- SAM 3.1 release: https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md
- DINOv3 reference implementation: https://github.com/facebookresearch/dinov3
- Ultralytics license: https://www.ultralytics.com/license
- SAM 3 license: https://github.com/facebookresearch/sam3/blob/main/LICENSE
- DINOv3 license: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
- Robust and Label-Efficient Deep Waste Detection:
  https://arxiv.org/abs/2508.18799
- SortWaste official repository: https://github.com/sarainacio/SortWaste
