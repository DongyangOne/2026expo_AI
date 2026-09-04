# AIHub 재현성 진단과 다음 단계 — 2026-09-04

## 1. 실제 확인한 기존 증거

이 문서는 **8/31 생성 V4 데이터에 대한 기존 drift 측정 보고서를 9/4에 회수해 읽은 결과**다.
오늘 AIHub 91,938장을 새로 추론하거나 학습한 결과가 아니다. 보고서 자체에는 측정 시각
필드가 없으며, 보관 run은 `proposal_verifier_multitask_v4_bgfix_20260831_drift_audit_0504f05`다.

- 원본 보고서: `/share/Container/runs/proposal_verifier_multitask_v4_bgfix_20260831_drift_audit_0504f05/historical_manifest_replay_drift.json`
- 회수본: `/share/Container/operational_refresh_80bf78a_20260904_101000/historical_replay_drift_snapshot_20260904.json`
- 회수본 크기: 157,273 bytes
- 직접 확인한 SHA-256: `c71b8d54289f416609231df8af7c497e7174b08e73f854516b8ef8d44b81ec9c`

보고서는 `diagnostic_only=true`이고 lineage/training/blind/deployment 권한이 모두
`false`다. 숫자 오차를 측정했을 뿐 허용 임계값 판정을 적용한 통과 보고서가 아니다.

| 결박 대상 | SHA-256 |
|---|---|
| 원본 `manifest.csv` | `5c9e0e933e75cd7318a5ca9b3f5baf4460fee02466eb0a572cdd1aec5dda2dbb` |
| `dataset_info.json` | `8e10c64d43e02d46c008deec0b7dc0c7567901196286af216516f9276717f550` |
| YOLO detector | `7b849c25c3983a54b4b6c922e425798f89326b2da21e862b90d2ee0c6a181f69` |
| inference spec | `c6ddaa1f7bc6dc58114a2dab52e9f2382bd2f4acaba4bd398c1c3b91cfc0e5de` |
| source/label/crop 결박 집합 | `91e86995d230d6dfb1c8af13317a230b7c3c127a19eef522d458033ab1d86b92` |

`bindings.loaded_code_sha256`에는 당시 auditor·prepare·validator·preprocessing 네 파일의
SHA도 들어 있다. 재현 시 현재 HEAD가 당시 코드와 같다고 가정하지 않고 이 값을 대조한다.

## 2. 측정 환경과 핵심 결과

측정 환경은 RTX 2000 Ada, logical CUDA device `0`, **batch=12**, Python 3.12.3,
PyTorch 2.11.0+cu128, Ultralytics 8.4.60, CUDA 12.8, cuDNN 91900이다.
`cudnn_benchmark=false`, `deterministic_algorithms_enabled=false`였다.
detector confidence 0.10/NMS IoU 0.70, crop 320/padding 0.08/fill 114를 사용했다.
보고서는 원본 source/label/crop 및 detector/loaded code의 종료 시 재해시도 기록한다.

- 입력·관측 source: 각각 **91,938**. 누락·중복·예상 밖 source는 0.
- 모든 source replay는 끝났지만 5개는 proposal이 없어 수치 비교는 **91,933개**만 존재한다.
- `contract_impacting_drift=true`; 단순한 저장 소수점 차이만 있는 결과가 아니다.

| 측정값 | p50 | p90 | p95 | p99 | 최대 |
|---|---:|---:|---:|---:|---:|
| confidence 절대 차이 | 4.8729e-9 | 0.0012853 | 0.0070480 | 0.0705875 | 0.8770116 |
| bbox 좌표 최대 절대 차이(px) | 0 | 0.057144 | 0.119659 | 0.482758 | 540.846039 |

`hard_semantic_mismatch_sources=4,070`은 아래 계약 변화가 하나 이상 있는 source의 합집합이다.
항목별 개수는 겹치므로 더하면 안 된다.

| 계약 변화 | 개수 |
|---|---:|
| crop 정수 경계 변경 | 4,048 |
| detector class 변경 | 65 |
| assignment material/reason 변경 | 각각 58 |
| strict-zero-intersection 결정 변경 | 58 |
| proposal 없음 | 5 |

이 지표의 `semantic`은 **기존 저장 예측과 replay 사이의 계약 변화**를 뜻한다.
4,070개가 실제 정답 오분류라는 의미가 아니며, 나머지가 정답이라는 의미도 아니다.
이 보고서로 기존 confidence mismatch가 해결됐다고 할 수 없다.

## 3. 원인 가설과 최소 진단

현재 및 기존 `841b703` prepare는 confidence를 소수점 8자리로 저장한다. 정상 0~1 값의
반올림 오차 상한은 5e-9이므로 이것만으로 고정 허용오차 1e-6 초과를 설명하지 못한다.
prepare는 source 순서로 batch 예측한 후 CSV를 split/material/hash 순서로 다시 정렬하고,
validator는 CSV 순서로 batch를 구성한다. **동반 이미지와 shape가 바뀌어 detector의
rectangular padding/배치 연산이 달라졌을 가능성**이 있다. 이는 코드상 가설이며
보고서도 한 번의 batch-N replay만으로 batch 인과관계를 분리하지 못한다고 명시한다.

과거 실패 로그의 `source-00000012-5f3737d5431d7792.jpg`는 당시 코드의 명명법상
원본 CSV 13행(데이터 12번째)이다. 현재 원본 CSV 전체 SHA와 그 source SHA 접두사부터
재확인한다. 임시 snapshot 폴더를 재사용하거나 원본 행의 confidence를 수정하지 않는다.
동일 모델·코드·컨테이너 image·GPU 환경에서 새 불변 진단 경로로 다음을 비교한다.

1. 실패행이 포함된 **기존 validator 순서의 완전한 batch**. 당시 설정이 12인지 원본
   `dataset_info.json`과 container inspect로 확인하고, 맞으면 첫 12개 데이터 행을 사용한다.
2. 같은 이미지가 포함된 **기존 generation source 순서의 완전한 batch**. 당시 YAML과
   source 목록을 확인해 재구성하고, 원래 목록/순서를 복원할 수 없으면 동일 조건 재현이라고 하지 않는다.
3. 같은 실패 이미지의 **batch=1 독립 반복 두 번**. 각 결과와 원래 저장 top1을 비교한다.

각 조건에서 source/model/code SHA, 이미지 순서·원본 크기, 실제 detector 입력 tensor
shape/dtype, `half/fp16`, `rect`, TF32/determinism 설정, class/confidence/bbox를 기록한다.
현재 `iter_yolo_predictions`는 `half/rect`를 명시하지 않으며 기존 보고서도 실제 dtype과
tensor shape를 기록하지 않았다. spec의 **crop 320 float32**를 detector 입력 dtype이나
640 letterbox 동작의 증거로 혼동하지 않는다. 원본 GT 및 허용오차는 변경하지 않는다.

`audit_v4_detector_replay_drift.py`에는 `--limit`이 없어 원본 CSV를 주면 전량을 읽는다.
소수행은 원본 SHA·행 번호·배치 구성에 결박한 별도 진단 입력/좁은 probe가 먼저 필요하다.
`run_v4_reproducible_generation.sh`는 신규 batch=1 생성용이고, QX3 validation wrapper는
봉인된 cohort 전체를 두 번 검증하는 용도다. 둘을 실패행 몇 개의 가벼운 진단으로 오인하지 않는다.
단기 목표는 원인 분리이며, 91k 재생성이나 임계값 완화부터 하지 않는다.

## 4. AIHub-only 연구 학습의 조건

trainer의 `--no-condition-heads`로 objectness 2분류+재질 9종 연구 학습은 가능하다.
기존 V4 CSV는 train/validation 모두 9종+background가 있지만 세 상태는 전부 `-1`이다.
따라서 이 경로는 상태 성능을 개선하는 최종 V4 후보가 아니며, 기존 상태 정답을 억지로
복사하거나 `-1`을 `0`으로 바꾸지 않는다.

연구용 학습 전에도 실제 pinned runtime replay·crop 검증, report에 결박한 strict lineage
upgrade, 최종 CSV 재검증, license/origin 및 source/crop/group/session/근접중복 감사가 필요하다.
운영 hold 9장·보호 hardware 41장·QX3 3,500장은 분리하고, 양쪽 role의 9종+background
coverage를 유지한다. 고정 입력·기존 pretrained 모델·독립 run으로
`--dry-run --no-condition-heads`를 먼저 확인한다.

현재 정식 candidate builder는 세 상태 head 및 각 role의 0/1 지원을 요구하며,
builder/launcher의 승인 policy pin은 `UNCONFIGURED`다. 별도 non-authoritative 연구
launcher는 별도 검토 대상이지 기존 candidate gate나 v3 watcher를 우회할 명분이 아니다.
학습 후에도 고정 validation/calibration과 신규 독립 하드웨어 end-to-end gate 없이
production/Pi 모델·Spring 계약을 바꾸지 않는다.

이 작업은 운영 9B teacher의 오라벨/전경 혼동과 독립된 **AIHub YOLO replay 재현성 문제**다.
운영 hold는 계속 유지한다. 위 1–4절은 당시 문서 작성 시점의 계획이다.

## 5. 실제 4조건 GPU 진단 완료 — 2026-09-04 12:37 KST

`probe_aihub_replay_batch_v2_20260904`가 03:37:16–03:37:27 UTC에 exit 0으로 끝났다.
실제 원본 첫 12장을 batch 12로, 실패 대상 한 장을 새 YOLO 인스턴스에서 두 번,
첫 12장을 `rect=false` 대조 조건으로 처리했다. 총 26 image inference이며 전량 재검증은 아니다.

- 원본 행/경로 확인 보고서 SHA: `3203cf7efd93eb86d4fe9b6d30ed0d72f71c5cc12888c934a1d34435bbd783ef`
- 진단 runner v2 SHA: `d8a634cddc2e7a12553a59979670bcd5d627194892602cfba94e9e1b28bb7576`
- 결과: 작업 루트의 `aihub_replay_batch_probe_v2_20260904/new.json`
- 결과 SHA: `75953771b5c491419157dd033cfee018cfcd836fd2cdbaa796976e32e7e42aab`
- 대상 source SHA: `5f3737d5431d77922771751704a72e2c414bb3648544ccc7a50110a1b382f7a5`

원본 첫 12장 모두 640×360이다. 실제 preprocessing을 교체하지 않고
`DetectionPredictor.preprocess` 입력/출력을 관측했다. 원본 bytes와 decoded BGR SHA,
각 결과의 `orig_img` SHA/순서/크기, 모델·manifest·metadata·runner SHA를 대조하고
종료 후 다시 확인했다. 모든 detector tensor는 CUDA:0, float32, half=false였다.

| 대상 사진 처리 조건 | 실제 NCHW tensor | can confidence | 과거 bbox 최대 차이(px) |
|---|---|---:|---:|
| 과거 CSV 저장값 | 당시 실제 shape 미기록 | 0.98830354 | 기준 |
| 원본 CSV 첫12, 기본 rect=true | 12×3×384×640 | 0.9882251024 | 0.02698517 |
| 대상 단독 반복 A | 1×3×384×640 | 0.9882209301 | 0.02168274 |
| 대상 단독 반복 B | 1×3×384×640 | 0.9882209301 | 0.02168274 |
| 첫12, rect=false 대조군 | 12×3×640×640 | 0.9883035421 | 0 |

단독 반복 A/B의 class/confidence/bbox는 정확히 같았다. 정사각형 대조군의 대상 bbox는
과거 저장값과 정확히 같고 confidence 차이는 2.14e-9로 8자리 저장 반올림 범위다.
**이 한 사례에서는 입력 padding 차이가 과거 mismatch를 설명한다는 직접 근거**다.
원래 generation batch의 동반 이미지/shape를 복원한 것은 아니며, 전체 91,938행의
원인이 모두 확정됐거나 기존 validator를 통과했다는 뜻은 아니다. 고정 임계값이나
원본 confidence/bbox를 변경하지 않는다. 새 데이터는 운영 단일 요청에 맞는 batch=1
생성 후 같은 조건으로 실제 재검증하는 경로가 타당하다.
동일한 384×640 padding 안에서도 batch12와 batch1의 confidence는 4.17e-6 차이가
있었다. 따라서 padding만 유일한 수치 변동 원인이라고 주장하지 않는다.

첫 실행은 CUDA 초기화에서 거부되어 추론이 없었다. 다른 GPU 작업이 없는 것을 확인한
뒤 문서화된 memory compaction을 실행했고, retry01은 CUDA 진입 후 결과 경로 검사에서
멈췄다. 실제 pinned Ultralytics 8.4.60은 list 입력을 PIL/NumPy로 바꾸고 filename이
없으면 `image{i}.jpg`를 사용한다. 이는 source 순서가 달라졌다는 증거가 아니므로 v2는
표시 경로를 정답으로 믿지 않고 실제 결과 픽셀 SHA·순서를 반드시 검증한다. 이전 실패
결과는 유지했다. NAS 재부팅, 운영 모델 및 서비스 변경은 없었다.

기존 QX3 3,500 source의 별도 batch=1 생성/검증 완료 marker도 실물로 확인했다.
`v4_repro_validation_qx3_retry2_5ac3031_20260902_223000/validation/control`의
`diagnostic_ready.json`은 `batch1_validator_ab_reproducibility_passed`, comparison은
`validator_ab_exact_reproduction`이다. 생성된 3,498행의 두 report와 validated CSV가
일치했다. 이는 과거 재현성 진단이며 새 학습 실행·학습 승인·현재 full-quality 계약 통과가
아니다. 3,500 source 전체를 계속 보호하고 학습에 넣지 않는다.

## 6. 다음 데이터 생성 입력을 정하는 순서

이미 확인한 91,938 manifest는 **source 후보 경로 목록으로만** 재사용할 수 있다.
기존 predicted class/confidence/bbox를 새 정답으로 복사하지 않는다. 전체 약 20만 원본을
다시 추론하기 전에 이 후보 pool의 출처·보호집합·원래 분할을 검사한다.

`yolo_commercial_single_v1_20260813` 안에는 AIHub 파생 이미지 외에
`hardware_rN_*` 복사본도 있다. 폴더 접두사만으로 AIHub origin을 부여하지 않는다.
실제 `selected_manifest.csv`의 exact stem join과 원본 경로/annotation까지 확인해야 한다.

- `selected_manifest.csv`: 18,744,451 bytes,
  SHA `2f026c4d914ff3b5c6a8e3bf89280678a847e37a5f22611a481d792e8223012a`.
  필드는 stem/category/class_id/source_id/source_path/area_bin/dent/raw_dirtiness다.
  여기의 20자리 source_id는 V4 CSV의 64자리 image SHA와 다른 식별자다.
- `selection_summary.json` SHA
  `b7c59da5f867ebd60584de650756fce9ef8bef09619bd7fa4f761983fef339b6`.
  당시 선택 원본 74,906, hardware 복사 528이라고 기록돼 있다. 이것은 새 학습 가능 수가 아니다.
- 기존 YAML validation은 `/app/yolo_dataset_9class_v2/val/images`를 가리킨다.
  commercial train의 selected_manifest에 없다는 이유로 validation origin을 추측하지 않는다.
  해당 원본의 독립 annotation/출처 연결을 추가로 확인한다.

보호 SHA는 QX3 `selected_sources[].source_sha256` 전체와 capture inventory의 모든 SHA,
known-audit의 모든 SHA key를 합친다. `teacher_required`만 제외하는 방식은 사용하지 않는다.
metadata SHA 중복 제외는 파일 재인코딩/근접중복 분리의 대체가 아니다.
다음 metadata-only 감사는 원본 이미지·GT를 읽거나 학습용 CSV/YAML을 게시하지 않는다.
그 결과에서 후보 수가 확보된 뒤 실제 bytes/단일 객체 GT/원본 연결/역할 분리를 검증하고,
새 불변 source lists/YAML에서 batch=1 raw generation을 실행한다.

AIHub 개방 데이터 정책과 dataset 71362 설명은 2026-09-04에 공식 페이지에서 다시 읽었다.
연구·개발 활용과 출처 표시, 재배포·국외 반출 등 제한을 구분한다.
[데이터 설명](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=71362&topMenu=),
[이용정책](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105).
이 페이지를 읽은 사실은 실제 다운로드 계정 동의나 모든 파생 source의 origin을 증명하지
않으므로 가짜 license receipt/승인 pin을 만들지 않는다. 원본·선별 이미지는 외부에 게시하지 않는다.

### Metadata-only 전량 집계 완료

2026-09-04 `audit_aihub_source_pool_20260904`에서 기존 91,938행 메타데이터를 모두 읽었다.
결과는 작업 루트의 `aihub_source_pool_audit_20260904/report.json`이며
SHA `4eb22923a2a2520e60b8e35c60126018f114c65841c773a864d2a49a83eb9a28`이다.
runner SHA는 `183f64eb14951e002570330ebfebe3c0fe60daf05461ee317783d84205a72987`이다.
입력 metadata SHA는 전후 확인했지만 원본 이미지/annotation을 읽거나 해시하지 않았다.

| 메타데이터 source 경로 | 기존 train | 기존 validation |
|---|---:|---:|
| commercial single root | 19,941 | 0 |
| original yolo_dataset_9class_v2 root | 55,516 | 16,479 |
| hardware_capture_prep root | 2 | 0 |

commercial 19,941행 중 19,919행은 selected_manifest의 원본 경로와 exact stem으로
연결되고, 22행은 hardware 복사본 이름이었다. QX3 3,500/캡처 133/known-audit 116의
보호 SHA 합집합은 3,636개였다. 기존 CSV에는 QX3 SHA/path 3,496행이 등장했다.
QX3 및 hardware 복사본 이름으로 제외할 행의 합집합은 **3,515행**이며 중복 사유를
더하지 않는다. 나머지 88,423행은 **학습 수용 완료 수가 아니라 다음 확인 대상 수**다.

- commercial exact join train: **19,283행**, 선언상 9종 존재.
- original v2 tree: train **53,661행**, validation **15,479행**. 이 원본들의 출처 연결은
  이번 exact commercial join 범위 밖이므로 추가 확인해야 한다.
- 남은 CSV의 선언값에는 train/validation 모두 9종+background가 있지만, 검증된 GT
  coverage가 아니다. 새 현재 이미지 hash·GT·재현성·근접중복 검증을 대체하지 않는다.
- 실제 원천/파생 이미지 삭제, 학습용 목록 게시, YOLO 추론, 학습 및 배포는 하지 않았다.

다음 단계는 이 88,423개를 바로 학습시키는 것이 아니다. commercial 목록의 원본·GT를
실제로 확인하고, validation의 AIHub 원본 연결을 복원한 뒤 보호집합을 분리한 새 source
입력을 만든다. 원본 v2의 `train_NNNNNNN.jpg`/`val_NNNNNNN.jpg`는 당시 converter의
순번 이름이므로 이름만으로 원 JSON을 추측하지 않는다. 변환 목록/설정이 없으면 원본
경로를 보존한 verifier manifest나 AIHub 원본 annotation에서 새 입력을 구성한다.
