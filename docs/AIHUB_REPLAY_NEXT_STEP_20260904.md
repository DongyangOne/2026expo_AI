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

## 7. 원본 이미지·JSON 검증 착수 (2026-09-04 13:10 KST)

순번으로 저장된 기존 v2 이미지의 원본을 추측하지 않고,
`crops_verifier_single_v3/manifest.csv`에서 고해상도 원본을 직접 연결한다.
이 CSV는 72,271,739 bytes, SHA
`c42f6a31382da5e060bfc784f0460ccc37d6a8e198577db3ba00b821968bafe7`이며
162,305개의 source_id가 중복 없이 존재한다. `source_path_b64`를 surrogateescape
왕복으로 복원한다. validation can 4개의 비 UTF-8 경로를 문자 대체로 고치지 않는다.

| 품목 | 원본 Training | 원본 Validation | 기존 commercial exact join |
|---|---:|---:|---:|
| can | 19,968 | 3,998 | 10,000 |
| pet | 19,999 | 4,000 | 10,000 |
| paper | 9,992 | 1,998 | 9,758 |
| plastic | 29,992 | 6,000 | 10,000 |
| styrofoam | 10,000 | 2,000 | 9,986 |
| vinyl | 9,999 | 1,999 | 9,814 |
| glass | 29,979 | 5,997 | 10,000 |
| battery | 3,271 | 409 | 3,105 |
| fluorescent | 2,403 | 301 | 2,243 |
| 합계 | 135,603 | 26,702 | 74,906 |

위 숫자는 **CSV 출처 연결 수**이며 전량 원본 검증 통과 수가 아니다.
commercial selected 74,906개는 모두 source_id/원본 path bytes/category/training/
source_object_count=1로 v3 CSV와 일치한다. 기존 train_balanced의 hardware 복사는
이 AIHub 원본 집합에 포함시키지 않는다.

### 실물 pilot 통과, 전량 검사는 진행 중

새 `scripts/audit_aihub_original_annotations.py`는 결정론적으로 선정한 원본에 대해
실제 컬러 이미지 decode/치수, 원본 JSON 파일명·단일 annotation·재질·bbox,
폴더/manifest/공식 split/source_id 일치를 검사한다. unknown 품목을 plastic으로
대체하지 않는다. malformed bbox, 다중 BOX, 중복 JSON key, symlink는 거부한다.
원본 DAMAGE는 참고 annotation_dent만 기록하고 세 상태 학습값은 전부 -1이다.
DIRTINESS를 label/foreign_material 정답으로 복사하지 않는다.

작업 루트 `operational_refresh_80bf78a_20260904_101000` 아래 실물 결과:

- `original_annotation_pilot_v1_20260904/result/report.json`: 36/36 원본 연결 통과,
  실제 읽기 142,975,442 bytes. SHA
  `393debb6e52905c416edbedaa751dc4ebb216f1624b49f9540c888142c7a7427`.
- `original_annotation_pilot_v2_20260904/result/report.json`: 위와 같은 36장을
  강화된 JSON/파일 안정성 검사와 기존 감사의 direct-grayscale/DCT pHash로 다시 검사.
  36/36 통과, elapsed 5.20초(입력 CSV 로딩/컨테이너 시작 제외), 4 workers.
  SHA `949f5081c55cb7fa358302327bb7e13e39e1b4a07d9faeb9934c12cbe4ab71b5`.
- v2 runner SHA
  `d3d82f6ee009a397716da7abde6e9499d54d85febb55a7c1f23ba337f5d70f99`.
  코드/단위·CLI 통합 테스트 44개가 통과했다(main `c03770f`).

36장은 9품목×Training/Validation×2장이다. 샘플 통과를 전체 데이터 품질이나
모델 정확도로 해석하지 않는다. `snapshot_only=true`이고 소비자는 원본/JSON SHA를
실제 사용 전후 다시 확인해야 한다. 학습·배포 권한은 계속 false다.

**진행 중:** `audit_original_annotation_full_v2_20260904` 컨테이너가 동일 v2 코드로
162,305장 전체를 읽는다(`--per-class-split 30000`, 현재 모든 그룹 크기보다 큼).
입력은 read-only, network none, CPU 4/RAM 4GiB, 원본 읽기 상한 1TiB이며
고해상도 이미지를 복제하지 않는다. 결과 예정 경로는
`original_annotation_full_v2_20260904/result/report.json`이다.
완료 여부는 exit code/OOM flag/최종 report·manifest SHA를 확인한다.
100장 단위 로그에서 실제 처리 수·elapsed와 선형 ETA를 확인할 수 있다.
첫 1,300장 당시 약 19.7장/초였지만 품목·파일 크기·캐시에 따라 달라질 수 있다.

### 전량 검사 이후 연결할 작업

1. 실제 통과/격리 원인을 집계하고 report SHA 및 입력 pin을 고정한다. 다시 읽은
   원본 SHA/JSON SHA가 불일치하면 새 입력에서 제외한다.
2. QX3/모든 운영·known-audit 보호집합을 원본 ID, 실제 SHA, 동일 pHash 규약으로
   대조한다. 공식 Training/Validation을 유지하고 교차 분할·근접중복을 격리한다.
3. 원본 JSON GT에서 새 YOLO label과 원본 path/source_id를 보존하는 sidecar를
   생성한다. 단일 원본·정답 파일·파생 바이트 해시를 함께 결박하고 기존 predicted
   class/conf/bbox는 재사용하지 않는다.
4. 신규 불변 source/YAML에서 batch1 raw 생성 → 같은 조건의 strict replay →
   lineage·누수 감사 → 조건 head 없는 품목 연구학습을 연결한다. 별도 상태 정답
   검증과 candidate policy pin 없이는 세 상태 head 후보로 승격하지 않는다.

이 단계들은 모델 성능 검증을 대체하지 않는다. 기존 41장과 QX3는 계속 보호하며,
신규 독립 blind와 하드웨어 이미지+무게→Spring callback gate 전 production은 유지한다.

### 보호 이미지 3,636장 실물 지문 완료 (13:20 KST)

기존 metadata 제외 목록을 실제 이미지 SHA와 같은 DCT pHash 규약으로 확인했다.
`audit_protected_fingerprints_v1_20260904`는 13:19:23~13:20:11 KST 실행 후
exit 0/OOM false, `failed.json` 없음으로 종료했다.

- QX3 3,500 + 운영 inventory 133 + known-audit 116의 SHA 합집합은 3,636개다.
  known 116 중 113개는 NAS 캡처에 있었고 나머지 3개는 노트북 기존 원본에서
  source SHA를 확인해 NAS의 새 보호 참조 폴더로 복사했다(총 203,234 bytes).
  누락 3개는 8월 1일 전 사진이므로 **보호 지문 참조 전용**이며 학습에 넣지 않는다.
  원본·기존 캡처·정답은 변경/삭제하지 않았다.
- `hardware_capture_prep_20260803/dataset_v2/resolved_audit_by_sha.json`의
  116 SHA와 membership은 known-audit에 정확히 연결된다. 원본 SHA는
  `aa835f2262482b1678754d99f547b598cb62ad8a794aa68dcf0033fb12af3982`다.
- 입력 `protected_fingerprints_inputs_20260904/inventory_strict.json` SHA:
  `26491c938d26aa90a1db51c187cb5b61d24360c6cb21b5779cc2319a002a74a0`.
  각 레코드는 SHA/path/roles만 있으며 `qx3`, `capture`, `known_audit` 역할을
  합친다. 과거 predicted class/bbox/label 값은 정답으로 전달하지 않는다.
- 출력 `protected_fingerprints_v1_20260904/result/report.json` SHA:
  `d33cb1105310bbc7da8dff3748d17350a9d5b67351e75d76a789521681c1aa41`.
  3,636/3,636 실물 이미지 SHA·치수·pHash 확인, missing 0, snapshot_complete다.
- helper `scripts/audit_protected_image_fingerprints.py` SHA:
  `c291de4f8ce4eb83f4f60b37c3a5e0b8e91933ffe08e86797c8f7eb2fac4ea6a`.
  원본 감사 helper와 합쳐 테스트 67개 통과. CPU 1/RAM 1GiB/network none으로
  실행했고 입력/metadata/code/원본을 게시 전후 다시 해시했다. 사용한 읽기 상한은
  2GiB이며 데이터 다운로드나 GPU 추론은 없었다.

이 결과는 **보호집합 자체의 바이트/지문 검증**이다. 아직 진행 중인 원본 162,305장과의
exact/near-duplicate 비교 완료나 학습 데이터 수용을 뜻하지 않는다. 보고서의
training/deployment/blind/selection 권한은 전부 false이고, 이후 소비 시 원본 SHA를
재확인한다. 전량 검사 종료 후 이 두 보고서를 연결해 cross-source/공식 split/보호집합
누수를 먼저 검사한다. 원본 1만 장 시점 elapsed 603.35초였으며 전체 검사 선형 잔여
예상은 약 2시간 33분이었다(품목/크기/캐시에 따라 변동, 학습 ETA가 아님).

## 8. 선별·YOLO 원본 변환 연결 (2026-09-04 13:43 KST)

`scripts/plan_aihub_original_cohort.py`는 전체 원본 감사와 보호 지문 보고서의 SHA,
메타데이터 5개 및 감사 코드를 고정한 뒤 source ID/exact SHA/pHash distance≤4로
보호 사진과 공식 분할 간 중복을 제외한다. 같은 분할의 exact duplicate는 source ID
순서로 하나만 남긴다. 교차 분할 중복은 양쪽을 제외하며 기존 예측값을 정답으로 쓰지 않는다.
전체 CLI는 18개 품목/분할의 원본 감사 범위가 완결되어야 실행된다. 36장 pilot을 전체
입력으로 가장하지 않으며 `full_cohort`는 감사 범위이지 학습·배포 승인이 아니다.

보호 이미지의 commercial stem에서 원본 ID 636개는 정확히 연결됐다. 그러나 과거
순번 이름의 legacy train 1,855장/val 1,000장, **총 2,855장은 원본 ID 연결이 미완료**다.
이들의 실제 SHA와 pHash는 보호되지만 재인코딩 전 원본 정체성의 완전한 증명은 아니다.
보고서에 `complete_original_lineage=false` 및 변환 이력·파생 이미지 누수·raw replay·
독립 하드웨어 gate의 `pending_checks`를 유지한다. 이 조건을 삭제해 수용하지 않는다.

### 실제 36장 선별 및 33장 이미지·정답 생성

다음 세 컨테이너는 13:43 KST 재조회에서 모두 exit 0/OOM false였다.

- `probe_original_cohort_36_20260904`: 원본 pilot 36장의 metadata 선별 36/36.
  `cohort_probe_36_20260904/report.json` SHA
  `de19c09ae882e6e58372a4e1299e72987b74d70fcd3bb8c07222f978161a100e`.
- `build_materializer_pilot_input_20260904`: 전체 학습용이 아닌 36장 진단 입력만 생성.
  `materializer_pilot_input_20260904/cohort.json` SHA
  `077a3384b947c06a80619da5df07e7584c47e74ad391d5dc39b022d134c2d4c6`.
- `materialize_original_pilot_36_20260904`: 13:37:09~13:37:15 KST 실행. 원본·JSON
  36장 재검증, 33장 변환, bbox 면적 비율 범위 밖 3장 제외. 이미지 33개와 YOLO 정답
  파일 33개가 실제 존재하며 `failed.json`은 없다. 약 3.19MB의 파생 파일만 생성했다.

변환 출력은 작업 루트 밖의 독립 경로
`/share/Container/aihub_original_materialization_pilot_20260904/dataset`이다.
`report.json` SHA는
`f4526132a62ede5abbb5bfe4b13cb1dcf0ebd695898a879d5add0204b13bb43b`이며
`snapshot_ready.json`의 report pin과 일치한다. Training 18장(9종 각 2장), Validation
15장이다. Validation battery가 0장이므로 **이 pilot은 9종 검증 데이터 충족이 아니다**.
`full_cohort=false`, training/deployment/blind authority=false다.

변환기는 원본 최소 변 320, bbox 면적 비율 0.04~0.80, 축소 grayscale 밝기 18~238,
Laplacian variance≥20으로 품질을 제외한다. 긴 변 최대 640/업스케일 없음/round/
INTER_AREA/JPEG90과 원본 JSON bbox의 정규화 8자리 YOLO label을 사용한다.
lineage에 원본·annotation·파생 이미지·YOLO label SHA를 결박하며 상태는 -1이다.
이 규칙은 이미지 품질 선별이지 실제 쓰레기통 적합성·상태 정답의 자동 증명은 아니다.

전체 감사는 중단하지 않았다. 13:43 KST 실제 로그 36,400/162,305(22.4%),
verified 36,400, elapsed 2,007.47초, 선형 잔여 6,944초(약 1시간 56분)였다.
원본 검사 ETA이며 학습 완료 ETA가 아니다. 보호 및 원본 입력은 계속 read-only다.

### 실제 33장 YOLO proposal 생성 완료

`generate_original_raw_pilot_33_20260904`는 13:46:32~13:46:45 KST 실행 후
exit 0/OOM false로 종료했다. 기존 모델
`runs/trash_v2_full-2/weights/best.pt`(SHA
`7b849c25c3983a54b4b6c922e425798f89326b2da21e862b90d2ee0c6a181f69`)와
현재 고정 batch1/raw-generation wrapper를 사용했다. GPU에서 33개 source를 실제
처리했고 proposal 34개 중 runtime top1로 33개를 선택했다. 33개 모두 원 JSON GT에
대한 positive IoU≥0.5이며 320px crop 33개/708,392 bytes를 생성했다. 이는
**이 pilot의 객체 위치 연결 결과**이지 9종 재질 정확도 100%라는 뜻이 아니다.
background와 Validation battery 지원도 이 pilot에는 없다.

출력 `/share/Container/aihub_original_raw_pilot_20260904/generation`:

- `raw/manifest.csv` SHA
  `105b6a01141fc4ec8a7aa37af778c2963bff2dc5a159a009bffc999c6ea31de2`.
- `raw/dataset_info.json` SHA
  `ab5cd47534dc01254b025c241bbbe3e8471197cfbd47429e8ede8405bee79fef`.
- `control/raw_generation_ready.json` SHA
  `92b8929b8bf8d48b10810fea32bb41602bf0e6c83aad55c577407417e1a6fe51`.
  `control/failed.txt`는 없다. 모든 downstream authority는 false다.

원본·annotation→파생 이미지·정답→실제 YOLO crop 생성 연결이 처음부터 확인됐다.
다음에는 frozen validator의 같은 batch1 replay를 검사한다. 과거 confidence/bbox
오차 허용 범위는 바꾸지 않으며 원본/기존 CSV를 수정하지 않는다.
읽기 전용 검사 컨테이너 `inspect_original_raw_pilot_20260904` 역시 exit 0/OOM false다.
관리용 foreground 요청은 60초 관측 timeout이 있었지만 실제 컨테이너는 약 2.35초 만에
성공 종료했다. 재접속 후 같은 컨테이너의 상태·로그로 확인했으며 작업은 재시작하지 않았다.

### NAS 자체 원본 검사→선별 자동 연결 설치

`scripts/nas/continue_original_audit_to_cohort_20260904.sh`를 새 불변 code 경로
`J/cohort_release_20260904_1352`에 복사하고 NAS `/bin/sh -n` 및 SHA를 확인했다.
J는 위 작업 루트다. bridge SHA는
`535a2e54921b66ac154b087efe4c131ca02d1ba734869e269ba08b42afded307`,
planner SHA는 `2bd66a9ed59b95f72265d8ac4f4a9c37b5e3f1533aa118d40c1b02dbbe5f3c33`다.

실제 CONTROL은 **`J/cohort_continuation_20260904_1356`**이며 다음 감시는 먼저 이
경로를 확인한다. 호스트 PID 26311/PPid 1과 자식 `docker wait` PID 26549가 실제로
원본 검사 컨테이너 ID
`306f871abf56624d3076ed01373905972b6d7e99b30e82dea599e1a6a9b0c0c1`을 대기하는 것을
확인했다. `producer_before.txt`에 running/OOM false 및 고정 image ID가 기록됐다.
bootstrap 로그는 `J/cohort_bridge_20260904_1356.bootstrap.log`다.

NAS에는 외부 `timeout`/`nohup` 명령이 없다. 초기 nohup 시작 실패 로그
`cohort_bridge_20260904_1352.bootstrap.log`는 보존했으며 해당 CONTROL은 생성되지
않았다. 실제 실행은 SIGHUP을 무시하는 분리된 shell과 전용 로그로 시작했다.
짧은 Docker 관리 명령의 watchdog은 자식 PID+Linux starttime이 같은 경우에만
TERM→KILL을 보낸다. 다른 서비스/PID 그룹을 건드리지 않으며 `docker wait`에는
시간 제한이 없다. watchdog의 stdin/stdout/stderr는 `/dev/null`로 분리한다.
Linux 정상 종료·실패 코드·TERM/KILL timeout·PID 변경·출력 pipe 검사를 포함한
6개 실행 검사와 shell syntax 검사를 통과했다.

원본 컨테이너의 같은 ID가 exit 0/OOM false로 끝나야 최종 report SHA를 고정하고
`cohort_original_20260904_1356`을 **한 번만** 생성한다(CPU 2/RAM 3GiB/network none,
입력 read-only/새 CONTROL만 writable). 그 출력은 `CONTROL/cohort/cohort.json`이다.
`cohort_ready.json`은 metadata 완료만 의미한다. failed.txt가 ready보다 우선하고,
관리 timeout은 observation_error.txt로 남긴 뒤 재시작하지 않는다. 자동 연결은 학습이나
배포를 수행하지 않는다. 감시에서 이 작업과 별도로 cohort를 중복 실행하지 않는다.

최신 코드의 원본 감사/보호 지문/선별/변환 테스트는 합계 **137 passed**다.
materializer의 기존 코드 SHA pin 덮어쓰기 결함도 수정했다(회귀 4개 포함).
최신 SHA `48598dde8491928f46ab64f9f479946e4a14a800f3e8d1dd1c396d9189e93b06`을
위 cohort_release에 배치했다. 앞선 36장 pilot은 코드 경로를 metadata로 pin하지 않아
그 결함 경로를 사용하지 않았다. 기존 pilot 코드·결과는 변경하지 않는다.

### 33장 frozen GPU replay A/B 완료 (13:57 KST)

`replay_original_raw_pilot_33_20260904`는 13:57:18~13:57:35 KST 실행 후
exit 0/OOM false로 종료했다. 동일한 원본 pilot raw 33행을 기존
`validate_v4_background_candidates.validate_manifest(prediction_provider=None,
diagnostic_only=True)`로 A/B 두 번 실제 GPU 재실행했다. 각 실행은 별도의
상대 symlink workspace를 사용하고 원본 generation의 파일·inventory는 변경하지 않았다.

- 실제 추론 A/B 모두 33/33. confidence tolerance **1e-6**, bbox tolerance **1e-4**
  그대로 통과했으며 임계값 완화·정답 변경·custom prediction provider는 없었다.
- 검증된 A/B manifest가 바이트 단위로 같다. SHA
  `9994dcb8bf9c2d9e021289afc709309f212d9338f3fe9379d6dd6ac197122437`.
- A/B report도 바이트 단위로 같다. SHA
  `1bd59878fd7b2aaa5f96b5f15a1430a44c4b59e01b3ebf193f3ad80d03dd8c15`.
- 출력 `/share/Container/aihub_original_replay_pilot_20260904/result`의
  `diagnostic_ready.json` SHA
  `8d1af01306a515bb9d47bf7777720d43efd6907df08b0a10274d6ddabf43dfc1`.
  실제 ready를 읽었고 `failed.json`은 없다. 내부 A 9.07초/B 2.27초는 초기화·캐시
  차이가 포함된 작은 진단 실행 시간이며 전체 학습/추론 처리량 예측으로 사용하지 않는다.
- runner는 `J/replay_pilot_code_20260904/replay_original_raw_pilot_20260904.py`,
  SHA `5655c9a4e662bc71c0a176c56b630f7cf0646b037e7b61c43c98f5cd4a08b785`.
  validator SHA `dd06258012ecd4cd92da06c33937d4d02ddd9f73f7c4f249c1bb55567ea1cbf8`,
  spec SHA `bc852128e8e7bee542b222c52ba8ff45d3ef0a648e4ac3884d2a0f37e1f9b841`.
  runner의 CPU 모의 연결 테스트 4개는 실행 순서·binding·실패 처리를 검사한 것이며
  위 실제 GPU 결과와 구분한다. input/model/code/spec/원본 source·label/raw inventory는
  A/B 실행 전후와 ready 게시 전후 재해시했다.

이는 **새 원본 기반 33장 pipeline의 실제 재현성 진단 통과**다. 전체 162,305장 또는
과거 91,938행 검증 완료·재질 성능 향상·독립 blind 통과가 아니다. training/lineage/
blind/deployment 권한은 계속 false다. 같은 pilot을 변화 없이 다시 돌리지 않는다.
다음 작업은 이미 실행 중인 전량 원본 검사와 NAS 자동 cohort 연결의 결과 확인이다.
13:57 KST 전량 로그는 processed 52,600/162,305(32.4%), verified 52,599,
선형 잔여 6,004초(약 1시간 40분)였다. 격리 1건의 상세 사유는 최종 보고서에서 확인한다.

## 9. Legacy 원본 연결 진단과 대기 bridge 보완 (2026-09-04)

과거 전체 train 804,421장은 base 변환 361,546장과 remainder 442,875장의 합이다.
별도 remainder 컨테이너/실제 로그를 확인했으며, 이전 `DATA_AUDIT.md`의 잔존 파일
추정을 정정했다. 파일 수만으로 원본을 삭제하거나 기존 train을 비우지 않는다.
상세 설정·로그 SHA·변환 순서는 `DATA_AUDIT.md`의 9/4 정정 절을 따른다.

새 `scripts/audit_legacy_aihub_links.py`는 기존 converter 순서/stride/나머지 선택으로
원본 **후보만** 찾고, 실제 BGR decode→floor 크기/640/AREA/JPEG90의 재생성 SHA가
보호 legacy 이미지 SHA와 같은 경우에만 `verified_source_link`로 기록한다.
원본·JSON·legacy 이미지/sidecar·코드·metadata를 소비 전후 재해시한다. 신규 GT 생성이나
역사적으로 유일한 원본이었다는 증명이 아니다. unresolved 행에도 검색 후보의 경로/SHA가
있을 수 있으므로 후속 소비는 반드시 status와 재생성 SHA 일치를 확인한다.

### NAS 실제 9장 probe 결과

- 최초 `audit_legacy_aihub_link_pilot_20260904`는 exit 1/OOM false다.
  `cap-drop ALL` 상태의 container UID 0이 host chunwol 소유 mode 0700 출력 경로를
  통과하지 못했다. 데이터 검사 전 실패이며 원본 연결 실패 9건으로 집계하지 않는다.
  해당 컨테이너·로그·출력 디렉터리는 보존했다.
- 별도 불변 출력에서 `audit_legacy_aihub_link_pilot_v2_20260904`를 실행했다.
  **14:19:45~14:19:55 KST, exit 0/OOM false**, 원본 연결 **9/9**,
  `failed.json` 없음. train/train_r/val에서 각각 첫 보호 인덱스 3개씩 검사했다.
- 출력 `/share/Container/legacy_aihub_link_pilot_v2_20260904/result/report.json` SHA:
  `a5b117f22485250bea6471060e1a8b24a8f40aa0e42bbc82291e685e0ac965de`.
  container 내부 읽기와 SMB `Get-FileHash`가 일치했다.
- 9개 모두 기존 sidecar 재현도 같았다. 9개 모두 `outside_v3`이며, 이는 **현재 v3
  manifest에 원본 경로 bytes가 없다**는 뜻이다. SHA/pHash alias 부재나 누수 검증 완료를
  뜻하지 않는다. 전체 보호 legacy는 train 1,255/train_r 600/val 1,000, 합계 2,855개다.
- `partial_selection=true`, `complete_original_lineage=false`,
  `original_alias_uniqueness_proven=false`, training/blind/deployment 권한 false다.
  전체 연결을 위해서는 `--max-per-kind 0`의 새 불변 실행과 미해결/별칭 후속 검사가 필요하다.
  원본 전량 검사가 실행 중이므로 큰 중복 I/O 작업을 즉시 추가하지 않았다.
- helper SHA `24a4fea8bb9c77d39ae5dd3d5de1ce243dbf33ec13e23c8a7eddaed0a17526f6`.
  helper 25개와 기존 cohort/materializer를 합쳐 **95 tests passed**다.

### 실제 대기 bridge 교체

위 권한 문제가 원본 검사 뒤의 cohort 컨테이너에도 적용됨을 확인했다. bridge는
`cap-drop ALL`에 **DAC_OVERRIDE 하나만** 추가했다. rootfs/input read-only, 새 CONTROL만
read-write, network none, no-new-privileges, CPU 2/RAM 3GiB 및 나머지 격리는 유지한다.
NAS 파일 권한/소유권은 변경하지 않았다. 회귀·POSIX syntax 검사 4개가 통과했다.

기존 대기 프로세스 PID 26311/starttime 632411687 및 그 자식 docker wait
PID 26549/starttime 632411906의 실제 명령과 대기 producer ID를 확인한 뒤 **그 둘만**
TERM으로 종료했다. 원본 컨테이너는 종료/재시작하지 않았다. 기존 CONTROL 1356의
`replacement_process_evidence.txt`와 `superseded.txt`에 근거와 교체 이유를 보존했다.

이후 사용하는 경로는 다음으로 바뀌었다. 앞의 8절 1356 경로는 역사 기록이다.

- CODE_ROOT: `J/cohort_release_20260904_1428`
- CONTROL: `J/cohort_continuation_20260904_1428`
- bootstrap: `J/cohort_bridge_20260904_1428.bootstrap.log`
- bridge SHA: `fc3ea4d3ce585a011a6984f63cc7309b5a48aef7dc8aa6224da854262a50b7b6`
- planner/helper SHA는 8절과 같다. 실제 새 bridge PID는 22398이다.
- 다음 cohort 컨테이너: `cohort_original_20260904_1428`

PID는 다음 실행 때 재사용을 확인한다. 원본 producer의 동일 ID/정상 종료 후 한 번만
metadata cohort를 생성하며 학습·배포는 하지 않는다. `failed.txt` 우선,
`observation_error.txt`는 관측 실패이며 중복 재시작 근거가 아니다.

14:23 KST 새 PID 22398/starttime 632597154/PPid 1의 실제 script 명령과
`producer_before.txt`의 원본 동일 ID/running/OOM false/image pin을 확인했다.
새 bootstrap과 오류 파일은 비어 있고 failed/observation_error/ready는 아직 없다.
원본 processed **80,500/162,305(49.6%)**, verified 80,492, 격리 8,
elapsed 4,437.59초/선형 잔여 4,510초(약 75분)다. 격리 이유는 최종 report로 확인한다.
이후 CPU 306.71%/RAM 1.485GiB, GPU 0%/2MiB/16,380MiB/42°C,
디스크 2.6TiB 여유(83% 사용)를 조회했다. GPU 조회는 QNAP 드라이버 lib 경로를
그 명령의 `LD_LIBRARY_PATH`에만 지정했다. timiroom 컨테이너는 계속 실행 중이며
naco-ollama 두 컨테이너는 기존 exited 상태다. 학습 ETA가 아닌 원본 검사 ETA다.

### Pi 접근 상태 (14:10 KST)

공개 `https://ai.oneexpo.kro.kr/health`는 HTTP 200, status ok,
main/state/verifier 모델 loaded를 반환했다. PC→Pi `100.121.110.75:22`는 5초 TCP 확인에서
도달하지 못했고 local Tailscale BackendState는 NoState였다. VPN/서비스를 바꾸거나
NAS 비밀번호로 Pi에 로그인하지 않았다. Pi host key는 이번에 live 대조하지 못했으며
신규 운영 사진 확보 여부는 여전히 미확인이다. health 성공은 신규 모델 성능·실제
하드웨어/Spring callback E2E 통과 증거가 아니다.

## 10. 원본 증거를 연결한 실제 GPU crop/replay (2026-09-04)

기존 8절 진단은 파생 JPEG/YOLO sidecar까지만 소비했다. 이번에는
`audited_aihub_snapshot.py`가 cohort/report/lineage와 **실제 원본 이미지·JSON에서
재생성한 파생 JPEG·sidecar 바이트**를 대조한 뒤, crop generator와 validator에
그 연결을 전달한다. 기존 완료 artifact는 수정하지 않았다.

- `source_sha256`은 실제 detector replay 대상인 resized JPEG SHA다.
  원본의 ID/이미지 SHA/annotation SHA/두 base64 경로와 materializer report SHA는
  별도 6개 필드로 보존한다. 공식 training/validation과 role/fold를 대조하고 상태는 -1이다.
- 기본은 full cohort 필수다. 이번 33장 partial 입력은 명시적 diagnostic 옵션이며
  validator도 diagnostic-only를 요구한다. metadata 게시 전후 원본/JSON·cohort·report·
  파생 파일·입력 membership 및 실패 marker를 다시 확인한다.
- 파일 번호/사진 SHA를 물리적 객체 ID로 가장하지 않는다. 실제 제공기관 PDF의
  다방향 촬영과 파일번호 정의, 라벨/외부 이물질 혼합 정의, 전체 metadata 진단은
  `AIHUB_SOURCE_SEMANTICS_20260904.md`를 따른다.

### 실제 NAS 산출물

코드 root는 `J/linked_pilot_code_20260904`다. 기존 materialized 33장과
원본 cohort/report pin(8절)을 그대로 사용했다. direct runc/device/QPKG libraries,
고정 image와 model/spec, batch1, confidence 1e-6/bbox 1e-4를 유지했다.

- `generate_original_linked_raw_pilot_33_20260904`: **14:52:01~14:52:19 KST,
  exit 0/OOM false**. 33개 원본 증거 연결, 34개 검출 중 top1 33개 crop 생성.
  `/share/Container/aihub_original_linked_raw_pilot_20260904/generation`의
  ready SHA `b895f376ca5cfc9b1c8075cfec9b0403e8ca3cd147d4ec4b83379a84272637cd`,
  raw manifest SHA `7ea619e2d1f96816b7629f772bfb589e4f02e84e754201de9107a7a6d1d8aec8`,
  dataset_info SHA `7d4c07dbca38874aa7a33eaeb9d3e5da60ad2f2481892ec8f14afed7070f54a5`.
- `replay_original_linked_pilot_33_20260904`: **14:54:36~14:55:06 KST,
  exit 0/OOM false**. actual GPU A/B 각각 33/33 strict replay 통과.
  `/share/Container/aihub_original_linked_replay_pilot_20260904/result`의
  `diagnostic_ready.json` SHA `5bb47dda88d10e718718d00d98d7e35bc0c71b8c07397f1547f2ae680409889c`.
  A/B manifest 동일 SHA `ffaca6edbc004d397ad94e5f6eccd20c423ac1b7f28be018918d71aa1e128023`,
  report 동일 SHA `97d09ffd789574ad5eaad5a863eee39a8317fb95178926f7ceb6b3d8d1645388`.
  `failed.json`은 없고 training/lineage/blind/deployment authority는 모두 false다.
- A 17.74초/B 8.18초는 원본 증거 재검증과 초기화가 포함된 작은 진단 시간이다.
  전체 학습 처리량이나 모델 정확도로 해석하지 않는다.
- frozen prepare SHA `7a7b67652f98923c8cc4e263917065ffceb08cb0b52d0f1ccd02ca8e7478aca9`,
  validator `f2b684ef59e275a1a2bcd21db233ec8d65df0de02ddc93df50bcf09e591ffbc9`,
  reader `c31ff7ebedede62e78270caad224b4f36cc306819c887e49c648a1e36ba39a32`.
  wrapper `be8132bb2991704b9c792445e5aa3d6a91eb5f7b6d77c9e49e759acb3066726a`,
  runner `5320b79eab7b846de1f0191998d78999c8ef21b3fb05a9a0b6ec547eae21ee5f`.
  runner는 generation의 11개 입력 pin, 별도 고정 spec, canonical helper 6개를
  포함한 실행 코드도 전후 대조했다.

이 결과는 **원본 증거→실제 YOLO crop→동일 GPU replay 연결의 33장 진단 통과**다.
val battery와 background 지원은 여전히 없으며 전체 학습이나 9종 정확도 개선 완료가
아니다. 현재 원본 162,305장 감사와 1428 bridge는 별도로 진행 중이다.
완료된 같은 33장 probe를 반복하지 않고 전량 report/격리 사유/cohort 결과를 확인한 뒤,
보호 legacy 2,855개 원본 연결과 alias/근접중복 검사를 이어간다.

최종 동결 코드에 대해 reader, audited prepare/validator 통합, wrapper, 기존
prepare/validator 및 운영 통합의 7개 테스트 파일을 함께 실행해 **159 passed
(176.05초)**를 확인했다. 실제 GPU 진단과 CPU fixture 회귀를 구분한다.

14:57 KST 원본 로그는 processed 116,600/162,305(71.8%), verified 116,585,
격리 15, 선형 잔여 2,523초(약 42분)다. 상세 격리 원인은 최종 report로 확인한다.
CPU 276.67%/RAM 1.788GiB(상한 4GiB), GPU 0%/2MiB/16,380MiB/42°C,
디스크 2.6TiB 여유(83% 사용)였다. 원본 검사 ETA이며 학습 종료 ETA가 아니다.

### 다음 연구 학습용 기존 backbone 확보

실제 중지된 `train_proposal_verifier_multitask_v3_20260827`의 변경 파일 목록에서
`/root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth`를 찾았다.
컨테이너를 실행하지 않고 `docker cp`로 새 `J/pretrained_reuse_v1_20260904/`에
10,306,551바이트를 복사했다. SHA256은
`047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`다.
원본 컨테이너 ID `2923d7a57e1a1c988a0c92975e047a971b62f77f78529955f517750725690532`와
exited/exit0/OOMfalse가 복사 전후 같았다. `source_container_before.txt`,
`source_container_after.txt`, `pretrained.sha256`를 함께 남겼다. 외부 다운로드나
기존 모델 교체는 하지 않았다. 이 파일은 backbone 초기값이지 새 쓰레기 분류 후보가 아니다.

## 11. 보호 legacy 전체 연결 자동 대기와 cohort 제외 연결 (2026-09-04)

원본 검사를 다시 실행하지 않고 **같은 producer 정상 종료 뒤 한 번만** 보호 legacy
2,855장 전체 연결 검사를 실행하도록 NAS 자체 대기를 추가했다.

- script: `scripts/nas/continue_original_audit_to_legacy_20260904.sh`
- NAS frozen script: `J/legacy_full_release_v1_20260904/scripts/nas/continue_original_audit_to_legacy_20260904.sh`
- SHA: `4662f9b64e1a443f1d925e9d2f6e55641cc11c0fcf04d23172cd7a5c4426b104`
- CONTROL: `J/legacy_full_continuation_v1_20260904`
- bootstrap: `J/legacy_full_bridge_v1_20260904.bootstrap.log`
- 실제 대기 PID 3906/starttime 632881529/PPid 1, 자식 3949가 원본 producer
  `306f871abf56624d3076ed01373905972b6d7e99b30e82dea599e1a6a9b0c0c1`을 `docker wait` 중이다.
  이후 조회에서는 PID 재사용을 다시 검사한다.
- 다음 컨테이너: `audit_legacy_aihub_link_full_v1_20260904`
- 다음 산출물: `/share/Container/legacy_aihub_link_full_v1_20260904/result/report.json`

15:17:46 KST에 두 대기 프로세스(기존 cohort PID 22398 포함)와 자식의 실제 명령을
확인했다. 새 CONTROL에는 `producer_before.txt`와 빈 `producer_wait.txt`만 있었다.
**legacy 전체 검사는 아직 시작 전**이며 원본은 139,200/162,305(85.8%),
verified 139,181, 격리 19, 선형 잔여 1,273초(약 21분)다. 학습 ETA가 아니다.

대기는 producer 동일 ID/image/exit0/OOM false와 원본 report/실패 marker를 확인한 뒤,
기존 9장 pilot에서 검증한 helper·converter·remainder·보호 snapshot·manifest SHA를
재확인하고 `--max-per-kind 0`으로 실행한다. CPU1/RAM2GiB, network none,
rootfs/입력 RO, 새 output만 RW, cap-drop ALL + DAC_OVERRIDE, no-new-privileges이며
GPU나 Docker socket을 전달하지 않는다. 기존 서비스와 모델은 변경하지 않았다.
`audit_container_id.txt`/`dispatched.txt`는 **dispatch** 근거이지 완료 근거가 아니다.
`failed.txt`가 우선이고 `observation_error.txt`는 중복 시작 근거가 아니다.
다음 감시에서는 이 대기와 실제 container ID를 먼저 확인한다.

### 전체 결과가 나온 후 연결할 최소 단계

1. 실제 full legacy report의 정상 종료·SHA·2,855개 coverage·verified/unresolved 수를
   확인한다. partial 9장 report를 full로 사용하지 않는다.
2. `build_legacy_protected_inventory.py`는 실제 검증된 원본 링크만 source-only 목록으로
   만든다. 기존 보호 역할을 상속하고 같은 원본 SHA를 합치며 unresolved의 추정 경로는
   사용하지 않는다. 이미지나 학습 정답은 생성하지 않는다.
3. 기존 `audit_protected_image_fingerprints.py`가 이 목록의 **실제 원본 픽셀**을 읽어
   SHA/pHash를 계산한다. 이전 3,636개 snapshot은 재실행하지 않고 보존한다.
4. 새 불변 경로에서 `plan_aihub_original_cohort.py`에 `--legacy-link-report`와
   `--legacy-original-fingerprint-report` 및 각 SHA를 함께 제공한다. 기존 원본/보호
   report·selected manifest·auditor SHA 입력도 그대로 필요하다. 새 원본 ID 및
   동일 SHA alias 전체를 제외하고 기존 보호 지문에 원본 pHash를 합친다.
   공식 split과 pHash 거리 4는 바꾸지 않는다.

기존 `cohort_release_20260904_1428`의 실행 코드와 결과를 덮어쓰지 않는다.
새 planner는 추가 입력이 없는 기존 mode를 유지한다. 원본 경로·이미지/JSON SHA·
공식 split·실제 link/fingerprint coverage와 소비 metadata를 전후 검증한다.
outside-pool 원본도 pHash 보호에 포함하지만, 미해결 링크·변환 alias·변환 후 이미지
누수 검사는 별도로 명시한다. 물리적 객체 정체성이 모두 입증됐다고 하지 않는다.

관련 테스트는 planner **77 passed**, inventory **17 passed**, NAS 대기 wrapper
**19 passed**다. fixture 회귀이며 실제 2,855장 연결이나 모델 정확도 통과가 아니다.
학습 입력은 이 단계 이후 full materialization→새 actual YOLO crop/strict replay→
lineage 및 source/crop 양쪽 근접중복 검사로 이어진다. `--no-condition-heads` 연구
학습은 상태 모델 최종 후보나 배포 승인이 아니며 production/Pi/Spring은 유지한다.

### 다음 코드 사전 배치와 기존 보호 crop 재사용 확인

main `a9b48e25d6a81dced53dc1ce83307d8de2770ea3`의 다음 네 파일을
`J/legacy_exclusion_code_a9b48e2_20260904/scripts/`에 새로 복사했다. SMB 복사 전후와
NAS `sha256sum` 결과가 일치했고, 기존 CPU 컨테이너에서 builder의 `--help` import가
exit 0으로 완료됐다. 이는 배치/실행 환경 확인이지 full report 처리 완료가 아니다.

- `build_legacy_protected_inventory.py`: `1fbc4e211caf2f7a8e8f920da631b4544f138e3d9ab4408b7c3644d3550a922b`
- `plan_aihub_original_cohort.py`: `be81331b74385880615ff2b9217eaac89c03cfafda50125f108e2cdb94efef9a`
- `audit_protected_image_fingerprints.py`: `c291de4f8ce4eb83f4f60b37c3a5e0b8e91933ffe08e86797c8f7eb2fac4ea6a`
- `audit_aihub_original_annotations.py`: `d3d82f6ee009a397716da7abde6e9499d54d85febb55a7c1f23ba337f5d70f99`

15:28 KST 원본 검사는 150,600/162,305(92.8%), verified 150,579,
격리 21, 선형 잔여 644초였다. 이후에는 최종 report와 producer 종료 상태를 먼저 확인한다.

보호 crop을 전량 다시 추론하기 전에 실제 기존 QX3 파일을 조회했다.

- `v4_batch1_repro_pilot_qx3_5ac3031_20260902_203411/input/selection_inventory.json`의
  전체 3,500 source와 `v4_repro_validation_qx3_retry2_5ac3031_20260902_223000/validation/validator-a/manifest.v4.validated.csv`를
  SHA로 join했다. CSV의 실제 SHA는 `686839341a25cfd81e7a01622db386df72625de657d1b6205712cf2b0d64db70`,
  3,498행/3,498 unique source다. 기존 diagnostic 역할과 lineage authority false를 유지한다.
- 기존 generation의 `generation/raw/dataset_info.json`을 실제 읽고 SHA
  `c15d8310a3f9cf779d483a9a996f7a851df30cfecc97d577d2873701332ddd17`을 대조했다.
  전체 경로의 root는 `v4_batch1_repro_generation_qx3_retry1_5ac3031_20260902_214200`이다.
  frames_seen 3,500, proposals_selected 3,498, frames_without_eligible_proposal 2,
  source/write rejection은 비어 있다. 원본 폴더는 입력 RO로 유지됐고 이 generation은
  exited 0이다. 새로운 추론은 실행하지 않았다.
- 누락 두 source는 selection에서 explicit_empty_label=true인 training/background다.
  `capture_9619af3f1aa23aaa.jpg` SHA `7344f55e8e5a3c11190ce8fd67c25997e239029ed7827dd92727fe866cb8834e`,
  `capture_d0ee74f0e4a719b6.jpg` SHA `f4c9fd25e7a25fa35e40acc52cbc4cff6c936c47a4977ac29ea198408702c1a4`다.
  이 두 SHA도 보호 집합에서 제거하지 않는다. 개별 빈 detector 결과의 새 attestation이나
  다른 crop 파일 존재까지 입증한 것은 아니다.

3,498개는 우선 **재사용 후보**다. 실제 crop bytes/상대경로와 원본 SHA를 다시 대조해야
formal inventory로 소비할 수 있다. 보호 전용 ROI는 정답 bbox crop인지 YOLO crop인지
출처를 구분한다. 현 formal v1은 모든 보호 SHA에 source/crop 각 1개를 요구하므로,
빈 source를 crop이라고 복사하거나 2개를 누락한 목록으로 통과시키지 않는다.
`crop_absent` 확장은 기존 계약 무변경이 아니므로 이번 작업에서는 적용하지 않았다.

`tmp/run_aihub_material_research_20260904.sh`는 same-process CUDA/dry-run/최대100 epoch
연구용 **미실행 초안**이다(SHA `cd97b8b022a1957d15f11403c20d5a7a431b779d27bcee3d672583b856474040`).
구문·정적 경계만 검사했으며 NAS에 실행하거나 학습 완료로 표시하지 않았다.
실제 역할별 CSV와 full 보호 감사 입력이 확보된 후 실행 검토한다.

## 12. 전체 원본/legacy 완료, annotation 충돌 격리와 회수 원본 지문 검사

2026-09-04 15:56 KST 기준 실제 상태다. 11절의 대기/진행 중 상태를 대체하며,
완료된 원본 및 legacy producer와 실패한 1428 cohort는 다시 실행하지 않는다.

- 원본 producer `306f871abf56624d3076ed01373905972b6d7e99b30e82dea599e1a6a9b0c0c1`:
  15:38:32 KST 종료, exit0/OOMfalse. `J/original_annotation_full_v2_20260904/result/report.json`
  SHA `aed28757966d86f2a53a461ea089edf577156cec3e6001d57f09670d703655e2`.
  전체 162,305건 중 검증 162,280건, 격리 25건이다. 격리 사유는 범위 밖/빈 bbox 6,
  알 수 없거나 충돌하는 재질 7, annotation 파일명 불일치 12다. 검증 데이터에는
  공식 training/validation × 9종의 18 strata가 모두 존재한다. 학습/배포 권한은 false다.
- full legacy producer `f24cbf8cfd9fc7004d4fd6fb76bb18eb008157401fa9a681e6c631f594754eef`:
  15:38:37 시작, 15:47:52 종료, exit0/OOMfalse. 실제 자동 대기가 이어서 시작했다.
  `/share/Container/legacy_aihub_link_full_v1_20260904/result/report.json`
  SHA `73da019af967787cb2ca93279bec0ce8c7e5f46bc0a114613a36fbd350be4747`.
  train 1,255 + train_r 600 + val 1,000 = 2,855건 모두 verified_source_link,
  unresolved 0, label reproduction 실패 0이다. 원본 pool 내부 1,144, 외부 1,711이며
  이 실행의 unique original SHA는 2,855다. `original_alias_uniqueness_proven=false`와
  `complete_original_lineage=false`는 유지된다. 원본 연결 성공이 전체 변환 alias 증명은 아니다.

### 실패한 기존 cohort를 통과 처리하지 않은 이유

`cohort_original_20260904_1428`/ID
`e186765c5d7339831fff5f806d0eb92891af8053ed842c51ce2aff6cc4ce10b1`은
15:38:47에 exit1/OOMfalse로 종료했다. 실제 오류는
`same image SHA has conflicting ground truth or image evidence`이며 OOM이 아니다.
기존 CONTROL의 failed.txt, 로그, release는 보존했다.

원본 report만 읽는 CPU 진단 `inspect_original_conflicts_v1_20260904`가 exit0으로
완료했고, `J/original_conflicts_diagnostic_v1_20260904/output/conflicts.json` SHA는
`af835e77ba33018b2572898dd32d3191c1f15865c1d724cc260bb90bef5764e8`이다.
사진 bytes/해상도/pHash 충돌은 0이고, 같은 PET 사진의 bbox annotation이 다른
2개 SHA 그룹/4개 원본 행이 원인이다.

- SHA `d5d3225770e62dadefc5a603bb97d024a7b78d9bf409402298b5bb813b656389`:
  `2f310e06c1e41d69a911`(training), `d113d77c390402219684`(validation).
- SHA `e94737689112d5400829bd78583ebf609c9956e8b4925157e2629309582d91ab`:
  `fdad0d1428cf9276d4d0`, `4ffa3224312bd7da3037`(모두 training).

새 planner의 `--quarantine-annotation-conflicts`는 기본 off다. 명시적으로 켜면
class/bbox 충돌 그룹 전원을 `annotation_conflict_same_sha256` 사유로 제외하고,
이미지와 annotation 경로/SHA/bbox 근거를 보고서에 기록한다. 대표 정답을 고르거나
bbox를 수정하지 않는다. 모든 행을 split/pHash 인덱스에 남겨 다른 near duplicate
제외까지 유지한다. bytes/해상도/pHash 충돌은 옵션과 관계없이 계속 실패한다.
공식 split, pHash 거리 4, 보호 SHA 및 학습/배포 권한 false를 변경하지 않았다.

새 코드 `J/legacy_exclusion_code_annfix_v1_20260904/scripts/`의 4개 파일은
SMB 복사 전후와 실제 NAS SHA가 일치한다. planner SHA는
`2ebc1612d86d6a36f0d0a56ee13bd129491b9e35f0e530539e5b1b14ff88541f`이며
나머지 3개 helper SHA는 11절과 같다. 이전 release는 덮어쓰지 않았다.
planner **99 passed**, inventory/reader/materializer 회귀 **71 passed(53.30초)**를
확인했다. 데이터 정제 코드 회귀이지 모델 정확도 검증이 아니다.

### 현재 실행 중인 다음 단계

`fingerprint_legacy_originals_v1_20260904`/ID
`19f2f6c6769ce930ebe36fa648f05acde9d8bed9877782c27c2f1fc9ab1ce859`가
15:55:57 KST 실제 시작됐다. CPU1/RAM3GiB, runc, network none, rootfs/전체 입력 RO,
새 `J/legacy_recovered_fingerprints_v1_20260904`만 RW, Devices=[]다.
builder가 2,855개 원본 inventory를 게시한 뒤 실제 원본 SHA/pHash 검사를 순차 실행한다.

- inventory: `J/legacy_recovered_fingerprints_v1_20260904/inventory/inventory.json`
- inventory SHA: `5014c0ab08307711172ffc14ca98f307ad8bbb996ea82e0af04cbbda9158134e`
- 완료 시 report: `J/legacy_recovered_fingerprints_v1_20260904/result/report.json`
- 15:56:33 로그: verified 300/2,855. 약 8.7장/초로 픽셀 처리 잔여 약 5분이며,
  이후 원본/metadata의 두 번 재해시가 추가된다. 완료 ETA나 학습 ETA로 혼동하지 않는다.
- max-read 128GiB/max-file 128MiB는 읽기 상한이다. 원본 이미지를 복사하지 않는다.

다음 실행은 실제 exit0/OOMfalse와 report SHA/coverage를 확인한 뒤 이 새 report와
full legacy report, 기존 원본/보호/manifest/auditor 핀을 모두 넣은 **새 cohort**다.
annotation 충돌 4행 전원 제외와 18 strata를 확인한 후 materialization으로 이어간다.
기존 QX3 보호 crop 3,498개는 재사용 후보이며 아직 전량 실제 bytes 검증 전이다.
추론이 없는 보호 source를 crop으로 위장하거나 보호 목록에서 빼지 않는다.

현재 앱은 연구용 objectness(2)+material(9) ONNX를 그대로 읽는 계약이 아니다.
별도 shadow adapter와 기존 상태 모델 유지, 독립 하드웨어 E2E 검증이 필요하므로
연구 학습 완료/오프라인 평가와 운영 교체를 구분한다. production/Pi/Spring,
timiroom 서비스, 중지된 naco-ollama 2개는 변경하지 않았다.

## 13. 회수 원본 지문과 full refined cohort 완료 (2026-09-04)

12절의 `fingerprint_legacy_originals_v1_20260904`는 16:02:15 KST에 exit0/OOMfalse로
정상 종료했다. 실제 report SHA는
`d35de267a695b0462e1f3e35e5e038060e2362bf5fca76138f5fedda7794f0cd`,
expected/verified/records 모두 2,855, missing 0, failed.json 없음이다.
report를 읽기 전후 SHA가 같았고 inventory/code SHA도 12절과 같았다.
원본 픽셀 검사와 게시 전후 재해시까지 끝났으며 학습/배포 권한은 false다.

이어 실행한 `cohort_original_legacy_refined_v1_20260904`/ID
`b27c511a92f93c26e2af58e8a5afeea66109c8a0792e269ae452659e43265944`는
16:05:00~16:05:24 KST exit0/OOMfalse다. 산출물은
`J/cohort_original_legacy_refined_v1_20260904/result/cohort.json`, SHA
`2383dd91fee9b98b18a108cbba3270484ad6b051ff0dd3bb1196547a2196e6de`다.
읽기 전후 SHA와 실제 레코드 membership/상태값/개수를 별도 read-only 컨테이너에서
검사했고 모두 통과했다. 전량 metadata 선별 완료이지 새 모델 학습 완료는 아니다.

- verified 원본 162,280 = accepted **155,537** + excluded **6,743**.
  최초 원본 감사 격리 25건을 합하면 전체 162,305건이다.
- annotation 충돌 2그룹/4행 모두 exclusions에 있고 accepted에는 없다.
- 사유별 수는 **중복 집계**다: annotation conflict 4, cross-split exact/near 4,953,
  protected exact SHA 1,142, protected near pHash 2,300, protected source ID 1,702.
- 회수 링크 verified 2,855/unresolved 0, recovered original SHA 2,855 모두 소비했다.
  최종 verified pool 기준 path/ID 일치는 1,142, pool ID 미연결 1,713이다.
  이는 full legacy report의 manifest pool membership 1,144/1,711과 다른 집계 범위다.
  pool 밖 원본도 pHash 보호에서 누락하지 않았다.
- `phash_distance=4`, `full_cohort=true`, 모든 상태값 `-1`, 학습/배포 권한 false를 확인했다.
  alias uniqueness와 complete_original_lineage는 여전히 false다.
- pending: legacy transform aliases, materialized-image leakage, raw proposal replay,
  independent hardware gate. 이 단계를 생략하거나 모두 완료로 바꾸지 않는다.

| 원본 분류 | training | validation |
| --- | ---: | ---: |
| battery | 3,011 | 298 |
| can | 19,397 | 3,706 |
| fluorescent | 2,156 | 190 |
| glass | 29,211 | 5,606 |
| paper | 9,714 | 1,814 |
| pet | 19,454 | 3,710 |
| plastic | 28,974 | 5,526 |
| styrofoam | 9,509 | 1,692 |
| vinyl | 9,763 | 1,806 |

원본 학습 분류의 pet/plastic 분리는 기존 9종 기준이며 Spring 반환 계약의
PET→plastic/class_id=3 통합은 변경하지 않는다.

### Materialization 출력 경로 사전 실패와 수정

첫 `materialize_original_legacy_refined_v1_20260904`/ID
`9fb560fccf95e2557a46a6d24a62a7372eec73c87d1897cfeea76d5559a11ef0`는
16:07:57~16:08:00 KST exit1/OOMfalse다. 첫 출력 위치를 `J` 아래에 두었는데,
cohort가 `J` 바로 아래 원본 auditor 파일도 immutable input으로 결박하므로
materializer의 보호 경로 검사에 걸렸다. read-only 재현에서 실제 원문은
`materialize_audited_aihub_sources.py:200: ValueError: output overlaps immutable input evidence`였다.
원본/이미지 처리 전 실패했고 `J/materialized_original_legacy_refined_v1_20260904/result`는
생성되지 않았음을 확인했다. 컨테이너와 로그를 보존하며 안전 검사를 완화하지 않았다.

복구 실행은 출력만 `J` **밖의 새 경로**로 옮긴다.

- container: `materialize_original_legacy_refined_v2_20260904`
- ID: `73ca9e2edb36b9a9196e04e71add905de07cbd91a63c68f54a46d875746ca623`
- OUT: `/share/Container/aihub_original_legacy_refined_materialized_v2_20260904/result`
- 기존 frozen 코드: `J/cohort_release_20260904_1428/scripts/materialize_audited_aihub_sources.py`
  SHA `48598dde8491928f46ab64f9f479946e4a14a800f3e8d1dd1c396d9189e93b06`와
  같은 디렉터리 원본 auditor SHA `d3d82f6ee009a397716da7abde6e9499d54d85febb55a7c1f23ba337f5d70f99`.
  실제 NAS에서 두 SHA를 다시 확인했고 코드는 수정하지 않았다.
- CPU2/RAM6GiB, runc, network none, rootfs/전체 입력 RO, 이 새 OUT 부모만 RW,
  cap-drop ALL+DAC_OVERRIDE/no-new-privileges, GPU devices 없음.
- CLI: 위 cohort와 SHA, dataset-root `/app/ai_dataset/학습용_데이터`,
  `--max-sources 0 --min-free-gib 300 --max-output-gib 30`.
  시작 전 여유 디스크 2.6TiB/83% 사용, buffers/cache 제외 사용 메모리 약 13.5GiB였다.

현재 materializer는 순차 처리로 `--workers`/`--resume`가 없고 100개마다 진행률을
출력한다. 초기 path/metadata 검사와 마지막 3회 재해시에는 별도 진행률이 없다.
원본/JSON은 논리적으로 5회 읽지만 OS cache 영향으로 실제 디스크 I/O와는 다를 수 있다.
30GiB는 축소 JPEG·label·lineage·report 전체 상한이며 고해상도 원본은 복제하지 않는다.
완료 판정은 실제 exit0/OOMfalse, snapshot_ready.json, report/lineage SHA,
failed.json 없음, 품질 제외 후 18 strata 지원까지 확인해야 한다.

16:14:07 KST 실측: v2는 16:10:38 시작 후 running/OOMfalse이며 실제 result 폴더와
이미지를 생성하고 있다. 16:13:07 verified600/materialized570에서
16:14:04 verified1,000/materialized960으로 진행됐다. 최근 400건/57초는 약
7건/초다. 이 초기 속도를 그대로 적용하면 **주 변환 처리만 잔여 약 6.1시간**이며
마지막 전체 재해시 3회가 추가되므로 최종 완료 ETA는 아직 확정하지 않는다.
품질 제외 40건은 최종 excluded.jsonl/report에서 사유를 확인한다.

자동 감시 `nas`는 30분 간격 ACTIVE를 유지하면서 이 v2 ID/새 OUT과 완료된
fingerprint/cohort SHA로 갱신했다. 저장된 prompt 일치를 다시 확인했다.
이미 끝난 원본·legacy·fingerprint·cohort를 재실행하지 않고 v2 완료 후 다음
YOLO 생성/strict replay로 이어가며, 변화 없는 정상 상태는 알림을 만들지 않는다.
