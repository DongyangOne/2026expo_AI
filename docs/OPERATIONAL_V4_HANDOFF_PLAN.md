# 운영 pseudo-label → V4 후보학습 연결 계획

2026-09-04 기준 **구현 계획이며 실행·학습 승인 문서가 아니다.** 저장소는
`D:\git\2026expo_AI`다. 기존 도구를 연결하며 신규 플랫폼·프레임워크는 추가하지 않는다.
NAS 클래스/상태별 표본 수와 모델 정확도는 아직 미확인이다.

## 아직 남은 세 가지

- 운영 teacher/localizer 출력과 V4 source CSV·실제 replay report 사이의 provenance 연결.
- train/model_validation 각각 9종 + background 및 세 상태 head의 0/1 지원 여부.
  NAS 집계 전에는 충족한다고 가정하지 않는다. 모르는 dent/label은 계속 `-1`이다.
- 실제 training config·license/protected evidence·host inspect를 결박한 승인 policy.
  `configs/v4_candidate_training_trusted_policy.json`은 아직 없고 builder/launcher의
  `APPROVED_TRUSTED_POLICY_SHA256`은 `UNCONFIGURED`다. 테스트 fixture는 승인 자료가 아니다.
- 추가 blocker: 새 20장에 확정 품질실패가 0건이어도 현재 producer(`build_v4_quality_exclusion_manifest.py:130`)와 shared contract(`operational_quality_assembly_contract.py:257`)는 빈 entries를 거부한다. 정상 empty 경로를 명시적으로 구현·검증해야 하며 가짜 실패를 만들면 안 된다.

## 다음 작업: 한 번에 한 단계씩

1. **실제 입력 목록을 고정한다.** 현재 운영 배치 식별자는
   `operational_refresh_80bf78a_20260904_101000`이다. 다음 실행에서 NAS의 실제
   queue/known-audit/capture-inventory, 새 8192-context teacher 결과, A/B provider
   manifest·실제 GGUF·spec, full quality assembly의 경로와 SHA를 확인해 기록한다.
   원본 상대경로·오류 출력을 보존한다. 아래는 **과거 watcher의 탐색 시작점이며 현재 미확인**이다.
   `/share/Container/runs/trash_v2_full-2/weights/best.pt`,
   `/share/Container/yolo_mixed_replay_v2_20260819/dataset_mixed.yaml`,
   `/share/Container/hardware_capture_prep_20260803/dataset_v2/yolo/images/val`.
   마지막 경로의 고정 41장은 진단/calibration 전용이다.

2. **source 단위 증거를 먼저 만든다.** 변경 위치는
   `scripts/build_operational_teacher_manifest.py`의 `MANIFEST_FIELDS`와 accepted-row/
   lineage 생성 부분이다. 기존 검증을 재사용해 실제 입력을 재검증한 source-evidence
   JSON을 봉인한다. 필요한 필드는 `source_filepath`(검증된 원본 절대경로), `captured_at`
   (`capture_timestamp`의 검증된 원래 시각), `teacher_output_sha256`,
   `localizer_output_sha256`, `auditor_sha256`다. 마지막 세 값은 임의 문자열이 아니라
   각각 원본 teacher row, 재계산한 independent-localization 객체, source-evidence
   JSON의 실제 digest로 정의한다. canonical JSON 규칙과 입력 파일 SHA를 명시한다.
   source-evidence에 **최종 V4 CSV hash를 포함하지 않아** auditor SHA 순환을 피한다.
   `vlm_teacher_pseudo_label_train_only` 권한을 보존한다.
   테스트: timestamp·증거/모델 변조, 보호 SHA, bool/int, symlink·root 이탈, 게시 중 변경.

3. **독립 annotation과 실제 YOLO proposal을 분리해 연결한다.** 변경 위치는
   `scripts/prepare_proposal_verifier_dataset.py`의 `collect_sources`,
   `candidates_from_frames`, `write_selected_crops` 및 manifest 필드다. 운영 입력은
   2단계 증거를 검증해 teacher 재질/독립 bbox를 pseudo-annotation으로 유지하고,
   고정 기존 YOLO를 실제 실행해 proposal class/confidence/bbox와 runtime-top1 crop을 만든다.
   미검출을 억지 양성으로 만들지 않는다. `materialize_operational_verifier_crops.py`의
   consensus-box crop을 runtime-YOLO crop이라고 바꿔 표시하지 않는다.
   운영 데이터는 train-only이고 AIHub validation은 기존 분리를 유지한다.
   테스트: teacher/YOLO 불일치 보존, crop 출처 혼동 거부, 미검출·다중 객체·빈 장면.

4. **V4 validator에 명시적 운영 provenance 경로를 추가한다.** 변경 위치는
   `scripts/validate_v4_background_candidates.py`의 `_validate_dataset_contract`,
   `validate_rows`, `validate_manifest`와 report 생성 부분이다. AIHub 기존 경로를 보존하고
   운영 증거를 재검증하는 versioned provenance를 추가한다. 운영 자료를
   `aihub_annotation_geometry_development_only`로 가장하지 않는다. 실제 detector replay,
   confidence `1e-6`/bbox `1e-4`, deterministic crop, background zero-intersection와
   margin `0.10`은 그대로 유지한다. 없는 label 파일을 빈 장면으로 간주하지 않는다.
   테스트: 가짜 AIHub/custom-provider 권한, 증거 교체, replay/source/crop 불일치 시 ready 없음.

5. **최종 CSV와 report의 바이트 연결을 맞춘다.** 변경/확인 위치는
   `scripts/upgrade_proposal_manifest_lineage.py`의 `_normalize_row`,
   `_load_validator_report`와 candidate builder의 `_validate_full_data_report`,
   `_build_candidate_bundle`이다. 운영의 실제 object_group/capture_session을 SHA 단위
   fallback으로 덮지 않고, 신규 필드를 보존한다. lineage upgrade로 CSV가 바뀌면
   그 **최종 CSV를 다시 실제 검증**해 정확한 SHA의 report를 받는다. 이전 report의
   SHA 필드만 고쳐 쓰지 않는다. candidate는 운영 report 종류를 검증된 증거와 함께
   명시적으로 수용하며 generic skip/allow-unsafe 옵션을 만들지 않는다.
   테스트: source CSV/report 바꿔치기, 역할·세션 누수, 다른 assembly/QX3 조합 거부.

6. **배제 범위와 라벨 지원을 CPU에서 집계한다.** quality 제외 SHA가 full-data
   manifests에 존재해야 한다는 현재 검사를 건너뛰지 않는다. accepted 운영 행만
   넘겨 배제 증거를 잃지 않도록 전체 source inventory와 제외 기록을 함께 보존한다.
   읽을 수 없는 원본처럼 crop/replay 자체가 불가능한 제외항목은 가짜 CSV 행을 만들지
   말고, source inventory에 결박된 exclusion coverage를 별도로 검증하도록 계약과
   테스트를 명시적으로 확장한다. 품질 사유·cutoff·제외 효과는 바꾸지 않는다.
   train/model_validation 각각의 클래스/상태 0·1·-1과 origin 수를 집계하고,
   부족하면 학습을 시작하지 않는다. 고정 41장, QX3 3,500장, 기타 보호 자료와
   source/crop SHA·object-group/session·근접중복을 분리한다.
   테스트: 누락된 제외 SHA, 중복 alias, 상태 한쪽 값 부재, protected/near-duplicate 누수.

7. **한 후보의 config와 authority를 실제 증거로 동결한다.** 기존
   `build_v4_candidate_training_authority.py --mode preaudit-proposal` →
   `audit_v4_near_duplicate_leakage.py`를 사용한다. AIHub를 주 데이터로 두고 운영
   expected sampling 3~8%를 실제 행 수×origin weight로 계산한다. 현재 계약은
   MobileNetV3-small 320 verifier와 세 상태 head이며 YOLO detector는 유지한다.
   full quality assembly와 QX3 증거, license/protected inventory, code snapshot,
   pretrained backbone, 새 container의 실물 inspect/device/library/마운트 증거로
   policy를 구성한다. 검토 후 policy와 builder/launcher의 두 SHA pin을 함께 고정한다.
   `scripts/nas/run_v4_candidate_training.sh`의 안전 검사를 제거하거나 과거 v3 watcher를
   우회 실행하지 않는다. 테스트 fixture의 dummy SHA·모델·license 값을 복사하지 않는다.

8. **회귀 → 소규모 실물 검증 → candidate 학습/calibration까지만 진행한다.**
   위 각 스크립트에 대응하는 기존 `tests/test_<스크립트명>.py`를 확장하고,
   실제 prepare→teacher/localizer→운영 provenance→validator→candidate preaudit 연결을 검증한다.
   CPU fixture 통과는 모델 정확도 증명이 아니다. 이후 NAS의 작은 실제 replay로
   연결을 확인하고 승인된 candidate run 한 번만 시작한다. 첫 epoch 완료 후 ETA를
   계산하고 별도 calibration 결과를 고정한다. 신규 독립 blind와 실제 하드웨어
   end-to-end 검증 전에는 production/Pi·Spring 계약을 바꾸지 않는다.

새 불변 출력 경로와 실패 원문을 보존한다. 정답·임계값·권한 조작, NAS 재부팅, `timiroom-*` 및 다른 서비스 변경은 이 계획에 포함되지 않는다.
