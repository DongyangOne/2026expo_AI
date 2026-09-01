#!/bin/sh
set -eu

# Offline v4 evidence pipeline. All paths are supplied by the caller. This
# script creates a new run directory and never manages services or deployment.

umask 077

require_env() {
  # Variable names are fixed by the loop below.  Avoid ``eval`` here so a
  # caller-supplied path can never be reinterpreted as shell syntax.
  value=$(printenv "$1" 2>/dev/null || true)
  if [ -z "$value" ]; then
    printf '%s\n' "missing required environment variable: $1" >&2
    exit 64
  fi
}

for name in \
  RUN_DIR CODE_ROOT RAW_MANIFEST DATASET_INFO DETECTOR_MODEL INFERENCE_SPEC \
  VALIDATED_MANIFEST VALIDATOR_REPORT CANDIDATE_ONNX CANDIDATE_METADATA \
  BASELINE_ONNX BASELINE_METADATA JUDGE_SPEC_1 JUDGE_SPEC_2
do
  require_env "$name"
done

PYTHON_BIN=${PYTHON_BIN:-python3}
PHASH_DISTANCE=${PHASH_DISTANCE:-4}
LINEAGE_ORIGIN=${LINEAGE_ORIGIN:-v4_independent_judge_gate}
TRUSTED_POLICY=${TRUSTED_POLICY:-$CODE_ROOT/configs/v4_candidate_judge_trusted_policy.json}

case "$PHASH_DISTANCE" in
  ''|*[!0-9]*) printf '%s\n' "PHASH_DISTANCE must be an integer in 0..7" >&2; exit 64 ;;
esac
if [ "$PHASH_DISTANCE" -gt 7 ]; then
  printf '%s\n' "PHASH_DISTANCE must be an integer in 0..7" >&2
  exit 64
fi

if ! mkdir "$RUN_DIR" 2>/dev/null; then
  printf '%s\n' "refusing to reuse immutable RUN_DIR: $RUN_DIR" >&2
  exit 73
fi

CONTROL=$RUN_DIR/control
LINEAGE=$RUN_DIR/lineage
REPLAY=$RUN_DIR/replay
VISUAL=$RUN_DIR/visual
FINAL=$RUN_DIR/final
if ! mkdir "$CONTROL"; then
  printf '%s\n' "failed to create control directory: $CONTROL" >&2
  exit 73
fi

terminal_state=0

publish_failure() {
  message=$1
  [ -d "$CONTROL" ] || return 1
  [ ! -e "$CONTROL/failed.txt" ] && [ ! -L "$CONTROL/failed.txt" ] || return 0
  temporary=$(mktemp "$CONTROL/.failed.XXXXXX") || return 1
  if ! printf '%s\n' "$message" > "$temporary" || \
     ! ln "$temporary" "$CONTROL/failed.txt" 2>/dev/null; then
    rm -f "$temporary"
    return 1
  fi
  rm -f "$temporary"
}

on_exit() {
  code=$?
  if [ "$terminal_state" -eq 0 ] && [ "$code" -ne 0 ]; then
    publish_failure "unexpected pipeline exit: code=$code" || true
  fi
}
trap on_exit 0

fail() {
  message=$1
  code=${2:-1}
  if publish_failure "$message"; then
    terminal_state=1
  fi
  printf '%s\n' "$message" >&2
  exit "$code"
}

mkdir "$LINEAGE" "$REPLAY" "$VISUAL" "$FINAL" || fail "failed to create stage directories" 73

require_file() {
  if [ ! -f "$1" ] || [ ! -s "$1" ]; then
    fail "missing or empty artifact: $1" 66
  fi
}

write_marker() {
  marker=$1
  shift
  temporary=$(mktemp "$CONTROL/.marker.XXXXXX") || fail "failed to create marker staging file"
  for artifact in "$@"; do
    require_file "$artifact"
    sha256sum "$artifact" >> "$temporary" || fail "failed to hash artifact: $artifact"
  done
  if ! ln "$temporary" "$marker" 2>/dev/null; then
    rm -f "$temporary"
    fail "refusing to overwrite stage marker: $marker" 73
  fi
  rm -f "$temporary"
}

verify_marker() {
  marker=$1
  require_file "$marker"
  sha256sum -c "$marker" >/dev/null 2>&1 || fail "stage marker hash verification failed: $marker"
}

for input in \
  "$CODE_ROOT/scripts/validate_v4_background_candidates.py" \
  "$CODE_ROOT/scripts/prepare_proposal_verifier_dataset.py" \
  "$CODE_ROOT/scripts/verifier_preprocessing_contract.py" \
  "$CODE_ROOT/scripts/upgrade_proposal_manifest_lineage.py" \
  "$CODE_ROOT/scripts/replay_v4_candidate_metrics.py" \
  "$CODE_ROOT/scripts/run_independent_visual_judges.py" \
  "$CODE_ROOT/scripts/evaluate_v4_candidate_judge.py" \
  "$RAW_MANIFEST" "$DATASET_INFO" "$DETECTOR_MODEL" "$INFERENCE_SPEC" \
  "$CANDIDATE_ONNX" "$CANDIDATE_METADATA" \
  "$BASELINE_ONNX" "$BASELINE_METADATA" "$JUDGE_SPEC_1" "$JUDGE_SPEC_2"
do
  require_file "$input"
done

write_marker "$CONTROL/00_inputs.sha256" \
  "$CODE_ROOT/scripts/validate_v4_background_candidates.py" \
  "$CODE_ROOT/scripts/prepare_proposal_verifier_dataset.py" \
  "$CODE_ROOT/scripts/verifier_preprocessing_contract.py" \
  "$CODE_ROOT/scripts/upgrade_proposal_manifest_lineage.py" \
  "$CODE_ROOT/scripts/replay_v4_candidate_metrics.py" \
  "$CODE_ROOT/scripts/run_independent_visual_judges.py" \
  "$CODE_ROOT/scripts/evaluate_v4_candidate_judge.py" \
  "$RAW_MANIFEST" "$DATASET_INFO" "$DETECTOR_MODEL" "$INFERENCE_SPEC" \
  "$CANDIDATE_ONNX" "$CANDIDATE_METADATA" \
  "$BASELINE_ONNX" "$BASELINE_METADATA" "$JUDGE_SPEC_1" "$JUDGE_SPEC_2"
verify_marker "$CONTROL/00_inputs.sha256"

# The validator requires its output pair adjacent to the source manifest. The
# caller must therefore allocate these paths inside a new generation directory.
if [ "$(dirname "$VALIDATED_MANIFEST")" != "$(dirname "$RAW_MANIFEST")" ] || \
   [ "$(dirname "$VALIDATOR_REPORT")" != "$(dirname "$RAW_MANIFEST")" ]; then
  fail "validator outputs must be adjacent to RAW_MANIFEST" 64
fi
if [ -e "$VALIDATED_MANIFEST" ] || [ -e "$VALIDATOR_REPORT" ]; then
  fail "refusing to overwrite validator outputs" 73
fi

verify_marker "$CONTROL/00_inputs.sha256"
if ! "$PYTHON_BIN" "$CODE_ROOT/scripts/validate_v4_background_candidates.py" \
  --input-manifest "$RAW_MANIFEST" \
  --dataset-info "$DATASET_INFO" \
  --detector-model "$DETECTOR_MODEL" \
  --inference-spec "$INFERENCE_SPEC" \
  --output-manifest "$VALIDATED_MANIFEST" \
  --output-report "$VALIDATOR_REPORT"
then
  fail "validator stage failed"
fi
write_marker "$CONTROL/01_validator.sha256" "$VALIDATED_MANIFEST" "$VALIDATOR_REPORT"
verify_marker "$CONTROL/01_validator.sha256"

validator_report_sha=$(sha256sum "$VALIDATOR_REPORT" | awk '{print $1}') || \
  fail "failed to hash validator report"
STRICT_CSV=$LINEAGE/strict.csv
STRICT_JSONL=$LINEAGE/strict.jsonl
LINEAGE_JSON=$LINEAGE/lineage.json
REJECTIONS_JSON=$LINEAGE/rejections.json
verify_marker "$CONTROL/00_inputs.sha256"
if ! "$PYTHON_BIN" "$CODE_ROOT/scripts/upgrade_proposal_manifest_lineage.py" \
  --input "$VALIDATED_MANIFEST" \
  --validator-report "$VALIDATOR_REPORT" \
  --validator-report-sha256 "$validator_report_sha" \
  --output-csv "$STRICT_CSV" \
  --output-jsonl "$STRICT_JSONL" \
  --lineage-json "$LINEAGE_JSON" \
  --rejections-json "$REJECTIONS_JSON" \
  --quarantine-validation-near-phash-distance "$PHASH_DISTANCE" \
  --origin "$LINEAGE_ORIGIN"
then
  fail "lineage stage failed"
fi
write_marker "$CONTROL/02_lineage.sha256" "$STRICT_CSV" "$STRICT_JSONL" "$LINEAGE_JSON" "$REJECTIONS_JSON"
verify_marker "$CONTROL/02_lineage.sha256"

CANDIDATE_PREDICTIONS=$REPLAY/candidate_predictions.jsonl
CANDIDATE_ATTESTATION=$REPLAY/candidate_attestation.json
verify_marker "$CONTROL/00_inputs.sha256"
if ! "$PYTHON_BIN" "$CODE_ROOT/scripts/replay_v4_candidate_metrics.py" \
  --manifest "$STRICT_CSV" \
  --verifier-onnx "$CANDIDATE_ONNX" \
  --verifier-metadata "$CANDIDATE_METADATA" \
  --inference-spec "$INFERENCE_SPEC" \
  --output-jsonl "$CANDIDATE_PREDICTIONS" \
  --output-attestation "$CANDIDATE_ATTESTATION"
then
  fail "candidate replay stage failed"
fi

BASELINE_PREDICTIONS=$REPLAY/baseline_predictions.jsonl
BASELINE_ATTESTATION=$REPLAY/baseline_attestation.json
verify_marker "$CONTROL/00_inputs.sha256"
if ! "$PYTHON_BIN" "$CODE_ROOT/scripts/replay_v4_candidate_metrics.py" \
  --manifest "$STRICT_CSV" \
  --verifier-onnx "$BASELINE_ONNX" \
  --verifier-metadata "$BASELINE_METADATA" \
  --inference-spec "$INFERENCE_SPEC" \
  --output-jsonl "$BASELINE_PREDICTIONS" \
  --output-attestation "$BASELINE_ATTESTATION"
then
  fail "baseline replay stage failed"
fi
write_marker "$CONTROL/03_replays.sha256" \
  "$CANDIDATE_PREDICTIONS" "$CANDIDATE_ATTESTATION" \
  "$BASELINE_PREDICTIONS" "$BASELINE_ATTESTATION"
verify_marker "$CONTROL/03_replays.sha256"

VISUAL_INPUT=$VISUAL/model_validation_background.csv
if ! "$PYTHON_BIN" - "$STRICT_CSV" "$VISUAL_INPUT" <<'PY'
import csv
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
if target.exists() or target.is_symlink():
    raise FileExistsError(target)
with source.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    if not reader.fieldnames:
        raise ValueError("strict manifest has no header")
    rows = [
        row for row in reader
        if row.get("role") == "model_validation"
        and row.get("split") == "validation"
        and row.get("material") == "9"
        and row.get("category") == "background"
        and row.get("crop_object_count") == "0"
    ]
if not rows:
    raise ValueError("strict manifest has no validation background rows")
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
os.link(temporary, target)
temporary.unlink()
PY
then
  fail "visual input projection failed"
fi

VISUAL_EVIDENCE=$VISUAL/evidence.jsonl
VISUAL_REPORT=$VISUAL/report.json
verify_marker "$CONTROL/00_inputs.sha256"
if ! "$PYTHON_BIN" "$CODE_ROOT/scripts/run_independent_visual_judges.py" \
  --input-manifest "$VISUAL_INPUT" \
  --judge-spec "$JUDGE_SPEC_1" \
  --judge-spec "$JUDGE_SPEC_2" \
  --output-jsonl "$VISUAL_EVIDENCE" \
  --output-report "$VISUAL_REPORT"
then
  fail "visual judge stage failed"
fi
write_marker "$CONTROL/04_visual.sha256" "$VISUAL_INPUT" "$VISUAL_EVIDENCE" "$VISUAL_REPORT"
verify_marker "$CONTROL/04_visual.sha256"

policy_pin=$(
  "$PYTHON_BIN" - "$CODE_ROOT/scripts/evaluate_v4_candidate_judge.py" <<'PY'
import ast
import sys
from pathlib import Path

tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "APPROVED_TRUSTED_POLICY_SHA256":
                value = ast.literal_eval(node.value)
                if not isinstance(value, str):
                    raise TypeError("policy pin must be a string")
                print(value)
                raise SystemExit(0)
raise RuntimeError("policy pin constant was not found")
PY
) || fail "failed to inspect trusted policy pin"

write_marker "$CONTROL/05_evidence_prepared.sha256" \
  "$STRICT_CSV" "$LINEAGE_JSON" \
  "$CANDIDATE_PREDICTIONS" "$CANDIDATE_ATTESTATION" \
  "$BASELINE_PREDICTIONS" "$BASELINE_ATTESTATION" \
  "$VISUAL_INPUT" "$VISUAL_EVIDENCE" "$VISUAL_REPORT"
verify_marker "$CONTROL/00_inputs.sha256"

if [ "$policy_pin" = "UNCONFIGURED" ]; then
  write_marker "$CONTROL/awaiting_policy_pin.sha256" \
    "$CONTROL/00_inputs.sha256" \
    "$CONTROL/01_validator.sha256" \
    "$CONTROL/02_lineage.sha256" \
    "$CONTROL/03_replays.sha256" \
    "$CONTROL/04_visual.sha256" \
    "$CONTROL/05_evidence_prepared.sha256"
  printf '%s\n' "evidence prepared; trusted policy pin is UNCONFIGURED; final gate not run"
  terminal_state=1
  exit 78
fi

case "$policy_pin" in
  *[!0-9a-f]*|'') fail "trusted policy pin is not a lowercase SHA-256" 65 ;;
esac
if [ "${#policy_pin}" -ne 64 ]; then
  fail "trusted policy pin is not a lowercase SHA-256" 65
fi
require_file "$TRUSTED_POLICY"
policy_actual=$(sha256sum "$TRUSTED_POLICY" | awk '{print $1}') || \
  fail "failed to hash trusted policy" 65
if [ "$policy_actual" != "$policy_pin" ]; then
  fail "trusted policy bytes differ from code pin" 65
fi

verify_marker "$CONTROL/01_validator.sha256"
verify_marker "$CONTROL/00_inputs.sha256"
verify_marker "$CONTROL/02_lineage.sha256"
verify_marker "$CONTROL/03_replays.sha256"
verify_marker "$CONTROL/04_visual.sha256"
verify_marker "$CONTROL/05_evidence_prepared.sha256"

if ! "$PYTHON_BIN" "$CODE_ROOT/scripts/evaluate_v4_candidate_judge.py" \
  --metadata "$CANDIDATE_METADATA" \
  --manifest "$STRICT_CSV" \
  --candidate-onnx "$CANDIDATE_ONNX" \
  --inference-spec "$INFERENCE_SPEC" \
  --replay-predictions "$CANDIDATE_PREDICTIONS" \
  --replay-attestation "$CANDIDATE_ATTESTATION" \
  --baseline-metadata "$BASELINE_METADATA" \
  --baseline-onnx "$BASELINE_ONNX" \
  --baseline-replay-predictions "$BASELINE_PREDICTIONS" \
  --baseline-replay-attestation "$BASELINE_ATTESTATION" \
  --trusted-policy "$TRUSTED_POLICY" \
  --visual-judge-report "$VISUAL_REPORT" \
  --visual-judge-evidence "$VISUAL_EVIDENCE" \
  --output-dir "$FINAL"
then
  fail "final offline judge gate rejected the candidate"
fi

FINAL_REPORT=$FINAL/v4_candidate_judge_report.json
EVALUATOR_READY=$FINAL/v4_candidate_judge_ready.txt
SEALED_EVALUATOR_READY=$FINAL/evaluator_offline_ready.txt
require_file "$FINAL_REPORT"
require_file "$EVALUATOR_READY"
if [ -e "$SEALED_EVALUATOR_READY" ] || [ -L "$SEALED_EVALUATOR_READY" ]; then
  fail "refusing to overwrite sealed evaluator marker" 73
fi
mv "$EVALUATOR_READY" "$SEALED_EVALUATOR_READY" || fail "failed to seal evaluator marker"

write_marker "$CONTROL/06_offline_gate.sha256" \
  "$CONTROL/00_inputs.sha256" \
  "$CONTROL/01_validator.sha256" \
  "$CONTROL/02_lineage.sha256" \
  "$CONTROL/03_replays.sha256" \
  "$CONTROL/04_visual.sha256" \
  "$CONTROL/05_evidence_prepared.sha256" \
  "$FINAL_REPORT" "$SEALED_EVALUATOR_READY"
verify_marker "$CONTROL/06_offline_gate.sha256"

WRAPPER_READY=$CONTROL/offline_gate_ready.json
if ! "$PYTHON_BIN" - \
  "$WRAPPER_READY" "$CONTROL/failed.txt" "$CONTROL/06_offline_gate.sha256" \
  "$FINAL_REPORT" "$SEALED_EVALUATOR_READY" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

ready, failed, chain, report, evaluator_ready = map(Path, sys.argv[1:6])
if ready.exists() or ready.is_symlink():
    raise FileExistsError(ready)
if failed.exists() or failed.is_symlink():
    raise RuntimeError("failure marker exists")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

payload = {
    "schema_version": 1,
    "status": "offline_judge_gate_passed",
    "artifact_role": "wrapper_offline_gate_ready_not_hardware_or_deployment_authority",
    "production_deployment_authorized": False,
    "requires_independent_blind_hardware_evidence": True,
    "bindings": {
        "evidence_chain_sha256": sha(chain),
        "final_report_sha256": sha(report),
        "sealed_evaluator_ready_sha256": sha(evaluator_ready),
    },
}
temporary = ready.with_name(f".{ready.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
os.link(temporary, ready)
temporary.unlink()
PY
then
  fail "failed to publish wrapper offline ready marker"
fi

# Publication above is the last fallible operation. Consumers must require this
# wrapper marker, a valid 06 chain, and absence of failed.txt.
terminal_state=1
exit 0
