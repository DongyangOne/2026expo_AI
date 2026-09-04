#!/bin/sh
set -eu

# Generate a new raw v4 proposal dataset. This stage has no validation,
# training, judge, blind-test, deployment, or service-management authority.

umask 077

require_env() {
  value=$(printenv "$1" 2>/dev/null || true)
  if [ -z "$value" ]; then
    printf '%s\n' "missing required environment variable: $1" >&2
    exit 64
  fi
}

for name in GEN_DIR CODE_ROOT MODEL_PATH DATA_PATH DATASET_DIR; do
  require_env "$name"
done

PYTHON_BIN=${PYTHON_BIN:-python3}
DEVICE=${DEVICE:-0}
SEED=${SEED:-20260901}
GENERATOR=$CODE_ROOT/scripts/prepare_proposal_verifier_dataset.py
PREPROCESSING=$CODE_ROOT/scripts/verifier_preprocessing_contract.py
WRAPPER=$CODE_ROOT/scripts/nas/run_v4_reproducible_generation.sh

case "$SEED" in
  ''|*[!0-9]*) printf '%s\n' "SEED must be a non-negative integer" >&2; exit 64 ;;
esac

if ! mkdir "$GEN_DIR" 2>/dev/null; then
  printf '%s\n' "refusing to reuse immutable GEN_DIR: $GEN_DIR" >&2
  exit 73
fi
CONTROL=$GEN_DIR/control
RAW_DIR=$GEN_DIR/raw
if ! mkdir "$CONTROL"; then
  printf '%s\n' "failed to create control directory: $CONTROL" >&2
  exit 73
fi

terminal_state=0

publish_failure() {
  message=$1
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
    publish_failure "unexpected generation exit: code=$code" || true
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

require_file() {
  if [ ! -f "$1" ] || [ ! -s "$1" ]; then
    fail "missing or empty input: $1" 66
  fi
}

require_file "$GENERATOR"
require_file "$PREPROCESSING"
require_file "$WRAPPER"
require_file "$MODEL_PATH"
require_file "$DATA_PATH"
AUDITED_AIHUB_REPORT=${AUDITED_AIHUB_REPORT:-}
AUDITED_AIHUB_REPORT_SHA256=${AUDITED_AIHUB_REPORT_SHA256:-}
AUDITED_AIHUB_COHORT=${AUDITED_AIHUB_COHORT:-}
AUDITED_AIHUB_DIAGNOSTIC=${AUDITED_AIHUB_DIAGNOSTIC:-0}
case "$AUDITED_AIHUB_DIAGNOSTIC" in 0|1) ;; *) fail "invalid audited AIHub diagnostic flag" 64;; esac
set --
if [ -n "$AUDITED_AIHUB_REPORT$AUDITED_AIHUB_REPORT_SHA256$AUDITED_AIHUB_COHORT" ]; then
  [ -n "$AUDITED_AIHUB_REPORT" ] && [ -n "$AUDITED_AIHUB_REPORT_SHA256" ] && \
    [ -n "$AUDITED_AIHUB_COHORT" ] || fail "audited AIHub report, SHA, and cohort must be supplied together" 64
  [ "${#AUDITED_AIHUB_REPORT_SHA256}" -eq 64 ] || fail "invalid audited AIHub report SHA" 64
  case "$AUDITED_AIHUB_REPORT_SHA256" in *[!0-9a-f]*) fail "invalid audited AIHub report SHA" 64;; esac
  require_file "$AUDITED_AIHUB_REPORT"
  require_file "$AUDITED_AIHUB_COHORT"
  for helper in audited_aihub_snapshot.py audit_aihub_original_annotations.py materialize_audited_aihub_sources.py; do
    require_file "$CODE_ROOT/scripts/$helper"
  done
  set -- --audited-aihub-report "$AUDITED_AIHUB_REPORT" \
    --audited-aihub-report-sha256 "$AUDITED_AIHUB_REPORT_SHA256" \
    --audited-aihub-cohort "$AUDITED_AIHUB_COHORT" \
    --aihub-origin aihub_original_annotation_v1
  if [ "$AUDITED_AIHUB_DIAGNOSTIC" = 1 ]; then set -- "$@" --audited-aihub-diagnostic; fi
elif [ "$AUDITED_AIHUB_DIAGNOSTIC" = 1 ]; then
  fail "audited AIHub diagnostic requires report, SHA, and cohort" 64
fi
if [ ! -d "$DATASET_DIR" ]; then
  fail "missing dataset directory: $DATASET_DIR" 66
fi

inventory_tree() {
  tree_root=$1
  output=$2
  "$PYTHON_BIN" - "$tree_root" "$output" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
if output.exists() or output.is_symlink():
    raise FileExistsError(output)
rows = []
for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.relative_to(root).as_posix()):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    rows.append({
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })
payload = {
    "schema_version": 1,
    "root": root.as_posix(),
    "file_count": len(rows),
    "files": rows,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
os.link(temporary, output)
temporary.unlink()
PY
}

verify_inventory() {
  tree_root=$1
  inventory=$2
  "$PYTHON_BIN" - "$tree_root" "$inventory" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
inventory = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if inventory.get("root") != root.as_posix():
    raise ValueError("inventory root mismatch")
rows = inventory.get("files")
if not isinstance(rows, list) or inventory.get("file_count") != len(rows):
    raise ValueError("inventory count mismatch")
seen = set()
for row in rows:
    relative = row.get("path")
    if not isinstance(relative, str) or relative in seen:
        raise ValueError("inventory path is invalid or duplicated")
    seen.add(relative)
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("inventory path escapes root") from error
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != row.get("size") or digest.hexdigest() != row.get("sha256"):
        raise ValueError(f"inventory artifact changed: {relative}")
actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file()
}
if actual != seen:
    raise ValueError("inventory file set changed")
PY
}

inventory_generation_sources() {
  output=$1
  shift
  "$PYTHON_BIN" - "$CODE_ROOT" "$DATA_PATH" "$DATASET_DIR" "$output" "$@" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

code_root = Path(sys.argv[1]).resolve()
data_path = Path(sys.argv[2]).resolve()
dataset_dir = Path(sys.argv[3]).resolve()
output = Path(sys.argv[4])
generate = len(sys.argv) > 5 and sys.argv[5] == "--generate"
generator_arguments = sys.argv[6:] if generate else []
if output.exists() or output.is_symlink():
    raise FileExistsError(output)

def stable_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        finished = os.fstat(handle.fileno())
    after = path.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if not identity(before) == identity(opened) == identity(finished) == identity(after):
        raise RuntimeError(f"input changed while hashing: {path}")
    return digest.hexdigest()

generator = code_root / "scripts/prepare_proposal_verifier_dataset.py"
preprocessing = code_root / "scripts/verifier_preprocessing_contract.py"
wrapper = code_root / "scripts/nas/run_v4_reproducible_generation.sh"
code_paths = [generator, preprocessing, wrapper]
audited_report = os.environ.get("AUDITED_AIHUB_REPORT", "")
if audited_report:
    code_paths.extend(code_root / "scripts" / name for name in (
        "audited_aihub_snapshot.py", "audit_aihub_original_annotations.py",
        "materialize_audited_aihub_sources.py",
    ))
# Only small code is read before CUDA; model and source inventories follow it.
code_pins = {path: stable_sha256(path) for path in code_paths}

def verify_code() -> None:
    for path, expected in code_pins.items():
        if stable_sha256(path) != expected:
            raise RuntimeError(f"generation code changed: {path}")

sys.path.insert(0, str(code_root))
from scripts import prepare_proposal_verifier_dataset as prepare  # noqa: E402
if Path(prepare.__file__).resolve() != generator:
    raise RuntimeError("generator import path mismatch")
verify_code()
_label_path = prepare._label_path
resolve_split_images = prepare.resolve_split_images
# Retain the allocation in this frame through inventory, marker and main().
cuda_guard = None
if generate:
    device = os.environ.get("DEVICE", "0")
    cuda_guard = prepare.eager_initialize_cuda_context(device)
    if device.strip().lower() in {"0", "cuda", "cuda:0"} and cuda_guard is None:
        raise RuntimeError("CUDA initialization did not retain a context guard")
    if audited_report and stable_sha256(Path(audited_report)) != os.environ["AUDITED_AIHUB_REPORT_SHA256"]:
        raise RuntimeError("audited AIHub report SHA mismatch")

def artifact(path: Path, *, kind: str, split: str) -> dict[str, object]:
    resolved = path.resolve(strict=False)
    value: dict[str, object] = {
        "kind": kind,
        "split": split,
        "path": resolved.as_posix(),
        "exists": resolved.is_file(),
        "size": None,
        "sha256": None,
    }
    if value["exists"]:
        before = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = resolved.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"input changed while being inventoried: {resolved}")
        value["size"] = after.st_size
        value["sha256"] = digest.hexdigest()
    return value

split_images = resolve_split_images(data_path, dataset_dir)
rows = []
seen = set()
for split in ("training", "validation"):
    for source in split_images.get(split, []):
        source = source.resolve(strict=False)
        source_key = (split, "source", source.as_posix())
        if source_key not in seen:
            seen.add(source_key)
            rows.append(artifact(source, kind="source", split=split))
        try:
            label = _label_path(source)
        except ValueError:
            rows.append({
                "kind": "unresolved_label_path",
                "split": split,
                "path": source.as_posix(),
                "exists": False,
                "size": None,
                "sha256": None,
            })
            continue
        label_key = (split, "label", label.resolve(strict=False).as_posix())
        if label_key not in seen:
            seen.add(label_key)
            rows.append(artifact(label, kind="label", split=split))
rows.sort(key=lambda row: (str(row["split"]), str(row["kind"]), str(row["path"])))
payload = {
    "schema_version": 1,
    "contract": "resolved_yolo_train_val_sources_and_label_sidecars_sha256.v1",
    "data_path": data_path.as_posix(),
    "dataset_dir": dataset_dir.as_posix(),
    "artifact_count": len(rows),
    "artifacts": rows,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
os.link(temporary, output)
temporary.unlink()
verify_code()

if generate:
    input_marker = output.parent / "inputs.sha256"
    # Preserve the six base pins and the five optional audited-input pins.
    input_paths = [Path(os.environ["MODEL_PATH"]), data_path, generator,
                   preprocessing, wrapper, output]
    if audited_report:
        input_paths.extend([Path(audited_report), Path(os.environ["AUDITED_AIHUB_COHORT"]),
                            *code_paths[3:]])
    input_pins = [(path, stable_sha256(path)) for path in input_paths]
    if audited_report and input_pins[6][1] != os.environ["AUDITED_AIHUB_REPORT_SHA256"]:
        raise RuntimeError("audited AIHub report changed before input marker")
    verify_code()
    marker_bytes = bytearray()
    for path, digest in input_pins:
        filename = os.fsencode(path.as_posix())
        escaped = any(char in filename for char in (b"\\", b"\n", b"\r"))
        filename = filename.replace(b"\\", b"\\\\").replace(b"\n", b"\\n").replace(b"\r", b"\\r")
        marker_bytes.extend((b"\\" if escaped else b"") + digest.encode("ascii") + b"  " + filename + b"\n")
    staging = input_marker.with_name(f".{input_marker.name}.{os.getpid()}.tmp")
    with staging.open("xb") as handle:
        handle.write(marker_bytes)
    os.link(staging, input_marker)
    staging.unlink()
    for path, digest in input_pins:
        if stable_sha256(path) != digest:
            raise RuntimeError(f"generation input changed before main: {path}")
    # Release large CPU bookkeeping, not the live CUDA allocation.
    del rows, seen, payload, split_images
    sys.argv = [str(generator), *generator_arguments]
    prepare.main()
    verify_code()
PY
}

DATASET_INPUT_INVENTORY=$CONTROL/dataset_input_inventory.json
INPUT_MARKER=$CONTROL/inputs.sha256
if ! inventory_generation_sources "$DATASET_INPUT_INVENTORY" --generate \
  --model "$MODEL_PATH" \
  --data "$DATA_PATH" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$RAW_DIR" \
  --device "$DEVICE" \
  --batch 1 \
  --imgsz 640 \
  --conf 0.10 \
  --nms-iou 0.70 \
  --positive-iou 0.50 \
  --negative-iou 0.10 \
  --crop-size 320 \
  --padding 0.08 \
  --jpeg-quality 92 \
  --proposal-selection runtime-top1 \
  --background-policy strict-zero-intersection \
  --background-gt-margin 0.10 \
  --max-per-class 10000 \
  --val-max-per-class 2000 \
  --max-background 10000 \
  --val-max-background 2000 \
  --seed "$SEED" \
  --min-free-gb 300 \
  --max-output-gb 30 "$@"
then
  fail "raw proposal generation failed"
fi

MANIFEST=$RAW_DIR/manifest.csv
DATASET_INFO=$RAW_DIR/dataset_info.json
require_file "$MANIFEST"
require_file "$DATASET_INFO"

if ! "$PYTHON_BIN" - \
  "$DATASET_INFO" "$MANIFEST" "$MODEL_PATH" "$DATA_PATH" "$DATASET_DIR" "$SEED" \
  "$AUDITED_AIHUB_REPORT" "$AUDITED_AIHUB_REPORT_SHA256" "$AUDITED_AIHUB_COHORT" "$AUDITED_AIHUB_DIAGNOSTIC" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

info_path, manifest_path, model_path, data_path, dataset_dir = map(Path, sys.argv[1:6])
seed = int(sys.argv[6])
info = json.loads(info_path.read_text(encoding="utf-8"))
audited_report, audited_sha, audited_cohort, audited_diagnostic = sys.argv[7:11]
binding = info.get("audited_aihub_snapshot")
if audited_report:
    expected_binding = {
        "report_path": Path(audited_report).resolve().as_posix(),
        "report_sha256": audited_sha,
        "cohort_path": Path(audited_cohort).resolve().as_posix(),
        "cohort_sha256": hashlib.sha256(Path(audited_cohort).read_bytes()).hexdigest(),
        "require_full_cohort": audited_diagnostic != "1",
    }
    if binding != expected_binding or type(binding.get("require_full_cohort")) is not bool:
        raise ValueError("audited AIHub snapshot binding differs from launch inputs")
elif binding is not None:
    raise ValueError("unexpected audited AIHub snapshot binding")
expected_paths = {
    "model": model_path.resolve(),
    "data": data_path.resolve(),
    "dataset_dir": dataset_dir.resolve(),
    "manifest": manifest_path.resolve(),
}
for field, expected in expected_paths.items():
    if Path(info.get(field, "")).resolve() != expected:
        raise ValueError(f"dataset_info {field} path mismatch")
exact = {
    ("inference", "batch"): 1,
    ("inference", "imgsz"): 640,
    ("inference", "conf"): 0.10,
    ("inference", "nms_iou"): 0.70,
    ("assignment", "positive_iou_inclusive"): 0.50,
    ("assignment", "negative_iou_inclusive"): 0.10,
    ("assignment", "ambiguous_iou_skipped"): True,
    ("proposal_policy", "selection_mode"): "runtime-top1",
    ("proposal_policy", "background_policy"): "strict-zero-intersection",
    ("proposal_policy", "background_gt_margin"): 0.10,
    ("crop", "size"): 320,
    ("crop", "padding"): 0.08,
    ("crop", "jpeg_quality"): 92,
    ("selection", "max_per_class"): 10000,
    ("selection", "val_max_per_class"): 2000,
    ("selection", "max_background"): 10000,
    ("selection", "val_max_background"): 2000,
    ("selection", "seed"): seed,
    ("storage_guards", "min_free_gb"): 300.0,
    ("storage_guards", "max_output_gb"): 30.0,
}
for (section, field), expected in exact.items():
    actual = info.get(section, {}).get(field)
    if actual != expected:
        raise ValueError(f"dataset_info {section}.{field} mismatch: {actual!r}")
written = info.get("written_crops")
if not isinstance(written, int) or isinstance(written, bool) or written <= 0:
    raise ValueError("dataset_info written_crops must be positive")
with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = sum(1 for _ in csv.DictReader(handle))
if rows <= 0 or rows != written:
    raise ValueError("manifest row count differs from positive written_crops")
PY
then
  fail "generated dataset contract verification failed"
fi

# Rebuild and compare the complete input inventory after generation. A changed
# source tree invalidates this generation even when individual output hashes exist.
DATASET_INPUT_INVENTORY_END=$CONTROL/dataset_input_inventory_end.json
sha256sum -c "$INPUT_MARKER" >/dev/null 2>&1 || fail "model, data, or code changed during generation"
inventory_generation_sources "$DATASET_INPUT_INVENTORY_END" || fail "failed to re-inventory resolved dataset inputs"
if ! cmp -s "$DATASET_INPUT_INVENTORY" "$DATASET_INPUT_INVENTORY_END"; then
  fail "resolved dataset inputs changed during generation"
fi
sha256sum -c "$INPUT_MARKER" >/dev/null 2>&1 || fail "model, data, or code changed during generation"

OUTPUT_INVENTORY=$CONTROL/raw_output_inventory.json
inventory_tree "$RAW_DIR" "$OUTPUT_INVENTORY" || fail "failed to inventory generated outputs"
verify_inventory "$RAW_DIR" "$OUTPUT_INVENTORY" || fail "generated output inventory verification failed"
OUTPUT_MARKER=$CONTROL/outputs.sha256
temporary=$(mktemp "$CONTROL/.outputs.XXXXXX") || fail "failed to create output marker staging file"
sha256sum "$MANIFEST" "$DATASET_INFO" "$OUTPUT_INVENTORY" > "$temporary" || fail "failed to hash generation outputs"
if ! ln "$temporary" "$OUTPUT_MARKER" 2>/dev/null; then
  rm -f "$temporary"
  fail "refusing to overwrite output marker" 73
fi
rm -f "$temporary"
sha256sum -c "$OUTPUT_MARKER" >/dev/null 2>&1 || fail "output marker verification failed"
verify_inventory "$RAW_DIR" "$OUTPUT_INVENTORY" || fail "generated outputs changed before ready marker"

READY=$CONTROL/raw_generation_ready.json
if ! "$PYTHON_BIN" - \
  "$READY" "$INPUT_MARKER" "$OUTPUT_MARKER" "$MANIFEST" "$DATASET_INFO" "$SEED" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

ready, inputs, outputs, manifest, info = map(Path, sys.argv[1:6])
seed = int(sys.argv[6])
if ready.exists() or ready.is_symlink():
    raise FileExistsError(ready)
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "artifact_role": "raw_v4_reproducible_generation_not_validation_or_promotion_authority",
    "status": "raw_generation_ready",
    "batch": 1,
    "seed": seed,
    "validator_authority": False,
    "judge_authority": False,
    "training_authority": False,
    "blind_test_authority": False,
    "production_deployment_authorized": False,
    "bindings": {
        "input_marker_sha256": sha(inputs),
        "output_marker_sha256": sha(outputs),
        "manifest_sha256": sha(manifest),
        "dataset_info_sha256": sha(info),
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
  fail "failed to publish raw generation ready marker"
fi

# Ready publication is the last fallible operation. Consumers must also require
# the absence of failed.txt; this marker grants no downstream authority.
terminal_state=1
exit 0
