#!/bin/sh
set -eu

# Re-run the authoritative v4 validator twice against one immutable batch=1
# generation. This is a reproducibility diagnostic only: it cannot authorize
# lineage, training, blind evaluation, model promotion, or deployment.

umask 077

require_env() {
  value=$(printenv "$1" 2>/dev/null || true)
  if [ -z "$value" ]; then
    printf '%s\n' "missing required environment variable: $1" >&2
    exit 64
  fi
}

for name in \
  VALIDATION_DIR CODE_ROOT GEN_DIR PILOT_INPUT_DIR DETECTOR_MODEL INFERENCE_SPEC
do
  require_env "$name"
done

PYTHON_BIN=${PYTHON_BIN:-python3}
VALIDATOR=$CODE_ROOT/scripts/validate_v4_background_candidates.py
WRAPPER=$CODE_ROOT/scripts/nas/run_v4_repro_pilot_validation.sh
GEN_WRAPPER=$CODE_ROOT/scripts/nas/run_v4_reproducible_generation.sh
PILOT_BUILDER=$CODE_ROOT/scripts/build_v4_repro_pilot_inputs.py
PROPOSAL_PREPARE=$CODE_ROOT/scripts/prepare_proposal_verifier_dataset.py
PREPROCESSING=$CODE_ROOT/scripts/verifier_preprocessing_contract.py
PILOT_READY=$PILOT_INPUT_DIR/input_ready.json
PILOT_INPUTS=$PILOT_INPUT_DIR/inputs.sha256
PILOT_INVENTORY=$PILOT_INPUT_DIR/selection_inventory.json
PILOT_YAML=$PILOT_INPUT_DIR/pilot_dataset.yaml
PILOT_TRAIN=$PILOT_INPUT_DIR/train_pilot.txt
PILOT_VALIDATION=$PILOT_INPUT_DIR/validation_pilot.txt
GEN_CONTROL=$GEN_DIR/control
RAW_DIR=$GEN_DIR/raw
RAW_MANIFEST=$RAW_DIR/manifest.csv
DATASET_INFO=$RAW_DIR/dataset_info.json
GEN_INPUTS=$GEN_CONTROL/inputs.sha256
GEN_OUTPUTS=$GEN_CONTROL/outputs.sha256
GEN_DATASET_INPUT_INVENTORY=$GEN_CONTROL/dataset_input_inventory.json
RAW_INVENTORY=$GEN_CONTROL/raw_output_inventory.json
GEN_READY=$GEN_CONTROL/raw_generation_ready.json

if ! mkdir "$VALIDATION_DIR" 2>/dev/null; then
  printf '%s\n' "refusing to reuse immutable VALIDATION_DIR: $VALIDATION_DIR" >&2
  exit 73
fi
CONTROL=$VALIDATION_DIR/control
WORK_A=$VALIDATION_DIR/validator-a
WORK_B=$VALIDATION_DIR/validator-b
if ! mkdir "$CONTROL"; then
  printf '%s\n' "failed to create validation control directory: $CONTROL" >&2
  exit 73
fi

READY=$CONTROL/diagnostic_ready.json
terminal_state=0

remove_ready() {
  if [ -e "$READY" ] || [ -L "$READY" ]; then
    rm -f "$READY" 2>/dev/null || true
  fi
}

publish_failure() {
  message=$1
  [ -d "$CONTROL" ] || return 1
  if [ ! -e "$CONTROL/failed.txt" ] && [ ! -L "$CONTROL/failed.txt" ]; then
    temporary=$(mktemp "$CONTROL/.failed.XXXXXX") || return 1
    if ! printf '%s\n' "$message" > "$temporary" || \
       ! ln "$temporary" "$CONTROL/failed.txt" 2>/dev/null; then
      rm -f "$temporary"
      return 1
    fi
    rm -f "$temporary"
  fi
  # A failure marker always wins over a prematurely/racing ready marker.
  remove_ready
}

on_exit() {
  code=$?
  if [ "$terminal_state" -eq 0 ] && [ "$code" -ne 0 ]; then
    publish_failure "unexpected reproducibility pilot exit: code=$code" || true
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
  sha256sum -c "$marker" >/dev/null 2>&1 || fail "stage marker verification failed: $marker"
}

for artifact in \
  "$VALIDATOR" "$WRAPPER" "$GEN_WRAPPER" "$PILOT_BUILDER" "$PROPOSAL_PREPARE" \
  "$PREPROCESSING" "$DETECTOR_MODEL" "$INFERENCE_SPEC" \
  "$PILOT_READY" "$PILOT_INPUTS" "$PILOT_INVENTORY" "$PILOT_YAML" \
  "$PILOT_TRAIN" "$PILOT_VALIDATION" \
  "$RAW_MANIFEST" "$DATASET_INFO" "$GEN_INPUTS" "$GEN_OUTPUTS" \
  "$GEN_DATASET_INPUT_INVENTORY" \
  "$RAW_INVENTORY" "$GEN_READY"
do
  require_file "$artifact"
done
if [ ! -d "$RAW_DIR/training" ] || [ ! -d "$RAW_DIR/validation" ]; then
  fail "raw generation is missing training or validation directory" 66
fi
if [ -e "$PILOT_INPUT_DIR/failed.txt" ] || [ -L "$PILOT_INPUT_DIR/failed.txt" ]; then
  fail "pilot input failure marker exists" 65
fi

if ! "$PYTHON_BIN" - "$VALIDATION_DIR" "$GEN_DIR" "$PILOT_INPUT_DIR" <<'PY'
import sys
from pathlib import Path

validation = Path(sys.argv[1]).resolve()
generation = Path(sys.argv[2]).resolve()
pilot = Path(sys.argv[3]).resolve(strict=True)
try:
    validation.relative_to(generation)
except ValueError:
    pass
else:
    raise ValueError("VALIDATION_DIR must not be inside GEN_DIR")
try:
    generation.relative_to(validation)
except ValueError:
    pass
else:
    raise ValueError("GEN_DIR must not be inside VALIDATION_DIR")
try:
    validation.relative_to(pilot)
except ValueError:
    pass
else:
    raise ValueError("VALIDATION_DIR must not be inside PILOT_INPUT_DIR")
try:
    pilot.relative_to(validation)
except ValueError:
    pass
else:
    raise ValueError("PILOT_INPUT_DIR must not be inside VALIDATION_DIR")
PY
then
  fail "generation and validation directories are not independent" 64
fi

verify_raw_inventory() {
  "$PYTHON_BIN" - "$RAW_DIR" "$RAW_INVENTORY" <<'PY'
import hashlib
import csv
import json
import os
import sys
from pathlib import Path

raw_arg = Path(sys.argv[1])
inventory_path = Path(sys.argv[2])
if raw_arg.is_symlink():
    raise ValueError("raw directory must not be a symlink")
root = raw_arg.resolve(strict=True)
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
if inventory.get("root") != root.as_posix():
    raise ValueError("raw inventory root mismatch")
rows = inventory.get("files")
if not isinstance(rows, list) or inventory.get("file_count") != len(rows):
    raise ValueError("raw inventory count mismatch")
seen = set()
for row in rows:
    relative = row.get("path")
    if not isinstance(relative, str) or not relative or relative in seen:
        raise ValueError("raw inventory path is invalid or duplicated")
    seen.add(relative)
    lexical = raw_arg / relative
    if lexical.is_symlink():
        raise ValueError(f"raw artifact became a symlink: {relative}")
    path = lexical.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("raw inventory path escapes root") from error
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        opened_after = os.fstat(handle.fileno())
    after = path.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if (
        identity(before) != identity(opened_before)
        or identity(opened_before) != identity(opened_after)
        or identity(opened_after) != identity(after)
    ):
        raise RuntimeError(f"raw artifact changed while hashing: {relative}")
    if after.st_size != row.get("size") or digest.hexdigest() != row.get("sha256"):
        raise ValueError(f"raw inventory artifact changed: {relative}")
actual = set()
for path in root.rglob("*"):
    if path.is_symlink():
        raise ValueError(f"raw tree contains a symlink: {path}")
    if path.is_file():
        actual.add(path.relative_to(root).as_posix())
if actual != seen:
    raise ValueError("raw inventory file set changed")
PY
}

verify_pilot_contract() {
  if [ -e "$PILOT_INPUT_DIR/failed.txt" ] || [ -L "$PILOT_INPUT_DIR/failed.txt" ]; then
    fail "pilot input failure marker exists" 65
  fi
  for artifact in \
    "$PILOT_READY" "$PILOT_INPUTS" "$PILOT_INVENTORY" "$PILOT_YAML" \
    "$PILOT_TRAIN" "$PILOT_VALIDATION"
  do
    require_file "$artifact"
  done
  if ! (cd "$PILOT_INPUT_DIR" && sha256sum -c inputs.sha256 >/dev/null 2>&1); then
    fail "pilot input marker verification failed" 65
  fi
  if ! "$PYTHON_BIN" - \
    "$PILOT_READY" "$PILOT_INPUTS" "$PILOT_INVENTORY" "$PILOT_YAML" \
    "$PILOT_TRAIN" "$PILOT_VALIDATION" "$PILOT_BUILDER" \
    "$PROPOSAL_PREPARE" "$PREPROCESSING" <<'PY'
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

(
    ready_path, marker_path, inventory_path, yaml_path, train_path, val_path,
    builder_path, proposal_path, preprocessing_path,
) = map(Path, sys.argv[1:10])
role = (
    "v4_batch1_reproducibility_pilot_inputs_diagnostic_only_"
    "not_training_blind_or_deployment_authority"
)
contract = (
    "v4_repro_pilot_inputs."
    "gt_stratified_historical_background_probe_blake2b.v2"
)
materials = (
    "can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass",
    "battery", "fluorescent",
)
strata = (*materials, "background")
expected_counts = {
    **{f"training/{name}": 250 for name in strata},
    **{f"validation/{name}": 100 for name in strata},
}
expected_shortages = {name: 0 for name in expected_counts}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

ready = json.loads(ready_path.read_text(encoding="utf-8"))
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
for value, description, status in (
    (ready, "pilot ready", "pilot_inputs_ready"),
    (inventory, "pilot inventory", "selection_complete_not_replay_validated"),
):
    if value.get("schema_version") != 1:
        raise ValueError(f"{description} schema mismatch")
    if value.get("artifact_role") != role:
        raise ValueError(f"{description} role mismatch")
    if value.get("selection_contract") != contract:
        raise ValueError(f"{description} selection contract mismatch")
    if value.get("status") != status:
        raise ValueError(f"{description} status mismatch")

if ready.get("selected_sources") != 3500:
    raise ValueError("pilot ready selected source count must be 3500")
if ready.get("selected_counts") != expected_counts:
    raise ValueError("pilot ready selected counts do not meet exact quotas")
if ready.get("full_quota_met") is not True:
    raise ValueError("pilot ready does not attest full quota")
for field in (
    "validator_authority", "training_authorized", "blind_test_authorized",
    "production_deployment_authorized",
):
    if ready.get(field) is not False:
        raise ValueError(f"pilot ready unexpectedly grants {field}")

if inventory.get("quota_per_stratum") != {"training": 250, "validation": 100}:
    raise ValueError("pilot inventory quota contract mismatch")
if inventory.get("classes") != list(materials) or inventory.get("strata") != list(strata):
    raise ValueError("pilot inventory strata mismatch")
source_contract = inventory.get("source_contract")
if not isinstance(source_contract, dict):
    raise ValueError("pilot source contract is missing")
for field in (
    "explicit_label_file_required",
    "background_prefers_current_explicit_empty_label",
    "historical_background_probe_requires_current_single_object_label",
    "historical_background_category_is_selection_only",
    "historical_background_category_is_not_ground_truth",
    "current_batch1_replay_decides_emitted_category",
    "material_requires_exactly_one_valid_yolo_label",
    "multi_object_excluded",
    "cross_split_content_duplicates_quarantined",
    "same_split_conflicting_ground_truth_quarantined",
):
    if source_contract.get(field) is not True:
        raise ValueError(f"pilot source contract does not require {field}")
if inventory.get("selected_counts") != expected_counts:
    raise ValueError("pilot inventory selected counts do not meet exact quotas")
if inventory.get("quota_shortages") != expected_shortages:
    raise ValueError("pilot inventory has a quota shortage")
if inventory.get("full_quota_met") is not True:
    raise ValueError("pilot inventory does not attest full quota")
for field in (
    "selected_current_gt_counts", "selected_cohort_counts",
    "background_quota_composition",
):
    if ready.get(field) != inventory.get(field):
        raise ValueError(f"pilot ready and inventory differ for {field}")
authority = inventory.get("authority")
if not isinstance(authority, dict):
    raise ValueError("pilot inventory authority is missing")
for field in (
    "raw_generation_authorized", "validator_authority", "training_authorized",
    "blind_test_authorized", "production_deployment_authorized",
):
    if authority.get(field) is not False:
        raise ValueError(f"pilot inventory unexpectedly grants {field}")

artifact_paths = {
    "pilot_dataset.yaml": yaml_path,
    "selection_inventory.json": inventory_path,
    "train_pilot.txt": train_path,
    "validation_pilot.txt": val_path,
}
artifact_hashes = {name: sha(path) for name, path in artifact_paths.items()}
declared = {}
for line in marker_path.read_text(encoding="ascii").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
    if not match or match.group(2) in declared:
        raise ValueError("pilot input marker line is invalid or duplicated")
    declared[match.group(2)] = match.group(1)
if declared != artifact_hashes:
    raise ValueError("pilot input marker does not bind the exact four artifacts")
bindings = ready.get("bindings")
if not isinstance(bindings, dict):
    raise ValueError("pilot ready bindings are missing")
if bindings.get("inputs_marker_sha256") != sha(marker_path):
    raise ValueError("pilot ready input marker binding mismatch")
if bindings.get("artifacts") != artifact_hashes:
    raise ValueError("pilot ready artifact bindings mismatch")
inventory_bindings = inventory.get("bindings")
if not isinstance(inventory_bindings, dict):
    raise ValueError("pilot inventory bindings are missing")
if bindings.get("resolved_universe_sha256") != inventory_bindings.get("resolved_universe_sha256"):
    raise ValueError("pilot ready universe binding mismatch")
if Path(str(inventory_bindings.get("selector_path", ""))).resolve() != builder_path.resolve():
    raise ValueError("pilot inventory selector path mismatch")
if inventory_bindings.get("selector_sha256") != sha(builder_path):
    raise ValueError("pilot inventory selector hash mismatch")
if Path(str(inventory_bindings.get("proposal_generator_path", ""))).resolve() != proposal_path.resolve():
    raise ValueError("pilot inventory proposal generator path mismatch")
if inventory_bindings.get("proposal_generator_sha256") != sha(proposal_path):
    raise ValueError("pilot inventory proposal generator hash mismatch")
# The current builder imports the proposal generator, which imports this crop
# contract. The preprocessing bytes have no inventory field yet, so the outer
# 00 marker is their immutable binding for this diagnostic run.
if not preprocessing_path.is_file() or preprocessing_path.stat().st_size <= 0:
    raise ValueError("pilot preprocessing dependency is missing or empty")

selected = inventory.get("selected_sources")
if not isinstance(selected, list) or len(selected) != 3500:
    raise ValueError("pilot selection inventory must contain 3500 rows")
actual_counts = Counter()
actual_current_gt_counts = Counter()
actual_cohort_counts = Counter()
actual_background_composition = {
    "training": Counter(
        {"current_explicit_empty_label": 0, "historical_background_probe": 0}
    ),
    "validation": Counter(
        {"current_explicit_empty_label": 0, "historical_background_probe": 0}
    ),
}
listed = {"training": [], "validation": []}
seen_paths = set()
seen_source_hashes = set()
for row in selected:
    if not isinstance(row, dict):
        raise ValueError("pilot selected source row must be an object")
    split = row.get("split")
    stratum = row.get("stratum")
    selection_stratum = row.get("selection_stratum")
    current_gt_stratum = row.get("current_gt_stratum")
    cohort = row.get("selection_cohort")
    path_text = row.get("path")
    if (
        split not in listed
        or stratum not in strata
        or selection_stratum != stratum
        or current_gt_stratum not in strata
    ):
        raise ValueError("pilot selected source split or stratum is invalid")
    if not isinstance(path_text, str) or not Path(path_text).is_absolute():
        raise ValueError("pilot selected source path must be absolute")
    if path_text in seen_paths:
        raise ValueError("pilot selected source path is duplicated")
    seen_paths.add(path_text)
    source_sha = row.get("source_sha256")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("pilot selected source SHA is invalid")
    if source_sha in seen_source_hashes:
        raise ValueError("pilot selected source SHA is duplicated")
    seen_source_hashes.add(source_sha)
    categories = row.get("historical_categories_selection_only")
    explicit = row.get("explicit_empty_label")
    probe = row.get("historical_background_probe_selection_only")
    if not isinstance(categories, list) or any(
        category not in strata for category in categories
    ):
        raise ValueError("pilot selected historical categories are invalid")
    if probe is True:
        if (
            stratum != "background"
            or current_gt_stratum == "background"
            or explicit is not False
            or cohort != "historical_background_probe"
            or "background" not in categories
            or not isinstance(row.get("gt_class_id"), int)
            or not isinstance(row.get("gt_xywhn"), list)
            or len(row["gt_xywhn"]) != 4
        ):
            raise ValueError("pilot historical background probe semantics are invalid")
        if row.get("selection_reason") not in {
            "historical_background_probe_blake2", "drift_anchor_priority",
        }:
            raise ValueError("pilot historical background probe reason is invalid")
        actual_background_composition[split]["historical_background_probe"] += 1
    else:
        if (
            probe is not False
            or cohort != "current_yolo_ground_truth"
            or current_gt_stratum != stratum
            or explicit is not (current_gt_stratum == "background")
        ):
            raise ValueError("pilot current GT selection semantics are invalid")
        if stratum == "background":
            if row.get("selection_reason") != "current_explicit_empty_label":
                raise ValueError("pilot explicit background reason is invalid")
            actual_background_composition[split]["current_explicit_empty_label"] += 1
    actual_counts[f"{split}/{stratum}"] += 1
    actual_current_gt_counts[f"{split}/{current_gt_stratum}"] += 1
    actual_cohort_counts[f"{split}/{cohort}"] += 1
    listed[split].append(path_text)
if dict(actual_counts) != expected_counts:
    raise ValueError("pilot selected rows do not meet exact quotas")
if dict(actual_current_gt_counts) != inventory.get("selected_current_gt_counts"):
    raise ValueError("pilot selected current GT counts mismatch")
if dict(actual_cohort_counts) != inventory.get("selected_cohort_counts"):
    raise ValueError("pilot selected cohort counts mismatch")
historical = inventory.get("historical_selection_evidence")
if not isinstance(historical, dict):
    raise ValueError("pilot historical selection evidence is missing")
if historical.get("ground_truth_authority") is not False:
    raise ValueError("pilot historical evidence grants ground-truth authority")
if historical.get("replay_validation_authority") is not False:
    raise ValueError("pilot historical evidence grants replay authority")
if historical.get("background_category_authority") is not False:
    raise ValueError("pilot historical evidence grants background authority")
eligible_empty = historical.get("eligible_current_explicit_empty_counts")
eligible_probe = historical.get("eligible_historical_background_probe_counts")
declared_composition = inventory.get("background_quota_composition")
if not all(isinstance(value, dict) for value in (eligible_empty, eligible_probe, declared_composition)):
    raise ValueError("pilot background selection evidence is missing")
expected_composition = {}
for split, quota in (("training", 250), ("validation", 100)):
    available_empty = eligible_empty.get(split)
    available_probe = eligible_probe.get(split)
    if (
        not isinstance(available_empty, int)
        or isinstance(available_empty, bool)
        or available_empty < 0
        or not isinstance(available_probe, int)
        or isinstance(available_probe, bool)
        or available_probe < 0
    ):
        raise ValueError("pilot eligible background counts are invalid")
    selected_empty = min(quota, available_empty)
    selected_probe = quota - selected_empty
    if available_probe < selected_probe:
        raise ValueError("pilot historical background probe supply is below quota")
    expected_composition[split] = {
        "current_explicit_empty_label": selected_empty,
        "historical_background_probe": selected_probe,
        "total": quota,
    }
    actual_background_composition[split]["total"] = sum(
        actual_background_composition[split].values()
    )
if declared_composition != expected_composition:
    raise ValueError("pilot declared background quota composition mismatch")
if {
    split: dict(counts) for split, counts in actual_background_composition.items()
} != expected_composition:
    raise ValueError("pilot selected background quota composition mismatch")

def list_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line for line in lines) or len(lines) != len(set(lines)):
        raise ValueError(f"pilot list is empty, blank, or duplicated: {path}")
    return lines

if list_lines(train_path) != sorted(listed["training"]):
    raise ValueError("training pilot list differs from selected inventory rows")
if list_lines(val_path) != sorted(listed["validation"]):
    raise ValueError("validation pilot list differs from selected inventory rows")
dataset = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
if not isinstance(dataset, dict):
    raise ValueError("pilot dataset YAML must be an object")
if Path(str(dataset.get("train", ""))).resolve() != train_path.resolve():
    raise ValueError("pilot YAML training list mismatch")
if Path(str(dataset.get("val", ""))).resolve() != val_path.resolve():
    raise ValueError("pilot YAML validation list mismatch")
if Path(str(dataset.get("path", ""))).resolve() != Path(
    str(inventory_bindings.get("dataset_dir", ""))
).resolve():
    raise ValueError("pilot YAML dataset root differs from selection inventory")
names = dataset.get("names")
if names != {index: name for index, name in enumerate(materials)}:
    raise ValueError("pilot YAML material names mismatch")
PY
  then
    fail "pilot input contract verification failed" 65
  fi
  if [ -e "$PILOT_INPUT_DIR/failed.txt" ] || [ -L "$PILOT_INPUT_DIR/failed.txt" ]; then
    fail "pilot input failure marker appeared during verification" 65
  fi
}

COHORT_BINDING=$CONTROL/cohort_binding.json

seal_or_verify_cohort() {
  mode=$1
  if ! "$PYTHON_BIN" - \
    "$mode" "$COHORT_BINDING" "$PILOT_READY" "$PILOT_INVENTORY" \
    "$PILOT_YAML" "$GEN_DATASET_INPUT_INVENTORY" "$RAW_INVENTORY" \
    "$RAW_MANIFEST" <<'PY'
import base64
import binascii
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

mode = sys.argv[1]
(
    output, ready_path, selection_path, pilot_yaml, generation_inventory_path,
    raw_inventory_path, raw_manifest,
) = map(Path, sys.argv[2:9])
if mode not in {"create", "verify"}:
    raise ValueError("invalid cohort binding mode")
sha_re = re.compile(r"^[0-9a-f]{64}$")

def stable_artifact(path: Path, *, description: str) -> tuple[int, str]:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        opened_after = os.fstat(handle.fileno())
    after = resolved.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if (
        identity(before) != identity(opened_before)
        or identity(opened_before) != identity(opened_after)
        or identity(opened_after) != identity(after)
    ):
        raise RuntimeError(f"{description} changed while hashing: {resolved}")
    return after.st_size, digest.hexdigest()

def stable_content(path: Path, *, description: str) -> tuple[bytes, int, str]:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    with resolved.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        content = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = resolved.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if (
        identity(before) != identity(opened_before)
        or identity(opened_before) != identity(opened_after)
        or identity(opened_after) != identity(after)
    ):
        raise RuntimeError(f"{description} changed while being read: {resolved}")
    return content, after.st_size, hashlib.sha256(content).hexdigest()

def sha(path: Path) -> str:
    return stable_artifact(path, description="bound evidence")[1]

ready = json.loads(ready_path.read_text(encoding="utf-8"))
selection = json.loads(selection_path.read_text(encoding="utf-8"))
if ready.get("historical_selection_only") is not True:
    raise ValueError("pilot ready is not bound to historical selection evidence")
historical = selection.get("historical_selection_evidence")
if not isinstance(historical, dict) or historical.get("used_for_selection_only") is not True:
    raise ValueError("pilot inventory lacks historical selection evidence")
old_manifest = historical.get("old_manifest")
drift_report = historical.get("drift_report")
if not isinstance(old_manifest, dict) or not isinstance(drift_report, dict):
    raise ValueError("pilot historical manifest or drift report binding is missing")
old_path = Path(str(old_manifest.get("path", "")))
drift_path = Path(str(drift_report.get("path", "")))
if not old_path.is_absolute() or not drift_path.is_absolute():
    raise ValueError("pilot historical evidence paths must be absolute")
old_content, old_size, old_sha = stable_content(
    old_path, description="historical manifest"
)
drift_content, drift_size, drift_sha = stable_content(
    drift_path, description="historical drift report"
)
if (
    old_manifest.get("sha256") != old_sha
    or not isinstance(old_manifest.get("rows"), int)
    or isinstance(old_manifest.get("rows"), bool)
    or old_manifest["rows"] <= 0
):
    raise ValueError("historical manifest binding mismatch")
if (
    drift_report.get("sha256") != drift_sha
    or not isinstance(drift_report.get("anchor_source_ids"), int)
    or isinstance(drift_report.get("anchor_source_ids"), bool)
    or drift_report["anchor_source_ids"] <= 0
):
    raise ValueError("historical drift report binding mismatch")
try:
    historical_lines = old_content.decode("utf-8-sig").splitlines()
except UnicodeError as error:
    raise ValueError("historical manifest is not valid UTF-8") from error
historical_reader = csv.DictReader(historical_lines)
historical_required = {"source_id", "split", "category"}
if not historical_reader.fieldnames or not historical_required.issubset(
    historical_reader.fieldnames
):
    raise ValueError("historical manifest lacks probe membership fields")
historical_membership = set()
historical_categories_by_source_split = {}
historical_row_count = 0
for line, row in enumerate(historical_reader, start=2):
    historical_row_count += 1
    source_id = str(row.get("source_id", "")).strip().lower()
    split = str(row.get("split", "")).strip()
    category = str(row.get("category", "")).strip().lower()
    if not sha_re.fullmatch(source_id):
        raise ValueError(f"historical manifest source_id is invalid at row {line}")
    if split not in {"training", "validation"}:
        raise ValueError(f"historical manifest split is invalid at row {line}")
    if category not in {
        "can", "pet", "paper", "plastic", "styrofoam", "vinyl",
        "glass", "battery", "fluorescent", "background",
    }:
        raise ValueError(f"historical manifest category is invalid at row {line}")
    historical_membership.add((source_id, split, category))
    historical_categories_by_source_split.setdefault((source_id, split), set()).add(
        category
    )
if historical_row_count != old_manifest["rows"]:
    raise ValueError("historical manifest row count differs from bound evidence")
if sha(old_path) != old_sha:
    raise RuntimeError("historical manifest changed while probe membership was parsed")

try:
    drift_value = json.loads(drift_content.decode("utf-8"))
except (UnicodeError, json.JSONDecodeError) as error:
    raise ValueError("historical drift report is invalid JSON") from error
if not isinstance(drift_value, dict):
    raise ValueError("historical drift report must contain an object")

def source_ids_from_examples(value, *, location):
    if value is None:
        return set()
    if not isinstance(value, list):
        raise ValueError(f"drift report {location} must contain a list")
    found = set()
    for index, example in enumerate(value):
        if not isinstance(example, dict):
            raise ValueError(f"drift report {location}[{index}] must be an object")
        raw = example.get("source_id")
        if raw is None:
            continue
        normalized = str(raw).strip().lower()
        if not sha_re.fullmatch(normalized):
            raise ValueError(
                f"drift report {location}[{index}] has invalid source_id"
            )
        found.add(normalized)
    return found

replay = drift_value.get("replay")
if not isinstance(replay, dict):
    raise ValueError("historical drift report must contain a replay object")
drift_anchor_source_ids = set()
hard = replay.get("hard_semantic_mismatch_examples")
if hard is not None:
    if not isinstance(hard, dict):
        raise ValueError(
            "drift report replay.hard_semantic_mismatch_examples must be an object"
        )
    for name, examples in hard.items():
        drift_anchor_source_ids.update(
            source_ids_from_examples(
                examples,
                location=f"replay.hard_semantic_mismatch_examples.{name}",
            )
        )
for section_name in (
    "confidence_abs_drift", "bbox_max_abs_drift",
    "declared_vs_replayed_crop_bounds",
):
    section = replay.get(section_name)
    if section is None:
        continue
    if not isinstance(section, dict):
        raise ValueError(f"drift report replay.{section_name} must be an object")
    drift_anchor_source_ids.update(
        source_ids_from_examples(
            section.get("max_examples"),
            location=f"replay.{section_name}.max_examples",
        )
    )
thresholds = replay.get("fixed_threshold_diagnostics")
if thresholds is not None:
    if not isinstance(thresholds, dict):
        raise ValueError(
            "drift report replay.fixed_threshold_diagnostics must be an object"
        )
    for name, examples in thresholds.items():
        if str(name).endswith("_nearest_examples"):
            drift_anchor_source_ids.update(
                source_ids_from_examples(
                    examples,
                    location=f"replay.fixed_threshold_diagnostics.{name}",
                )
            )
if len(drift_anchor_source_ids) != drift_report["anchor_source_ids"]:
    raise ValueError("historical drift anchor count differs from bound evidence")
historical_source_ids = {
    source_id for source_id, _, _ in historical_membership
}
if not drift_anchor_source_ids.issubset(historical_source_ids):
    raise ValueError("historical drift anchor is absent from historical manifest")
if sha(drift_path) != drift_sha:
    raise RuntimeError("historical drift report changed while anchors were parsed")

selected = selection.get("selected_sources")
if not isinstance(selected, list) or len(selected) != 3500:
    raise ValueError("pilot selection must contain exactly 3500 rows")
selected_by_path = {}
expected_inventory = {}
selected_source_hashes = set()
selected_anchor_paths = set()
selected_background_probe_paths = set()
priority_anchors = 0
for row in selected:
    if not isinstance(row, dict):
        raise ValueError("pilot selected row must be an object")
    split = row.get("split")
    if split not in {"training", "validation"}:
        raise ValueError("pilot selected row split is invalid")
    source = Path(str(row.get("path", "")))
    label = Path(str(row.get("label_path", "")))
    if not source.is_absolute() or not label.is_absolute():
        raise ValueError("pilot selected source and label paths must be absolute")
    source = source.resolve(strict=True)
    label = label.resolve(strict=True)
    source_key = source.as_posix()
    if source_key in selected_by_path:
        raise ValueError("pilot selected source path is duplicated")
    source_size, source_sha = stable_artifact(source, description="selected source")
    label_content, label_size, label_sha = stable_content(
        label, description="selected label"
    )
    if row.get("source_sha256") != source_sha or row.get("label_sha256") != label_sha:
        raise ValueError("selected source or label current bytes differ from inventory")
    if not sha_re.fullmatch(source_sha) or source_sha in selected_source_hashes:
        raise ValueError("pilot selected source SHA is invalid or duplicated")
    selected_source_hashes.add(source_sha)
    actual_historical_categories = sorted(
        historical_categories_by_source_split.get((source_sha, split), set())
    )
    if row.get("historical_categories_selection_only") != actual_historical_categories:
        raise ValueError("selected historical categories differ from bound manifest")
    expected_anchor = bool(actual_historical_categories) and (
        source_sha in drift_anchor_source_ids
    )
    if row.get("drift_anchor") is not expected_anchor:
        raise ValueError("selected drift anchor differs from bound drift allowlist")
    if row.get("selection_reason") == "drift_anchor_priority" and not expected_anchor:
        raise ValueError("pilot drift anchor priority is not allowlisted")
    try:
        label_text = label_content.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("selected current YOLO label is not UTF-8") from error
    label_lines = [line.strip() for line in label_text.splitlines() if line.strip()]
    parsed_class_id = None
    parsed_xywhn = None
    if label_lines:
        if len(label_lines) != 1:
            raise ValueError("selected current YOLO label is not single-object")
        parts = label_lines[0].split()
        if len(parts) != 5:
            raise ValueError("selected current YOLO label has invalid column count")
        try:
            raw_class, cx, cy, width, height = (float(value) for value in parts)
        except ValueError as error:
            raise ValueError("selected current YOLO label is non-numeric") from error
        if not math.isfinite(raw_class):
            raise ValueError("selected current YOLO label class is non-finite")
        parsed_class_id = int(raw_class)
        if raw_class != parsed_class_id or not 0 <= parsed_class_id < 9:
            raise ValueError("selected current YOLO label class is invalid")
        parsed_xywhn = [cx, cy, width, height]
        if not all(math.isfinite(value) for value in parsed_xywhn):
            raise ValueError("selected current YOLO label bbox is non-finite")
        if not (
            0 <= cx <= 1
            and 0 <= cy <= 1
            and 0 < width <= 1
            and 0 < height <= 1
            and cx - width / 2 >= -0.01
            and cx + width / 2 <= 1.01
            and cy - height / 2 >= -0.01
            and cy + height / 2 <= 1.01
        ):
            raise ValueError("selected current YOLO label bbox is invalid")
    parsed_gt_stratum = (
        "background"
        if parsed_class_id is None
        else (
            "can", "pet", "paper", "plastic", "styrofoam", "vinyl",
            "glass", "battery", "fluorescent",
        )[parsed_class_id]
    )
    if (
        row.get("current_gt_stratum") != parsed_gt_stratum
        or row.get("gt_class_id") != parsed_class_id
        or row.get("gt_xywhn") != parsed_xywhn
        or row.get("explicit_empty_label") is not (parsed_class_id is None)
    ):
        raise ValueError("selected current YOLO label semantics differ from inventory")
    selected_by_path[source_key] = {
        "split": split,
        "source_sha256": source_sha,
        "label_sha256": label_sha,
        "selection_cohort": row.get("selection_cohort"),
    }
    for kind, path, size, digest in (
        ("source", source, source_size, source_sha),
        ("label", label, label_size, label_sha),
    ):
        key = (split, kind, path.as_posix())
        if key in expected_inventory:
            raise ValueError("pilot generation inventory key is duplicated")
        expected_inventory[key] = {"exists": True, "size": size, "sha256": digest}
    if row.get("drift_anchor") is True:
        selected_anchor_paths.add(source_key)
        if row.get("selection_reason") == "drift_anchor_priority":
            priority_anchors += 1
    if row.get("historical_background_probe_selection_only") is True:
        if (
            row.get("selection_stratum") != "background"
            or row.get("stratum") != "background"
            or row.get("current_gt_stratum") == "background"
            or row.get("selection_cohort") != "historical_background_probe"
            or row.get("explicit_empty_label") is not False
            or (source_sha, split, "background") not in historical_membership
        ):
            raise ValueError("pilot background probe lacks bound historical membership")
        selected_background_probe_paths.add(source_key)

if not selected_anchor_paths or priority_anchors <= 0:
    raise ValueError("pilot selection must include historical drift anchors and priority anchors")
if historical.get("anchors_selected") != len(selected_anchor_paths):
    raise ValueError("selected drift anchor count differs from historical evidence")
if historical.get("anchors_priority_selected") != priority_anchors:
    raise ValueError("priority drift anchor count differs from historical evidence")

generation_inventory = json.loads(generation_inventory_path.read_text(encoding="utf-8"))
if generation_inventory.get("schema_version") != 1 or generation_inventory.get("contract") != "resolved_yolo_train_val_sources_and_label_sidecars_sha256.v1":
    raise ValueError("generation dataset input inventory contract mismatch")
selection_bindings = selection.get("bindings")
if not isinstance(selection_bindings, dict):
    raise ValueError("pilot selection bindings are missing")
if Path(str(generation_inventory.get("data_path", ""))).resolve() != pilot_yaml.resolve():
    raise ValueError("generation dataset input inventory pilot YAML mismatch")
if Path(str(generation_inventory.get("dataset_dir", ""))).resolve() != Path(
    str(selection_bindings.get("dataset_dir", ""))
).resolve():
    raise ValueError("generation dataset input inventory dataset root mismatch")
artifacts = generation_inventory.get("artifacts")
if not isinstance(artifacts, list) or generation_inventory.get("artifact_count") != len(artifacts):
    raise ValueError("generation dataset input inventory count mismatch")
declared_inventory = {}
for row in artifacts:
    if not isinstance(row, dict):
        raise ValueError("generation dataset input artifact must be an object")
    path = Path(str(row.get("path", ""))).resolve()
    key = (row.get("split"), row.get("kind"), path.as_posix())
    if key in declared_inventory:
        raise ValueError("generation dataset input inventory key is duplicated")
    declared_inventory[key] = {
        "exists": row.get("exists"),
        "size": row.get("size"),
        "sha256": row.get("sha256"),
    }
if declared_inventory != expected_inventory:
    raise ValueError("generation dataset input inventory differs from selected current bytes")

raw_root = raw_manifest.parent.resolve(strict=True)
raw_inventory = json.loads(raw_inventory_path.read_text(encoding="utf-8"))
if raw_inventory.get("root") != raw_root.as_posix():
    raise ValueError("raw output inventory root mismatch during cohort binding")
raw_inventory_files = raw_inventory.get("files")
if not isinstance(raw_inventory_files, list):
    raise ValueError("raw output inventory files are missing")
raw_inventory_by_path = {
    row.get("path"): row for row in raw_inventory_files if isinstance(row, dict)
}
if len(raw_inventory_by_path) != len(raw_inventory_files):
    raise ValueError("raw output inventory contains invalid or duplicate paths")

with raw_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    required = {
        "filepath", "crop_bytes", "source_path_b64", "source_id", "split",
        "category",
    }
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("raw manifest lacks source cohort binding fields")
    raw_rows = list(reader)
emitted_paths = set()
emitted_source_ids = set()
emitted_split_counts = Counter()
emitted_background_probe_paths = set()
background_probe_replay_categories = Counter()
emitted_crop_paths = set()
for line, row in enumerate(raw_rows, start=2):
    crop_value = str(row.get("filepath", "")).strip()
    if not crop_value:
        raise ValueError(f"raw manifest crop path is empty at row {line}")
    crop_lexical = Path(crop_value)
    if not crop_lexical.is_absolute():
        crop_lexical = raw_root / crop_lexical
    try:
        crop = crop_lexical.resolve(strict=True)
        crop_relative = crop.relative_to(raw_root).as_posix()
    except (OSError, ValueError) as error:
        raise ValueError("raw manifest crop escapes the inventoried raw directory") from error
    crop_inventory = raw_inventory_by_path.get(crop_relative)
    if not isinstance(crop_inventory, dict):
        raise ValueError("raw manifest crop is absent from the raw output inventory")
    inventory_size = crop_inventory.get("size")
    inventory_sha = crop_inventory.get("sha256")
    if (
        not isinstance(inventory_size, int)
        or inventory_size <= 0
        or not isinstance(inventory_sha, str)
        or not sha_re.fullmatch(inventory_sha)
    ):
        raise ValueError("raw manifest crop inventory binding is invalid")
    try:
        crop_bytes = int(str(row.get("crop_bytes", "")).strip())
    except ValueError as error:
        raise ValueError("raw manifest crop_bytes is invalid") from error
    crop_size, crop_sha = stable_artifact(crop, description="raw manifest crop")
    if (
        crop_bytes <= 0
        or crop_bytes != crop_size
        or crop_size != inventory_size
        or crop_sha != inventory_sha
    ):
        raise ValueError("raw manifest crop current bytes differ from the raw output inventory")
    if crop_relative in emitted_crop_paths:
        raise ValueError("raw manifest crop path is duplicated")
    emitted_crop_paths.add(crop_relative)
    try:
        decoded = base64.urlsafe_b64decode(str(row["source_path_b64"]).encode("ascii"))
        source = Path(os.fsdecode(decoded)).resolve(strict=True)
    except (UnicodeError, ValueError, binascii.Error, OSError) as error:
        raise ValueError(f"raw manifest source path is invalid at row {line}") from error
    source_key = source.as_posix()
    selected_row = selected_by_path.get(source_key)
    if selected_row is None:
        raise ValueError("raw manifest contains a source outside the selected pilot cohort")
    source_id = str(row.get("source_id", "")).strip().lower()
    if source_id != selected_row["source_sha256"]:
        raise ValueError("raw manifest source_id differs from selected current bytes")
    if row.get("split") != selected_row["split"]:
        raise ValueError("raw manifest split differs from selected pilot row")
    if source_key in emitted_paths or source_id in emitted_source_ids:
        raise ValueError("raw manifest contains a duplicate emitted source")
    emitted_paths.add(source_key)
    emitted_source_ids.add(source_id)
    emitted_split_counts[row["split"]] += 1
    category = str(row.get("category", "")).strip().lower()
    if category not in {
        "can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass",
        "battery", "fluorescent", "background",
    }:
        raise ValueError("raw manifest category is invalid")
    if selected_row["selection_cohort"] == "historical_background_probe":
        emitted_background_probe_paths.add(source_key)
        background_probe_replay_categories[f"{row['split']}/{category}"] += 1
expected_raw_files = {"manifest.csv", "dataset_info.json", *emitted_crop_paths}
if set(raw_inventory_by_path) != expected_raw_files:
    raise ValueError("raw output inventory is not exactly manifest, info, and emitted crops")
selected_count = len(selected_by_path)
emitted_count = len(emitted_paths)
minimum_emitted = math.ceil(selected_count * 0.99)
if emitted_count < minimum_emitted:
    raise ValueError(
        f"raw manifest selected-source coverage is below 99 percent: {emitted_count}/{selected_count}"
    )
if not selected_anchor_paths.issubset(emitted_paths):
    raise ValueError("raw manifest omits one or more selected drift anchors")
if not selected_background_probe_paths.issubset(emitted_background_probe_paths):
    raise ValueError("raw manifest omits one or more selected background probes")

payload = {
    "schema_version": 1,
    "status": "pilot_cohort_current_bytes_and_raw_membership_bound",
    "artifact_role": "v4_reproducibility_pilot_cohort_diagnostic_only",
    "selected_sources": selected_count,
    "emitted_unique_sources": emitted_count,
    "minimum_emitted_sources": minimum_emitted,
    "selected_source_coverage": emitted_count / selected_count,
    "emitted_split_counts": dict(sorted(emitted_split_counts.items())),
    "selected_drift_anchors": len(selected_anchor_paths),
    "emitted_drift_anchors": len(selected_anchor_paths & emitted_paths),
    "priority_drift_anchors": priority_anchors,
    "selected_background_probes": len(selected_background_probe_paths),
    "emitted_background_probes": len(emitted_background_probe_paths),
    "background_probe_replay_categories": dict(
        sorted(background_probe_replay_categories.items())
    ),
    "historical_background_probe_is_selection_only": True,
    "historical_background_probe_replay_result_assumed": False,
    "historical_selection_only": True,
    "lineage_execution_authorized": False,
    "training_authority": False,
    "blind_test_authority": False,
    "production_deployment_authorized": False,
    "bindings": {
        "pilot_ready_sha256": sha(ready_path),
        "selection_inventory_sha256": sha(selection_path),
        "generation_dataset_input_inventory_sha256": sha(generation_inventory_path),
        "raw_manifest_sha256": sha(raw_manifest),
        "raw_output_inventory_sha256": sha(raw_inventory_path),
        "historical_manifest_sha256": old_sha,
        "historical_manifest_size": old_size,
        "historical_drift_report_sha256": drift_sha,
        "historical_drift_report_size": drift_size,
    },
}
content = (
    json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    + "\n"
).encode("utf-8")
if mode == "create":
    if output.exists() or output.is_symlink():
        raise FileExistsError("cohort binding already exists")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
    os.link(temporary, output)
    temporary.unlink()
else:
    if output.read_bytes() != content:
        raise ValueError("pilot cohort or current source bytes changed after sealing")
PY
  then
    fail "pilot cohort provenance verification failed" 65
  fi
}

verify_generation_contract() {
  verify_pilot_contract
  if [ -e "$GEN_CONTROL/failed.txt" ] || [ -L "$GEN_CONTROL/failed.txt" ]; then
    fail "generation failure marker exists" 65
  fi
  for artifact in \
    "$RAW_MANIFEST" "$DATASET_INFO" "$GEN_INPUTS" "$GEN_OUTPUTS" \
    "$GEN_DATASET_INPUT_INVENTORY" "$RAW_INVENTORY" "$GEN_READY"
  do
    require_file "$artifact"
  done
  sha256sum -c "$GEN_INPUTS" >/dev/null 2>&1 || fail "generation input marker verification failed" 65
  sha256sum -c "$GEN_OUTPUTS" >/dev/null 2>&1 || fail "generation output marker verification failed" 65
  if ! "$PYTHON_BIN" - \
    "$GEN_READY" "$GEN_INPUTS" "$GEN_OUTPUTS" "$RAW_MANIFEST" \
    "$DATASET_INFO" "$RAW_INVENTORY" "$PILOT_YAML" "$DETECTOR_MODEL" \
    "$PROPOSAL_PREPARE" "$PREPROCESSING" "$GEN_WRAPPER" \
    "$GEN_DATASET_INPUT_INVENTORY" <<'PY'
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

(
    ready_path, inputs, outputs, manifest, info, inventory, pilot_yaml,
    detector, proposal_prepare, preprocessing, generation_wrapper,
    dataset_input_inventory,
) = map(Path, sys.argv[1:13])
ready = json.loads(ready_path.read_text(encoding="utf-8"))
if ready.get("schema_version") != 1:
    raise ValueError("generation ready schema mismatch")
if ready.get("status") != "raw_generation_ready":
    raise ValueError("generation ready status mismatch")
if ready.get("artifact_role") != "raw_v4_reproducible_generation_not_validation_or_promotion_authority":
    raise ValueError("generation ready role mismatch")
if ready.get("batch") != 1:
    raise ValueError("generation was not batch=1")
for field in (
    "validator_authority", "judge_authority", "training_authority",
    "blind_test_authority", "production_deployment_authorized",
):
    if ready.get(field) is not False:
        raise ValueError(f"generation ready unexpectedly grants {field}")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected_bindings = {
    "input_marker_sha256": sha(inputs),
    "output_marker_sha256": sha(outputs),
    "manifest_sha256": sha(manifest),
    "dataset_info_sha256": sha(info),
}
if ready.get("bindings") != expected_bindings:
    raise ValueError("generation ready bindings mismatch")

dataset_info = json.loads(info.read_text(encoding="utf-8"))
if Path(str(dataset_info.get("data", ""))).resolve() != pilot_yaml.resolve():
    raise ValueError("generation dataset_info is not bound to the pilot YAML")
if Path(str(dataset_info.get("model", ""))).resolve() != detector.resolve():
    raise ValueError("generation dataset_info detector model mismatch")
if Path(str(dataset_info.get("manifest", ""))).resolve() != manifest.resolve():
    raise ValueError("generation dataset_info manifest mismatch")
expected_contract = {
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
    ("storage_guards", "min_free_gb"): 300.0,
    ("storage_guards", "max_output_gb"): 30.0,
}
for (section, field), expected in expected_contract.items():
    if dataset_info.get(section, {}).get(field) != expected:
        raise ValueError(f"generation dataset_info contract mismatch: {section}.{field}")
seed = ready.get("seed")
if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
    raise ValueError("generation ready seed is invalid")
if dataset_info.get("selection", {}).get("seed") != seed:
    raise ValueError("generation dataset_info seed differs from ready marker")
written = dataset_info.get("written_crops")
with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    manifest_rows = sum(1 for _ in csv.DictReader(handle))
if not isinstance(written, int) or isinstance(written, bool) or written != manifest_rows or written <= 0:
    raise ValueError("generation written_crops differs from raw manifest")

expected_outputs = {
    manifest.resolve(): sha(manifest),
    info.resolve(): sha(info),
    inventory.resolve(): sha(inventory),
}
declared = {}
for line in outputs.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
    if not match:
        raise ValueError("invalid generation output marker line")
    path = Path(match.group(2)).resolve()
    if path in declared:
        raise ValueError("duplicate generation output marker path")
    declared[path] = match.group(1)
if declared != expected_outputs:
    raise ValueError("generation output marker does not bind the exact raw artifacts")

expected_inputs = {
    detector.resolve(): sha(detector),
    pilot_yaml.resolve(): sha(pilot_yaml),
    proposal_prepare.resolve(): sha(proposal_prepare),
    preprocessing.resolve(): sha(preprocessing),
    generation_wrapper.resolve(): sha(generation_wrapper),
    dataset_input_inventory.resolve(): sha(dataset_input_inventory),
}
declared_inputs = {}
for line in inputs.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
    if not match:
        raise ValueError("invalid generation input marker line")
    path = Path(match.group(2)).resolve()
    if path in declared_inputs:
        raise ValueError("duplicate generation input marker path")
    declared_inputs[path] = match.group(1)
if declared_inputs != expected_inputs:
    raise ValueError("generation input marker does not bind the exact pilot dependencies")
PY
  then
    fail "generation ready binding verification failed" 65
  fi
  verify_raw_inventory || fail "raw output inventory verification failed" 65
}

create_workspace() {
  workspace=$1
  if ! "$PYTHON_BIN" - "$workspace" "$RAW_DIR" <<'PY'
import os
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
raw = Path(sys.argv[2]).resolve(strict=True)
workspace.mkdir(mode=0o700)
for name in ("manifest.csv", "dataset_info.json", "training", "validation"):
    source = (raw / name).resolve(strict=True)
    link = workspace / name
    relative = os.path.relpath(source, start=workspace.resolve())
    os.symlink(relative, link, target_is_directory=source.is_dir())
    if not link.is_symlink() or link.resolve(strict=True) != source:
        raise RuntimeError(f"workspace link did not resolve to raw input: {name}")
    if os.path.isabs(os.readlink(link)):
        raise RuntimeError(f"workspace link is not relative: {name}")
PY
  then
    fail "failed to create independent validator workspace: $workspace" 73
  fi
}

validate_report_contract() {
  manifest=$1
  report=$2
  if ! "$PYTHON_BIN" - \
    "$manifest" "$report" "$RAW_MANIFEST" "$DATASET_INFO" \
    "$DETECTOR_MODEL" "$INFERENCE_SPEC" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

validated, report_path, raw, info, detector, spec = map(Path, sys.argv[1:7])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("schema_version") != 1:
    raise ValueError("validator report schema mismatch")
if report.get("artifact_role") != "v4_runtime_replay_diagnostic_not_lineage_blind_or_deployment_authority":
    raise ValueError("validator report is not runtime diagnostic evidence")
if report.get("ready_for_lineage_upgrade") is not False:
    raise ValueError("diagnostic validator report unexpectedly grants lineage readiness")
if report.get("lineage_execution_authorized") is not False:
    raise ValueError("diagnostic validator report unexpectedly grants lineage execution")
rows = report.get("rows")
if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
    raise ValueError("validator report has no validated rows")
if report.get("blind_test_eligible") is not False:
    raise ValueError("validator report grants blind-test authority")
if report.get("production_deployment_authorized") is not False:
    raise ValueError("validator report grants production authority")
provenance = report.get("contract", {}).get("proposal_provenance", {})
expected = {
    "provider_kind": "frozen_yolo_runtime",
    "runtime_detector_executed": True,
    "runtime_top1_replayed": True,
    "provided_top1_predictions_matched": True,
    "proposal_class_confidence_bbox_matched": True,
    "confidence_abs_tolerance": 1e-6,
    "bbox_abs_tolerance": 1e-4,
}
for field, value in expected.items():
    if provenance.get(field) != value:
        raise ValueError(f"validator proposal provenance mismatch: {field}")
if provenance.get("production_or_blind_authority") is not False:
    raise ValueError("validator provenance grants production or blind authority")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected_bindings = {
    "input_manifest_sha256": sha(raw),
    "dataset_info_sha256": sha(info),
    "detector_model_sha256": sha(detector),
    "inference_spec_sha256": sha(spec),
    "validated_manifest_sha256": sha(validated),
}
if report.get("bindings") != expected_bindings:
    raise ValueError("validator report bindings mismatch")
with validated.open("r", encoding="utf-8-sig", newline="") as handle:
    actual_rows = sum(1 for _ in csv.DictReader(handle))
if actual_rows != rows:
    raise ValueError("validated manifest row count differs from report")
with raw.open("r", encoding="utf-8-sig", newline="") as handle:
    raw_rows = sum(1 for _ in csv.DictReader(handle))
if raw_rows != rows:
    raise ValueError("validator report row count differs from raw manifest")
materials = (
    "can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass",
    "battery", "fluorescent",
)
minimum = {
    **{f"training/{name}": 25 for name in materials},
    **{f"validation/{name}": 10 for name in materials},
    "training/background": 100,
    "validation/background": 50,
}
counts = report.get("counts")
if not isinstance(counts, dict):
    raise ValueError("validator report coverage counts are missing")
if set(counts) - set(minimum):
    raise ValueError("validator report contains an unexpected coverage stratum")
if any(
    not isinstance(value, int) or isinstance(value, bool) or value < minimum[name]
    for name, value in counts.items()
):
    raise ValueError("validator report coverage count is invalid")
for name, required in minimum.items():
    if counts.get(name, 0) < required:
        raise ValueError(f"validator report coverage is below minimum: {name}")
if sum(counts.values()) != rows:
    raise ValueError("validator coverage counts do not sum to report rows")
PY
  then
    fail "validator report contract verification failed: $report" 65
  fi
}

verify_generation_contract
seal_or_verify_cohort create
write_marker "$CONTROL/00_raw_generation.sha256" \
  "$VALIDATOR" "$WRAPPER" "$GEN_WRAPPER" "$PILOT_BUILDER" "$PROPOSAL_PREPARE" \
  "$PREPROCESSING" "$DETECTOR_MODEL" "$INFERENCE_SPEC" \
  "$PILOT_READY" "$PILOT_INPUTS" "$PILOT_INVENTORY" "$PILOT_YAML" \
  "$PILOT_TRAIN" "$PILOT_VALIDATION" \
  "$RAW_MANIFEST" "$DATASET_INFO" "$GEN_INPUTS" "$GEN_OUTPUTS" \
  "$GEN_DATASET_INPUT_INVENTORY" \
  "$RAW_INVENTORY" "$GEN_READY" "$COHORT_BINDING"
verify_marker "$CONTROL/00_raw_generation.sha256"

create_workspace "$WORK_A"
create_workspace "$WORK_B"

A_MANIFEST=$WORK_A/manifest.v4.validated.csv
A_REPORT=$WORK_A/manifest.v4.validation.json
A_STDOUT=$WORK_A/validator.stdout.json
B_MANIFEST=$WORK_B/manifest.v4.validated.csv
B_REPORT=$WORK_B/manifest.v4.validation.json
B_STDOUT=$WORK_B/validator.stdout.json

verify_generation_contract
verify_marker "$CONTROL/00_raw_generation.sha256"
if ! "$PYTHON_BIN" "$VALIDATOR" \
  --diagnostic-only \
  --input-manifest "$WORK_A/manifest.csv" \
  --dataset-info "$WORK_A/dataset_info.json" \
  --detector-model "$DETECTOR_MODEL" \
  --inference-spec "$INFERENCE_SPEC" \
  --output-manifest "$A_MANIFEST" \
  --output-report "$A_REPORT" > "$A_STDOUT"
then
  fail "validator A failed"
fi
require_file "$A_STDOUT"
validate_report_contract "$A_MANIFEST" "$A_REPORT"
verify_generation_contract
write_marker "$CONTROL/01_validator_a.sha256" "$A_MANIFEST" "$A_REPORT" "$A_STDOUT"
verify_marker "$CONTROL/01_validator_a.sha256"

verify_generation_contract
verify_marker "$CONTROL/00_raw_generation.sha256"
if ! "$PYTHON_BIN" "$VALIDATOR" \
  --diagnostic-only \
  --input-manifest "$WORK_B/manifest.csv" \
  --dataset-info "$WORK_B/dataset_info.json" \
  --detector-model "$DETECTOR_MODEL" \
  --inference-spec "$INFERENCE_SPEC" \
  --output-manifest "$B_MANIFEST" \
  --output-report "$B_REPORT" > "$B_STDOUT"
then
  fail "validator B failed"
fi
require_file "$B_STDOUT"
validate_report_contract "$B_MANIFEST" "$B_REPORT"
verify_generation_contract
write_marker "$CONTROL/02_validator_b.sha256" "$B_MANIFEST" "$B_REPORT" "$B_STDOUT"
verify_marker "$CONTROL/02_validator_b.sha256"

if ! cmp -s "$A_MANIFEST" "$B_MANIFEST"; then
  fail "validator A/B validated manifest bytes differ" 65
fi
COMPARISON=$CONTROL/reproducibility_comparison.json
if ! "$PYTHON_BIN" - \
  "$A_MANIFEST" "$A_REPORT" "$B_MANIFEST" "$B_REPORT" "$COHORT_BINDING" \
  "$COMPARISON" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

a_manifest, a_report_path, b_manifest, b_report_path, cohort_path, output = map(
    Path, sys.argv[1:7]
)
a_report = json.loads(a_report_path.read_text(encoding="utf-8"))
b_report = json.loads(b_report_path.read_text(encoding="utf-8"))
core_fields = (
    "schema_version", "artifact_role", "ready_for_lineage_upgrade",
    "blind_test_eligible", "production_deployment_authorized", "rows",
    "counts", "contract", "bindings",
)
a_core = {field: a_report.get(field) for field in core_fields}
b_core = {field: b_report.get(field) for field in core_fields}
if a_core != b_core:
    raise ValueError("validator A/B report contracts or bindings differ")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

a_manifest_sha = sha(a_manifest)
b_manifest_sha = sha(b_manifest)
if a_manifest_sha != b_manifest_sha:
    raise ValueError("validator A/B validated manifest hashes differ")
cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
if cohort.get("status") != "pilot_cohort_current_bytes_and_raw_membership_bound":
    raise ValueError("pilot cohort binding status mismatch")
if cohort.get("historical_background_probe_is_selection_only") is not True:
    raise ValueError("pilot cohort does not keep background probes selection-only")
if cohort.get("historical_background_probe_replay_result_assumed") is not False:
    raise ValueError("pilot cohort assumes historical background replay results")
if cohort.get("selected_background_probes") != cohort.get("emitted_background_probes"):
    raise ValueError("pilot cohort did not emit every selected background probe")
if not isinstance(cohort.get("background_probe_replay_categories"), dict):
    raise ValueError("pilot cohort background probe replay distribution is missing")
payload = {
    "schema_version": 1,
    "status": "validator_ab_exact_reproduction",
    "artifact_role": "v4_batch1_validator_reproducibility_diagnostic_only",
    "validated_manifest_bytes_equal": True,
    "report_core_contract_and_bindings_equal": True,
    "rows": a_report["rows"],
    "selected_sources": cohort["selected_sources"],
    "emitted_unique_sources": cohort["emitted_unique_sources"],
    "minimum_emitted_sources": cohort["minimum_emitted_sources"],
    "selected_source_coverage": cohort["selected_source_coverage"],
    "selected_drift_anchors": cohort["selected_drift_anchors"],
    "emitted_drift_anchors": cohort["emitted_drift_anchors"],
    "priority_drift_anchors": cohort["priority_drift_anchors"],
    "selected_background_probes": cohort["selected_background_probes"],
    "emitted_background_probes": cohort["emitted_background_probes"],
    "background_probe_replay_categories": cohort[
        "background_probe_replay_categories"
    ],
    "historical_background_probe_is_selection_only": True,
    "historical_background_probe_replay_result_assumed": False,
    "bindings": {
        "validated_manifest_sha256": a_manifest_sha,
        "validator_a_report_sha256": sha(a_report_path),
        "validator_b_report_sha256": sha(b_report_path),
        "cohort_binding_sha256": sha(cohort_path),
    },
    "lineage_execution_authorized": False,
    "training_authority": False,
    "blind_test_authority": False,
    "production_deployment_authorized": False,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
os.link(temporary, output)
temporary.unlink()
PY
then
  fail "validator A/B reproducibility comparison failed" 65
fi

verify_generation_contract
verify_marker "$CONTROL/00_raw_generation.sha256"
verify_marker "$CONTROL/01_validator_a.sha256"
verify_marker "$CONTROL/02_validator_b.sha256"
write_marker "$CONTROL/03_reproducibility.sha256" \
  "$CONTROL/00_raw_generation.sha256" \
  "$CONTROL/01_validator_a.sha256" \
  "$CONTROL/02_validator_b.sha256" \
  "$COMPARISON" "$COHORT_BINDING"
verify_marker "$CONTROL/03_reproducibility.sha256"

# Recheck the complete raw tree and every sealed stage immediately before the
# one terminal publication. A late raw mutation cannot inherit a ready marker.
verify_generation_contract
seal_or_verify_cohort verify
verify_marker "$CONTROL/00_raw_generation.sha256"
verify_marker "$CONTROL/01_validator_a.sha256"
verify_marker "$CONTROL/02_validator_b.sha256"
verify_marker "$CONTROL/03_reproducibility.sha256"

if ! "$PYTHON_BIN" - \
  "$READY" "$CONTROL/failed.txt" "$CONTROL/03_reproducibility.sha256" \
  "$GEN_READY" "$COMPARISON" "$A_MANIFEST" "$A_REPORT" "$B_REPORT" \
  "$PILOT_READY" "$PILOT_INPUTS" "$PILOT_INVENTORY" "$PILOT_YAML" \
  "$PILOT_TRAIN" "$PILOT_VALIDATION" "$COHORT_BINDING" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    ready, failed, chain, generation_ready, comparison, manifest, report_a,
    report_b, pilot_ready, pilot_inputs, pilot_inventory, pilot_yaml,
    pilot_train, pilot_validation, cohort_binding,
) = map(Path, sys.argv[1:16])
if ready.exists() or ready.is_symlink():
    raise FileExistsError("diagnostic ready marker already exists")
if failed.exists() or failed.is_symlink():
    raise RuntimeError("failure marker exists")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

comparison_value = json.loads(comparison.read_text(encoding="utf-8"))
if comparison_value.get("status") != "validator_ab_exact_reproduction":
    raise ValueError("comparison status mismatch")
payload = {
    "schema_version": 1,
    "status": "batch1_validator_ab_reproducibility_passed",
    "artifact_role": "v4_reproducibility_diagnostic_not_candidate_or_deployment_authority",
    "raw_generation_mutation_detected": False,
    "validator_runs": 2,
    "validated_rows": comparison_value["rows"],
    "selected_sources": comparison_value["selected_sources"],
    "emitted_unique_sources": comparison_value["emitted_unique_sources"],
    "minimum_emitted_sources": comparison_value["minimum_emitted_sources"],
    "selected_source_coverage": comparison_value["selected_source_coverage"],
    "selected_drift_anchors": comparison_value["selected_drift_anchors"],
    "emitted_drift_anchors": comparison_value["emitted_drift_anchors"],
    "priority_drift_anchors": comparison_value["priority_drift_anchors"],
    "selected_background_probes": comparison_value["selected_background_probes"],
    "emitted_background_probes": comparison_value["emitted_background_probes"],
    "background_probe_replay_categories": comparison_value[
        "background_probe_replay_categories"
    ],
    "historical_background_probe_is_selection_only": True,
    "historical_background_probe_replay_result_assumed": False,
    "drift_hypothesis_diagnostic_only": True,
    "lineage_execution_authorized": False,
    "judge_authority": False,
    "training_authority": False,
    "blind_test_authority": False,
    "candidate_promotion_authorized": False,
    "production_deployment_authorized": False,
    "bindings": {
        "reproducibility_chain_sha256": sha(chain),
        "raw_generation_ready_sha256": sha(generation_ready),
        "comparison_sha256": sha(comparison),
        "validated_manifest_sha256": sha(manifest),
        "validator_a_report_sha256": sha(report_a),
        "validator_b_report_sha256": sha(report_b),
        "pilot_input_ready_sha256": sha(pilot_ready),
        "pilot_inputs_marker_sha256": sha(pilot_inputs),
        "pilot_selection_inventory_sha256": sha(pilot_inventory),
        "pilot_dataset_yaml_sha256": sha(pilot_yaml),
        "pilot_train_list_sha256": sha(pilot_train),
        "pilot_validation_list_sha256": sha(pilot_validation),
        "cohort_binding_sha256": sha(cohort_binding),
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
  fail "failed to publish reproducibility diagnostic ready marker"
fi

# Ready publication is the final fallible operation. Consumers must require the
# complete 03 chain and absence of failed.txt. This marker grants no authority.
terminal_state=1
exit 0
