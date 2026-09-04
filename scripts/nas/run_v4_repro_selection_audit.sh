#!/bin/sh
set -eu

# CPU-only, immutable pre-generation audit of the deterministic v4 selector.
# Passing this stage is reproducibility evidence only. It never authorizes raw
# generation, validation, training, promotion, or deployment.

umask 077

require_env() {
  value=$(printenv "$1" 2>/dev/null || true)
  if [ -z "$value" ]; then
    printf '%s\n' "missing required environment variable: $1" >&2
    exit 64
  fi
}

for name in AUDIT_DIR CODE_ROOT PILOT_INPUT_DIR QUALITY_EXCLUSION_MANIFEST QUALITY_EXCLUSION_ASSEMBLY_RECEIPT
do
  require_env "$name"
done

PYTHON_BIN=${PYTHON_BIN:-python3}

# Reject relative or newline-bearing roots before AUDIT_DIR is created or any
# CODE_ROOT/PILOT_INPUT_DIR artifact path is derived from them.
if ! "$PYTHON_BIN" - "$AUDIT_DIR" "$CODE_ROOT" "$PILOT_INPUT_DIR" \
  "$QUALITY_EXCLUSION_MANIFEST" "$QUALITY_EXCLUSION_ASSEMBLY_RECEIPT" <<'PY'
import json
import os
import sys
from pathlib import Path

raw_values = dict(zip(
    ("AUDIT_DIR", "CODE_ROOT", "PILOT_INPUT_DIR", "QUALITY_EXCLUSION_MANIFEST", "QUALITY_EXCLUSION_ASSEMBLY_RECEIPT"),
    sys.argv[1:6],
    strict=True,
))
def reject_symlink_components(path: Path, *, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path contains a symlink component")

for name, raw in raw_values.items():
    if "\n" in raw or "\r" in raw or not Path(raw).is_absolute():
        raise ValueError(f"{name} must be an absolute newline-free path")
audit = Path(raw_values["AUDIT_DIR"]).resolve(strict=False)
code_arg = Path(raw_values["CODE_ROOT"])
pilot_arg = Path(raw_values["PILOT_INPUT_DIR"])
quality_arg = Path(raw_values["QUALITY_EXCLUSION_MANIFEST"])
receipt_arg = Path(raw_values["QUALITY_EXCLUSION_ASSEMBLY_RECEIPT"])
if (
    code_arg.is_symlink()
    or pilot_arg.is_symlink()
    or not code_arg.is_dir()
    or not pilot_arg.is_dir()
    or quality_arg.is_symlink()
    or not quality_arg.is_file()
    or receipt_arg.is_symlink()
    or not receipt_arg.is_file()
):
    raise ValueError("code, pilot, and quality exclusion inputs must be regular")
reject_symlink_components(quality_arg, description="QUALITY_EXCLUSION_MANIFEST")
reject_symlink_components(receipt_arg, description="QUALITY_EXCLUSION_ASSEMBLY_RECEIPT")
code = code_arg.resolve(strict=True)
pilot = pilot_arg.resolve(strict=True)
quality = quality_arg.resolve(strict=True)
inventory_path = pilot / "selection_inventory.json"
if inventory_path.is_symlink() or not inventory_path.is_file():
    raise ValueError("pilot selection inventory must be a regular file")
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
dataset_raw = inventory.get("bindings", {}).get("dataset_dir")
if not isinstance(dataset_raw, str) or not Path(dataset_raw).is_absolute():
    raise ValueError("pilot dataset directory binding is invalid")
dataset_arg = Path(dataset_raw)
if dataset_arg.is_symlink() or not dataset_arg.is_dir():
    raise ValueError("pilot dataset directory must be a regular directory")
dataset = dataset_arg.resolve(strict=True)
for protected_name, protected in (
    ("CODE_ROOT", code),
    ("PILOT_INPUT_DIR", pilot),
    ("dataset directory", dataset),
    ("quality assembly directory", receipt_arg.resolve(strict=True).parent),
):
    try:
        audit.relative_to(protected)
    except ValueError:
        pass
    else:
        raise ValueError(f"AUDIT_DIR must not be inside {protected_name}")
    try:
        protected.relative_to(audit)
    except ValueError:
        pass
    else:
        raise ValueError(f"AUDIT_DIR must not contain {protected_name}")
try:
    quality.relative_to(audit)
except ValueError:
    pass
else:
    raise ValueError("AUDIT_DIR must not contain the quality exclusion manifest")
PY
then
  printf '%s\n' "audit roots must be absolute paths" >&2
  exit 64
fi

WRAPPER=$CODE_ROOT/scripts/nas/run_v4_repro_selection_audit.sh
PILOT_BUILDER=$CODE_ROOT/scripts/build_v4_repro_pilot_inputs.py
PROPOSAL_PREPARE=$CODE_ROOT/scripts/prepare_proposal_verifier_dataset.py
QUALITY_MANIFEST=$QUALITY_EXCLUSION_MANIFEST
QUALITY_RECEIPT=$QUALITY_EXCLUSION_ASSEMBLY_RECEIPT

if [ -e "$AUDIT_DIR" ] || [ -L "$AUDIT_DIR" ]; then
  printf '%s\n' "refusing to reuse immutable AUDIT_DIR: $AUDIT_DIR" >&2
  exit 73
fi
if ! mkdir "$AUDIT_DIR" 2>/dev/null; then
  printf '%s\n' "failed to create immutable AUDIT_DIR: $AUDIT_DIR" >&2
  exit 73
fi

READY=$AUDIT_DIR/selection_audit_ready.json
FAILED=$AUDIT_DIR/failed.txt
RECOMPUTE=$AUDIT_DIR/recompute
terminal_state=0

remove_ready() {
  if [ -e "$READY" ] || [ -L "$READY" ]; then
    rm -f "$READY" 2>/dev/null || true
  fi
}

publish_failure() {
  message=$1
  [ -d "$AUDIT_DIR" ] || return 1
  if [ ! -e "$FAILED" ] && [ ! -L "$FAILED" ]; then
    temporary=$(mktemp "$AUDIT_DIR/.failed.XXXXXX") || return 1
    if ! printf '%s\n' "$message" > "$temporary" || \
       ! ln "$temporary" "$FAILED" 2>/dev/null; then
      rm -f "$temporary"
      return 1
    fi
    rm -f "$temporary"
  fi
  remove_ready
}

on_exit() {
  code=$?
  if [ "$terminal_state" -eq 0 ] && [ "$code" -ne 0 ]; then
    publish_failure "unexpected v4 selection audit exit: code=$code" || true
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

for artifact in "$WRAPPER" "$PILOT_BUILDER" "$PROPOSAL_PREPARE" "$QUALITY_MANIFEST" "$QUALITY_RECEIPT"
do
  if [ ! -f "$artifact" ] || [ ! -s "$artifact" ]; then
    fail "missing or empty audit dependency: $artifact" 66
  fi
done
if [ -e "$PILOT_INPUT_DIR/failed.txt" ] || [ -L "$PILOT_INPUT_DIR/failed.txt" ]; then
  fail "pilot input failure marker exists" 65
fi

if ! "$PYTHON_BIN" - \
  "$AUDIT_DIR" "$RECOMPUTE" "$CODE_ROOT" "$PILOT_INPUT_DIR" \
  "$WRAPPER" "$PILOT_BUILDER" "$PROPOSAL_PREPARE" "$QUALITY_MANIFEST" "$QUALITY_RECEIPT" <<'PY'
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

(
    audit_arg,
    recompute_arg,
    code_arg,
    pilot_arg,
    wrapper_arg,
    builder_arg,
    proposal_arg,
    quality_arg,
    receipt_arg,
) = map(Path, sys.argv[1:10])

ROLE = (
    "v4_repro_selection_audit_cpu_only_diagnostic_"
    "not_generation_training_blind_or_deployment_authority"
)
AUDIT_CONTRACT = "v4_repro_selection_audit.cpu_only_byte_exact.v2"
SELECTION_CONTRACT = (
    "v4_repro_pilot_inputs."
    "gt_stratified_historical_observation_priority_blake2b.v4"
)
PILOT_ROLE = (
    "v4_batch1_reproducibility_pilot_inputs_diagnostic_only_"
    "not_training_blind_or_deployment_authority"
)
ARTIFACT_NAMES = (
    "pilot_dataset.yaml",
    "selection_inventory.json",
    "train_pilot.txt",
    "validation_pilot.txt",
)
SEALED_NAMES = (*ARTIFACT_NAMES, "inputs.sha256", "input_ready.json")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
QUALITY_CONTRACT = "v4_capture_quality_exclusions.sha256_reason_only.v1"
QUALITY_ROLE = (
    "v4_capture_quality_exclusion_manifest_selection_only_"
    "not_ground_truth_or_authority"
)
# Contract v1 retains legacy immutable manifests and accepts current canonical reasons.
QUALITY_REASONS = {
    "captured_before_2026_08_01",
    "severe_frame_crop",
    "person_occlusion_or_dominance",
    "clutter_or_multiple_objects",
    "boundary_unreadable",
    "objective_unreadable",
    "resolution_too_low",
    "extreme_exposure",
    "excessive_background_or_multi_object",
    "unreadable_boundary",
    "too_low_resolution",
}
QUALITY_AUTHORITY = {
    "selection": False,
    "ground_truth": False,
    "replay": False,
    "training": False,
    "calibration": False,
    "blind_test": False,
    "deployment": False,
}
QUALITY_MAX_SOURCES = 100


def identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def reject_symlink_components(path: Path, *, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path contains a symlink component")


def stable_bytes(path: Path, *, description: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")
    before = path.stat()
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        content = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = path.stat()
    if not (
        identity(before)
        == identity(opened_before)
        == identity(opened_after)
        == identity(after)
    ):
        raise RuntimeError(f"{description} changed while reading: {path}")
    if not content:
        raise ValueError(f"{description} is empty: {path}")
    return content


def sha_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def stable_sha(path: Path, *, description: str) -> str:
    return sha_bytes(stable_bytes(path, description=description))


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, item in pairs:
        if key in parsed:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        parsed[key] = item
    return parsed


def load_object(path: Path, *, description: str) -> tuple[dict[str, object], bytes]:
    content = stable_bytes(path, description=description)
    try:
        value = json.loads(
            content.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value, content


def require_false_fields(
    value: dict[str, object], fields: tuple[str, ...], *, description: str
) -> None:
    for field in fields:
        if value.get(field) is not False:
            raise ValueError(f"{description} unexpectedly grants {field}")


def parse_inputs_marker(content: bytes, expected: dict[str, str]) -> None:
    try:
        lines = content.decode("ascii").splitlines()
    except UnicodeError as error:
        raise ValueError("pilot inputs marker is not ASCII") from error
    declared: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if match is None or match.group(2) in declared:
            raise ValueError("pilot inputs marker line is invalid or duplicated")
        declared[match.group(2)] = match.group(1)
    if declared != expected:
        raise ValueError("pilot inputs marker does not bind the exact four artifacts")


def resolve_bound_file(info: object, *, description: str) -> tuple[Path, str]:
    if not isinstance(info, dict):
        raise ValueError(f"{description} binding is missing")
    raw_path = info.get("path")
    digest = info.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not isinstance(digest, str)
        or not SHA_RE.fullmatch(digest)
    ):
        raise ValueError(f"{description} binding is invalid")
    path = Path(raw_path)
    if not path.is_absolute() or stable_sha(path, description=description) != digest:
        raise ValueError(f"{description} path or hash mismatch")
    return Path(os.path.abspath(path)), digest


def ensure_disjoint(left: Path, right: Path, *, description: str) -> None:
    try:
        left.relative_to(right)
    except ValueError:
        pass
    else:
        raise ValueError(f"{description}: {left} is inside {right}")
    try:
        right.relative_to(left)
    except ValueError:
        pass
    else:
        raise ValueError(f"{description}: {right} is inside {left}")


def publish_exclusive(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(
                f"refusing to overwrite immutable artifact: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def exact_json_equal(actual: object, expected: object) -> bool:
    try:
        options = {
            "ensure_ascii": False,
            "sort_keys": True,
            "separators": (",", ":"),
            "allow_nan": False,
        }
        return json.dumps(actual, **options) == json.dumps(expected, **options)
    except (TypeError, ValueError):
        return False


def canonical_quality_entries_bytes(entries: list[dict[str, str]]) -> bytes:
    return (
        json.dumps(
            entries,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_quality_manifest(
    value: dict[str, object], content: bytes, *, assembly_bundle: object | None = None
) -> tuple[set[str], dict[str, object], dict[str, object]]:
    # Empty selections are legitimate only after the full assembly is validated.
    # A naked zero-entry manifest must never grant this exception by itself.
    candidate_authority._validate_quality_manifest(
        value, assembly_bundle=assembly_bundle
    )
    expected_keys = {
        "schema_version", "artifact_role", "quality_exclusion_contract",
        "status", "excluded_source_count", "reason_counts",
        "max_excluded_sources", "source_list_sha256", "entries", "authority",
    }
    if set(value) != expected_keys:
        raise ValueError("quality exclusion manifest schema keys are not exact")
    authority = value.get("authority")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("artifact_role") != QUALITY_ROLE
        or value.get("quality_exclusion_contract") != QUALITY_CONTRACT
        or value.get("status") != "quality_exclusions_ready"
        or not exact_json_equal(authority, QUALITY_AUTHORITY)
        or type(value.get("max_excluded_sources")) is not int
        or value.get("max_excluded_sources") != QUALITY_MAX_SOURCES
    ):
        raise ValueError("quality exclusion manifest contract or authority mismatch")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ValueError("quality exclusion entries must be a list")
    if len(entries) > QUALITY_MAX_SOURCES:
        raise ValueError("quality exclusion entries exceed bounded maximum")
    normalized: list[dict[str, str]] = []
    excluded: set[str] = set()
    reason_counts: dict[str, int] = {}
    for index, row in enumerate(entries):
        if not isinstance(row, dict) or set(row) != {"source_sha256", "reason"}:
            raise ValueError(f"quality exclusion entry {index} is malformed")
        source_sha = row.get("source_sha256")
        reason = row.get("reason")
        if not isinstance(source_sha, str) or not SHA_RE.fullmatch(source_sha):
            raise ValueError(f"quality exclusion entry {index} source SHA is invalid")
        if source_sha in excluded:
            raise ValueError(f"quality exclusion entry {index} duplicates source SHA")
        if reason not in QUALITY_REASONS:
            raise ValueError(f"quality exclusion entry {index} reason is unknown")
        excluded.add(source_sha)
        normalized.append({"source_sha256": source_sha, "reason": str(reason)})
        reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    normalized.sort(key=lambda row: row["source_sha256"])
    if entries != normalized:
        raise ValueError("quality exclusion entries are not canonical SHA order")
    canonical_sha = sha_bytes(canonical_quality_entries_bytes(normalized))
    if value.get("source_list_sha256") != canonical_sha:
        raise ValueError("quality exclusion canonical source-list hash mismatch")
    expected_counts = dict(sorted(reason_counts.items()))
    declared_counts = value.get("reason_counts")
    if (
        type(value.get("excluded_source_count")) is not int
        or value.get("excluded_source_count") != len(excluded)
        or not isinstance(declared_counts, dict)
        or any(
            type(count) is not int or count <= 0
            for count in declared_counts.values()
        )
        or not exact_json_equal(declared_counts, expected_counts)
    ):
        raise ValueError("quality exclusion counts are inconsistent")
    full = {
        "required": True,
        "manifest_contract": QUALITY_CONTRACT,
        "manifest_path": quality.resolve(strict=True).as_posix(),
        "manifest_sha256": sha_bytes(content),
        "source_list_sha256": canonical_sha,
        "excluded_source_count": len(excluded),
        "max_excluded_sources": QUALITY_MAX_SOURCES,
        "reason_counts": expected_counts,
        "selection_authority": False,
        "ground_truth_authority": False,
        "replay_authority": False,
        "training_authority": False,
        "calibration_authority": False,
        "blind_test_authority": False,
        "deployment_authority": False,
    }
    sanitized = dict(full)
    sanitized.pop("manifest_path")
    return excluded, full, sanitized


def verify_selected_lists_exclude_quality_sources(
    inventory_value: dict[str, object],
    train_content: bytes,
    validation_content: bytes,
    excluded: set[str],
    *,
    description: str,
) -> None:
    rows = inventory_value.get("selected_sources")
    if not isinstance(rows, list):
        raise ValueError(f"{description} selected source rows are missing")
    expected_paths: dict[str, set[str]] = {"training": set(), "validation": set()}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{description} selected row {index} is malformed")
        split = row.get("split")
        raw_path = row.get("path")
        source_sha = row.get("source_sha256")
        if (
            split not in expected_paths
            or not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
            or not isinstance(source_sha, str)
            or not SHA_RE.fullmatch(source_sha)
        ):
            raise ValueError(f"{description} selected row {index} binding is invalid")
        if source_sha in excluded:
            raise ValueError(f"{description} contains a quality-excluded source SHA")
        source_path = Path(raw_path)
        if stable_sha(source_path, description=f"{description} selected source") != source_sha:
            raise ValueError(f"{description} selected source SHA binding mismatch")
        if raw_path in expected_paths[str(split)]:
            raise ValueError(f"{description} selected source path is duplicated")
        expected_paths[str(split)].add(raw_path)
    for split, content in (
        ("training", train_content),
        ("validation", validation_content),
    ):
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeError as error:
            raise ValueError(f"{description} {split} list is not UTF-8") from error
        if len(lines) != len(set(lines)) or set(lines) != expected_paths[split]:
            raise ValueError(f"{description} {split} list differs from inventory")


audit = audit_arg.resolve(strict=True)
normalized_recompute = (audit / "recompute").resolve(strict=False)
code = code_arg.resolve(strict=True)
pilot = pilot_arg.resolve(strict=True)
wrapper = wrapper_arg.resolve(strict=True)
builder = builder_arg.resolve(strict=True)
proposal = proposal_arg.resolve(strict=True)
if (
    audit_arg.is_symlink()
    or code_arg.is_symlink()
    or pilot_arg.is_symlink()
    or wrapper_arg.is_symlink()
    or builder_arg.is_symlink()
    or proposal_arg.is_symlink()
    or quality_arg.is_symlink()
    or not audit.is_dir()
    or not code.is_dir()
    or not pilot.is_dir()
    or not quality_arg.is_file()
):
    raise ValueError("audit, code, pilot, and quality paths are invalid")
reject_symlink_components(quality_arg, description="quality exclusion manifest")
quality = quality_arg.resolve(strict=True)
ensure_disjoint(audit, pilot, description="AUDIT_DIR and PILOT_INPUT_DIR must be independent")
ensure_disjoint(audit, code, description="AUDIT_DIR and CODE_ROOT must be independent")
try:
    quality.relative_to(audit)
except ValueError:
    pass
else:
    raise ValueError("quality exclusion manifest must be outside AUDIT_DIR")
if recompute_arg.resolve(strict=False) != normalized_recompute:
    raise ValueError("recompute path does not normalize beneath AUDIT_DIR")
if recompute_arg.exists() or recompute_arg.is_symlink():
    raise ValueError("recompute directory already exists")

pilot_paths = {name: pilot / name for name in SEALED_NAMES}
if (pilot / "failed.txt").exists() or (pilot / "failed.txt").is_symlink():
    raise ValueError("pilot input failure marker exists")
pilot_contents = {
    name: stable_bytes(path, description=f"pilot {name}")
    for name, path in pilot_paths.items()
}
pilot_hashes = {name: sha_bytes(content) for name, content in pilot_contents.items()}
quality_value, quality_content = load_object(
    quality, description="quality exclusion manifest"
)
quality_sha = sha_bytes(quality_content)
validator_path = code / "scripts" / "operational_quality_assembly_contract.py"
reject_symlink_components(validator_path, description="quality assembly validator")
validator_content = stable_bytes(validator_path, description="quality assembly validator")
validator_sha = sha_bytes(validator_content)
sys.dont_write_bytecode = True
candidate_authority = types.ModuleType("audit_quality_authority")
candidate_authority.__file__ = str(validator_path)
sys.modules[candidate_authority.__name__] = candidate_authority
exec(compile(validator_content, str(validator_path), "exec"), candidate_authority.__dict__)
quality_bundle = candidate_authority._validate_operational_quality_assembly(
    receipt_path=receipt_arg, quality_path=quality,
    quality_value=quality_value, quality_content=quality_content, output_dir=audit,
)
excluded_source_shas, expected_quality, sanitized_quality = validate_quality_manifest(
    quality_value, quality_content, assembly_bundle=quality_bundle
)
receipt_value = json.loads(quality_bundle.receipt_content.decode("utf-8"))
receipt_sha = sha_bytes(quality_bundle.receipt_content)
assembly_marker_sha = sha_bytes(quality_bundle.marker_content)
expected_quality.update({
    "assembly_schema": receipt_value["assembly_schema"],
    "assembly_mode": receipt_value["assembly_mode"],
    "operational_capture_cutoff_kst": receipt_value["operational_capture_cutoff_kst"],
    "objective_prepare_bundle_validated": receipt_value["scope"]["objective_prepare_bundle_validated"],
    "assembly_receipt_path": quality_bundle.receipt_path.as_posix(),
    "assembly_receipt_sha256": receipt_sha,
    "assembly_marker_path": quality_bundle.marker_path.as_posix(),
    "assembly_marker_sha256": assembly_marker_sha,
    "assembly_validator_path": validator_path.as_posix(),
    "assembly_validator_sha256": validator_sha,
})
sanitized_quality = {key: value for key, value in expected_quality.items() if key not in {
    "manifest_path", "assembly_receipt_path", "assembly_marker_path", "assembly_validator_path",
}}
quality_digest_bindings = {
    "quality_exclusions_sha256": quality_sha,
    "quality_exclusion_manifest_sha256": quality_sha,
    "quality_exclusion_assembly_receipt_sha256": receipt_sha,
    "quality_exclusion_assembly_marker_sha256": assembly_marker_sha,
    "quality_assembly_validator_sha256": validator_sha,
}
quality_path_bindings = {
    "quality_exclusion_manifest_path": quality.as_posix(),
    "quality_exclusion_assembly_receipt_path": quality_bundle.receipt_path.as_posix(),
    "quality_exclusion_assembly_marker_path": quality_bundle.marker_path.as_posix(),
    "quality_assembly_validator_path": validator_path.as_posix(),
}

def validate_quality_bindings(value: object, *, description: str, paths: bool) -> None:
    expected = {**quality_digest_bindings, **(quality_path_bindings if paths else {})}
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
        raise ValueError(f"{description} quality assembly binding mismatch")

def rehash_quality_bundle() -> None:
    candidate_authority._rehash_operational_quality_assembly(quality_bundle)
    reject_symlink_components(validator_path, description="quality assembly validator rehash")
    if stable_sha(validator_path, description="quality assembly validator rehash") != validator_sha:
        raise RuntimeError("quality assembly validator changed")

rehash_quality_bundle()
artifact_hashes = {name: pilot_hashes[name] for name in ARTIFACT_NAMES}
parse_inputs_marker(pilot_contents["inputs.sha256"], artifact_hashes)
try:
    ready = json.loads(
        pilot_contents["input_ready.json"].decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    inventory = json.loads(
        pilot_contents["selection_inventory.json"].decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
except (UnicodeError, json.JSONDecodeError) as error:
    raise ValueError("pilot JSON artifact is invalid") from error
if not isinstance(ready, dict) or not isinstance(inventory, dict):
    raise ValueError("pilot ready and inventory must be JSON objects")

for value, description, status in (
    (ready, "pilot ready", "pilot_inputs_ready"),
    (inventory, "pilot inventory", "selection_complete_not_replay_validated"),
):
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise ValueError(f"{description} schema mismatch")
    if value.get("artifact_role") != PILOT_ROLE:
        raise ValueError(f"{description} role mismatch")
    if value.get("selection_contract") != SELECTION_CONTRACT:
        raise ValueError(f"{description} selection contract mismatch")
    if value.get("status") != status:
        raise ValueError(f"{description} status mismatch")

seed = inventory.get("seed")
ready_seed = ready.get("seed")
if type(seed) is not int or seed != 20260901:
    raise ValueError("pilot selection seed must be the exact integer 20260901")
if type(ready_seed) is not int or ready_seed != seed:
    raise ValueError("pilot ready seed is invalid or inconsistent")
quota = inventory.get("quota_per_stratum")
if not isinstance(quota, dict):
    raise ValueError("pilot quota contract is missing")
train_quota = quota.get("training")
validation_quota = quota.get("validation")
if (
    type(train_quota) is not int
    or train_quota < 1
    or type(validation_quota) is not int
    or validation_quota < 1
):
    raise ValueError("pilot quotas must be exact positive integers")
if inventory.get("full_quota_met") is not True or ready.get("full_quota_met") is not True:
    raise ValueError("pilot does not attest full quota")
selected_counts = inventory.get("selected_counts")
if not isinstance(selected_counts, dict) or not exact_json_equal(ready.get("selected_counts"), selected_counts):
    raise ValueError("pilot selected counts are missing or inconsistent")
if not all(type(value) is int and value >= 0 for value in selected_counts.values()):
    raise ValueError("pilot selected counts must be exact non-negative integers")
selected_sources = sum(selected_counts.values())
if selected_sources < 1 or type(ready.get("selected_sources")) is not int or ready.get("selected_sources") != selected_sources:
    raise ValueError("pilot selected source total is invalid or inconsistent")

require_false_fields(
    ready,
    (
        "validator_authority",
        "training_authorized",
        "blind_test_authorized",
        "production_deployment_authorized",
    ),
    description="pilot ready",
)
authority = inventory.get("authority")
if not isinstance(authority, dict):
    raise ValueError("pilot inventory authority is missing")
require_false_fields(
    authority,
    (
        "raw_generation_authorized",
        "validator_authority",
        "training_authorized",
        "blind_test_authorized",
        "production_deployment_authorized",
    ),
    description="pilot inventory",
)

bindings = ready.get("bindings")
inventory_bindings = inventory.get("bindings")
if not isinstance(bindings, dict) or not isinstance(inventory_bindings, dict):
    raise ValueError("pilot bindings are missing")
validate_quality_bindings(bindings, description="pilot ready", paths=False)
validate_quality_bindings(inventory_bindings, description="pilot inventory", paths=True)
if bindings.get("inputs_marker_sha256") != pilot_hashes["inputs.sha256"]:
    raise ValueError("pilot ready marker binding mismatch")
if bindings.get("artifacts") != artifact_hashes:
    raise ValueError("pilot ready artifact binding mismatch")
universe_sha = inventory_bindings.get("resolved_universe_sha256")
if not isinstance(universe_sha, str) or not SHA_RE.fullmatch(universe_sha):
    raise ValueError("pilot universe binding is invalid")
if bindings.get("resolved_universe_sha256") != universe_sha:
    raise ValueError("pilot universe binding mismatch")

inventory_quality = inventory.get("quality_exclusion")
ready_quality = ready.get("quality_exclusion")
if not isinstance(inventory_quality, dict):
    raise ValueError("pilot quality exclusion binding is missing")
matched_resolved_sources = inventory_quality.get("matched_resolved_sources")
if (
    type(matched_resolved_sources) is not int
    or matched_resolved_sources != len(excluded_source_shas)
):
    raise ValueError("pilot quality exclusion resolved-source count is invalid")
expected_inventory_quality = {
    **expected_quality,
    "matched_resolved_sources": matched_resolved_sources,
}
if not exact_json_equal(inventory_quality, expected_inventory_quality):
    raise ValueError("pilot quality exclusion binding does not match exact manifest")
if not exact_json_equal(ready_quality, sanitized_quality):
    raise ValueError("pilot ready quality exclusion binding mismatch")
if (
    inventory_bindings.get("quality_exclusion_manifest_path")
    != quality.as_posix()
    or inventory_bindings.get("quality_exclusion_manifest_sha256") != quality_sha
):
    raise ValueError("pilot inventory quality exclusion path/hash mismatch")
selected_rows = inventory.get("selected_sources")
if not isinstance(selected_rows, list):
    raise ValueError("pilot selected source rows are missing")
for index, row in enumerate(selected_rows):
    if not isinstance(row, dict):
        raise ValueError(f"pilot selected source row {index} is malformed")
    if row.get("source_sha256") in excluded_source_shas:
        raise ValueError("pilot selection contains a quality-excluded source SHA")
verify_selected_lists_exclude_quality_sources(
    inventory,
    pilot_contents["train_pilot.txt"],
    pilot_contents["validation_pilot.txt"],
    excluded_source_shas,
    description="pilot selection",
)

bound_selector_raw = inventory_bindings.get("selector_path")
if not isinstance(bound_selector_raw, str) or not Path(bound_selector_raw).is_absolute():
    raise ValueError("pilot selector path binding is invalid")
builder_sha = stable_sha(builder, description="pilot selector")
if inventory_bindings.get("selector_sha256") != builder_sha:
    raise ValueError("pilot selector hash mismatch")
bound_proposal_raw = inventory_bindings.get("proposal_generator_path")
if not isinstance(bound_proposal_raw, str) or not Path(bound_proposal_raw).is_absolute():
    raise ValueError("pilot proposal generator path binding is invalid")
proposal_sha = stable_sha(proposal, description="pilot proposal generator")
if inventory_bindings.get("proposal_generator_sha256") != proposal_sha:
    raise ValueError("pilot proposal generator hash mismatch")
wrapper_sha = stable_sha(wrapper, description="selection audit wrapper")

data_raw = inventory_bindings.get("data_path")
data_sha = inventory_bindings.get("data_sha256")
dataset_raw = inventory_bindings.get("dataset_dir")
if (
    not isinstance(data_raw, str)
    or not isinstance(data_sha, str)
    or not SHA_RE.fullmatch(data_sha)
):
    raise ValueError("pilot data binding is invalid")
data_path = Path(data_raw)
if not data_path.is_absolute() or stable_sha(data_path, description="pilot data YAML") != data_sha:
    raise ValueError("pilot data YAML binding mismatch")
if not isinstance(dataset_raw, str):
    raise ValueError("pilot dataset directory binding is invalid")
dataset_dir = Path(dataset_raw)
if not dataset_dir.is_absolute() or dataset_dir.is_symlink() or not dataset_dir.is_dir():
    raise ValueError("pilot dataset directory must be an absolute regular directory")
dataset_dir = dataset_dir.resolve(strict=True)
ensure_disjoint(audit, dataset_dir, description="AUDIT_DIR and dataset directory must be independent")

historical = inventory.get("historical_selection_evidence")
if not isinstance(historical, dict):
    raise ValueError("pilot historical selection evidence is missing")
used_historical = historical.get("used_for_selection_only")
if type(used_historical) is not bool:
    raise ValueError("pilot historical selection usage flag is invalid")
old_info = historical.get("old_manifest")
drift_info = historical.get("drift_report")
old_manifest: Path | None = None
drift_report: Path | None = None
old_sha: str | None = None
drift_sha: str | None = None
if used_historical:
    old_manifest, old_sha = resolve_bound_file(old_info, description="historical manifest")
    if drift_info is not None:
        drift_report, drift_sha = resolve_bound_file(
            drift_info, description="historical drift report"
        )
elif old_info is not None or drift_info is not None:
    raise ValueError("unused historical evidence contains input bindings")
if drift_report is not None and old_manifest is None:
    raise ValueError("historical drift report lacks a bound manifest")

command = [
    sys.executable,
    os.fspath(builder),
    "--data",
    os.fspath(data_path),
    "--dataset-dir",
    os.fspath(dataset_dir),
    "--output-dir",
    os.fspath(normalized_recompute),
    "--quality-exclusion-manifest",
    os.fspath(quality),
    "--quality-exclusion-assembly-receipt",
    os.fspath(quality_bundle.receipt_path),
    "--seed",
    str(seed),
    "--train-quota-per-stratum",
    str(train_quota),
    "--validation-quota-per-stratum",
    str(validation_quota),
]
if old_manifest is not None:
    command.extend(("--old-manifest", os.fspath(old_manifest)))
if drift_report is not None:
    command.extend(("--drift-report", os.fspath(drift_report)))
child_env = os.environ.copy()
child_env.update(
    {
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
    }
)
subprocess.run(command, cwd=code, env=child_env, check=True)
rehash_quality_bundle()

# The selector scans a large tree. Recheck every immutable authority input
# after it exits so a concurrent same-run mutation cannot be sealed as valid.
post_run_expected = {
    wrapper_arg: (wrapper_sha, "selection audit wrapper"),
    builder_arg: (builder_sha, "pilot selector"),
    proposal_arg: (proposal_sha, "pilot proposal generator"),
    data_path: (data_sha, "pilot data YAML"),
    quality: (quality_sha, "quality exclusion manifest"),
    **{
        pilot_paths[name]: (pilot_hashes[name], f"pilot {name}")
        for name in SEALED_NAMES
    },
}
if old_manifest is not None and old_sha is not None:
    post_run_expected[old_manifest] = (old_sha, "historical manifest")
if drift_report is not None and drift_sha is not None:
    post_run_expected[drift_report] = (drift_sha, "historical drift report")
for path, (expected_sha, description) in post_run_expected.items():
    if stable_sha(path, description=f"post-selector {description}") != expected_sha:
        raise RuntimeError(f"{description} changed while selector was running")
if (
    dataset_dir.is_symlink()
    or not dataset_dir.is_dir()
    or dataset_dir.resolve(strict=True) != Path(dataset_raw).resolve(strict=True)
):
    raise RuntimeError("pilot dataset directory changed while selector was running")

recompute = normalized_recompute.resolve(strict=True)
if recompute_arg.is_symlink() or not recompute.is_dir():
    raise ValueError("selector recompute output must be a regular directory")
if (recompute / "failed.txt").exists() or (recompute / "failed.txt").is_symlink():
    raise ValueError("selector recompute published a failure marker")
entries = list(recompute.iterdir())
actual_files = {
    path.name for path in entries if path.is_file() and not path.is_symlink()
}
if (
    len(entries) != len(SEALED_NAMES)
    or actual_files != set(SEALED_NAMES)
    or any(path.is_symlink() or not path.is_file() for path in entries)
):
    raise ValueError("selector recompute output file set is not exact")
recomputed_contents = {
    name: stable_bytes(recompute / name, description=f"recomputed {name}")
    for name in SEALED_NAMES
}
recomputed_hashes = {
    name: sha_bytes(content) for name, content in recomputed_contents.items()
}
recomputed_artifact_hashes = {
    name: recomputed_hashes[name] for name in ARTIFACT_NAMES
}
parse_inputs_marker(recomputed_contents["inputs.sha256"], recomputed_artifact_hashes)
try:
    recomputed_ready = json.loads(
        recomputed_contents["input_ready.json"].decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    recomputed_inventory = json.loads(
        recomputed_contents["selection_inventory.json"].decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
except (UnicodeError, json.JSONDecodeError) as error:
    raise ValueError("recomputed selector JSON artifact is invalid") from error
if not isinstance(recomputed_ready, dict) or not isinstance(recomputed_inventory, dict):
    raise ValueError("recomputed selector JSON artifacts must be objects")
for value, description, status in (
    (recomputed_ready, "recomputed ready", "pilot_inputs_ready"),
    (recomputed_inventory, "recomputed inventory", "selection_complete_not_replay_validated"),
):
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("artifact_role") != PILOT_ROLE
        or value.get("selection_contract") != SELECTION_CONTRACT
        or value.get("status") != status
        or type(value.get("seed")) is not int
        or value.get("seed") != seed
    ):
        raise ValueError(f"{description} contract, status, or seed mismatch")
recomputed_quality = recomputed_inventory.get("quality_exclusion")
recomputed_matched = (
    recomputed_quality.get("matched_resolved_sources")
    if isinstance(recomputed_quality, dict)
    else None
)
if (
    type(recomputed_matched) is not int
    or recomputed_matched != len(excluded_source_shas)
    or not exact_json_equal(
        recomputed_quality,
        {**expected_quality, "matched_resolved_sources": recomputed_matched},
    )
    or not exact_json_equal(
        recomputed_ready.get("quality_exclusion"), sanitized_quality
    )
):
    raise ValueError("recomputed selector quality exclusion binding mismatch")
recomputed_selected = recomputed_inventory.get("selected_sources")
if not isinstance(recomputed_selected, list) or any(
    not isinstance(row, dict)
    or row.get("source_sha256") in excluded_source_shas
    for row in recomputed_selected
):
    raise ValueError("recomputed selection contains a quality-excluded source")
verify_selected_lists_exclude_quality_sources(
    recomputed_inventory,
    recomputed_contents["train_pilot.txt"],
    recomputed_contents["validation_pilot.txt"],
    excluded_source_shas,
    description="recomputed selection",
)
recomputed_ready_bindings = recomputed_ready.get("bindings")
if not isinstance(recomputed_ready_bindings, dict):
    raise ValueError("recomputed ready bindings are missing")
validate_quality_bindings(recomputed_ready_bindings, description="recomputed ready", paths=False)
validate_quality_bindings(recomputed_inventory.get("bindings"), description="recomputed inventory", paths=True)
if recomputed_ready_bindings.get("inputs_marker_sha256") != recomputed_hashes["inputs.sha256"]:
    raise ValueError("recomputed ready marker binding mismatch")
if recomputed_ready_bindings.get("artifacts") != recomputed_artifact_hashes:
    raise ValueError("recomputed ready artifact binding mismatch")
if recomputed_ready_bindings.get("resolved_universe_sha256") != universe_sha:
    raise ValueError("recomputed selector universe binding mismatch")
require_false_fields(
    recomputed_ready,
    (
        "validator_authority",
        "training_authorized",
        "blind_test_authorized",
        "production_deployment_authorized",
    ),
    description="recomputed ready",
)
recomputed_authority = recomputed_inventory.get("authority")
if not isinstance(recomputed_authority, dict):
    raise ValueError("recomputed inventory authority is missing")
require_false_fields(
    recomputed_authority,
    (
        "raw_generation_authorized",
        "validator_authority",
        "training_authorized",
        "blind_test_authorized",
        "production_deployment_authorized",
    ),
    description="recomputed inventory",
)

for name in ("selection_inventory.json", "train_pilot.txt", "validation_pilot.txt"):
    if recomputed_contents[name] != pilot_contents[name]:
        raise ValueError(f"fresh selector output differs byte-for-byte: {name}")

authority_false = {
    "raw_generation_authorized": False,
    "validator_authority": False,
    "judge_authority": False,
    "training_authority": False,
    "blind_test_authority": False,
    "candidate_promotion_authorized": False,
    "production_deployment_authorized": False,
}
evidence = {
    "schema_version": 1,
    "artifact_role": ROLE,
    "audit_contract": AUDIT_CONTRACT,
    "selection_contract": SELECTION_CONTRACT,
    "status": "selection_recomputed_byte_exact",
    "cpu_only": True,
    "seed": seed,
    "quota_per_stratum": {"training": train_quota, "validation": validation_quota},
    "selected_sources": selected_sources,
    "quality_exclusion": expected_inventory_quality,
    "quality_exclusion_dataset_membership_verified_by_selector_replay": True,
    "comparisons": {
        "selection_inventory_json_byte_exact": True,
        "train_pilot_txt_byte_exact": True,
        "validation_pilot_txt_byte_exact": True,
    },
    "bindings": {
        "wrapper_path": wrapper.as_posix(),
        "wrapper_sha256": wrapper_sha,
        "selector_path": builder.as_posix(),
        "pilot_bound_selector_path": bound_selector_raw,
        "selector_sha256": builder_sha,
        "proposal_generator_path": proposal.as_posix(),
        "pilot_bound_proposal_generator_path": bound_proposal_raw,
        "proposal_generator_sha256": proposal_sha,
        "data_path": data_path.resolve(strict=True).as_posix(),
        "data_sha256": data_sha,
        "dataset_dir": dataset_dir.as_posix(),
        "resolved_universe_sha256": universe_sha,
        "quality_exclusion_manifest_path": quality.as_posix(),
        "quality_exclusion_manifest_sha256": quality_sha,
        **quality_digest_bindings,
        **quality_path_bindings,
        "historical_manifest_path": old_manifest.as_posix() if old_manifest else None,
        "historical_manifest_sha256": old_sha,
        "drift_report_path": drift_report.as_posix() if drift_report else None,
        "drift_report_sha256": drift_sha,
        "pilot_artifacts": pilot_hashes,
        "recompute_artifacts": recomputed_hashes,
    },
    "authority": authority_false,
}
evidence_path = audit / "selection_audit_evidence.json"
evidence_content = json_bytes(evidence)
publish_exclusive(evidence_path, evidence_content)

# The external validation stage consumes this marker. Bind the exact six
# recompute artifacts plus the audit evidence; ready itself stays out to avoid
# a hash cycle.
marker_rows = []
for name in SEALED_NAMES:
    path = (recompute / name).resolve(strict=True)
    rendered = path.as_posix()
    if "\n" in rendered or "\r" in rendered:
        raise ValueError("selection audit marker path contains a newline")
    current_sha = stable_sha(path, description=f"marker recompute {name}")
    if current_sha != recomputed_hashes[name]:
        raise RuntimeError(f"recomputed artifact changed before marker: {name}")
    marker_rows.append(f"{current_sha}  {rendered}")
evidence_rendered = evidence_path.resolve(strict=True).as_posix()
if "\n" in evidence_rendered or "\r" in evidence_rendered:
    raise ValueError("selection audit evidence path contains a newline")
evidence_sha = stable_sha(evidence_path, description="selection audit evidence")
if evidence_sha != sha_bytes(evidence_content):
    raise RuntimeError("selection audit evidence changed before marker")
marker_rows.append(f"{evidence_sha}  {evidence_rendered}")
marker_content = ("\n".join(marker_rows) + "\n").encode("utf-8")
marker_path = audit / "selection_audit.sha256"
publish_exclusive(marker_path, marker_content)

for name in SEALED_NAMES:
    if stable_sha(
        recompute / name, description=f"pre-ready recompute {name}"
    ) != recomputed_hashes[name]:
        raise RuntimeError(f"recomputed artifact changed before ready: {name}")
if stable_sha(
    evidence_path, description="pre-ready selection audit evidence"
) != evidence_sha:
    raise RuntimeError("selection audit evidence changed before ready")
if stable_sha(marker_path, description="selection audit marker") != sha_bytes(
    marker_content
):
    raise RuntimeError("selection audit marker changed before ready")
if stable_sha(
    quality, description="pre-ready quality exclusion manifest"
) != quality_sha:
    raise RuntimeError("quality exclusion manifest changed before ready")
rehash_quality_bundle()

ready_out = {
    "schema_version": 1,
    "artifact_role": ROLE,
    "audit_contract": AUDIT_CONTRACT,
    "selection_contract": SELECTION_CONTRACT,
    "status": "selection_audit_ready",
    "cpu_only": True,
    "seed": seed,
    "quota_per_stratum": {"training": train_quota, "validation": validation_quota},
    "selected_sources": selected_sources,
    "quality_exclusion": sanitized_quality,
    "quality_exclusion_dataset_membership_verified_by_selector_replay": True,
    "byte_exact_artifacts": [
        "selection_inventory.json",
        "train_pilot.txt",
        "validation_pilot.txt",
    ],
    "bindings": {
        "selector_sha256": builder_sha,
        "selection_audit_marker_sha256": sha_bytes(marker_content),
        "pilot_artifacts": pilot_hashes,
        "recompute_artifacts": recomputed_hashes,
        "selection_audit_evidence_sha256": evidence_sha,
        "resolved_universe_sha256": universe_sha,
        "quality_exclusion_manifest_path": quality.as_posix(),
        "quality_exclusion_manifest_sha256": quality_sha,
        **quality_digest_bindings,
        **quality_path_bindings,
    },
    **authority_false,
}
# Validate again after publication; the shell failure trap removes ready on mutation.
publish_exclusive(audit / "selection_audit_ready.json", json_bytes(ready_out))
rehash_quality_bundle()
PY
then
  fail "v4 selection audit failed closed" 65
fi

terminal_state=1
printf '%s\n' "v4 selection audit passed as CPU-only diagnostic evidence: $READY"
