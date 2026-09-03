#!/bin/sh
# This case must remain the first executable logic. Direct/manual shell launch
# is unsupported because Bash startup files and inherited tracing can execute
# before this file's first line. The sole authoritative launch is the pinned
# Docker command whose argv starts with absolute /usr/bin/env -i and supplies
# V4_CLEAN_REEXEC=1 to /bin/sh.
case "${V4_CLEAN_REEXEC-}" in
1)
set -eu

# Candidate-only v4 verifier training entrypoint.
#
# The host must create the container first, write a hash-bound docker-inspect
# attestation, and start it with all of these properties:
#   * the global Container tree mounted read-only at /app;
#   * one new run parent mounted read-write at RUN_ROOT (normally /run-root);
#   * network mode none, restart policy no, 8 GiB shm, privileged=false;
#   * exactly the seven NVIDIA device nodes checked below, no DeviceRequests;
#   * the full immutable image ID supplied as CONTAINER_IMAGE_ID.
# Additional QNAP driver/library mounts are allowed only when read-only.  This
# entrypoint never creates containers, changes services, evaluates a candidate,
# promotes a model, deploys to Pi/production, or modifies Spring contracts.

umask 077

# The executable search path and Python process environment are part of the
# image-bound launcher contract, not caller-controlled inputs.
if [ "${PYTHON_BIN+x}" = x ]; then
  printf '%s\n' "PYTHON_BIN override is forbidden" >&2
  exit 64
fi
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS \
  PYTHONBREAKPOINT PYTHONUSERBASE LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT \
  BASH_ENV ENV PS4 BASH_XTRACEFD
PYTHONNOUSERSITE=1
PYTHONHASHSEED=0
export PYTHONNOUSERSITE PYTHONHASHSEED
PYTHON_BIN=/usr/local/bin/python3
APPROVED_TRUSTED_POLICY_SHA256="UNCONFIGURED"
QNAP_SNAPSHOT_ROOT=/dev/shm/v4-qnap-libraries
QNAP_SNAPSHOT_LIBRARY_PATH=$QNAP_SNAPSHOT_ROOT/nvidia/lib:$QNAP_SNAPSHOT_ROOT/cuda/lib64

require_env() {
  case "$1" in
    RUN_ROOT) value=${RUN_ROOT-} ;;
    RUN_DIR) value=${RUN_DIR-} ;;
    GLOBAL_ROOT) value=${GLOBAL_ROOT-} ;;
    CODE_ROOT) value=${CODE_ROOT-} ;;
    AUTHORITY_JSON) value=${AUTHORITY_JSON-} ;;
    AUTHORITY_MARKER) value=${AUTHORITY_MARKER-} ;;
    CODE_INVENTORY) value=${CODE_INVENTORY-} ;;
    TRAINING_CONFIG) value=${TRAINING_CONFIG-} ;;
    HOST_LAUNCH_CONTRACT) value=${HOST_LAUNCH_CONTRACT-} ;;
    PRETRAINED_BACKBONE) value=${PRETRAINED_BACKBONE-} ;;
    CONTAINER_IMAGE_ID) value=${CONTAINER_IMAGE_ID-} ;;
    *) value= ;;
  esac
  if [ -z "$value" ]; then
    printf '%s\n' "missing required environment variable: $1" >&2
    exit 64
  fi
}

for name in \
  RUN_ROOT RUN_DIR CODE_ROOT AUTHORITY_JSON AUTHORITY_MARKER CODE_INVENTORY \
  TRAINING_CONFIG HOST_LAUNCH_CONTRACT PRETRAINED_BACKBONE CONTAINER_IMAGE_ID
do
  require_env "$name"
done

GLOBAL_ROOT=${GLOBAL_ROOT:-/app}
TRAINER=$CODE_ROOT/scripts/train_multitask_verifier.py
WRAPPER=$CODE_ROOT/scripts/nas/run_v4_candidate_training.sh

case "$CONTAINER_IMAGE_ID" in
  sha256:[0-9a-f][0-9a-f]*) ;;
  *) printf '%s\n' "CONTAINER_IMAGE_ID must be a full sha256 image ID" >&2; exit 64 ;;
esac
if [ "${#CONTAINER_IMAGE_ID}" -ne 71 ]; then
  printf '%s\n' "CONTAINER_IMAGE_ID must be a full sha256 image ID" >&2
  exit 64
fi

if [ ! -d "$RUN_ROOT" ] || [ -L "$RUN_ROOT" ]; then
  printf '%s\n' "RUN_ROOT must be an existing non-symlink directory" >&2
  exit 66
fi
if ! "$PYTHON_BIN" - \
  "$RUN_ROOT" "$RUN_DIR" "$GLOBAL_ROOT" "$CODE_ROOT" \
  "$AUTHORITY_JSON" "$AUTHORITY_MARKER" "$CODE_INVENTORY" \
  "$TRAINING_CONFIG" "$HOST_LAUNCH_CONTRACT" "$PRETRAINED_BACKBONE" <<'PY'
import os
import sys
from pathlib import Path

(
    run_root_arg, run_dir_arg, global_root_arg, code_root_arg,
    authority_arg, marker_arg, inventory_arg, config_arg, host_arg, backbone_arg,
) = map(Path, sys.argv[1:])

def reject_symlink_components(path: Path, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path contains a symlink: {cursor}")

for path, description in (
    (run_root_arg, "RUN_ROOT"), (run_dir_arg, "RUN_DIR"),
    (global_root_arg, "GLOBAL_ROOT"), (code_root_arg, "CODE_ROOT"),
    (authority_arg, "AUTHORITY_JSON"), (marker_arg, "AUTHORITY_MARKER"),
    (inventory_arg, "CODE_INVENTORY"), (config_arg, "TRAINING_CONFIG"),
    (host_arg, "HOST_LAUNCH_CONTRACT"),
    (backbone_arg, "PRETRAINED_BACKBONE"),
):
    if not path.is_absolute():
        raise ValueError(f"{description} must be absolute")
    reject_symlink_components(path, description)
run_root = run_root_arg.resolve(strict=True)
global_root = global_root_arg.resolve(strict=True)
code_root = code_root_arg.resolve(strict=True)
if not run_root.is_dir() or not global_root.is_dir() or not code_root.is_dir():
    raise ValueError("RUN_ROOT, GLOBAL_ROOT, and CODE_ROOT must be existing directories")
if any(run_root.iterdir()):
    raise ValueError("RUN_ROOT must be a dedicated empty per-container workspace")
run_dir = Path(os.path.abspath(run_dir_arg))
if run_dir.exists() or run_dir.is_symlink():
    raise FileExistsError("RUN_DIR must not exist")
if run_dir.parent.resolve(strict=True) != run_root:
    raise ValueError("RUN_DIR must be one new direct child of RUN_ROOT")
if global_root == run_root or run_root in global_root.parents or global_root in run_root.parents:
    raise ValueError("GLOBAL_ROOT and RUN_ROOT must be fully disjoint")
try:
    code_root.relative_to(global_root)
except ValueError as error:
    raise ValueError("CODE_ROOT must be beneath GLOBAL_ROOT") from error
for path, description in (
    (authority_arg, "AUTHORITY_JSON"), (marker_arg, "AUTHORITY_MARKER"),
    (inventory_arg, "CODE_INVENTORY"), (config_arg, "TRAINING_CONFIG"),
    (host_arg, "HOST_LAUNCH_CONTRACT"),
    (backbone_arg, "PRETRAINED_BACKBONE"),
):
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{description} must be a regular non-symlink file")
    try:
        resolved.relative_to(global_root)
    except ValueError as error:
        raise ValueError(f"{description} must be beneath GLOBAL_ROOT") from error
PY
then
  printf '%s\n' "RUN_DIR path precheck failed; no directory was created" >&2
  exit 64
fi
if ! command mkdir "$RUN_DIR" 2>/dev/null; then
  printf '%s\n' "refusing to reuse immutable RUN_DIR: $RUN_DIR" >&2
  exit 73
fi

CONTROL=$RUN_DIR/control
CANDIDATE=$RUN_DIR/candidate
if ! command mkdir "$CONTROL"; then
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
    publish_failure "unexpected candidate training exit: code=$code" || true
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

for input in \
  "$TRAINER" "$WRAPPER" "$AUTHORITY_JSON" "$AUTHORITY_MARKER" \
  "$CODE_INVENTORY" "$TRAINING_CONFIG" "$HOST_LAUNCH_CONTRACT" \
  "$PRETRAINED_BACKBONE"
do
  if [ ! -f "$input" ] || [ ! -s "$input" ] || [ -L "$input" ]; then
    fail "missing, empty, or symlink input: $input" 66
  fi
done

QNAP_SNAPSHOT_REPORT=$CONTROL/qnap_library_snapshot.json
# This bootstrap uses only the pinned absolute Python and its standard library
# while LD_LIBRARY_PATH is unset. It binds the policy-approved QNAP library
# inventory, verifies the recursively read-only source mounts, and copies the
# exact bytes without dereferencing links into a new private tmpfs directory.
if ! "$PYTHON_BIN" - \
  "$CODE_ROOT" "$HOST_LAUNCH_CONTRACT" "$CONTAINER_IMAGE_ID" \
  "$APPROVED_TRUSTED_POLICY_SHA256" "$QNAP_SNAPSHOT_REPORT" \
  "$QNAP_SNAPSHOT_ROOT" <<'PY'
import hashlib
import json
import os
import posixpath
import re
import stat
import sys
from pathlib import Path, PurePosixPath

(
    code_root_arg, host_arg, image_id, approved_policy_sha,
    report_arg, snapshot_root_arg,
) = sys.argv[1:]
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_SNAPSHOT_BYTES = 3221225472
TRUSTED_POLICY_RELATIVE_PATH = "configs/v4_candidate_training_trusted_policy.json"
ALLOWED_TREES = {
    "/share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/usr/lib": "/qnap/nvidia/lib",
    "/share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/cuda-12.9/lib64": "/qnap/cuda/lib64",
}
SNAPSHOT_DESTINATIONS = {
    "/qnap/nvidia/lib": "nvidia/lib",
    "/qnap/cuda/lib64": "cuda/lib64",
}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(content: bytes, description: str):
    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error


def stable_bytes(path: Path, description: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file")
    before = path.stat(follow_symlinks=False)
    flags = (
        os.O_RDONLY | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        opened_before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
    )
    if not (
        identity(before) == identity(opened_before)
        == identity(opened_after) == identity(after)
    ):
        raise RuntimeError(f"{description} changed while being read")
    return b"".join(chunks)


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_entries(entries) -> bytes:
    return (
        json.dumps(
            entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def normalized_entry_path(value, description: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{description} must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{description} is not a normalized relative path")
    return value


def validate_inventory(value):
    if not isinstance(value, dict) or set(value) != {
        "schema", "snapshot_max_bytes", "trees", "required_mapped_libraries"
    }:
        raise ValueError("QNAP library inventory root schema mismatch")
    if value.get("schema") != "v4_qnap_library_inventory.v1":
        raise ValueError("QNAP library inventory schema mismatch")
    if type(value.get("snapshot_max_bytes")) is not int or value[
        "snapshot_max_bytes"
    ] != MAX_SNAPSHOT_BYTES:
        raise ValueError("QNAP library snapshot byte limit mismatch")
    trees = value.get("trees")
    if not isinstance(trees, list) or len(trees) != 2:
        raise ValueError("QNAP library inventory must contain exactly two trees")
    if [tree.get("source_root") for tree in trees if isinstance(tree, dict)] != sorted(
        ALLOWED_TREES
    ):
        raise ValueError("QNAP library trees must be source-root sorted")
    seen_sources = set()
    total_bytes = 0
    entries_by_container_root = {}
    for tree in trees:
        if not isinstance(tree, dict) or set(tree) != {
            "source_root", "container_root", "total_regular_bytes",
            "tree_sha256", "entries",
        }:
            raise ValueError("QNAP library tree schema mismatch")
        source = tree.get("source_root")
        container = tree.get("container_root")
        if source in seen_sources or ALLOWED_TREES.get(source) != container:
            raise ValueError("QNAP library tree root pair is not approved")
        seen_sources.add(source)
        entries = tree.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("QNAP library tree entries must be nonempty")
        paths = []
        by_path = {}
        regular_bytes = 0
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") not in {
                "file", "symlink"
            }:
                raise ValueError("QNAP inventory permits only file/symlink entries")
            relative = normalized_entry_path(
                entry.get("path"), "QNAP library entry path"
            )
            if relative in by_path:
                raise ValueError("duplicate QNAP library inventory path")
            if entry["type"] == "file":
                if set(entry) != {"path", "type", "size", "sha256"}:
                    raise ValueError("QNAP regular-file entry schema mismatch")
                if type(entry.get("size")) is not int or entry["size"] < 0:
                    raise ValueError("QNAP regular-file size is invalid")
                if not isinstance(entry.get("sha256"), str) or not SHA_RE.fullmatch(
                    entry["sha256"]
                ):
                    raise ValueError("QNAP regular-file SHA is invalid")
                regular_bytes += entry["size"]
            else:
                if set(entry) != {"path", "type", "target"}:
                    raise ValueError("QNAP symlink entry schema mismatch")
                target = entry.get("target")
                if (
                    not isinstance(target, str) or not target or "\\" in target
                    or target.startswith("/") or posixpath.normpath(target) != target
                ):
                    raise ValueError("QNAP symlink target must be normalized and relative")
                resolved_target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(relative), target)
                )
                if resolved_target == ".." or resolved_target.startswith("../"):
                    raise ValueError("QNAP symlink target escapes its library tree")
            paths.append(relative)
            by_path[relative] = entry
        if paths != sorted(paths):
            raise ValueError("QNAP library inventory entries must be path-sorted")
        for relative, entry in by_path.items():
            if entry["type"] != "symlink":
                continue
            target = posixpath.normpath(
                posixpath.join(posixpath.dirname(relative), entry["target"])
            )
            visited = {relative}
            while True:
                target_entry = by_path.get(target)
                if target_entry is None:
                    raise ValueError("QNAP symlink points to an unregistered entry")
                if target in visited:
                    raise ValueError("QNAP symlink cycle is forbidden")
                visited.add(target)
                if target_entry["type"] == "file":
                    break
                target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(target), target_entry["target"])
                )
                if target == ".." or target.startswith("../"):
                    raise ValueError("QNAP symlink chain escapes its library tree")
        if tree.get("total_regular_bytes") != regular_bytes:
            raise ValueError("QNAP library tree total byte count mismatch")
        if not isinstance(tree.get("tree_sha256"), str) or tree[
            "tree_sha256"
        ] != sha(canonical_entries(entries)):
            raise ValueError("QNAP library tree inventory SHA mismatch")
        total_bytes += regular_bytes
        entries_by_container_root[container] = by_path
    if seen_sources != set(ALLOWED_TREES):
        raise ValueError("QNAP library inventory roots are incomplete")
    if total_bytes > MAX_SNAPSHOT_BYTES:
        raise ValueError("QNAP library inventory exceeds the snapshot byte limit")
    required = value.get("required_mapped_libraries")
    if not isinstance(required, list) or not required:
        raise ValueError("QNAP required mapped library list must be nonempty")
    normalized_required = []
    for item in required:
        if not isinstance(item, dict) or set(item) != {"container_root", "path"}:
            raise ValueError("QNAP required mapped library schema mismatch")
        container_root = item.get("container_root")
        if container_root not in entries_by_container_root:
            raise ValueError("QNAP required mapped library root is not approved")
        relative = normalized_entry_path(
            item.get("path"), "QNAP required mapped library path"
        )
        entries = entries_by_container_root[container_root]
        cursor = relative
        visited = set()
        while True:
            entry = entries.get(cursor)
            if entry is None:
                raise ValueError("QNAP required mapped library is not inventoried")
            if cursor in visited:
                raise ValueError("QNAP required mapped library resolves through a cycle")
            visited.add(cursor)
            if entry["type"] == "file":
                break
            cursor = posixpath.normpath(
                posixpath.join(posixpath.dirname(cursor), entry["target"])
            )
        normalized_required.append((container_root, relative))
    if normalized_required != sorted(normalized_required) or len(
        normalized_required
    ) != len(set(normalized_required)):
        raise ValueError("QNAP required mapped libraries must be unique and sorted")
    if not any(
        root == "/qnap/nvidia/lib" and posixpath.basename(path) == "libcuda.so.1"
        for root, path in normalized_required
    ):
        raise ValueError("QNAP required mapped libraries must include libcuda.so.1")
    if {root for root, _ in normalized_required} != set(ALLOWED_TREES.values()):
        raise ValueError(
            "QNAP required mapped libraries must include at least one library from each tree"
        )
    return total_bytes


def expected_directories(entries):
    result = set()
    for entry in entries:
        parent = posixpath.dirname(entry["path"])
        while parent:
            result.add(parent)
            parent = posixpath.dirname(parent)
    return result


def stable_file_contract(path: Path, description: str):
    before = path.stat(follow_symlinks=False)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    size = 0
    try:
        opened_before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
    )
    if not (
        identity(before) == identity(opened_before)
        == identity(opened_after) == identity(after)
    ):
        raise RuntimeError(f"{description} changed while being hashed")
    return size, digest.hexdigest()


def enumerate_tree(root: Path):
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"QNAP library root must be a directory: {root}")
    entries = []
    directories = set()

    def scan(directory: Path, relative_parent: str) -> None:
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name)
        for child in children:
            relative = posixpath.join(relative_parent, child.name) if relative_parent else child.name
            normalized_entry_path(relative, "live QNAP library path")
            value = child.stat(follow_symlinks=False)
            mode = value.st_mode
            path = directory / child.name
            if stat.S_ISDIR(mode):
                directories.add(relative)
                scan(path, relative)
            elif stat.S_ISREG(mode):
                size, digest = stable_file_contract(
                    path, f"QNAP library {relative}"
                )
                entries.append({
                    "path": relative, "type": "file", "size": size,
                    "sha256": digest,
                })
            elif stat.S_ISLNK(mode):
                entries.append({
                    "path": relative, "type": "symlink", "target": os.readlink(path),
                })
            else:
                raise ValueError(f"special QNAP library entry is forbidden: {relative}")

    scan(root, "")
    entries.sort(key=lambda row: row["path"])
    return entries, directories


def unescape_mount(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value
    )


def mount_contracts():
    result = {}
    for line in Path("/proc/self/mountinfo").read_text(
        encoding="utf-8", errors="strict"
    ).splitlines():
        left, right = line.split(" - ", 1)
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 6 or len(right_fields) < 1:
            raise ValueError("malformed /proc/self/mountinfo")
        result[unescape_mount(left_fields[4])] = {
            "modes": set(left_fields[5].split(",")),
            "filesystem": right_fields[0],
        }
    return result


def require_recursive_read_only(mounts, root: str) -> None:
    prefix = root.rstrip("/") + "/"
    relevant = {
        path: value for path, value in mounts.items()
        if path == root or path.startswith(prefix)
    }
    if root not in relevant:
        raise ValueError(f"live QNAP library mount is missing: {root}")
    writable = sorted(
        path for path, value in relevant.items()
        if "ro" not in value["modes"] or "rw" in value["modes"]
    )
    if writable:
        raise ValueError(f"QNAP library tree contains writable mounts: {writable}")


def copy_regular(source: Path, destination: Path, expected) -> None:
    source_flags = (
        os.O_RDONLY | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    destination_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_fd = os.open(source, source_flags)
    destination_fd = os.open(destination, destination_flags, 0o600)
    digest = hashlib.sha256()
    size = 0
    try:
        source_before = os.fstat(source_fd)
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
        source_after = os.fstat(source_fd)
        if (
            source_before.st_dev, source_before.st_ino, source_before.st_size,
            source_before.st_mtime_ns,
        ) != (
            source_after.st_dev, source_after.st_ino, source_after.st_size,
            source_after.st_mtime_ns,
        ):
            raise RuntimeError(f"QNAP library changed during snapshot: {source}")
        if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"QNAP library bytes differ from inventory: {source}")
        os.fchmod(destination_fd, 0o444)
    finally:
        os.close(source_fd)
        os.close(destination_fd)


code_root = Path(code_root_arg).resolve(strict=True)
host_path = Path(host_arg).resolve(strict=True)
report_path = Path(report_arg)
snapshot_root = Path(snapshot_root_arg)
snapshot_root_contract = "/dev/shm/v4-qnap-libraries"
if not SHA_RE.fullmatch(approved_policy_sha):
    raise ValueError("approved trusted policy SHA256 is unconfigured or invalid")
policy_path = (code_root / TRUSTED_POLICY_RELATIVE_PATH).resolve(strict=True)
policy_content = stable_bytes(policy_path, "trusted policy")
if sha(policy_content) != approved_policy_sha:
    raise ValueError(
        "trusted policy bytes differ from the fixed launcher pin: "
        f"expected={approved_policy_sha} actual={sha(policy_content)}"
    )
policy = load_json(policy_content, "trusted policy JSON")
host_content = stable_bytes(host_path, "host launch contract")
if not isinstance(policy, dict) or policy.get(
    "host_launch_contract_sha256"
) != sha(host_content):
    raise ValueError("trusted policy does not bind the host launch contract")
host = load_json(host_content, "host launch contract JSON")
if not isinstance(host, dict) or host.get("container_image_id") != image_id:
    raise ValueError("host launch contract image ID mismatch")
inventory = host.get("qnap_library_inventory")
total_bytes = validate_inventory(inventory)

report = {
    "schema": "v4_qnap_library_snapshot.v1",
    "status": "not_applicable_non_linux_test_host",
    "snapshot_root": snapshot_root_contract,
    "snapshot_dev": None,
    "snapshot_ino": None,
    "snapshot_mode": None,
    "source_mounts_recursively_read_only": False,
    "snapshot_max_bytes": MAX_SNAPSHOT_BYTES,
    "total_regular_bytes": total_bytes,
    "inventory": inventory,
    "trees": [],
}
if sys.platform.startswith("linux"):
    if snapshot_root != Path(snapshot_root_contract):
        raise ValueError("QNAP snapshot root is not the fixed tmpfs direct child")
    mounts = mount_contracts()
    shm_contract = mounts.get("/dev/shm")
    if shm_contract is None or shm_contract["filesystem"] != "tmpfs":
        raise ValueError("/dev/shm is not an exact tmpfs mount")
    if "rw" not in shm_contract["modes"] or "ro" in shm_contract["modes"]:
        raise ValueError("/dev/shm must be writable while creating the snapshot")
    unexpected_shm_submounts = sorted(
        path for path in mounts
        if path.startswith("/dev/shm/")
    )
    if unexpected_shm_submounts:
        raise ValueError(
            f"/dev/shm contains unexpected nested mounts: {unexpected_shm_submounts}"
        )
    shm_capacity = os.statvfs("/dev/shm")
    if shm_capacity.f_frsize * shm_capacity.f_bavail < total_bytes:
        raise ValueError("/dev/shm lacks free space for the policy-bound QNAP snapshot")
    for tree in inventory["trees"]:
        require_recursive_read_only(mounts, tree["container_root"])
        live_entries, live_directories = enumerate_tree(Path(tree["container_root"]))
        if live_entries != tree["entries"]:
            raise ValueError(
                f"live QNAP library bytes/tree differ from inventory: {tree['container_root']}"
            )
        if live_directories != expected_directories(tree["entries"]):
            raise ValueError(
                f"live QNAP library directory set differs from inventory: {tree['container_root']}"
            )
    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise FileExistsError("QNAP library snapshot root must not already exist")
    if snapshot_root.parent.resolve(strict=True) != Path("/dev/shm"):
        raise ValueError("QNAP snapshot parent differs from /dev/shm")
    os.mkdir(snapshot_root, 0o700)
    snapshot_stat = snapshot_root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(snapshot_stat.st_mode):
        raise ValueError("QNAP snapshot root is not a directory")
    if snapshot_stat.st_dev != Path("/dev/shm").stat().st_dev:
        raise ValueError("QNAP snapshot root is not on /dev/shm tmpfs")
    for tree in inventory["trees"]:
        destination_root = snapshot_root / SNAPSHOT_DESTINATIONS[tree["container_root"]]
        directories = expected_directories(tree["entries"])
        for relative in sorted(directories, key=lambda value: (value.count("/"), value)):
            (destination_root / relative).mkdir(parents=True, exist_ok=True, mode=0o700)
        destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        source_root = Path(tree["container_root"])
        for entry in tree["entries"]:
            if entry["type"] == "file":
                copy_regular(
                    source_root / entry["path"],
                    destination_root / entry["path"], entry,
                )
        for entry in tree["entries"]:
            if entry["type"] == "symlink":
                os.symlink(entry["target"], destination_root / entry["path"])
        copied_entries, copied_directories = enumerate_tree(destination_root)
        if copied_entries != tree["entries"] or copied_directories != directories:
            raise ValueError("private QNAP library snapshot differs from inventory")
        source_entries_after, source_directories_after = enumerate_tree(source_root)
        if (
            source_entries_after != tree["entries"]
            or source_directories_after != directories
        ):
            raise RuntimeError("QNAP source library tree changed during snapshot creation")
        report["trees"].append({
            "source_root": tree["source_root"],
            "container_root": tree["container_root"],
            "snapshot_root": destination_root.as_posix(),
            "tree_sha256": tree["tree_sha256"],
            "total_regular_bytes": tree["total_regular_bytes"],
        })
    all_directories = sorted(
        (path for path in snapshot_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts), reverse=True,
    )
    for directory in all_directories:
        os.chmod(directory, 0o555, follow_symlinks=False)
    os.chmod(snapshot_root, 0o555, follow_symlinks=False)
    snapshot_stat = snapshot_root.stat(follow_symlinks=False)
    report.update({
        "status": "qnap_library_snapshot_verified",
        "snapshot_dev": snapshot_stat.st_dev,
        "snapshot_ino": snapshot_stat.st_ino,
        "snapshot_mode": stat.S_IMODE(snapshot_stat.st_mode),
        "source_mounts_recursively_read_only": True,
    })

if report_path.exists() or report_path.is_symlink():
    raise FileExistsError(report_path)
content = (
    json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    + "\n"
).encode("utf-8")
temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
with temporary.open("xb") as handle:
    handle.write(content)
    handle.flush()
    os.fsync(handle.fileno())
os.link(temporary, report_path)
temporary.unlink()
PY
then
  fail "QNAP library inventory/snapshot bootstrap failed" 65
fi
LD_LIBRARY_PATH=$QNAP_SNAPSHOT_LIBRARY_PATH
export LD_LIBRARY_PATH

verify_qnap_snapshot() {
  "$PYTHON_BIN" - "$QNAP_SNAPSHOT_REPORT" "$QNAP_SNAPSHOT_ROOT" <<'PY'
import hashlib
import json
import os
import posixpath
import stat
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
snapshot_root = Path(sys.argv[2])
snapshot_root_contract = "/dev/shm/v4-qnap-libraries"
if report_path.is_symlink() or not report_path.is_file():
    raise ValueError("QNAP snapshot report must remain a regular file")
report = json.loads(report_path.read_text(encoding="utf-8"))
if set(report) != {
    "schema", "status", "snapshot_root", "snapshot_dev", "snapshot_ino",
    "snapshot_mode", "source_mounts_recursively_read_only",
    "snapshot_max_bytes", "total_regular_bytes", "inventory", "trees",
}:
    raise ValueError("QNAP snapshot report schema changed")
if report.get("schema") != "v4_qnap_library_snapshot.v1":
    raise ValueError("QNAP snapshot report version changed")
if report.get("snapshot_root") != snapshot_root_contract:
    raise ValueError("QNAP snapshot report root changed")
if not sys.platform.startswith("linux"):
    if report.get("status") != "not_applicable_non_linux_test_host":
        raise ValueError("non-Linux QNAP snapshot report status mismatch")
    raise SystemExit(0)
if snapshot_root != Path(snapshot_root_contract):
    raise ValueError("QNAP snapshot root argument is not fixed")
if report.get("status") != "qnap_library_snapshot_verified":
    raise ValueError("Linux QNAP library snapshot is not verified")
if report.get("source_mounts_recursively_read_only") is not True:
    raise ValueError("QNAP source mount read-only evidence is missing")
if snapshot_root != Path("/dev/shm/v4-qnap-libraries"):
    raise ValueError("QNAP snapshot root is not fixed")
if snapshot_root.parent.resolve(strict=True) != Path("/dev/shm"):
    raise ValueError("QNAP snapshot is not a direct child of /dev/shm")
root_stat = snapshot_root.stat(follow_symlinks=False)
if not stat.S_ISDIR(root_stat.st_mode) or snapshot_root.is_symlink():
    raise ValueError("QNAP snapshot root identity changed")
if (
    root_stat.st_dev, root_stat.st_ino, stat.S_IMODE(root_stat.st_mode)
) != (
    report.get("snapshot_dev"), report.get("snapshot_ino"), report.get("snapshot_mode")
):
    raise ValueError("QNAP snapshot root inode/mode changed")
if stat.S_IMODE(root_stat.st_mode) != 0o555:
    raise ValueError("QNAP snapshot root is not read-only")
if root_stat.st_dev != Path("/dev/shm").stat().st_dev:
    raise ValueError("QNAP snapshot moved off tmpfs")

inventory = report.get("inventory")
if not isinstance(inventory, dict) or report.get(
    "snapshot_max_bytes"
) != inventory.get("snapshot_max_bytes"):
    raise ValueError("QNAP snapshot inventory binding changed")
destinations = {
    "/qnap/nvidia/lib": "nvidia/lib",
    "/qnap/cuda/lib64": "cuda/lib64",
}

def stable_file_contract(path: Path):
    before = path.stat(follow_symlinks=False)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    size = 0
    try:
        opened_before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
    )
    if not (
        identity(before) == identity(opened_before)
        == identity(opened_after) == identity(after)
    ):
        raise RuntimeError(f"snapshot file changed while read: {path}")
    return size, digest.hexdigest()


def enumerate_tree(root: Path):
    entries = []
    directories = set()

    def scan(directory: Path, parent: str):
        if stat.S_IMODE(directory.stat(follow_symlinks=False).st_mode) != 0o555:
            raise ValueError(f"QNAP snapshot directory is writable: {directory}")
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name)
        for child in children:
            relative = posixpath.join(parent, child.name) if parent else child.name
            value = child.stat(follow_symlinks=False)
            path = directory / child.name
            if stat.S_ISDIR(value.st_mode):
                directories.add(relative)
                scan(path, relative)
            elif stat.S_ISREG(value.st_mode):
                if stat.S_IMODE(value.st_mode) != 0o444:
                    raise ValueError(f"QNAP snapshot file is not mode 0444: {relative}")
                size, digest = stable_file_contract(path)
                entries.append({
                    "path": relative, "type": "file", "size": size,
                    "sha256": digest,
                })
            elif stat.S_ISLNK(value.st_mode):
                entries.append({
                    "path": relative, "type": "symlink", "target": os.readlink(path),
                })
            else:
                raise ValueError(f"special QNAP snapshot entry is forbidden: {relative}")

    scan(root, "")
    entries.sort(key=lambda row: row["path"])
    return entries, directories


def expected_directories(entries):
    result = set()
    for entry in entries:
        parent = posixpath.dirname(entry["path"])
        while parent:
            result.add(parent)
            parent = posixpath.dirname(parent)
    return result


expected_tree_reports = []
for tree in inventory.get("trees", []):
    destination = snapshot_root / destinations[tree["container_root"]]
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("QNAP snapshot tree is missing")
    entries, directories = enumerate_tree(destination)
    if entries != tree["entries"] or directories != expected_directories(tree["entries"]):
        raise ValueError("QNAP snapshot tree bytes changed")
    expected_tree_reports.append({
        "source_root": tree["source_root"],
        "container_root": tree["container_root"],
        "snapshot_root": destination.as_posix(),
        "tree_sha256": tree["tree_sha256"],
        "total_regular_bytes": tree["total_regular_bytes"],
    })
if report.get("trees") != expected_tree_reports:
    raise ValueError("QNAP snapshot tree report changed")
PY
}

PREFLIGHT=$CONTROL/preflight.json
INPUT_MARKER=$CONTROL/inputs.sha256
if ! "$PYTHON_BIN" - \
  "$RUN_ROOT" "$RUN_DIR" "$GLOBAL_ROOT" "$CODE_ROOT" "$AUTHORITY_JSON" "$AUTHORITY_MARKER" \
  "$CODE_INVENTORY" "$TRAINING_CONFIG" "$HOST_LAUNCH_CONTRACT" \
  "$PRETRAINED_BACKBONE" "$CONTAINER_IMAGE_ID" "$TRAINER" "$WRAPPER" \
  "$PREFLIGHT" "$INPUT_MARKER" "$APPROVED_TRUSTED_POLICY_SHA256" \
  "$QNAP_SNAPSHOT_REPORT" <<'PY'
import ast
import csv
import hashlib
import io
import json
import math
import os
import posixpath
import re
import socket
import stat
import sys
import types
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import onnx
import onnxruntime as ort
import torch
import torchvision
from torchvision import models

(
    run_root_arg, run_dir_arg, global_root_arg, code_root_arg, authority_arg, marker_arg,
    inventory_arg, config_arg, host_arg, backbone_arg, image_id,
    trainer_arg, wrapper_arg, preflight_arg, input_marker_arg,
    approved_policy_sha_arg, qnap_snapshot_report_arg,
) = sys.argv[1:]

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
TRUSTED_POLICY_RELATIVE_PATH = "configs/v4_candidate_training_trusted_policy.json"
TRUST_ROOT_CODE_PATHS = {
    "configs/v4_candidate_training_trusted_policy.json",
    "scripts/build_v4_candidate_training_authority.py",
    "scripts/nas/run_v4_candidate_training.sh",
}
EXPECTED_DEVICES = sorted((
    "/dev/nvidia0",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
    "/dev/nvidia-caps/nvidia-cap1",
    "/dev/nvidia-caps/nvidia-cap2",
))
REQUIRED_MANIFEST_FIELDS = {
    "filepath", "split", "source_id", "material", "category", "dent",
    "label", "foreign_material", "source_object_count", "sample_id", "role",
    "fold", "source_sha256", "image_sha256", "object_group",
    "capture_session", "origin", "source_filepath",
    "crop_object_count", "captured_at", "auditor_sha256",
    "teacher_output_sha256", "localizer_output_sha256",
}
FORBIDDEN_DIAGNOSTIC_TOKENS = (
    "diagnostic", "runtime_replay", "repro_pilot", "repro_selection",
    "repro_validation", "qx1", "qx2", "qx3",
)
REQUIRED_CONTAINER_ENV = {
    "RUN_ROOT", "RUN_DIR", "GLOBAL_ROOT", "CODE_ROOT", "AUTHORITY_JSON",
    "AUTHORITY_MARKER", "CODE_INVENTORY", "TRAINING_CONFIG",
    "HOST_LAUNCH_CONTRACT", "PRETRAINED_BACKBONE", "CONTAINER_IMAGE_ID",
}
CLEAN_CONTAINER_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
FORBIDDEN_CONTAINER_ENV = {
    "PYTHON_BIN", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
    "PYTHONINSPECT", "PYTHONWARNINGS", "PYTHONBREAKPOINT",
    "PYTHONUSERBASE", "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT",
    "BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "PS4", "BASH_XTRACEFD",
}
BASH_EXPORTED_FUNCTION_PREFIX = "BASH_FUNC_"
ALLOWED_QNAP_LIBRARY_MOUNTS = {
    ("/share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/usr/lib", "/qnap/nvidia/lib"),
    (
        "/share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/cuda-12.9/lib64",
        "/qnap/cuda/lib64",
    ),
}
POLICY_BINDING_FIELDS = {
    "qx3_diagnostic_ready_sha256", "qx3_diagnostic_report_sha256",
    "license_allowlist_sha256", "quality_exclusions_sha256",
    "protected_sources_sha256", "code_inventory_sha256",
    "training_config_sha256", "host_launch_contract_sha256",
    "raw_container_inspect_sha256", "pretrained_backbone_sha256",
    "container_image_id", "candidate_train_manifest_sha256",
    "candidate_model_validation_manifest_sha256",
    "candidate_dataset_snapshot_sha256",
    "candidate_dataset_consumption_contract_sha256",
    "candidate_near_duplicate_audit_sha256",
    "protected_reference_inventory_sha256",
}
AUTHORITY_BINDING_FIELDS = {
    "source_manifest_sha256", "full_data_validator_report_sha256",
    "trusted_policy_sha256", "dataset_content_inventory_sha256",
    "dataset_snapshot_publish_receipt_sha256",
    "dataset_consumption_contract_sha256",
    *POLICY_BINDING_FIELDS,
}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_bytes(content: bytes, description: str):
    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error


def reject_symlink_components(path: Path, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path contains a symlink: {cursor}")


def stable_bytes(path: Path, description: str) -> bytes:
    reject_symlink_components(path, description)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")
    before = path.stat()
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        content = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = path.stat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
    )
    if not (
        identity(before) == identity(opened_before) ==
        identity(opened_after) == identity(after)
    ):
        raise RuntimeError(f"{description} changed while being read: {path}")
    return content


def sha_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compact_json_value(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def compact_json(value) -> bytes:
    return compact_json_value(value) + b"\n"


def resolved_file(path: str, description: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError(f"{description} path must be absolute")
    reject_symlink_components(value, description)
    return value.resolve(strict=True)


def require_sha(value, description: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def require_exact_bool(value, expected: bool, description: str) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{description} must be {expected!r}")


run_root = Path(run_root_arg).resolve(strict=True)
run_dir = Path(run_dir_arg).resolve(strict=True)
global_root = Path(global_root_arg).resolve(strict=True)
code_root = Path(code_root_arg).resolve(strict=True)
authority_path = resolved_file(authority_arg, "training authority")
marker_path = resolved_file(marker_arg, "training authority marker")
inventory_path = resolved_file(inventory_arg, "code inventory")
config_path = resolved_file(config_arg, "training config")
host_path = resolved_file(host_arg, "host launch contract")
backbone_path = resolved_file(backbone_arg, "pretrained backbone")
trainer_path = resolved_file(trainer_arg, "trainer")
wrapper_path = resolved_file(wrapper_arg, "wrapper")
preflight_path = Path(preflight_arg)
input_marker_path = Path(input_marker_arg)
qnap_snapshot_report_path = resolved_file(
    qnap_snapshot_report_arg, "QNAP library snapshot report"
)

if run_dir.parent != run_root:
    raise ValueError("RUN_DIR must be one new direct child of RUN_ROOT")
if global_root == run_root or run_root in global_root.parents or global_root in run_root.parents:
    raise ValueError("GLOBAL_ROOT and RUN_ROOT must be fully disjoint")
try:
    code_root.relative_to(global_root)
except ValueError as error:
    raise ValueError("CODE_ROOT must stay beneath GLOBAL_ROOT") from error
for path, description in ((trainer_path, "trainer"), (wrapper_path, "wrapper")):
    try:
        path.relative_to(code_root)
    except ValueError as error:
        raise ValueError(f"{description} must stay beneath CODE_ROOT") from error

input_bytes = {
    authority_path: stable_bytes(authority_path, "training authority"),
    marker_path: stable_bytes(marker_path, "training authority marker"),
    inventory_path: stable_bytes(inventory_path, "code inventory"),
    config_path: stable_bytes(config_path, "training config"),
    host_path: stable_bytes(host_path, "host launch contract"),
    backbone_path: stable_bytes(backbone_path, "pretrained backbone"),
    trainer_path: stable_bytes(trainer_path, "trainer"),
    wrapper_path: stable_bytes(wrapper_path, "wrapper"),
    qnap_snapshot_report_path: stable_bytes(
        qnap_snapshot_report_path, "QNAP library snapshot report"
    ),
}

authority = load_json_bytes(input_bytes[authority_path], "training authority JSON")
if not isinstance(authority, dict):
    raise ValueError("training authority JSON must be an object")
authority_fields = {
    "schema", "artifact_role", "status", "candidate_only",
    "candidate_training_input_authorized", "training_authority",
    "lineage_execution_authorized", "ready_for_lineage_upgrade",
    "diagnostic_only", "production_runtime_modified", "blind_test_authority",
    "candidate_promotion_authorized", "production_deployment_authorized",
    "pi_deployment_authorized", "spring_contract_modified", "local_only",
    "portable", "operational_cutoff_kst", "material_classes",
    "objectness_classes", "condition_heads", "artifacts", "trust_root",
    "dataset_content_inventory", "dataset_snapshot_publish_receipt",
    "dataset_consumption_contract", "near_duplicate_audit",
    "counts", "bindings",
}
if set(authority) != authority_fields:
    raise ValueError("training authority top-level schema mismatch")
exact_authority = {
    "schema": "v4_candidate_training_authority.v3",
    "artifact_role": "v4_candidate_training_input_authority_not_blind_or_deployment",
    "status": "candidate_training_inputs_ready",
}
for field, expected in exact_authority.items():
    if authority.get(field) != expected:
        raise ValueError(f"training authority {field} must be {expected!r}")
for field in (
    "candidate_only", "candidate_training_input_authorized", "training_authority",
    "lineage_execution_authorized", "ready_for_lineage_upgrade",
):
    require_exact_bool(authority.get(field), True, f"training authority {field}")
for field in (
    "diagnostic_only", "production_runtime_modified", "blind_test_authority",
    "candidate_promotion_authorized", "production_deployment_authorized",
    "pi_deployment_authorized", "spring_contract_modified",
):
    require_exact_bool(authority.get(field), False, f"training authority {field}")
require_exact_bool(authority.get("local_only"), True, "training authority local_only")
require_exact_bool(authority.get("portable"), False, "training authority portable")
if authority.get("operational_cutoff_kst") != "2026-08-01T00:00:00+09:00":
    raise ValueError("training authority operational cutoff mismatch")
if authority.get("material_classes") != [
    "can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass",
    "battery", "fluorescent",
]:
    raise ValueError("training authority material class order mismatch")
if authority.get("objectness_classes") != ["background", "material"]:
    raise ValueError("training authority objectness classes mismatch")
if authority.get("condition_heads") != ["dent", "label", "foreign_material"]:
    raise ValueError("training authority condition heads mismatch")
role_text = f"{authority.get('artifact_role', '')} {authority.get('status', '')}".casefold()
if any(token in role_text for token in FORBIDDEN_DIAGNOSTIC_TOKENS):
    raise ValueError("diagnostic authority artifacts cannot authorize training")

artifacts = authority.get("artifacts")
bindings = authority.get("bindings")
if not isinstance(artifacts, dict) or not isinstance(bindings, dict):
    raise ValueError("training authority requires artifacts and bindings objects")
if set(artifacts) != {
    "manifests", "code_inventory", "training_config",
    "host_launch_contract", "pretrained_backbone", "dataset_snapshot_report",
}:
    raise ValueError("training authority artifacts schema mismatch")
if set(bindings) != AUTHORITY_BINDING_FIELDS:
    raise ValueError("training authority bindings schema mismatch")
if not IMAGE_ID_RE.fullmatch(image_id):
    raise ValueError("CONTAINER_IMAGE_ID is not a full image ID")
if bindings.get("container_image_id") != image_id:
    raise ValueError("container image ID differs from training authority")

approved_policy_sha = require_sha(
    approved_policy_sha_arg, "approved trusted policy SHA256"
)
policy_path = resolved_file(
    (code_root / TRUSTED_POLICY_RELATIVE_PATH).as_posix(), "trusted policy"
)
policy_content = stable_bytes(policy_path, "trusted policy")
policy_sha = sha_bytes(policy_content)
if policy_sha != approved_policy_sha or bindings.get("trusted_policy_sha256") != policy_sha:
    raise ValueError("trusted policy SHA differs from the fixed launcher trust root")
policy = load_json_bytes(policy_content, "trusted policy JSON")
if not isinstance(policy, dict):
    raise ValueError("trusted policy JSON must be an object")
expected_trust_root = {
    "method": "git_bundled_code_sha256_pin",
    "repository_relative_policy_path": TRUSTED_POLICY_RELATIVE_PATH,
    "approved_policy_sha256": approved_policy_sha,
    "actual_policy_sha256": policy_sha,
    "verified": True,
}
if authority.get("trust_root") != expected_trust_root:
    raise ValueError("authority trust_root does not exactly match the launcher trust root")
input_bytes[policy_path] = policy_content


def artifact_entry(name: str, supplied: Path, binding_name: str) -> None:
    value = artifacts.get(name)
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"authority artifacts.{name} must contain path and sha256 only")
    declared_path = resolved_file(value.get("path", ""), f"authority {name}")
    if declared_path != supplied:
        raise ValueError(f"authority artifacts.{name}.path mismatch")
    actual = sha_bytes(input_bytes[supplied])
    declared = require_sha(value.get("sha256"), f"authority {name} sha256")
    if declared != actual or bindings.get(binding_name) != actual:
        raise ValueError(f"authority {name} SHA binding mismatch")


artifact_entry("code_inventory", inventory_path, "code_inventory_sha256")
artifact_entry("training_config", config_path, "training_config_sha256")
artifact_entry("host_launch_contract", host_path, "host_launch_contract_sha256")
artifact_entry("pretrained_backbone", backbone_path, "pretrained_backbone_sha256")

snapshot_artifact = artifacts.get("dataset_snapshot_report")
if not isinstance(snapshot_artifact, dict) or set(snapshot_artifact) != {"path", "sha256"}:
    raise ValueError("authority artifacts.dataset_snapshot_report schema mismatch")
snapshot_report_path = resolved_file(
    snapshot_artifact.get("path", ""), "candidate dataset snapshot report"
)
if (
    snapshot_report_path.name != "candidate_dataset_snapshot.json"
    or snapshot_report_path.parent != authority_path.parent
    or snapshot_report_path in input_bytes
):
    raise ValueError("candidate dataset snapshot report path mismatch")
snapshot_report_content = stable_bytes(
    snapshot_report_path, "candidate dataset snapshot report"
)
snapshot_report_sha = sha_bytes(snapshot_report_content)
if (
    require_sha(snapshot_artifact.get("sha256"), "dataset snapshot report SHA")
    != snapshot_report_sha
    or bindings.get("candidate_dataset_snapshot_sha256") != snapshot_report_sha
):
    raise ValueError("candidate dataset snapshot report SHA binding mismatch")
input_bytes[snapshot_report_path] = snapshot_report_content
dataset_snapshot_report = load_json_bytes(
    snapshot_report_content, "candidate dataset snapshot report JSON"
)

manifest_values = artifacts.get("manifests")
if not isinstance(manifest_values, list) or len(manifest_values) != 2:
    raise ValueError("authority artifacts.manifests must contain exactly two entries")
expected_roles = {"train", "model_validation"}
manifest_paths = {}
manifest_hashes = {}
for value in manifest_values:
    if not isinstance(value, dict) or set(value) != {"role", "path", "sha256"}:
        raise ValueError("each manifest authority entry requires role,path,sha256 only")
    role = value.get("role")
    if role not in expected_roles or role in manifest_paths:
        raise ValueError("manifest authority roles must be unique train/model_validation")
    path = resolved_file(value.get("path", ""), f"{role} manifest")
    expected_name = "train_manifest.csv" if role == "train" else "model_validation_manifest.csv"
    if path.name != expected_name:
        raise ValueError(f"{role} manifest filename must be {expected_name}")
    if path in input_bytes:
        raise ValueError("manifest paths must be distinct from other authority artifacts")
    content = stable_bytes(path, f"{role} manifest")
    input_bytes[path] = content
    actual = sha_bytes(content)
    if require_sha(value.get("sha256"), f"{role} manifest sha256") != actual:
        raise ValueError(f"{role} manifest SHA mismatch")
    manifest_paths[role] = path
    manifest_hashes[role] = actual
if set(manifest_paths) != expected_roles or len(set(manifest_paths.values())) != 2:
    raise ValueError("authority must bind two distinct train/model_validation manifests")

# Consume the complete, immutable separation report embedded by the authority
# builder.  pHash is separation evidence only: it cannot delete, relabel,
# select, promote, or deploy any sample.  Any cross-role cluster fails closed.
near_duplicate_audit = authority.get("near_duplicate_audit")
near_duplicate_fields = {
    "schema", "status", "ok", "artifact_role", "authority", "algorithm",
    "bindings", "coverage", "summary", "entries", "edges", "clusters",
}
if not isinstance(near_duplicate_audit, dict) or set(near_duplicate_audit) != near_duplicate_fields:
    raise ValueError("near-duplicate audit top-level schema mismatch")
if near_duplicate_audit.get("schema") != "v4_near_duplicate_leakage_audit.v1":
    raise ValueError("near-duplicate audit schema mismatch")
if near_duplicate_audit.get("status") != "passed" or near_duplicate_audit.get("ok") is not True:
    raise ValueError("near-duplicate audit did not pass")
if near_duplicate_audit.get("artifact_role") != "candidate_dataset_separation_evidence_only":
    raise ValueError("near-duplicate audit artifact role mismatch")
expected_near_duplicate_authority = {
    "candidate_only": True,
    "label_authority": False,
    "blind_authority": False,
    "promotion_authority": False,
    "deployment_authority": False,
    "automatic_delete_or_relabel": False,
}
if near_duplicate_audit.get("authority") != expected_near_duplicate_authority:
    raise ValueError("near-duplicate audit grants forbidden authority")

algorithm = near_duplicate_audit.get("algorithm")
algorithm_fields = {
    "id", "threshold", "decode", "views", "resize", "dct", "bit_rule",
    "byte_cap", "pixel_cap", "graph_edge_cap",
    "exact_right_angle_rotation_invariant",
    "crop_invariant", "runtime",
}
if not isinstance(algorithm, dict) or set(algorithm) != algorithm_fields:
    raise ValueError("near-duplicate algorithm schema mismatch")
expected_algorithm = {
    "id": "oneexpo_phash_rot4_v1",
    "threshold": 4,
    "decode": "verified_bytes_cv2_grayscale_ignore_exif_orientation",
    "views": ["rot0", "rot90", "rot180", "rot270"],
    "resize": {"width": 32, "height": 32, "interpolation": "INTER_AREA"},
    "dct": {"dtype": "float32", "low_frequency_block": [8, 8]},
    "bit_rule": "row_major_msb_first; median(coefficients[1:]); coefficient>median; dc=0",
    "byte_cap": 67108864,
    "pixel_cap": 16000000,
    "graph_edge_cap": 1000000,
    "exact_right_angle_rotation_invariant": True,
    "crop_invariant": False,
}
for field, expected in expected_algorithm.items():
    if algorithm.get(field) != expected:
        raise ValueError(f"near-duplicate algorithm {field} mismatch")
runtime = algorithm.get("runtime")
if not isinstance(runtime, dict) or set(runtime) != {
    "python", "opencv", "numpy", "pillow", "opencv_build_information_sha256",
}:
    raise ValueError("near-duplicate algorithm runtime schema mismatch")
for field in ("python", "opencv", "numpy", "pillow"):
    if not isinstance(runtime.get(field), str) or not runtime[field]:
        raise ValueError(f"near-duplicate runtime {field} is invalid")
require_sha(
    runtime.get("opencv_build_information_sha256"),
    "near-duplicate OpenCV build information SHA",
)

near_bindings = near_duplicate_audit.get("bindings")
if not isinstance(near_bindings, dict) or set(near_bindings) != {
    "candidate_manifest_sha256", "candidate_payload_set_sha256",
    "protected_payload_set_sha256", "protected_sources",
    "protected_inventory", "auditor",
}:
    raise ValueError("near-duplicate audit bindings schema mismatch")
if near_bindings.get("candidate_manifest_sha256") != {
    role: manifest_hashes[role] for role in sorted(expected_roles)
}:
    raise ValueError("near-duplicate audit candidate manifest binding mismatch")
for field in ("candidate_payload_set_sha256", "protected_payload_set_sha256"):
    require_sha(near_bindings.get(field), f"near-duplicate audit {field}")
protected_source_binding = near_bindings.get("protected_sources")
if not isinstance(protected_source_binding, dict) or set(protected_source_binding) != {
    "file_sha256", "payload_sha256", "canonical_union_sha256",
}:
    raise ValueError("near-duplicate protected-sources binding schema mismatch")
for field, digest in protected_source_binding.items():
    require_sha(digest, f"near-duplicate protected_sources.{field}")
protected_inventory_binding = near_bindings.get("protected_inventory")
if not isinstance(protected_inventory_binding, dict) or set(protected_inventory_binding) != {
    "file_sha256", "payload_sha256",
}:
    raise ValueError("near-duplicate protected-inventory binding schema mismatch")
for field, digest in protected_inventory_binding.items():
    require_sha(digest, f"near-duplicate protected_inventory.{field}")
auditor_binding = near_bindings.get("auditor")
if not isinstance(auditor_binding, dict) or set(auditor_binding) != {
    "path", "sha256", "runtime_code_sha256",
}:
    raise ValueError("near-duplicate auditor binding schema mismatch")
if auditor_binding.get("path") != "scripts/audit_v4_near_duplicate_leakage.py":
    raise ValueError("near-duplicate auditor path mismatch")
require_sha(auditor_binding.get("sha256"), "near-duplicate auditor SHA")
require_sha(
    auditor_binding.get("runtime_code_sha256"),
    "near-duplicate auditor runtime code SHA",
)

coverage = near_duplicate_audit.get("coverage")
if not isinstance(coverage, dict) or set(coverage) != {
    "candidate_assets", "protected_assets", "protected_source_union",
    "verified_assets", "complete",
}:
    raise ValueError("near-duplicate coverage schema mismatch")
require_exact_bool(coverage.get("complete"), True, "near-duplicate coverage complete")
for field in (
    "candidate_assets", "protected_assets", "protected_source_union", "verified_assets",
):
    if type(coverage.get(field)) is not int or coverage[field] <= 0:
        raise ValueError(f"near-duplicate coverage {field} must be a positive integer")
if coverage["verified_assets"] != coverage["candidate_assets"] + coverage["protected_assets"]:
    raise ValueError("near-duplicate verified asset count mismatch")
if coverage["protected_assets"] != 2 * coverage["protected_source_union"]:
    raise ValueError("near-duplicate protected union coverage is incomplete")

entries = near_duplicate_audit.get("entries")
edges = near_duplicate_audit.get("edges")
clusters = near_duplicate_audit.get("clusters")
if not isinstance(entries, list) or not isinstance(edges, list) or not isinstance(clusters, list):
    raise ValueError("near-duplicate entries/edges/clusters must be arrays")
if len(entries) != coverage["verified_assets"] or not entries:
    raise ValueError("near-duplicate entry coverage mismatch")
if len(edges) > algorithm["graph_edge_cap"]:
    raise ValueError("near-duplicate supplied edge array exceeds its graph edge cap")
entry_fields = {
    "asset_id", "role", "cohort", "view_kind", "sample_id", "source_sha256",
    "image_sha256", "size", "width", "height", "phash_rot4",
}
asset_ids = set()
asset_entries = {}
phash_signatures = {}
candidate_entry_count = 0
protected_entry_count = 0
protected_source_shas = set()
protected_source_view_counts = Counter()
protected_crop_view_counts = Counter()
candidate_source_view_counts = Counter()
candidate_crop_view_counts = Counter()
candidate_audit_manifest_rows = set()
candidate_payload_shas = set()
protected_payload_shas = set()
previous_asset_id = None
protected_cohorts = {"qx3_diagnostic", "hardware41", "known_audit", "calibration", "blind_test"}
for index, entry in enumerate(entries):
    if not isinstance(entry, dict) or set(entry) != entry_fields:
        raise ValueError(f"near-duplicate entry {index} schema mismatch")
    asset_id = require_sha(entry.get("asset_id"), f"near-duplicate entry {index} asset_id")
    if previous_asset_id is not None and asset_id <= previous_asset_id:
        raise ValueError("near-duplicate entries are not strictly asset-id sorted")
    previous_asset_id = asset_id
    if asset_id in asset_ids:
        raise ValueError("near-duplicate entries contain duplicate asset IDs")
    asset_ids.add(asset_id)
    asset_entries[asset_id] = entry
    role = entry.get("role")
    cohort = entry.get("cohort")
    view_kind = entry.get("view_kind")
    sample_id = entry.get("sample_id")
    source_sha = require_sha(
        entry.get("source_sha256"), f"near-duplicate entry {index} source SHA"
    )
    image_sha = require_sha(
        entry.get("image_sha256"), f"near-duplicate entry {index} image SHA"
    )
    expected_asset_id = sha_bytes(compact_json_value({
        "cohort": cohort,
        "image_sha256": image_sha,
        "role": role,
        "sample_id": sample_id,
        "source_sha256": source_sha,
        "view_kind": view_kind,
    }))
    if asset_id != expected_asset_id:
        raise ValueError("near-duplicate entry asset ID is not deterministic")
    if cohort == "candidate":
        if role not in expected_roles:
            raise ValueError("near-duplicate candidate entry role mismatch")
        candidate_entry_count += 1
        candidate_payload_shas.add(image_sha)
        if view_kind == "source":
            if sample_id != f"source:{role}:{source_sha}":
                raise ValueError("near-duplicate candidate source sample ID mismatch")
            candidate_source_view_counts[(role, source_sha)] += 1
        elif view_kind == "crop":
            candidate_crop_view_counts[(role, source_sha)] += 1
            candidate_audit_manifest_rows.add(
                (role, sample_id, source_sha, image_sha)
            )
    else:
        if cohort not in protected_cohorts or role != cohort:
            raise ValueError("near-duplicate protected entry role/cohort mismatch")
        protected_entry_count += 1
        protected_source_shas.add(source_sha)
        protected_payload_shas.add(image_sha)
        if view_kind == "source":
            protected_source_view_counts[source_sha] += 1
        elif view_kind == "crop":
            protected_crop_view_counts[source_sha] += 1
    if view_kind not in {"source", "crop"}:
        raise ValueError("near-duplicate entry view_kind mismatch")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("near-duplicate entry sample_id is invalid")
    if view_kind == "source" and image_sha != source_sha:
        raise ValueError("near-duplicate source-view image/source SHA mismatch")
    for field in ("size", "width", "height"):
        if type(entry.get(field)) is not int or entry[field] <= 0:
            raise ValueError(f"near-duplicate entry {index} {field} is invalid")
    phash = entry.get("phash_rot4")
    if not isinstance(phash, list) or len(phash) != 4 or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{16}", value) is None
        for value in phash
    ):
        raise ValueError("near-duplicate entry pHash signature mismatch")
    phash_signatures[asset_id] = tuple(int(value, 16) for value in phash)
if candidate_entry_count != coverage["candidate_assets"]:
    raise ValueError("near-duplicate candidate asset coverage mismatch")
if protected_entry_count != coverage["protected_assets"]:
    raise ValueError("near-duplicate protected asset coverage mismatch")
if len(protected_source_shas) != coverage["protected_source_union"]:
    raise ValueError("near-duplicate protected source union count mismatch")
candidate_role_sources = {
    (entry["role"], entry["source_sha256"])
    for entry in entries if entry["cohort"] == "candidate"
}
if set(candidate_source_view_counts) != candidate_role_sources or any(
    count != 1 for count in candidate_source_view_counts.values()
):
    raise ValueError("near-duplicate candidate source-view coverage mismatch")
if candidate_crop_view_counts != candidate_source_view_counts:
    raise ValueError("near-duplicate candidate crop-view coverage mismatch")
if set(protected_source_view_counts) != protected_source_shas or any(
    count != 1 for count in protected_source_view_counts.values()
):
    raise ValueError("near-duplicate protected source-view coverage mismatch")
if set(protected_crop_view_counts) != protected_source_shas or any(
    count != 1 for count in protected_crop_view_counts.values()
):
    raise ValueError("near-duplicate protected crop-view coverage mismatch")
if near_bindings["candidate_payload_set_sha256"] != sha_bytes(
    compact_json_value(sorted(candidate_payload_shas))
):
    raise ValueError("near-duplicate candidate payload-set binding mismatch")
if near_bindings["protected_payload_set_sha256"] != sha_bytes(
    compact_json_value(sorted(protected_payload_shas))
):
    raise ValueError("near-duplicate protected payload-set binding mismatch")
if protected_source_binding["canonical_union_sha256"] != sha_bytes(
    compact_json_value(sorted(protected_source_shas))
):
    raise ValueError("near-duplicate protected canonical union binding mismatch")

edge_fields = {
    "left_asset_id", "right_asset_id", "distance", "evidence", "blocking",
}


def phash_bucket_keys(value, threshold):
    widths = [64 // (threshold + 1)] * (threshold + 1)
    for index in range(64 % (threshold + 1)):
        widths[index] += 1
    keys = []
    offset = 0
    for index, width in enumerate(widths):
        keys.append((index, (value >> offset) & ((1 << width) - 1)))
        offset += width
    return tuple(keys)


def phash_distance(left, right):
    return min((a ^ b).bit_count() for a in left for b in right)


# Reconstruct the complete edge set from the reported signatures and exact
# identities.  Merely checking supplied edges would let an omitted cross-role
# edge masquerade as two harmless singleton clusters.
expected_edges = {}
phash_buckets = {}
for current_id in sorted(asset_ids):
    signature = tuple(sorted(set(phash_signatures[current_id])))
    candidates = set()
    for value in signature:
        for key in phash_bucket_keys(value, algorithm["threshold"]):
            candidates.update(phash_buckets.get(key, ()))
    for other_id in sorted(candidates):
        distance = phash_distance(phash_signatures[other_id], signature)
        if distance <= algorithm["threshold"]:
            pair = (other_id, current_id)
            expected_edges.setdefault(pair, {"evidence": set(), "distance": distance})[
                "evidence"
            ].add("perceptual_hash")
            if len(expected_edges) > algorithm["graph_edge_cap"]:
                raise ValueError("near-duplicate audit exceeds its graph edge cap")
    for value in signature:
        for key in phash_bucket_keys(value, algorithm["threshold"]):
            phash_buckets.setdefault(key, set()).add(current_id)

for identity_field, evidence_name in (
    ("image_sha256", "exact_image_sha256"),
    ("source_sha256", "source_sha256"),
):
    groups = {}
    for asset_id, entry in asset_entries.items():
        groups.setdefault(entry[identity_field], []).append(asset_id)
    for members in groups.values():
        members.sort()
        for position, left in enumerate(members):
            for right in members[position + 1:]:
                edge = expected_edges.setdefault(
                    (left, right), {"evidence": set(), "distance": None}
                )
                if len(expected_edges) > algorithm["graph_edge_cap"]:
                    raise ValueError("near-duplicate audit exceeds its graph edge cap")
                edge["evidence"].add(evidence_name)
                if evidence_name == "exact_image_sha256":
                    edge["distance"] = 0

if len(expected_edges) > algorithm["graph_edge_cap"]:
    raise ValueError("near-duplicate audit exceeds its graph edge cap")
for (left, right), expected in expected_edges.items():
    if asset_entries[left]["role"] != asset_entries[right]["role"]:
        raise ValueError("near-duplicate audit omitted a forbidden cross-role edge")

edge_pairs = set()
actual_edges = {}
for index, edge in enumerate(edges):
    if not isinstance(edge, dict) or set(edge) != edge_fields:
        raise ValueError(f"near-duplicate edge {index} schema mismatch")
    left = require_sha(edge.get("left_asset_id"), f"near-duplicate edge {index} left")
    right = require_sha(edge.get("right_asset_id"), f"near-duplicate edge {index} right")
    if left >= right or left not in asset_ids or right not in asset_ids:
        raise ValueError("near-duplicate edge endpoints are invalid")
    pair = (left, right)
    if pair in edge_pairs:
        raise ValueError("near-duplicate audit contains duplicate edges")
    edge_pairs.add(pair)
    left_entry = asset_entries[left]
    right_entry = asset_entries[right]
    if left_entry["role"] != right_entry["role"]:
        raise ValueError("near-duplicate audit contains a forbidden cross-role edge")
    require_exact_bool(edge.get("blocking"), False, "near-duplicate edge blocking")
    evidence = edge.get("evidence")
    if not isinstance(evidence, list) or not evidence or evidence != sorted(set(evidence)) or any(
        value not in {"perceptual_hash", "exact_image_sha256", "source_sha256"}
        for value in evidence
    ):
        raise ValueError("near-duplicate edge evidence mismatch")
    distance = edge.get("distance")
    if "perceptual_hash" in evidence:
        actual_distance = min(
            (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()
            for left_hash in left_entry["phash_rot4"]
            for right_hash in right_entry["phash_rot4"]
        )
        if (
            type(distance) is not int
            or distance != actual_distance
            or not 0 <= distance <= algorithm["threshold"]
        ):
            raise ValueError("near-duplicate perceptual distance mismatch")
    elif distance is not None and not (
        "exact_image_sha256" in evidence and type(distance) is int and distance == 0
    ):
        raise ValueError("near-duplicate non-perceptual distance mismatch")
    if (
        "exact_image_sha256" in evidence
        and left_entry["image_sha256"] != right_entry["image_sha256"]
    ):
        raise ValueError("near-duplicate exact-image evidence is false")
    if (
        "source_sha256" in evidence
        and left_entry["source_sha256"] != right_entry["source_sha256"]
    ):
        raise ValueError("near-duplicate source evidence is false")
    actual_edges[pair] = {
        "evidence": set(evidence),
        "distance": distance,
    }
if set(actual_edges) != set(expected_edges):
    raise ValueError("near-duplicate audit edge set is incomplete")
for pair, expected in expected_edges.items():
    if actual_edges[pair] != expected:
        raise ValueError("near-duplicate audit edge evidence is incomplete")

cluster_fields = {
    "cluster_id", "member_asset_ids", "member_image_sha256s", "roles",
    "cohorts", "view_kinds", "edge_count", "multi_role", "blocking",
}
cluster_ids = set()
cluster_members = set()
asset_cluster = {}
cluster_edge_counts = {}
derived_same_role_duplicate_clusters = 0
for index, cluster in enumerate(clusters):
    if not isinstance(cluster, dict) or set(cluster) != cluster_fields:
        raise ValueError(f"near-duplicate cluster {index} schema mismatch")
    cluster_id = require_sha(cluster.get("cluster_id"), f"near-duplicate cluster {index} ID")
    if cluster_id in cluster_ids:
        raise ValueError("near-duplicate audit contains duplicate cluster IDs")
    cluster_ids.add(cluster_id)
    members = cluster.get("member_asset_ids")
    image_shas = cluster.get("member_image_sha256s")
    roles = cluster.get("roles")
    cohorts = cluster.get("cohorts")
    view_kinds = cluster.get("view_kinds")
    if not isinstance(members, list) or not members or members != sorted(set(members)):
        raise ValueError("near-duplicate cluster member schema mismatch")
    if any(member not in asset_ids or member in cluster_members for member in members):
        raise ValueError("near-duplicate cluster members overlap or are unknown")
    cluster_members.update(members)
    for member in members:
        asset_cluster[member] = cluster_id
    if not isinstance(image_shas, list) or not image_shas or image_shas != sorted(set(image_shas)):
        raise ValueError("near-duplicate cluster image SHA schema mismatch")
    for image_sha in image_shas:
        require_sha(image_sha, "near-duplicate cluster image SHA")
    expected_cluster_id = sha_bytes(compact_json_value({
        "algorithm_id": algorithm["id"],
        "threshold": algorithm["threshold"],
        "member_image_sha256s": image_shas,
    }))
    if cluster_id != expected_cluster_id:
        raise ValueError("near-duplicate cluster ID is not deterministic")
    if not isinstance(roles, list) or len(roles) != 1 or roles != sorted(set(roles)):
        raise ValueError("near-duplicate cluster is multi-role or malformed")
    if not isinstance(cohorts, list) or not cohorts or cohorts != sorted(set(cohorts)):
        raise ValueError("near-duplicate cluster cohorts are malformed")
    if not isinstance(view_kinds, list) or not view_kinds or view_kinds != sorted(set(view_kinds)):
        raise ValueError("near-duplicate cluster views are malformed")
    member_entries = [asset_entries[member] for member in members]
    if roles != sorted({entry["role"] for entry in member_entries}):
        raise ValueError("near-duplicate cluster roles differ from its members")
    if cohorts != sorted({entry["cohort"] for entry in member_entries}):
        raise ValueError("near-duplicate cluster cohorts differ from its members")
    if view_kinds != sorted({entry["view_kind"] for entry in member_entries}):
        raise ValueError("near-duplicate cluster views differ from its members")
    if image_shas != sorted({entry["image_sha256"] for entry in member_entries}):
        raise ValueError("near-duplicate cluster image SHAs differ from its members")
    edge_count = cluster.get("edge_count")
    if type(edge_count) is not int or edge_count < 0:
        raise ValueError("near-duplicate cluster edge count is invalid")
    if edge_count > 0:
        derived_same_role_duplicate_clusters += 1
    cluster_edge_counts[cluster_id] = edge_count
    require_exact_bool(cluster.get("multi_role"), False, "near-duplicate cluster multi_role")
    require_exact_bool(cluster.get("blocking"), False, "near-duplicate cluster blocking")
if cluster_members != asset_ids:
    raise ValueError("near-duplicate clusters do not partition all verified assets")
if any(asset_cluster[left] != asset_cluster[right] for left, right in edge_pairs):
    raise ValueError("near-duplicate edge crosses deterministic clusters")
derived_edge_counts = Counter(asset_cluster[left] for left, _right in edge_pairs)
if any(
    cluster_edge_counts[cluster_id] != derived_edge_counts[cluster_id]
    for cluster_id in cluster_ids
):
    raise ValueError("near-duplicate cluster edge counts mismatch")

summary = near_duplicate_audit.get("summary")
if not isinstance(summary, dict) or set(summary) != {
    "edges", "clusters", "blocking_multi_role_clusters",
    "same_role_duplicate_clusters_nonblocking",
}:
    raise ValueError("near-duplicate summary schema mismatch")
if summary != {
    "edges": len(edges),
    "clusters": len(clusters),
    "blocking_multi_role_clusters": 0,
    "same_role_duplicate_clusters_nonblocking": derived_same_role_duplicate_clusters,
}:
    raise ValueError("near-duplicate summary does not match complete cluster evidence")

near_duplicate_audit_bytes = compact_json(near_duplicate_audit)
near_duplicate_audit_sha = sha_bytes(near_duplicate_audit_bytes)
if bindings.get("candidate_near_duplicate_audit_sha256") != near_duplicate_audit_sha:
    raise ValueError("near-duplicate audit SHA binding mismatch")
if bindings.get("protected_sources_sha256") != protected_source_binding["file_sha256"]:
    raise ValueError("near-duplicate protected-sources file binding mismatch")
if bindings.get("protected_reference_inventory_sha256") != protected_inventory_binding["file_sha256"]:
    raise ValueError("near-duplicate protected-inventory file binding mismatch")

policy_fields = {
    "schema", "artifact_role", "status", "approved",
    "operational_cutoff_kst", "source_manifest_sha256",
    "full_data_validator_report_sha256", "operational_sources",
    "license_origins", "candidate_counts",
    *POLICY_BINDING_FIELDS,
}
if set(policy) != policy_fields:
    raise ValueError("trusted policy top-level schema mismatch")
if policy.get("schema") != "v4_candidate_training_trusted_policy.v1":
    raise ValueError("trusted policy schema mismatch")
if policy.get("artifact_role") != "approved_v4_candidate_training_policy":
    raise ValueError("trusted policy artifact_role mismatch")
if policy.get("status") != "approved":
    raise ValueError("trusted policy status mismatch")
require_exact_bool(policy.get("approved"), True, "trusted policy approved")
if policy.get("operational_cutoff_kst") != authority.get("operational_cutoff_kst"):
    raise ValueError("trusted policy operational cutoff mismatch")
for field in ("source_manifest_sha256", "full_data_validator_report_sha256"):
    values = policy.get(field)
    if (
        not isinstance(values, list) or not values
        or any(not isinstance(value, str) or not SHA_RE.fullmatch(value) for value in values)
    ):
        raise ValueError(f"trusted policy {field} binding mismatch")
    if len(values) != len(set(values)) or bindings.get(field) != values:
        raise ValueError(f"trusted policy {field} binding mismatch")
for field in POLICY_BINDING_FIELDS:
    expected = bindings.get(field)
    if policy.get(field) != expected:
        raise ValueError(f"trusted policy {field} differs from training authority")
    if field == "container_image_id":
        if expected != image_id:
            raise ValueError("trusted policy container image ID mismatch")
    else:
        require_sha(expected, f"training authority bindings.{field}")
if bindings.get("candidate_train_manifest_sha256") != manifest_hashes["train"]:
    raise ValueError("trusted policy does not bind the actual train manifest")
if bindings.get(
    "candidate_model_validation_manifest_sha256"
) != manifest_hashes["model_validation"]:
    raise ValueError("trusted policy does not bind the actual model-validation manifest")
operational_sources = policy.get("operational_sources")
if not isinstance(operational_sources, dict):
    raise ValueError("trusted policy operational_sources must be an object")
for source_sha, evidence in operational_sources.items():
    require_sha(source_sha, "trusted policy operational source")
    if not isinstance(evidence, dict) or set(evidence) != {
        "auditor_sha256", "teacher_output_sha256", "localizer_output_sha256"
    }:
        raise ValueError("trusted policy operational evidence schema mismatch")
    for field, digest in evidence.items():
        require_sha(digest, f"trusted policy operational evidence.{field}")
license_origins = policy.get("license_origins")
if not isinstance(license_origins, dict) or not license_origins:
    raise ValueError("trusted policy license_origins must be a nonempty object")
for origin, rule in license_origins.items():
    if not isinstance(origin, str) or not origin or not isinstance(rule, dict):
        raise ValueError("trusted policy license origin entry is invalid")
    if set(rule) != {
        "kind", "dataset_id", "commercial_training_allowed",
        "redistribution_allowed", "evidence_sha256",
    }:
        raise ValueError("trusted policy license origin schema mismatch")
    if rule.get("kind") not in {"aihub", "operational"}:
        raise ValueError("trusted policy license kind mismatch")
    if not isinstance(rule.get("dataset_id"), str) or not rule["dataset_id"]:
        raise ValueError("trusted policy license dataset_id mismatch")
    require_exact_bool(
        rule.get("commercial_training_allowed"), True,
        f"trusted policy license {origin}.commercial_training_allowed",
    )
    if type(rule.get("redistribution_allowed")) is not bool:
        raise ValueError("trusted policy license redistribution flag mismatch")
    require_sha(rule.get("evidence_sha256"), "trusted policy license evidence")
policy_candidate_counts = policy.get("candidate_counts")
if not isinstance(policy_candidate_counts, dict):
    raise ValueError("trusted policy candidate_counts must be an object")
policy_excluded_counts = policy_candidate_counts.get("excluded")
if not isinstance(policy_excluded_counts, dict) or any(
    not isinstance(reason, str) or not reason
    or type(count) is not int or count < 1
    for reason, count in policy_excluded_counts.items()
):
    raise ValueError("trusted policy excluded counts are invalid")

# A standard sha256sum marker is accepted only when it contains exactly the
# eight authority files. Unexpected paths cannot be smuggled into verification.
marker_rows = {}
for number, line in enumerate(input_bytes[marker_path].decode("utf-8").splitlines(), start=1):
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    if match is None:
        raise ValueError(f"invalid training_authority.sha256 line {number}")
    path = resolved_file(match.group(2), f"authority marker line {number}")
    if path in marker_rows:
        raise ValueError("authority marker contains a duplicate path")
    marker_rows[path] = match.group(1)
expected_marker_paths = {
    authority_path, inventory_path, config_path, host_path, backbone_path,
    snapshot_report_path, *manifest_paths.values(),
}
if set(marker_rows) != expected_marker_paths or len(marker_rows) != 8:
    raise ValueError("training_authority.sha256 must bind exactly eight expected files")
for path, declared in marker_rows.items():
    if declared != sha_bytes(input_bytes[path]):
        raise ValueError(f"authority marker SHA mismatch: {path}")


def canonical_json(value) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
    ).encode("utf-8")


snapshot_report_fields = {
    "schema", "artifact_role", "status", "candidate_only",
    "production_deployment_authorized", "payload_kind",
    "source_lineage_bytes_snapshotted", "snapshot_root_relative",
    "snapshot_max_bytes", "object_max_bytes", "object_count",
    "total_regular_bytes", "tree_sha256", "payload_set_sha256", "objects",
}
if not isinstance(dataset_snapshot_report, dict) or set(dataset_snapshot_report) != snapshot_report_fields:
    raise ValueError("candidate dataset snapshot report schema mismatch")
if dataset_snapshot_report.get("schema") != "v4_candidate_dataset_snapshot.v2":
    raise ValueError("candidate dataset snapshot report version mismatch")
if dataset_snapshot_report.get("artifact_role") != (
    "candidate_training_crop_bytes_not_blind_or_deployment_authority"
):
    raise ValueError("candidate dataset snapshot artifact role mismatch")
if dataset_snapshot_report.get("status") != "candidate_dataset_snapshot_ready":
    raise ValueError("candidate dataset snapshot status mismatch")
require_exact_bool(
    dataset_snapshot_report.get("candidate_only"), True,
    "candidate dataset snapshot candidate_only",
)
require_exact_bool(
    dataset_snapshot_report.get("production_deployment_authorized"), False,
    "candidate dataset snapshot production authority",
)
require_exact_bool(
    dataset_snapshot_report.get("source_lineage_bytes_snapshotted"), False,
    "candidate dataset snapshot source lineage flag",
)
if dataset_snapshot_report.get("payload_kind") != "training_crop_only":
    raise ValueError("candidate dataset snapshot payload kind mismatch")
if dataset_snapshot_report.get("snapshot_root_relative") != "dataset_snapshot":
    raise ValueError("candidate dataset snapshot root contract mismatch")
if dataset_snapshot_report.get("snapshot_max_bytes") != 68719476736:
    raise ValueError("candidate dataset snapshot byte cap mismatch")
if dataset_snapshot_report.get("object_max_bytes") != 67108864:
    raise ValueError("candidate dataset snapshot object byte cap mismatch")
snapshot_objects = dataset_snapshot_report.get("objects")
if not isinstance(snapshot_objects, list) or not snapshot_objects:
    raise ValueError("candidate dataset snapshot objects must be nonempty")
snapshot_object_fields = {"sample_id", "role", "path", "size", "sha256"}
snapshot_by_sample = {}
snapshot_paths = set()
snapshot_shas = set()
previous_snapshot_key = None
snapshot_total = 0
for index, row in enumerate(snapshot_objects):
    if not isinstance(row, dict) or set(row) != snapshot_object_fields:
        raise ValueError(f"candidate dataset snapshot object {index} schema mismatch")
    sample_id = row.get("sample_id")
    role = row.get("role")
    relative = row.get("path")
    size = row.get("size")
    digest = require_sha(row.get("sha256"), f"dataset snapshot object {index} SHA")
    if not isinstance(sample_id, str) or not sample_id or role not in {"train", "model_validation"}:
        raise ValueError("candidate dataset snapshot object identity mismatch")
    key = (role, sample_id)
    if previous_snapshot_key is not None and key <= previous_snapshot_key:
        raise ValueError("candidate dataset snapshot objects are not strictly role/sample sorted")
    previous_snapshot_key = key
    if (
        not isinstance(relative, str)
        or posixpath.normpath(relative) != relative
        or relative.startswith("/")
        or relative != f"dataset_snapshot/objects/{digest[:2]}/{digest}"
    ):
        raise ValueError("candidate dataset snapshot object path mismatch")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or size > dataset_snapshot_report["object_max_bytes"]
    ):
        raise ValueError("candidate dataset snapshot object size mismatch")
    if sample_id in snapshot_by_sample or relative in snapshot_paths or digest in snapshot_shas:
        raise ValueError("candidate dataset snapshot contains duplicate sample/path/SHA")
    snapshot_by_sample[sample_id] = row
    snapshot_paths.add(relative)
    snapshot_shas.add(digest)
    snapshot_total += size
if dataset_snapshot_report.get("object_count") != len(snapshot_objects):
    raise ValueError("candidate dataset snapshot object count mismatch")
if dataset_snapshot_report.get("total_regular_bytes") != snapshot_total:
    raise ValueError("candidate dataset snapshot total bytes mismatch")
if snapshot_total > dataset_snapshot_report["snapshot_max_bytes"]:
    raise ValueError("candidate dataset snapshot exceeds its byte cap")
snapshot_tree_rows = [
    {"path": row["path"], "size": row["size"], "sha256": row["sha256"]}
    for row in sorted(snapshot_objects, key=lambda value: value["path"])
]
snapshot_tree_sha = sha_bytes(
    (json.dumps(
        snapshot_tree_rows, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")
)
if dataset_snapshot_report.get("tree_sha256") != snapshot_tree_sha:
    raise ValueError("candidate dataset snapshot tree SHA mismatch")
snapshot_payload_set_sha = sha_bytes(
    (json.dumps(
        snapshot_objects, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")
)
if dataset_snapshot_report.get("payload_set_sha256") != snapshot_payload_set_sha:
    raise ValueError("candidate dataset snapshot payload-set SHA mismatch")

dataset_consumption_contract = authority.get("dataset_consumption_contract")
expected_dataset_consumption_contract = {
    "schema": "v4_candidate_dataset_consumption.v1",
    "version": "multitask_verifier.image_consumption.v1",
    "evidence_scope": "per_access_fail_closed_no_complete_access_receipt",
    "authority_platform": "linux_qnap",
    "read_semantics": "single_descriptor_fstat_sha256_then_bytesio_decode",
    "trainer_path": "scripts/train_multitask_verifier.py",
    "trainer_sha256": sha_bytes(input_bytes[trainer_path]),
    "dataset_snapshot_report_sha256": snapshot_report_sha,
    "dataset_snapshot_tree_sha256": snapshot_tree_sha,
    "manifest_payload_set_sha256": snapshot_payload_set_sha,
    "max_image_bytes": 67108864,
    "max_image_pixels": 16777216,
    "complete_access_receipt": False,
}
if dataset_consumption_contract != expected_dataset_consumption_contract:
    raise ValueError("candidate dataset consumption contract mismatch")
dataset_consumption_contract_sha = sha_bytes(
    canonical_json(dataset_consumption_contract)
)
if (
    bindings.get("dataset_consumption_contract_sha256")
    != dataset_consumption_contract_sha
    or bindings.get("candidate_dataset_consumption_contract_sha256")
    != dataset_consumption_contract_sha
):
    raise ValueError("candidate dataset consumption contract SHA binding mismatch")

snapshot_root = snapshot_report_path.parent / "dataset_snapshot"
reject_symlink_components(snapshot_root, "candidate dataset snapshot root")
snapshot_root = snapshot_root.resolve(strict=True)
if (
    snapshot_root.parent != authority_path.parent
    or snapshot_root.is_symlink()
    or not snapshot_root.is_dir()
):
    raise ValueError("candidate dataset snapshot root identity mismatch")
try:
    snapshot_root.relative_to(global_root)
except ValueError as error:
    raise ValueError("candidate dataset snapshot escapes GLOBAL_ROOT") from error
expected_snapshot_files = {}
expected_snapshot_directories = {snapshot_root}
for row in snapshot_objects:
    relative_parts = Path(row["path"]).parts[1:]
    path = snapshot_root.joinpath(*relative_parts)
    expected_snapshot_files[path] = row
    cursor = path.parent
    while cursor != snapshot_root:
        expected_snapshot_directories.add(cursor)
        cursor = cursor.parent
snapshot_entries = list(snapshot_root.rglob("*"))
if any(path.is_symlink() for path in snapshot_entries):
    raise ValueError("candidate dataset snapshot tree contains a symlink")
actual_snapshot_files = {path for path in snapshot_entries if path.is_file()}
actual_snapshot_directories = {
    snapshot_root, *(path for path in snapshot_entries if path.is_dir())
}
if actual_snapshot_files != set(expected_snapshot_files):
    raise ValueError("candidate dataset snapshot regular-file set mismatch")
if actual_snapshot_directories != expected_snapshot_directories:
    raise ValueError("candidate dataset snapshot directory set mismatch")
if any(not path.is_file() and not path.is_dir() for path in snapshot_entries):
    raise ValueError("candidate dataset snapshot contains a special file")

receipt = authority.get("dataset_snapshot_publish_receipt")
receipt_fields = {
    "schema", "snapshot_root", "root_dev", "root_ino", "root_mode", "files"
}
if not isinstance(receipt, dict) or set(receipt) != receipt_fields:
    raise ValueError("dataset snapshot publish receipt schema mismatch")
if receipt.get("schema") != "v4_candidate_dataset_snapshot_publish_receipt.v1":
    raise ValueError("dataset snapshot publish receipt version mismatch")
if receipt.get("snapshot_root") != snapshot_root.as_posix():
    raise ValueError("dataset snapshot publish receipt root mismatch")
if bindings.get("dataset_snapshot_publish_receipt_sha256") != sha_bytes(
    canonical_json(receipt)
):
    raise ValueError("dataset snapshot publish receipt SHA binding mismatch")
root_stat = snapshot_root.stat(follow_symlinks=False)
if (
    receipt.get("root_dev"), receipt.get("root_ino"), receipt.get("root_mode")
) != (root_stat.st_dev, root_stat.st_ino, stat.S_IMODE(root_stat.st_mode)):
    raise ValueError("dataset snapshot root device/inode/mode changed")
if sys.platform.startswith("linux") and stat.S_IMODE(root_stat.st_mode) != 0o555:
    raise ValueError("dataset snapshot root is not mode 0555")
for directory in actual_snapshot_directories:
    directory_stat = directory.stat(follow_symlinks=False)
    if sys.platform.startswith("linux") and stat.S_IMODE(directory_stat.st_mode) != 0o555:
        raise ValueError("dataset snapshot directory is not mode 0555")
runtime_snapshot_rows = []
for path in sorted(actual_snapshot_files, key=lambda value: value.as_posix()):
    expected = expected_snapshot_files[path]
    content = stable_bytes(path, "candidate dataset snapshot object")
    current = path.stat(follow_symlinks=False)
    if (
        current.st_nlink != 1
        or stat.S_IMODE(current.st_mode) != 0o444
        or current.st_size != expected["size"]
        or sha_bytes(content) != expected["sha256"]
    ):
        raise ValueError("candidate dataset snapshot object bytes/identity mismatch")
    runtime_snapshot_rows.append({
        "path": path.relative_to(snapshot_root.parent).as_posix(),
        "dev": current.st_dev,
        "ino": current.st_ino,
        "mode": stat.S_IMODE(current.st_mode),
        "nlink": current.st_nlink,
        "size": current.st_size,
        "sha256": sha_bytes(content),
    })
if receipt.get("files") != runtime_snapshot_rows:
    raise ValueError("dataset snapshot publish receipt differs from live tree")
dataset_snapshot_runtime_contract = {
    "report_path": snapshot_report_path.as_posix(),
    "report_sha256": snapshot_report_sha,
    "tree_sha256": snapshot_tree_sha,
    "snapshot_root": snapshot_root.as_posix(),
    "publish_receipt": receipt,
}

# Verify the complete immutable code tree, not merely the inventory file hash.
inventory = load_json_bytes(input_bytes[inventory_path], "code inventory JSON")
if not isinstance(inventory, dict) or inventory.get("schema") != "v4_candidate_code_inventory.v1":
    raise ValueError("unsupported code inventory schema")
if Path(str(inventory.get("root", ""))).resolve(strict=True) != code_root:
    raise ValueError("code inventory root mismatch")
files = inventory.get("files")
if not isinstance(files, list) or type(inventory.get("file_count")) is not int:
    raise ValueError("code inventory files/count are invalid")
if inventory["file_count"] != len(files) or not files:
    raise ValueError("code inventory count mismatch or empty inventory")
inventory_paths = set()
for row in files:
    if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
        raise ValueError("invalid code inventory row")
    relative = row.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("code inventory path must be nonempty and relative")
    normalized_relative = Path(relative).as_posix()
    if normalized_relative in TRUST_ROOT_CODE_PATHS:
        raise ValueError("code inventory must exclude fixed trust-root scripts")
    path = (code_root / relative).resolve(strict=True)
    try:
        path.relative_to(code_root)
    except ValueError as error:
        raise ValueError("code inventory path escapes CODE_ROOT") from error
    if path in inventory_paths:
        raise ValueError("duplicate code inventory path")
    inventory_paths.add(path)
    content = stable_bytes(path, "inventoried code file")
    if type(row.get("size")) is not int or row["size"] != len(content):
        raise ValueError(f"code inventory size mismatch: {relative}")
    if require_sha(row.get("sha256"), "code inventory file sha256") != sha_bytes(content):
        raise ValueError(f"code inventory SHA mismatch: {relative}")
auditor_path = (code_root / "scripts/audit_v4_near_duplicate_leakage.py").resolve(strict=True)
try:
    auditor_path.relative_to(code_root)
except ValueError as error:
    raise ValueError("near-duplicate auditor escapes CODE_ROOT") from error
if auditor_path not in inventory_paths:
    raise ValueError("near-duplicate auditor is absent from the complete code inventory")
auditor_content = stable_bytes(auditor_path, "near-duplicate auditor")
if sha_bytes(auditor_content) != auditor_binding["sha256"]:
    raise ValueError("near-duplicate auditor differs from the report/code inventory binding")
auditor_module_name = "_v4_near_duplicate_auditor_" + auditor_binding["sha256"]
auditor_module = types.ModuleType(auditor_module_name)
auditor_module.__file__ = str(auditor_path)
auditor_module.__package__ = ""
previous_auditor_module = sys.modules.get(auditor_module_name)
sys.modules[auditor_module_name] = auditor_module
try:
    auditor_code = compile(
        auditor_content,
        str(auditor_path),
        "exec",
        dont_inherit=True,
    )
    exec(auditor_code, auditor_module.__dict__)
    runtime_fingerprint_function = auditor_module.__dict__.get(
        "runtime_code_fingerprint_sha256"
    )
    if not callable(runtime_fingerprint_function):
        raise ValueError(
            "near-duplicate auditor runtime fingerprint function is absent"
        )
    actual_auditor_runtime_code_sha = runtime_fingerprint_function()
finally:
    if previous_auditor_module is None:
        sys.modules.pop(auditor_module_name, None)
    else:
        sys.modules[auditor_module_name] = previous_auditor_module
require_sha(
    actual_auditor_runtime_code_sha,
    "actual near-duplicate auditor runtime code SHA",
)
if actual_auditor_runtime_code_sha != auditor_binding["runtime_code_sha256"]:
    raise ValueError(
        "near-duplicate auditor executed-code fingerprint differs from the report"
    )
if sha_bytes(stable_bytes(auditor_path, "near-duplicate auditor final rehash")) != auditor_binding["sha256"]:
    raise ValueError("near-duplicate auditor changed during runtime validation")
actual_code_paths = {
    path.resolve()
    for path in code_root.rglob("*")
    if path.is_file() and not path.is_symlink()
}
excluded_code_paths = {
    (code_root / relative).resolve(strict=True)
    for relative in TRUST_ROOT_CODE_PATHS
    if (code_root / relative).exists()
}
code_symlinks = [path for path in code_root.rglob("*") if path.is_symlink()]
if code_symlinks:
    raise ValueError("CODE_ROOT must not contain symlinks")
if actual_code_paths - excluded_code_paths != inventory_paths:
    raise ValueError("code inventory does not exactly cover CODE_ROOT regular files")

host = load_json_bytes(input_bytes[host_path], "host launch contract JSON")
if not isinstance(host, dict) or host.get("schema") != "v4_candidate_training_host_launch.v1":
    raise ValueError("unsupported host launch contract schema")
if set(host) != {
    "schema", "container_id", "container_name", "container_image_id",
    "network_mode", "restart_policy", "shm_size_bytes", "privileged",
    "device_requests", "devices", "mounts", "command", "raw_inspect_path",
    "raw_inspect_sha256", "qnap_library_inventory",
}:
    raise ValueError("host launch contract top-level schema mismatch")
if host.get("container_image_id") != image_id:
    raise ValueError("host launch image ID mismatch")
qnap_inventory = host.get("qnap_library_inventory")
qnap_snapshot_report = load_json_bytes(
    input_bytes[qnap_snapshot_report_path], "QNAP library snapshot report JSON"
)
if not isinstance(qnap_snapshot_report, dict) or set(qnap_snapshot_report) != {
    "schema", "status", "snapshot_root", "snapshot_dev", "snapshot_ino",
    "snapshot_mode", "source_mounts_recursively_read_only",
    "snapshot_max_bytes", "total_regular_bytes", "inventory", "trees",
}:
    raise ValueError("QNAP library snapshot report schema mismatch")
if qnap_snapshot_report.get("schema") != "v4_qnap_library_snapshot.v1":
    raise ValueError("QNAP library snapshot report version mismatch")
if qnap_snapshot_report.get("inventory") != qnap_inventory:
    raise ValueError("QNAP snapshot report differs from policy-bound inventory")
if qnap_snapshot_report.get("snapshot_root") != "/dev/shm/v4-qnap-libraries":
    raise ValueError("QNAP snapshot report root mismatch")
if sys.platform.startswith("linux"):
    if qnap_snapshot_report.get("status") != "qnap_library_snapshot_verified":
        raise ValueError("QNAP snapshot was not verified on Linux")
    if qnap_snapshot_report.get("source_mounts_recursively_read_only") is not True:
        raise ValueError("QNAP source mount read-only proof is missing")
    qnap_snapshot_stat = Path(qnap_snapshot_report["snapshot_root"]).stat(
        follow_symlinks=False
    )
    if (
        qnap_snapshot_stat.st_dev,
        qnap_snapshot_stat.st_ino,
        stat.S_IMODE(qnap_snapshot_stat.st_mode),
    ) != (
        qnap_snapshot_report.get("snapshot_dev"),
        qnap_snapshot_report.get("snapshot_ino"),
        qnap_snapshot_report.get("snapshot_mode"),
    ):
        raise ValueError("QNAP snapshot root identity differs from its bootstrap report")
else:
    if qnap_snapshot_report.get("status") != "not_applicable_non_linux_test_host":
        raise ValueError("non-Linux QNAP snapshot report status mismatch")
raw_inspect_path = resolved_file(str(host.get("raw_inspect_path", "")), "raw docker inspect")
try:
    raw_inspect_path.relative_to(global_root)
except ValueError as error:
    raise ValueError("raw docker inspect must be beneath GLOBAL_ROOT") from error
raw_inspect_content = stable_bytes(raw_inspect_path, "raw docker inspect")
raw_inspect_sha = sha_bytes(raw_inspect_content)
if require_sha(host.get("raw_inspect_sha256"), "raw inspect sha256") != raw_inspect_sha:
    raise ValueError("raw docker inspect SHA mismatch")
input_bytes[raw_inspect_path] = raw_inspect_content
raw_inspect_value = load_json_bytes(raw_inspect_content, "raw docker inspect JSON")
if not isinstance(raw_inspect_value, list) or len(raw_inspect_value) != 1:
    raise ValueError("raw docker inspect must contain exactly one inspect object")
raw_inspect = raw_inspect_value[0]
if not isinstance(raw_inspect, dict):
    raise ValueError("raw docker inspect entry must be an object")
container_id = host.get("container_id")
container_name = host.get("container_name")
if not isinstance(container_id, str) or not re.fullmatch(r"[0-9a-f]{64}", container_id):
    raise ValueError("host contract container_id must be a full lowercase ID")
if (
    not isinstance(container_name, str)
    or re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,62}", container_name) is None
):
    raise ValueError("host contract container_name is invalid")
if raw_inspect.get("Id") != container_id or raw_inspect.get("Image") != image_id:
    raise ValueError("raw docker inspect identity/image mismatch")
if raw_inspect.get("Name") != f"/{container_name}":
    raise ValueError("raw docker inspect container name mismatch")
exact_host = {
    "network_mode": "none",
    "restart_policy": "no",
    "shm_size_bytes": 8589934592,
    "privileged": False,
    "device_requests": None,
}
for field, expected in exact_host.items():
    actual = host.get(field)
    if type(expected) is bool:
        require_exact_bool(actual, expected, f"host launch {field}")
    elif actual != expected or (type(expected) is int and type(actual) is not int):
        raise ValueError(f"host launch {field} must be {expected!r}")
raw_host = raw_inspect.get("HostConfig")
raw_config = raw_inspect.get("Config")
if not isinstance(raw_host, dict) or not isinstance(raw_config, dict):
    raise ValueError("raw docker inspect lacks HostConfig/Config")
restart = raw_host.get("RestartPolicy")
if not isinstance(restart, dict) or restart.get("Name") != "no" or restart.get("MaximumRetryCount") != 0:
    raise ValueError("raw docker inspect restart policy mismatch")
for field, expected in (
    ("NetworkMode", "none"), ("ShmSize", 8589934592),
    ("Privileged", False), ("DeviceRequests", None),
):
    if raw_host.get(field) != expected or (type(expected) is int and type(raw_host.get(field)) is not int):
        raise ValueError(f"raw docker inspect HostConfig.{field} mismatch")
devices = host.get("devices")
if not isinstance(devices, list) or sorted(devices) != EXPECTED_DEVICES or len(devices) != 7:
    raise ValueError("host launch must expose exactly the seven approved NVIDIA devices")
raw_devices = raw_host.get("Devices")
if not isinstance(raw_devices, list) or len(raw_devices) != 7:
    raise ValueError("raw docker inspect must expose exactly seven devices")
raw_device_paths = []
for device in raw_devices:
    if not isinstance(device, dict):
        raise ValueError("raw docker inspect device entry must be an object")
    source = device.get("PathOnHost")
    destination = device.get("PathInContainer")
    if source != destination or device.get("CgroupPermissions") != "rwm":
        raise ValueError("raw docker inspect device mapping/mode mismatch")
    raw_device_paths.append(destination)
if sorted(raw_device_paths) != EXPECTED_DEVICES:
    raise ValueError("raw docker inspect NVIDIA devices mismatch")
command = host.get("command")
if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
    raise ValueError("host launch command is malformed")
if raw_config.get("Cmd") != command:
    raise ValueError("raw docker inspect Config.Cmd mismatch")
if raw_config.get("Hostname") != container_id[:12]:
    raise ValueError("raw docker inspect Config.Hostname mismatch")
if raw_config.get("Entrypoint") not in (None, []):
    raise ValueError("raw docker inspect Config.Entrypoint must be empty")
if raw_config.get("User") != "" or raw_config.get("WorkingDir") != "":
    raise ValueError("raw docker inspect Config.User/WorkingDir mismatch")
raw_environment = raw_config.get("Env")
if not isinstance(raw_environment, list):
    raise ValueError("raw docker inspect Config.Env must be a list")
environment = {}
for entry in raw_environment:
    if not isinstance(entry, str) or "=" not in entry:
        raise ValueError("raw docker inspect Config.Env entry is malformed")
    name, content = entry.split("=", 1)
    if not name or name in environment:
        raise ValueError("raw docker inspect Config.Env contains an empty/duplicate name")
    environment[name] = content
forbidden_environment = sorted(
    name
    for name in environment
    if name in FORBIDDEN_CONTAINER_ENV
    or name.startswith(BASH_EXPORTED_FUNCTION_PREFIX)
)
if forbidden_environment:
    raise ValueError(
        f"raw docker inspect Config.Env contains injection variables: "
        f"{forbidden_environment}"
    )
live_forbidden_environment = sorted(
    name
    for name in os.environ
    if name in (FORBIDDEN_CONTAINER_ENV - {"LD_LIBRARY_PATH"})
    or name.startswith(BASH_EXPORTED_FUNCTION_PREFIX)
)
if live_forbidden_environment:
    raise ValueError(
        f"live environment contains injection variables: {live_forbidden_environment}"
    )
if sys.platform.startswith("linux") and os.environ.get(
    "LD_LIBRARY_PATH"
) != "/dev/shm/v4-qnap-libraries/nvidia/lib:/dev/shm/v4-qnap-libraries/cuda/lib64":
    raise ValueError("live LD_LIBRARY_PATH differs from the private QNAP snapshot path")
missing_environment = sorted(REQUIRED_CONTAINER_ENV.difference(environment))
if missing_environment:
    raise ValueError(
        f"raw docker inspect Config.Env lacks launcher variables: {missing_environment}"
    )
for name in REQUIRED_CONTAINER_ENV:
    if environment[name] != os.environ.get(name):
        raise ValueError(f"live environment differs from raw docker inspect: {name}")
if environment["CONTAINER_IMAGE_ID"] != image_id:
    raise ValueError("raw docker inspect CONTAINER_IMAGE_ID mismatch")
expected_command = [
    "/usr/bin/env", "-i", f"PATH={CLEAN_CONTAINER_PATH}",
    "V4_CLEAN_REEXEC=1",
    *[
        f"{name}={environment[name]}"
        for name in sorted(REQUIRED_CONTAINER_ENV)
    ],
    "/bin/sh", wrapper_path.as_posix(),
]
if command != expected_command:
    raise ValueError("host launch command does not clean and reconstruct the environment")
if "V4_CLEAN_REEXEC" in environment:
    raise ValueError("raw docker inspect may not inject the internal clean-env marker")
if os.environ.get("V4_CLEAN_REEXEC") != "1":
    raise ValueError("launcher did not enter through the clean environment gate")
for field in ("CapAdd", "CapDrop", "SecurityOpt"):
    if raw_host.get(field) not in (None, []):
        raise ValueError(f"raw docker inspect HostConfig.{field} must be empty")
for field in ("PidMode", "UTSMode", "UsernsMode"):
    if raw_host.get(field) != "":
        raise ValueError(f"raw docker inspect HostConfig.{field} must be empty")
if raw_host.get("IpcMode") != "private":
    raise ValueError("raw docker inspect HostConfig.IpcMode must be exactly private")
mounts = host.get("mounts")
if not isinstance(mounts, list) or len(mounts) < 2:
    raise ValueError("host launch mounts are missing")
mount_by_destination = {}
for mount in mounts:
    if not isinstance(mount, dict) or set(mount) != {"source", "destination", "read_only"}:
        raise ValueError("host mount entries require source,destination,read_only only")
    destination = mount.get("destination")
    source = mount.get("source")
    read_only = mount.get("read_only")
    if not isinstance(source, str) or not source or not isinstance(destination, str):
        raise ValueError("host mount source/destination must be nonempty strings")
    if type(read_only) is not bool or destination in mount_by_destination:
        raise ValueError("host mount mode or destination is invalid")
    mount_by_destination[destination] = mount
raw_mounts = raw_inspect.get("Mounts")
if not isinstance(raw_mounts, list):
    raise ValueError("raw docker inspect Mounts must be a list")
raw_mount_contract = []
for mount in raw_mounts:
    if not isinstance(mount, dict):
        raise ValueError("raw docker inspect mount entry must be an object")
    if mount.get("Type") != "bind" or mount.get("Propagation") != "rprivate":
        raise ValueError("raw docker inspect mount type/propagation mismatch")
    writable = mount.get("RW")
    if type(writable) is not bool or mount.get("Mode") != ("rw" if writable else "ro"):
        raise ValueError("raw docker inspect mount mode mismatch")
    raw_mount_contract.append({
        "source": mount.get("Source"),
        "destination": mount.get("Destination"),
        "read_only": not writable,
    })
mount_key = lambda item: (
    str(item.get("destination")), str(item.get("source")), item.get("read_only")
)
if sorted(raw_mount_contract, key=mount_key) != sorted(mounts, key=mount_key):
    raise ValueError("raw docker inspect mounts differ from host launch contract")
global_mount = mount_by_destination.get(global_root.as_posix())
run_mount = mount_by_destination.get(run_root.as_posix())
if global_mount is None or global_mount["read_only"] is not True:
    raise ValueError("global Container mount must be read-only")
if run_mount is None or run_mount["read_only"] is not False:
    raise ValueError("RUN_ROOT must be the sole read-write host mount")
if global_mount["source"] != "/share/Container":
    raise ValueError("GLOBAL_ROOT must come from the exact /share/Container source")
expected_run_source = f"/share/Container/runs/{container_name}-workspace"
if run_mount["source"] != expected_run_source:
    raise ValueError("RUN_ROOT must come from the dedicated per-container run workspace")
if [item["destination"] for item in mounts if item["read_only"] is False] != [run_root.as_posix()]:
    raise ValueError("RUN_ROOT must be the only read-write mount")
observed_qnap_mounts = {
    (item["source"], item["destination"])
    for item in mounts
    if item not in (global_mount, run_mount)
}
if observed_qnap_mounts != ALLOWED_QNAP_LIBRARY_MOUNTS:
    raise ValueError("host launch must contain both exact QNAP library mounts")
for item in mounts:
    if item in (global_mount, run_mount):
        continue
    if item["read_only"] is not True or (
        item["source"], item["destination"]
    ) not in ALLOWED_QNAP_LIBRARY_MOUNTS:
        raise ValueError("host launch contains an unapproved extra mount")

if sys.platform.startswith("linux"):
    hostname = socket.gethostname()
    cgroup_texts = {
        "self": Path("/proc/self/cgroup").read_text(encoding="utf-8", errors="strict"),
        "pid1": Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="strict"),
    }
    if hostname != container_id[:12]:
        raise ValueError("live Linux container identity differs from raw docker inspect")
    observed_cgroup_ids = sorted({
        match
        for content in cgroup_texts.values()
        for match in re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", content)
    })
    if observed_cgroup_ids and observed_cgroup_ids != [container_id]:
        raise ValueError("live cgroup container ID differs from raw docker inspect")
    runtime_container_identity_contract = {
        "platform": "linux",
        "hostname": hostname,
        "cgroup_container_ids": observed_cgroup_ids,
        "identity_evidence": (
            "cgroup_and_hostname" if observed_cgroup_ids
            else "hostname_with_trusted_host_attestation_fallback"
        ),
    }
    mount_modes = {}
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        left = line.split(" - ", 1)[0].split()
        if len(left) >= 6:
            mountpoint = left[4].replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")
            mount_modes[mountpoint] = set(left[5].split(","))
    def require_recursive_read_only(root: str, description: str) -> None:
        prefix = root.rstrip("/") + "/"
        tree_mounts = {
            mountpoint: modes
            for mountpoint, modes in mount_modes.items()
            if mountpoint == root or mountpoint.startswith(prefix)
        }
        if root not in tree_mounts:
            raise ValueError(f"live {description} mount is missing")
        writable = sorted(
            mountpoint for mountpoint, modes in tree_mounts.items()
            if "ro" not in modes or "rw" in modes
        )
        if writable:
            raise ValueError(
                f"live {description} tree contains writable mountpoints: {writable}"
            )

    require_recursive_read_only(global_root.as_posix(), "GLOBAL_ROOT")
    for qnap_root in sorted(destination for _, destination in ALLOWED_QNAP_LIBRARY_MOUNTS):
        require_recursive_read_only(qnap_root, "QNAP library")
    if "rw" not in mount_modes.get(run_root.as_posix(), set()):
        raise ValueError("live RUN_ROOT mount is not read-write")
    shm = os.statvfs("/dev/shm")
    if shm.f_frsize * shm.f_blocks < 8589934592:
        raise ValueError("live /dev/shm is smaller than 8 GiB")
    for device in EXPECTED_DEVICES:
        if not stat.S_ISCHR(os.stat(device).st_mode):
            raise ValueError(f"live NVIDIA device is not a character device: {device}")
    if set(os.listdir("/sys/class/net")) != {"lo"}:
        raise ValueError("live network namespace contains an interface other than lo")
else:
    runtime_container_identity_contract = {
        "platform": sys.platform,
        "identity_evidence": "not_applicable_non_linux_test_host",
    }
for path, description in (
    (code_root, "CODE_ROOT"), (authority_path, "authority"),
    (inventory_path, "code inventory"), (config_path, "training config"),
    (host_path, "host contract"), (backbone_path, "pretrained backbone"),
    *[(path, f"{role} manifest") for role, path in manifest_paths.items()],
):
    try:
        path.relative_to(global_root)
    except ValueError as error:
        raise ValueError(f"{description} must be beneath the read-only global mount") from error

if backbone_path.name != "mobilenet_v3_small-047dcff4.pth":
    raise ValueError("unexpected pretrained MobileNetV3 checkpoint filename")
if backbone_path.parent.name != "checkpoints" or backbone_path.parent.parent.name != "hub":
    raise ValueError("pretrained backbone must live beneath TORCH_HOME/hub/checkpoints")
torch_home = backbone_path.parent.parent.parent

config = load_json_bytes(input_bytes[config_path], "training config JSON")
required_config_fields = {
    "schema", "backbone", "pretrained", "input_size", "epochs", "patience",
    "batch", "workers", "lr", "backbone_lr", "head_lr", "label_smoothing",
    "class_weight_mode", "class_weight_beta", "objectness_weight",
    "material_weight", "condition_weight", "condition_heads", "origin_weights",
    "seed", "optimizer", "optimizer_betas", "weight_decay", "scheduler",
    "scheduler_t_max", "scheduler_eta_min",
    "sampling_mode", "sampling_samples_per_epoch",
    "sampling_expected_fraction_by_origin",
    "image_consumption_contract_version", "image_max_bytes",
    "image_max_pixels",
}
if not isinstance(config, dict) or set(config) != required_config_fields:
    raise ValueError("training config fields differ from the frozen contract")
if config.get("schema") != "v4_candidate_training_config.v2":
    raise ValueError("unsupported training config schema")
if config.get("backbone") != "mobilenet_v3_small":
    raise ValueError("candidate training backbone must be mobilenet_v3_small")
require_exact_bool(config.get("pretrained"), True, "training config pretrained")
if config.get("condition_heads") != ["dent", "label", "foreign_material"]:
    raise ValueError("all three condition heads must be explicitly enabled in frozen order")
if type(config.get("input_size")) is not int or config["input_size"] != 320:
    raise ValueError("training input_size must be 320")
for field in ("epochs", "patience", "batch"):
    if type(config.get(field)) is not int or config[field] < 1:
        raise ValueError(f"training config {field} must be a positive integer")
if type(config.get("workers")) is not int or config["workers"] < 0:
    raise ValueError("training config workers must be a non-negative integer")
if type(config.get("seed")) is not int or config["seed"] < 0:
    raise ValueError("training config seed must be a non-negative integer")
if config.get("image_consumption_contract_version") != (
    "multitask_verifier.image_consumption.v1"
):
    raise ValueError("training image consumption contract version mismatch")
if config.get("image_max_bytes") != 67108864:
    raise ValueError("training image_max_bytes mismatch")
if config.get("image_max_pixels") != 16777216:
    raise ValueError("training image_max_pixels mismatch")
for field in (
    "lr", "backbone_lr", "head_lr", "objectness_weight", "material_weight",
    "condition_weight",
):
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"training config {field} must be finite and positive")
label_smoothing = config.get("label_smoothing")
if isinstance(label_smoothing, bool) or not isinstance(label_smoothing, (int, float)) or not math.isfinite(label_smoothing) or not 0 <= label_smoothing < 1:
    raise ValueError("training config label_smoothing must be finite in [0,1)")
if config.get("class_weight_mode") not in {"none", "inverse", "effective-number"}:
    raise ValueError("unsupported class_weight_mode")
if config.get("optimizer") != "AdamW":
    raise ValueError("training optimizer must be AdamW")
if config.get("optimizer_betas") != [0.9, 0.999]:
    raise ValueError("training optimizer betas mismatch")
if config.get("weight_decay") != 0.0001 or isinstance(config.get("weight_decay"), bool):
    raise ValueError("training weight_decay must be exactly 0.0001")
if config.get("scheduler") != "CosineAnnealingLR":
    raise ValueError("training scheduler must be CosineAnnealingLR")
if type(config.get("scheduler_t_max")) is not int or config.get(
    "scheduler_t_max"
) != config.get("epochs"):
    raise ValueError("training scheduler_t_max must exactly equal epochs")
if (
    isinstance(config.get("scheduler_eta_min"), bool)
    or not isinstance(config.get("scheduler_eta_min"), (int, float))
    or float(config["scheduler_eta_min"]) != 0.0
):
    raise ValueError("training scheduler_eta_min must be exactly zero")
if config.get("sampling_mode") not in {
    "weighted_replacement", "shuffle_without_replacement"
}:
    raise ValueError("training sampling_mode is unsupported")
if type(config.get("sampling_samples_per_epoch")) is not int or config[
    "sampling_samples_per_epoch"
] < 1:
    raise ValueError("training sampling_samples_per_epoch must be positive")
sampling_fractions = config.get("sampling_expected_fraction_by_origin")
if not isinstance(sampling_fractions, dict) or not sampling_fractions:
    raise ValueError("training sampling fractions must be a nonempty object")
for origin, fraction in sampling_fractions.items():
    if (
        not isinstance(origin, str) or not origin
        or isinstance(fraction, bool) or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction) or not 0 < fraction <= 1
    ):
        raise ValueError("training sampling fraction is invalid")
if not math.isclose(
    math.fsum(sampling_fractions.values()), 1.0,
    rel_tol=0.0, abs_tol=1e-12,
):
    raise ValueError("training sampling fractions must sum to one")

# The trainer has no optimizer CLI flags yet, so bind the declarative config to
# the exact reviewed implementation in the already hash-pinned trainer bytes.
try:
    trainer_tree = ast.parse(
        input_bytes[trainer_path].decode("utf-8"), filename=trainer_path.as_posix()
    )
except (UnicodeDecodeError, SyntaxError) as error:
    raise ValueError(f"pinned trainer source is not valid UTF-8 Python: {error}") from error
optimizer_functions = [
    node for node in trainer_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "_build_optimizer"
]
if len(optimizer_functions) != 1:
    raise ValueError("pinned trainer must define exactly one _build_optimizer")
adamw_calls = [
    node for node in ast.walk(optimizer_functions[0])
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "AdamW"
]
if len(adamw_calls) != 1:
    raise ValueError("pinned trainer must create exactly one AdamW optimizer")
adamw_keywords = {keyword.arg: keyword.value for keyword in adamw_calls[0].keywords}
weight_decay_node = adamw_keywords.get("weight_decay")
if (
    not isinstance(weight_decay_node, ast.Constant)
    or isinstance(weight_decay_node.value, bool)
    or weight_decay_node.value != config["weight_decay"]
):
    raise ValueError("pinned trainer AdamW weight_decay differs from config")
betas_node = adamw_keywords.get("betas")
if betas_node is None:
    # PyTorch's positional defaults are not a stable index contract; create a
    # harmless one-parameter optimizer to observe the effective defaults.
    probe_parameter = torch.nn.Parameter(torch.zeros(1))
    probe_optimizer = torch.optim.AdamW([probe_parameter], lr=1e-3)
    effective_betas = tuple(probe_optimizer.defaults["betas"])
else:
    try:
        effective_betas = tuple(ast.literal_eval(betas_node))
    except (ValueError, TypeError) as error:
        raise ValueError("pinned trainer AdamW betas are not literal") from error
if list(effective_betas) != config["optimizer_betas"]:
    raise ValueError("pinned trainer AdamW betas differ from config")
scheduler_calls = [
    node for node in ast.walk(trainer_tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "CosineAnnealingLR"
]
if len(scheduler_calls) != 1:
    raise ValueError("pinned trainer must create exactly one CosineAnnealingLR")
if len(scheduler_calls[0].args) < 2:
    raise ValueError("pinned trainer CosineAnnealingLR lacks T_max")
scheduler_t_max_node = scheduler_calls[0].args[1]
if not (
    isinstance(scheduler_t_max_node, ast.Subscript)
    and isinstance(scheduler_t_max_node.value, ast.Name)
    and scheduler_t_max_node.value.id == "effective"
    and isinstance(scheduler_t_max_node.slice, ast.Constant)
    and scheduler_t_max_node.slice.value == "epochs"
):
    raise ValueError("pinned trainer scheduler T_max is not effective epochs")
scheduler_keywords = {
    keyword.arg: keyword.value for keyword in scheduler_calls[0].keywords
}
eta_min_node = scheduler_keywords.get("eta_min")
if eta_min_node is None:
    probe_parameter = torch.nn.Parameter(torch.zeros(1))
    probe_optimizer = torch.optim.AdamW([probe_parameter], lr=1e-3)
    effective_eta_min = torch.optim.lr_scheduler.CosineAnnealingLR(
        probe_optimizer, config["scheduler_t_max"]
    ).eta_min
else:
    try:
        effective_eta_min = ast.literal_eval(eta_min_node)
    except (ValueError, TypeError) as error:
        raise ValueError("pinned trainer scheduler eta_min is not literal") from error
if effective_eta_min != config["scheduler_eta_min"]:
    raise ValueError("pinned trainer scheduler eta_min differs from config")
optimizer_runtime_contract = {
    "optimizer": "AdamW",
    "optimizer_betas": list(effective_betas),
    "weight_decay": config["weight_decay"],
    "scheduler": "CosineAnnealingLR",
    "scheduler_t_max": config["scheduler_t_max"],
    "scheduler_eta_min": effective_eta_min,
}
runtime_providers = sorted(ort.get_available_providers())
if "CPUExecutionProvider" not in runtime_providers:
    raise ValueError("onnxruntime lacks required CPUExecutionProvider")
runtime_dependency_contract = {
    "onnx": str(onnx.__version__),
    "onnxruntime": str(ort.__version__),
    "onnxruntime_providers": runtime_providers,
    "torch": str(torch.__version__),
    "torchvision": str(torchvision.__version__),
}
if any(
    not value
    for key, value in runtime_dependency_contract.items()
    if key != "onnxruntime_providers"
):
    raise ValueError("runtime dependency version is empty")
if not callable(models.mobilenet_v3_small):
    raise ValueError("torchvision MobileNetV3-small constructor is unavailable")
dependency_probe_model = models.mobilenet_v3_small(weights=None)
if not dependency_probe_model.state_dict():
    raise ValueError("torchvision MobileNetV3-small construction failed")


def collect_qnap_mapped_library_contract(snapshot_report):
    inventory_value = snapshot_report["inventory"]
    snapshot_root = Path(snapshot_report["snapshot_root"])
    tree_reports = {
        row["container_root"]: row for row in snapshot_report["trees"]
    }
    entries_by_root = {
        tree["container_root"]: {entry["path"]: entry for entry in tree["entries"]}
        for tree in inventory_value["trees"]
    }

    def terminal_relative(container_root, relative):
        entries = entries_by_root[container_root]
        cursor = relative
        visited = set()
        while True:
            if cursor in visited:
                raise ValueError("required QNAP mapped library contains a symlink cycle")
            visited.add(cursor)
            entry = entries[cursor]
            if entry["type"] == "file":
                return cursor, entry
            cursor = posixpath.normpath(
                posixpath.join(posixpath.dirname(cursor), entry["target"])
            )

    required_terminal_rows = []
    required_absolute_paths = set()
    inventoried_files = {}
    inventoried_basenames = set()
    for container_root, entries in entries_by_root.items():
        if sys.platform.startswith("linux") and container_root not in tree_reports:
            raise ValueError("QNAP snapshot tree report is incomplete")
        destination_root = (
            Path(tree_reports[container_root]["snapshot_root"])
            if container_root in tree_reports
            else snapshot_root / (
                "nvidia/lib" if container_root == "/qnap/nvidia/lib" else "cuda/lib64"
            )
        )
        for relative, entry in entries.items():
            inventoried_basenames.add(posixpath.basename(relative))
            if entry["type"] == "file":
                inventoried_files[(destination_root / relative).as_posix()] = (
                    container_root, relative, entry
                )
    for required in inventory_value["required_mapped_libraries"]:
        container_root = required["container_root"]
        terminal, entry = terminal_relative(container_root, required["path"])
        destination_root = (
            Path(tree_reports[container_root]["snapshot_root"])
            if container_root in tree_reports
            else snapshot_root / (
                "nvidia/lib" if container_root == "/qnap/nvidia/lib" else "cuda/lib64"
            )
        )
        absolute = (destination_root / terminal).as_posix()
        required_absolute_paths.add(absolute)
        required_terminal_rows.append({
            "container_root": container_root,
            "declared_path": required["path"],
            "terminal_path": terminal,
            "snapshot_path": absolute,
            "size": entry["size"],
            "sha256": entry["sha256"],
        })
    required_terminal_rows.sort(
        key=lambda row: (row["container_root"], row["declared_path"])
    )
    if not sys.platform.startswith("linux"):
        return {
            "status": "not_applicable_non_linux_test_host",
            "observer_role": "launcher_boundary_process_not_trainer_process",
            "platform": sys.platform,
            "required_terminal_files": required_terminal_rows,
            "mapped_files": [],
        }

    original_roots = ("/qnap/nvidia/lib", "/qnap/cuda/lib64")
    mapped_snapshot_paths = set()
    deleted_collisions = []
    outside_collisions = []
    for line in Path("/proc/self/maps").read_text(
        encoding="utf-8", errors="strict"
    ).splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        mapped_text = fields[5]
        deleted = mapped_text.endswith(" (deleted)")
        if deleted:
            mapped_text = mapped_text[:-10]
        basename = posixpath.basename(mapped_text)
        under_original = any(
            mapped_text == root or mapped_text.startswith(root + "/")
            for root in original_roots
        )
        under_snapshot = (
            mapped_text == snapshot_root.as_posix()
            or mapped_text.startswith(snapshot_root.as_posix() + "/")
        )
        collision = basename in inventoried_basenames
        if deleted and (under_original or under_snapshot or collision):
            deleted_collisions.append(fields[5])
            continue
        if under_original or (collision and not under_snapshot):
            outside_collisions.append(mapped_text)
            continue
        if under_snapshot:
            mapped_snapshot_paths.add(mapped_text)
    if deleted_collisions:
        raise ValueError(
            f"QNAP-related mapped libraries are deleted: {sorted(set(deleted_collisions))}"
        )
    if outside_collisions:
        raise ValueError(
            "QNAP-inventoried library basenames mapped outside the private snapshot: "
            f"{sorted(set(outside_collisions))}"
        )
    if not mapped_snapshot_paths:
        raise ValueError("no QNAP library was mapped from the private snapshot")
    missing_required = sorted(required_absolute_paths - mapped_snapshot_paths)
    if missing_required:
        raise ValueError(
            f"policy-required QNAP libraries were not mapped: {missing_required}"
        )
    rows = []
    for mapped_text in sorted(mapped_snapshot_paths):
        expected = inventoried_files.get(mapped_text)
        if expected is None:
            raise ValueError(f"mapped snapshot library is not an inventoried file: {mapped_text}")
        container_root, relative, entry = expected
        path = Path(mapped_text)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"mapped QNAP snapshot path is not a regular file: {path}")
        content = stable_bytes(path, "mapped QNAP snapshot library")
        value = path.stat(follow_symlinks=False)
        if len(content) != entry["size"] or sha_bytes(content) != entry["sha256"]:
            raise ValueError(f"mapped QNAP snapshot bytes differ from inventory: {path}")
        rows.append({
            "container_root": container_root,
            "relative_path": relative,
            "path": mapped_text,
            "size": len(content),
            "sha256": sha_bytes(content),
            "dev": value.st_dev,
            "ino": value.st_ino,
        })
    return {
        "status": "qnap_snapshot_mappings_verified",
        "observer_role": "launcher_boundary_process_not_trainer_process",
        "platform": "linux",
        "required_terminal_files": required_terminal_rows,
        "mapped_files": rows,
    }


if sys.platform.startswith("linux"):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ValueError("CUDA is unavailable before candidate training")
    cuda_device = torch.device("cuda:0")
    cuda_properties = torch.cuda.get_device_properties(cuda_device)
    cuda_probe = torch.arange(4096, dtype=torch.float32, device=cuda_device)
    cuda_probe_result = float((cuda_probe.square().sum()).cpu().item())
    dependency_probe_model = dependency_probe_model.eval().to(cuda_device)
    model_probe = torch.zeros(
        1, 3, 320, 320, dtype=torch.float32, device=cuda_device
    )
    with torch.no_grad():
        model_probe_result = dependency_probe_model(model_probe)
    if tuple(model_probe_result.shape) != (1, 1000):
        raise ValueError("CUDA MobileNetV3-small smoke output shape mismatch")
    torch.cuda.synchronize(cuda_device)
    if not math.isfinite(cuda_probe_result) or cuda_probe_result <= 0:
        raise ValueError("CUDA allocation/kernel smoke probe returned an invalid value")
    driver_version_path = Path("/proc/driver/nvidia/version")
    driver_version_bytes = stable_bytes(
        driver_version_path, "live NVIDIA driver version"
    )
    cuda_runtime_contract = {
        "required": True,
        "device_count": torch.cuda.device_count(),
        "device_name": str(cuda_properties.name),
        "compute_capability": [cuda_properties.major, cuda_properties.minor],
        "total_memory_bytes": int(cuda_properties.total_memory),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_version_sha256": sha_bytes(driver_version_bytes),
        "smoke_probe": "cuda:0 arange-square-sum-synchronize",
    }
    mapped_qnap_library_contract = collect_qnap_mapped_library_contract(
        qnap_snapshot_report
    )
    del cuda_probe, model_probe, model_probe_result, dependency_probe_model
    torch.cuda.empty_cache()
else:
    del dependency_probe_model
    cuda_runtime_contract = {
        "required": False,
        "platform": sys.platform,
        "smoke_probe": "not_applicable_non_linux_test_host",
    }
    mapped_qnap_library_contract = collect_qnap_mapped_library_contract(
        qnap_snapshot_report
    )
beta = config.get("class_weight_beta")
if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 <= beta < 1:
    raise ValueError("training config class_weight_beta must be finite in [0,1)")
origin_weights = config.get("origin_weights")
if not isinstance(origin_weights, dict):
    raise ValueError("training config origin_weights must be an object")
for origin, weight in origin_weights.items():
    if not isinstance(origin, str) or not origin.strip() or "=" in origin:
        raise ValueError("origin weight names must be nonempty and cannot contain =")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("origin weights must be finite and positive")

# Perform a cheap independent role/provenance scan before the trainer's two
# complete content/hash/leakage passes (dry-run and actual training).
manifest_rows = {}
manifest_payload_files = {}
dataset_content_inventory = []
manifest_audit_rows = set()
snapshot_samples_seen = set()
train_origin_counts = Counter()
selected_by_role = Counter()
selected_by_origin = Counter()
material_by_role = {role: Counter() for role in ("train", "model_validation")}
objectness_by_role = {role: Counter() for role in ("train", "model_validation")}
origin_by_role = {role: Counter() for role in ("train", "model_validation")}
condition_targets_by_role = {
    role: {
        head: Counter() for head in ("dent", "label", "foreign_material")
    }
    for role in ("train", "model_validation")
}
condition_targets = {
    head: Counter() for head in ("dent", "label", "foreign_material")
}
license_kind_by_role = {
    role: Counter() for role in ("train", "model_validation")
}
dataset_by_role = {role: Counter() for role in ("train", "model_validation")}

def bind_manifest_payload(path: Path, declared_sha: str, description: str) -> None:
    reject_symlink_components(path, description)
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"manifest payload is not a regular file: {path}")
    try:
        path.relative_to(global_root)
    except ValueError as error:
        raise ValueError("manifest payload escapes the read-only global mount") from error
    previous = manifest_payload_files.get(path)
    if previous is not None:
        if previous["sha256"] != declared_sha:
            raise ValueError(f"one payload path declares conflicting hashes: {path}")
        return
    content = stable_bytes(path, description)
    actual = sha_bytes(content)
    if actual != declared_sha:
        raise ValueError(f"manifest payload SHA mismatch: {path}")
    current = path.stat(follow_symlinks=False)
    if current.st_nlink != 1:
        raise ValueError(f"manifest payload hardlinks are forbidden: {path}")
    manifest_payload_files[path] = {
        "path": path.as_posix(), "size": len(content), "sha256": actual,
        "dev": current.st_dev, "ino": current.st_ino,
        "mode": stat.S_IMODE(current.st_mode), "nlink": current.st_nlink,
    }

for role, path in manifest_paths.items():
    content = input_bytes[path]
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{role} manifest is not UTF-8") from error
    reader = csv.DictReader(text.splitlines())
    fields = list(reader.fieldnames or [])
    if len(fields) != len(set(fields)) or set(fields) != REQUIRED_MANIFEST_FIELDS:
        raise ValueError(f"{role} manifest schema differs from the exact contract")
    count = 0
    expected_split = "training" if role == "train" else "validation"
    for number, row in enumerate(reader, start=2):
        count += 1
        if row.get("role") != role or row.get("split", "").casefold() != expected_split:
            raise ValueError(f"{role} manifest contains another role/split at line {number}")
        if row.get("fold") != role:
            raise ValueError(f"{role} manifest fold mismatch at line {number}")
        origin = str(row.get("origin", ""))
        normalized_origin = origin.casefold()
        if not origin or any(
            token in normalized_origin for token in FORBIDDEN_DIAGNOSTIC_TOKENS
        ):
            raise ValueError(f"diagnostic or empty row origin is forbidden at {path}:{number}")
        if role == "train":
            train_origin_counts[origin] += 1
        rule = license_origins.get(origin)
        if not isinstance(rule, dict):
            raise ValueError(f"unapproved license origin at {path}:{number}")
        try:
            material = int(str(row.get("material", "")))
            source_count = int(str(row.get("source_object_count", "")))
            crop_count = int(str(row.get("crop_object_count", "")))
        except ValueError as error:
            raise ValueError(f"invalid numeric manifest contract at {path}:{number}") from error
        if material not in range(10):
            raise ValueError(f"material must be 0..9 at {path}:{number}")
        material_names = (
            "can", "pet", "paper", "plastic", "styrofoam", "vinyl",
            "glass", "battery", "fluorescent",
        )
        expected_category = "background" if material == 9 else material_names[material]
        expected_crop_count = 0 if material == 9 else 1
        if row.get("category") != expected_category:
            raise ValueError(f"material/category mismatch at {path}:{number}")
        if source_count not in {0, 1} or crop_count != expected_crop_count or crop_count > source_count:
            raise ValueError(f"object-count contract mismatch at {path}:{number}")
        for head in ("dent", "label", "foreign_material"):
            target = str(row.get(head, ""))
            if target not in {"-1", "0", "1"}:
                raise ValueError(f"invalid {head} target at {path}:{number}")
            condition_targets_by_role[role][head][target] += 1
            condition_targets[head][target] += 1
        if rule["kind"] == "operational":
            captured_text = str(row.get("captured_at", ""))
            try:
                captured = datetime.fromisoformat(captured_text.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"invalid operational captured_at at {path}:{number}") from error
            if captured.tzinfo is None or captured.utcoffset() is None:
                raise ValueError(f"operational captured_at lacks timezone at {path}:{number}")
            cutoff = datetime(2026, 8, 1, tzinfo=ZoneInfo("Asia/Seoul"))
            if captured.astimezone(ZoneInfo("Asia/Seoul")) < cutoff or role != "train":
                raise ValueError(f"operational row violates cutoff/train-only policy at {path}:{number}")
            source_sha = str(row.get("source_sha256", ""))
            if operational_sources.get(source_sha) != {
                "auditor_sha256": row.get("auditor_sha256"),
                "teacher_output_sha256": row.get("teacher_output_sha256"),
                "localizer_output_sha256": row.get("localizer_output_sha256"),
            }:
                raise ValueError(f"operational evidence mismatch at {path}:{number}")
        selected_by_role[role] += 1
        selected_by_origin[origin] += 1
        material_by_role[role][expected_category] += 1
        objectness_by_role[role]["background" if material == 9 else "material"] += 1
        origin_by_role[role][origin] += 1
        license_kind_by_role[role][str(rule["kind"])] += 1
        dataset_by_role[role][str(rule["dataset_id"])] += 1
        for field in ("source_sha256", "image_sha256"):
            if not SHA_RE.fullmatch(str(row.get(field, ""))):
                raise ValueError(f"invalid {field} at {path}:{number}")
        image_path = Path(str(row.get("filepath", "")).strip())
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        source_path = Path(str(row.get("source_filepath", "")).strip())
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        bind_manifest_payload(
            image_path, str(row["image_sha256"]), f"{role} image line {number}"
        )
        bind_manifest_payload(
            source_path, str(row["source_sha256"]), f"{role} source line {number}"
        )
        resolved_image = image_path.resolve(strict=True)
        resolved_source = source_path.resolve(strict=True)
        for payload_path in (resolved_image, resolved_source):
            normalized_payload_path = payload_path.as_posix().casefold()
            if any(
                token in normalized_payload_path
                for token in FORBIDDEN_DIAGNOSTIC_TOKENS
            ):
                raise ValueError(
                    f"diagnostic payload path is forbidden at {path}:{number}"
                )
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"empty sample_id at {path}:{number}")
        manifest_audit_rows.add((
            role,
            sample_id,
            str(row["source_sha256"]),
            str(row["image_sha256"]),
        ))
        snapshot_entry = snapshot_by_sample.get(sample_id)
        if (
            not isinstance(snapshot_entry, dict)
            or snapshot_entry.get("role") != role
            or snapshot_entry.get("sha256") != row.get("image_sha256")
            or snapshot_entry.get("size") != manifest_payload_files[resolved_image]["size"]
            or resolved_image != (
                snapshot_root.joinpath(*Path(str(snapshot_entry.get("path"))).parts[1:])
            )
            or sample_id in snapshot_samples_seen
        ):
            raise ValueError(f"manifest row differs from dataset snapshot at {path}:{number}")
        snapshot_samples_seen.add(sample_id)
        dataset_content_inventory.append({
            "sample_id": sample_id,
            "role": role,
            "source_path": resolved_source.as_posix(),
            "source_size": manifest_payload_files[resolved_source]["size"],
            "source_sha256": str(row["source_sha256"]),
            "crop_path": resolved_image.as_posix(),
            "crop_size": manifest_payload_files[resolved_image]["size"],
            "crop_sha256": str(row["image_sha256"]),
        })
    if count == 0:
        raise ValueError(f"{role} manifest is empty")
    manifest_rows[role] = count

if snapshot_samples_seen != set(snapshot_by_sample):
    raise ValueError("manifest rows do not consume the exact dataset snapshot object set")
if manifest_audit_rows != candidate_audit_manifest_rows:
    raise ValueError(
        "near-duplicate candidate crop entries differ from the exact manifest rows"
    )

derived_candidate_counts = {
    "selected_by_role": dict(sorted(selected_by_role.items())),
    "selected_by_origin": dict(sorted(selected_by_origin.items())),
    "material_by_role": {
        role: dict(sorted(material_by_role[role].items()))
        for role in ("train", "model_validation")
    },
    "objectness_by_role": {
        role: dict(sorted(objectness_by_role[role].items()))
        for role in ("train", "model_validation")
    },
    "origin_by_role": {
        role: dict(sorted(origin_by_role[role].items()))
        for role in ("train", "model_validation")
    },
    "condition_targets_by_role": {
        role: {
            head: dict(sorted(condition_targets_by_role[role][head].items()))
            for head in ("dent", "label", "foreign_material")
        }
        for role in ("train", "model_validation")
    },
    "license_kind_by_role": {
        role: dict(sorted(license_kind_by_role[role].items()))
        for role in ("train", "model_validation")
    },
    "dataset_by_role": {
        role: dict(sorted(dataset_by_role[role].items()))
        for role in ("train", "model_validation")
    },
    "excluded": policy_excluded_counts,
    "condition_targets": {
        head: dict(sorted(condition_targets[head].items()))
        for head in ("dent", "label", "foreign_material")
    },
}
if policy_candidate_counts != derived_candidate_counts:
    raise ValueError("trusted policy candidate counts differ from actual manifests")
if authority.get("counts") != derived_candidate_counts:
    raise ValueError("training authority counts differ from actual manifests")

origin_weights = config["origin_weights"]
missing_weighted_origins = sorted(set(origin_weights).difference(train_origin_counts))
if missing_weighted_origins:
    raise ValueError(
        f"origin weights reference absent selected origins: {missing_weighted_origins}"
    )
weighted_mass = {
    origin: count * float(origin_weights.get(origin, 1.0))
    for origin, count in sorted(train_origin_counts.items())
}
total_weighted_mass = math.fsum(weighted_mass.values())
derived_sampling = {
    "mode": (
        "weighted_replacement"
        if len({float(origin_weights.get(origin, 1.0)) for origin in train_origin_counts}) > 1
        else "shuffle_without_replacement"
    ),
    "samples_per_epoch": sum(train_origin_counts.values()),
    "configured_origin_weights": dict(sorted(origin_weights.items())),
    "row_counts_by_origin": dict(sorted(train_origin_counts.items())),
    "weighted_mass_by_origin": {
        origin: float(mass) for origin, mass in weighted_mass.items()
    },
    "expected_fraction_by_origin": {
        origin: float(mass / total_weighted_mass)
        for origin, mass in weighted_mass.items()
    },
    "manifest_rows_remain_unique": True,
}
if config["sampling_mode"] != derived_sampling["mode"]:
    raise ValueError("configured sampling mode differs from manifests")
if config["sampling_samples_per_epoch"] != derived_sampling["samples_per_epoch"]:
    raise ValueError("configured samples_per_epoch differs from manifests")
configured_fractions = config["sampling_expected_fraction_by_origin"]
derived_fractions = derived_sampling["expected_fraction_by_origin"]
if set(configured_fractions) != set(derived_fractions) or any(
    not math.isclose(
        float(configured_fractions[origin]), fraction,
        rel_tol=0.0, abs_tol=1e-12,
    )
    for origin, fraction in derived_fractions.items()
):
    raise ValueError("configured sampling fractions differ from manifests")

dataset_content_inventory.sort(key=lambda row: (row["role"], row["sample_id"]))
if authority.get("dataset_content_inventory") != dataset_content_inventory:
    raise ValueError("authority dataset_content_inventory differs from manifest bytes")
dataset_inventory_bytes = (
    json.dumps(
        dataset_content_inventory, ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False,
    ) + "\n"
).encode("utf-8")
dataset_inventory_sha = sha_bytes(dataset_inventory_bytes)
if bindings.get("dataset_content_inventory_sha256") != dataset_inventory_sha:
    raise ValueError("dataset content inventory SHA binding mismatch")

bound_paths = [
    authority_path, marker_path, inventory_path, config_path, host_path,
    backbone_path, trainer_path, wrapper_path,
    policy_path, raw_inspect_path, qnap_snapshot_report_path,
    snapshot_report_path,
    manifest_paths["train"], manifest_paths["model_validation"],
]
snapshot_rows = [
    {"path": path.as_posix(), "size": len(input_bytes[path]), "sha256": sha_bytes(input_bytes[path])}
    for path in bound_paths
]
payload = {
    "schema": "v4_candidate_training_preflight.v1",
    "status": "candidate_training_preflight_passed",
    "candidate_only": True,
    "production_deployment_authorized": False,
    "container_image_id": image_id,
    "run_root": run_root.as_posix(),
    "run_dir": run_dir.as_posix(),
    "global_root": global_root.as_posix(),
    "code_root": code_root.as_posix(),
    "code_inventory": inventory_path.as_posix(),
    "code_inventory_sha256": sha_bytes(input_bytes[inventory_path]),
    "trusted_policy_path": policy_path.as_posix(),
    "trusted_policy_sha256": policy_sha,
    "raw_inspect_path": raw_inspect_path.as_posix(),
    "raw_inspect_sha256": raw_inspect_sha,
    "qnap_library_snapshot_report_path": qnap_snapshot_report_path.as_posix(),
    "qnap_library_snapshot_report_sha256": sha_bytes(
        input_bytes[qnap_snapshot_report_path]
    ),
    "qnap_library_snapshot": qnap_snapshot_report,
    "dataset_content_inventory_sha256": dataset_inventory_sha,
    "candidate_dataset_snapshot": dataset_snapshot_report,
    "candidate_dataset_snapshot_runtime": dataset_snapshot_runtime_contract,
    "dataset_consumption_contract": dataset_consumption_contract,
    "dataset_consumption_contract_sha256": dataset_consumption_contract_sha,
    "near_duplicate_audit": near_duplicate_audit,
    "near_duplicate_audit_sha256": near_duplicate_audit_sha,
    "manifests": [
        {
            "role": role,
            "path": manifest_paths[role].as_posix(),
            "sha256": manifest_hashes[role],
            "rows": manifest_rows[role],
        }
        for role in ("train", "model_validation")
    ],
    "training_config": config,
    "sampling": derived_sampling,
    "optimizer_runtime_contract": optimizer_runtime_contract,
    "runtime_dependency_contract": runtime_dependency_contract,
    "cuda_runtime_contract": cuda_runtime_contract,
    "mapped_qnap_library_contract": mapped_qnap_library_contract,
    "runtime_container_identity_contract": runtime_container_identity_contract,
    "training_config_path": config_path.as_posix(),
    "host_launch_contract_path": host_path.as_posix(),
    "pretrained_backbone_path": backbone_path.as_posix(),
    "torch_home": torch_home.as_posix(),
    "trainer_path": trainer_path.as_posix(),
    "trainer_sha256": sha_bytes(input_bytes[trainer_path]),
    "wrapper_path": wrapper_path.as_posix(),
    "bound_inputs": snapshot_rows,
    "manifest_payload_files": [
        manifest_payload_files[path] for path in sorted(
            manifest_payload_files, key=lambda value: value.as_posix()
        )
    ],
}


def publish(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


preflight_bytes = (
    json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
).encode("utf-8")
publish(preflight_path, preflight_bytes)
marker_lines = [
    f"{row['sha256']}  {row['path']}\n" for row in snapshot_rows
]
marker_lines.append(f"{sha_bytes(preflight_bytes)}  {preflight_path.resolve().as_posix()}\n")
publish(input_marker_path, "".join(marker_lines).encode("utf-8"))
PY
then
  fail "candidate training authority preflight failed" 65
fi

verify_inputs() {
  verify_qnap_snapshot || return 1
  if [ ! -f "$INPUT_MARKER" ] || [ ! -s "$INPUT_MARKER" ] || [ -L "$INPUT_MARKER" ]; then
    return 1
  fi
  sha256sum -c "$INPUT_MARKER" >/dev/null 2>&1 || return 1
  if ! "$PYTHON_BIN" - "$PREFLIGHT" "$CODE_INVENTORY" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

preflight_path = Path(sys.argv[1])
inventory_path = Path(sys.argv[2])
if preflight_path.is_symlink() or not preflight_path.is_file():
    raise ValueError("preflight must remain a regular non-symlink file")
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
consumption = preflight.get("dataset_consumption_contract")
if not isinstance(consumption, dict):
    raise ValueError("dataset consumption contract is missing")
consumption_sha = hashlib.sha256(
    (json.dumps(
        consumption, ensure_ascii=False, indent=2, sort_keys=True,
        allow_nan=False,
    ) + "\n").encode("utf-8")
).hexdigest()
if preflight.get("dataset_consumption_contract_sha256") != consumption_sha:
    raise ValueError("dataset consumption contract SHA changed")
if (
    preflight.get("trainer_sha256") != consumption.get("trainer_sha256")
    or consumption.get("version") != "multitask_verifier.image_consumption.v1"
    or consumption.get("max_image_bytes") != 67108864
    or consumption.get("max_image_pixels") != 16777216
    or consumption.get("complete_access_receipt") is not False
):
    raise ValueError("dataset consumption contract semantics changed")

def reject_symlink_components(path: Path, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path contains a symlink: {cursor}")

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for row in preflight["bound_inputs"]:
    path = Path(row["path"])
    reject_symlink_components(path, "bound input")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"bound input is no longer a regular file: {path}")
    if path.stat().st_size != row["size"] or sha(path) != row["sha256"]:
        raise ValueError(f"bound input changed: {path}")
global_root = Path(preflight["global_root"]).resolve(strict=True)
run_root = Path(preflight["run_root"]).resolve(strict=True)
if global_root == run_root or run_root in global_root.parents or global_root in run_root.parents:
    raise ValueError("GLOBAL_ROOT and RUN_ROOT are no longer disjoint")
for row in preflight["manifest_payload_files"]:
    path = Path(row["path"])
    reject_symlink_components(path, "manifest payload")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"manifest payload is no longer a regular file: {path}")
    try:
        path.resolve(strict=True).relative_to(global_root)
    except ValueError as error:
        raise ValueError("manifest payload escaped GLOBAL_ROOT") from error
    current = path.stat(follow_symlinks=False)
    if (
        current.st_size != row["size"]
        or current.st_dev != row["dev"]
        or current.st_ino != row["ino"]
        or stat.S_IMODE(current.st_mode) != row["mode"]
        or current.st_nlink != row["nlink"]
        or sha(path) != row["sha256"]
    ):
        raise ValueError(f"manifest payload changed: {path}")

snapshot_report = preflight["candidate_dataset_snapshot"]
snapshot_runtime = preflight["candidate_dataset_snapshot_runtime"]
snapshot_root = Path(snapshot_runtime["snapshot_root"])
reject_symlink_components(snapshot_root, "dataset snapshot root")
snapshot_root = snapshot_root.resolve(strict=True)
receipt = snapshot_runtime["publish_receipt"]
root_stat = snapshot_root.stat(follow_symlinks=False)
if (
    snapshot_root.is_symlink()
    or not snapshot_root.is_dir()
    or (root_stat.st_dev, root_stat.st_ino, stat.S_IMODE(root_stat.st_mode))
    != (receipt["root_dev"], receipt["root_ino"], receipt["root_mode"])
):
    raise ValueError("dataset snapshot root identity changed")
if sys.platform.startswith("linux") and stat.S_IMODE(root_stat.st_mode) != 0o555:
    raise ValueError("dataset snapshot root mode changed")
expected_snapshot_files = {}
expected_snapshot_directories = {snapshot_root}
for row in snapshot_report["objects"]:
    path = snapshot_root.joinpath(*Path(row["path"]).parts[1:])
    expected_snapshot_files[path] = row
    cursor = path.parent
    while cursor != snapshot_root:
        expected_snapshot_directories.add(cursor)
        cursor = cursor.parent
snapshot_entries = list(snapshot_root.rglob("*"))
if any(path.is_symlink() for path in snapshot_entries):
    raise ValueError("dataset snapshot gained a symlink")
actual_snapshot_files = {path for path in snapshot_entries if path.is_file()}
actual_snapshot_directories = {
    snapshot_root, *(path for path in snapshot_entries if path.is_dir())
}
if actual_snapshot_files != set(expected_snapshot_files):
    raise ValueError("dataset snapshot file set changed")
if actual_snapshot_directories != expected_snapshot_directories:
    raise ValueError("dataset snapshot directory set changed")
if any(not path.is_file() and not path.is_dir() for path in snapshot_entries):
    raise ValueError("dataset snapshot gained a special file")
for directory in actual_snapshot_directories:
    if sys.platform.startswith("linux") and stat.S_IMODE(
        directory.stat(follow_symlinks=False).st_mode
    ) != 0o555:
        raise ValueError("dataset snapshot directory mode changed")
snapshot_rows = []
tree_rows = []
for path in sorted(actual_snapshot_files, key=lambda value: value.as_posix()):
    expected = expected_snapshot_files[path]
    current = path.stat(follow_symlinks=False)
    digest = sha(path)
    if (
        current.st_nlink != 1
        or stat.S_IMODE(current.st_mode) != 0o444
        or current.st_size != expected["size"]
        or current.st_size > snapshot_report["object_max_bytes"]
        or digest != expected["sha256"]
    ):
        raise ValueError("dataset snapshot object changed")
    relative = path.relative_to(snapshot_root.parent).as_posix()
    snapshot_rows.append({
        "path": relative, "dev": current.st_dev, "ino": current.st_ino,
        "mode": stat.S_IMODE(current.st_mode), "nlink": current.st_nlink,
        "size": current.st_size, "sha256": digest,
    })
    tree_rows.append({"path": relative, "size": current.st_size, "sha256": digest})
if snapshot_rows != receipt["files"]:
    raise ValueError("dataset snapshot publish receipt changed")
tree_sha = hashlib.sha256(
    (json.dumps(
        tree_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("utf-8")
).hexdigest()
if tree_sha != snapshot_report["tree_sha256"] or tree_sha != snapshot_runtime["tree_sha256"]:
    raise ValueError("dataset snapshot tree SHA changed")

inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
root = Path(inventory["root"]).resolve(strict=True)
excluded_relatives = {
    "configs/v4_candidate_training_trusted_policy.json",
    "scripts/build_v4_candidate_training_authority.py",
    "scripts/nas/run_v4_candidate_training.sh",
}
expected = {}
for row in inventory["files"]:
    if Path(row["path"]).as_posix() in excluded_relatives:
        raise ValueError("inventory unexpectedly contains a fixed trust-root script")
    path = (root / row["path"]).resolve(strict=True)
    reject_symlink_components(path, "inventoried code")
    expected[path] = (row["size"], row["sha256"])
actual = {
    path.resolve()
    for path in root.rglob("*")
    if path.is_file() and not path.is_symlink()
}
excluded = {
    (root / relative).resolve(strict=True)
    for relative in excluded_relatives
    if (root / relative).exists()
}
if actual - excluded != set(expected):
    raise ValueError("CODE_ROOT file set changed")
for path, (size, digest) in expected.items():
    if path.stat().st_size != size or sha(path) != digest:
        raise ValueError(f"inventoried code changed: {path}")
PY
  then
    return 1
  fi
  verify_qnap_snapshot || return 1
}

verify_inputs || fail "bound inputs changed after preflight" 65

DRY_RUN_REPORT=$CONTROL/training_dry_run.json
run_verified_trainer() {
  mode=$1
  "$PYTHON_BIN" - "$PREFLIGHT" "$DRY_RUN_REPORT" "$CANDIDATE" "$mode" <<'PY'
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

VERIFIED_WORKER_BOOTSTRAP = r'''
import hashlib as _hashlib
import importlib.machinery as _machinery
import json as _json
import os as _os
import sys as _sys
import types as _types
import zipfile as _zipfile

_ENTRY_NAMES = (
    "__main__.py",
    "scripts/__init__.py",
    "scripts/train_verifier.py",
    "trainer.py",
    "verified_contract.json",
)
_suffix = "/__main__.py"
_bootstrap_file = globals().get("__file__")
if not isinstance(_bootstrap_file, str) or not _bootstrap_file.endswith(_suffix):
    raise RuntimeError("verified worker bootstrap did not start from its sealed archive")
_archive_path = _bootstrap_file[:-len(_suffix)]
with _zipfile.ZipFile(_archive_path, "r") as _archive:
    _infos = _archive.infolist()
    if [item.filename for item in _infos] != list(_ENTRY_NAMES):
        raise RuntimeError("verified worker archive entry set/order mismatch")
    if any(
        item.is_dir()
        or item.compress_type != _zipfile.ZIP_STORED
        or item.file_size > 16 * 1024 * 1024
        for item in _infos
    ):
        raise RuntimeError("verified worker archive contains an invalid entry")
    _entries = {item.filename: _archive.read(item) for item in _infos}
if _entries["scripts/__init__.py"] != b"":
    raise RuntimeError("verified scripts package marker must be empty")
_contract = _json.loads(_entries["verified_contract.json"].decode("utf-8"))
if not isinstance(_contract, dict) or set(_contract) != {
    "schema", "trainer_path", "trainer_sha256",
    "legacy_path", "legacy_sha256",
}:
    raise RuntimeError("verified worker contract schema mismatch")
if _contract.get("schema") != "v4_verified_trainer_archive.v1":
    raise RuntimeError("unsupported verified worker contract")

def _require_sha256(value, description):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{description} is not a lowercase SHA-256")
    return value

_trainer_path = _contract.get("trainer_path")
_legacy_path = _contract.get("legacy_path")
if not isinstance(_trainer_path, str) or not _os.path.isabs(_trainer_path):
    raise RuntimeError("verified trainer path is not absolute")
if not isinstance(_legacy_path, str) or not _os.path.isabs(_legacy_path):
    raise RuntimeError("verified legacy dependency path is not absolute")
_trainer_sha256 = _require_sha256(
    _contract.get("trainer_sha256"), "verified trainer SHA-256"
)
_legacy_sha256 = _require_sha256(
    _contract.get("legacy_sha256"), "verified legacy dependency SHA-256"
)
_trainer_source = _entries["trainer.py"]
_legacy_source = _entries["scripts/train_verifier.py"]
if _hashlib.sha256(_trainer_source).hexdigest() != _trainer_sha256:
    raise RuntimeError("sealed worker trainer bytes differ from the bound SHA-256")
if _hashlib.sha256(_legacy_source).hexdigest() != _legacy_sha256:
    raise RuntimeError("sealed worker dependency bytes differ from the code inventory")

_trainer_parent = _os.path.dirname(_trainer_path)
if not _sys.argv:
    _sys.argv = [_trainer_path]
else:
    _sys.argv[0] = _trainer_path
if not _sys.path or _sys.path[0] != _archive_path:
    raise RuntimeError("spawn worker did not enter through the sealed archive")
# Keep the direct-script import context while leaving one archive entry for
# runpy.run_path() to remove after this bootstrap returns.
_sys.path[0] = _trainer_parent
_sys.path.insert(1, _archive_path)
if hasattr(_sys, "orig_argv"):
    _sys.orig_argv = [_sys.executable, *_sys.argv]

_scripts_package = _types.ModuleType("scripts")
_scripts_package.__file__ = None
_scripts_package.__package__ = "scripts"
_scripts_package.__path__ = [_trainer_parent]
_scripts_package.__loader__ = None
_scripts_package.__spec__ = _machinery.ModuleSpec(
    "scripts", loader=None, is_package=True
)
_scripts_package.__spec__.submodule_search_locations = [_trainer_parent]
_sys.modules["scripts"] = _scripts_package
_legacy_module = _types.ModuleType("scripts.train_verifier")
_legacy_module.__file__ = _legacy_path
_legacy_module.__cached__ = None
_legacy_module.__package__ = "scripts"
_legacy_module.__loader__ = None
_legacy_module.__spec__ = _machinery.ModuleSpec(
    "scripts.train_verifier", loader=None
)
_sys.modules["scripts.train_verifier"] = _legacy_module
_sys.modules["train_verifier"] = _legacy_module
exec(
    compile(_legacy_source, _legacy_path, "exec", dont_inherit=True),
    _legacy_module.__dict__,
    _legacy_module.__dict__,
)

globals()["__file__"] = _trainer_path
globals()["__cached__"] = None
globals()["__package__"] = None
globals()["__loader__"] = None
globals()["__spec__"] = None
exec(
    compile(_trainer_source, _trainer_path, "exec", dont_inherit=True),
    globals(),
    globals(),
)
'''

VERIFIED_TRAINER_LOADER = r'''
import hashlib
import importlib.machinery
import io
import json
import os
import stat
import sys
import types
import zipfile

(
    expected_archive_sha256,
    expected_bootstrap_sha256,
    expected_trainer_sha256,
    trainer_path,
    expected_legacy_sha256,
    legacy_path,
    *trainer_argv,
) = sys.argv[1:]

def require_sha256(value, description):
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} is invalid")
    return value

for value, description in (
    (expected_archive_sha256, "expected archive SHA-256"),
    (expected_bootstrap_sha256, "expected bootstrap SHA-256"),
    (expected_trainer_sha256, "expected trainer SHA-256"),
    (expected_legacy_sha256, "expected legacy dependency SHA-256"),
):
    require_sha256(value, description)
if not os.path.isabs(trainer_path) or not os.path.isabs(legacy_path):
    raise ValueError("verified trainer and dependency paths must be absolute")

archive_bytes = sys.stdin.buffer.read(40 * 1024 * 1024 + 1)
if not archive_bytes or len(archive_bytes) > 40 * 1024 * 1024:
    raise ValueError("verified trainer archive is empty or oversized")
if hashlib.sha256(archive_bytes).hexdigest() != expected_archive_sha256:
    raise ValueError("verified trainer archive transfer SHA-256 mismatch")
expected_names = (
    "__main__.py",
    "scripts/__init__.py",
    "scripts/train_verifier.py",
    "trainer.py",
    "verified_contract.json",
)
with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
    infos = archive.infolist()
    if [item.filename for item in infos] != list(expected_names):
        raise ValueError("verified trainer archive entry set/order mismatch")
    if any(
        item.is_dir()
        or item.compress_type != zipfile.ZIP_STORED
        or item.file_size > 16 * 1024 * 1024
        for item in infos
    ):
        raise ValueError("verified trainer archive contains an invalid entry")
    entries = {item.filename: archive.read(item) for item in infos}
if entries["scripts/__init__.py"] != b"":
    raise ValueError("verified scripts package marker must be empty")
if hashlib.sha256(entries["__main__.py"]).hexdigest() != expected_bootstrap_sha256:
    raise ValueError("verified worker bootstrap SHA-256 mismatch")
if hashlib.sha256(entries["trainer.py"]).hexdigest() != expected_trainer_sha256:
    raise ValueError("verified trainer source SHA-256 mismatch")
if (
    hashlib.sha256(entries["scripts/train_verifier.py"]).hexdigest()
    != expected_legacy_sha256
):
    raise ValueError("verified legacy dependency SHA-256 mismatch")
contract = json.loads(entries["verified_contract.json"].decode("utf-8"))
if contract != {
    "schema": "v4_verified_trainer_archive.v1",
    "trainer_path": trainer_path,
    "trainer_sha256": expected_trainer_sha256,
    "legacy_path": legacy_path,
    "legacy_sha256": expected_legacy_sha256,
}:
    raise ValueError("verified trainer archive contract mismatch")

archive_descriptor = None
archive_path = None
if sys.platform.startswith("linux"):
    import fcntl
    import multiprocessing.popen_spawn_posix as popen_spawn_posix
    import multiprocessing.spawn as multiprocessing_spawn

    required_os_names = ("memfd_create", "MFD_ALLOW_SEALING", "MFD_CLOEXEC")
    required_fcntl_names = (
        "F_ADD_SEALS", "F_GET_SEALS", "F_SEAL_SEAL",
        "F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_WRITE",
    )
    if any(not hasattr(os, name) for name in required_os_names):
        raise RuntimeError("Linux runtime lacks sealed memfd support")
    if any(not hasattr(fcntl, name) for name in required_fcntl_names):
        raise RuntimeError("Linux runtime lacks required memfd seal constants")
    archive_descriptor = os.memfd_create(
        "v4-verified-trainer",
        os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    try:
        offset = 0
        while offset < len(archive_bytes):
            written = os.write(archive_descriptor, archive_bytes[offset:])
            if written <= 0:
                raise OSError("verified trainer archive memfd write made no progress")
            offset += written
        os.lseek(archive_descriptor, 0, os.SEEK_SET)
        required_seals = (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
        fcntl.fcntl(archive_descriptor, fcntl.F_ADD_SEALS, required_seals)
        actual_seals = fcntl.fcntl(archive_descriptor, fcntl.F_GET_SEALS)
        if actual_seals & required_seals != required_seals:
            raise RuntimeError("verified trainer archive memfd is not fully sealed")
        sealed_copy = bytearray()
        while len(sealed_copy) < len(archive_bytes):
            chunk = os.read(
                archive_descriptor,
                min(1024 * 1024, len(archive_bytes) - len(sealed_copy)),
            )
            if not chunk:
                break
            sealed_copy.extend(chunk)
        if bytes(sealed_copy) != archive_bytes:
            raise RuntimeError("sealed trainer archive bytes changed")
        os.lseek(archive_descriptor, 0, os.SEEK_SET)
        archive_path = f"/proc/self/fd/{archive_descriptor}"
        with zipfile.ZipFile(archive_path, "r") as sealed_archive:
            if sealed_archive.namelist() != list(expected_names):
                raise RuntimeError("sealed trainer archive cannot be reopened exactly")
    except BaseException:
        os.close(archive_descriptor)
        raise

    original_get_preparation_data = multiprocessing_spawn.get_preparation_data
    original_launch = popen_spawn_posix.Popen._launch
    if not callable(original_get_preparation_data) or not callable(original_launch):
        raise RuntimeError("pinned multiprocessing spawn hooks are unavailable")

    def verified_get_preparation_data(process_name):
        data = original_get_preparation_data(process_name)
        if "init_main_from_name" in data:
            raise RuntimeError("spawn unexpectedly selected module-name main import")
        original_main = data.get("init_main_from_path")
        if (
            not isinstance(original_main, str)
            or os.path.normpath(original_main) != os.path.normpath(trainer_path)
        ):
            raise RuntimeError("spawn main path differs from the verified trainer")
        data["init_main_from_path"] = archive_path
        return data

    def verified_launch(self, process_object):
        descriptors = getattr(self, "_fds", None)
        if not isinstance(descriptors, list):
            raise RuntimeError("pinned spawn Popen descriptor list is unavailable")
        if archive_descriptor not in descriptors:
            descriptors.append(archive_descriptor)
        return original_launch(self, process_object)

    multiprocessing_spawn.get_preparation_data = verified_get_preparation_data
    popen_spawn_posix.Popen._launch = verified_launch

trainer_source = entries["trainer.py"]
legacy_source = entries["scripts/train_verifier.py"]
trainer_parent = os.path.dirname(trainer_path)
sys.argv = [trainer_path, *trainer_argv]
if hasattr(sys, "orig_argv"):
    sys.orig_argv = [sys.executable, trainer_path, *trainer_argv]
if sys.path:
    sys.path[0] = trainer_parent
else:
    sys.path.append(trainer_parent)

scripts_package = types.ModuleType("scripts")
scripts_package.__file__ = None
scripts_package.__package__ = "scripts"
scripts_package.__path__ = [trainer_parent]
scripts_package.__loader__ = None
scripts_package.__spec__ = importlib.machinery.ModuleSpec(
    "scripts", loader=None, is_package=True
)
scripts_package.__spec__.submodule_search_locations = [trainer_parent]
sys.modules["scripts"] = scripts_package
legacy_module = types.ModuleType("scripts.train_verifier")
legacy_module.__file__ = legacy_path
legacy_module.__cached__ = None
legacy_module.__package__ = "scripts"
legacy_module.__loader__ = None
legacy_module.__spec__ = importlib.machinery.ModuleSpec(
    "scripts.train_verifier", loader=None
)
sys.modules["scripts.train_verifier"] = legacy_module
sys.modules["train_verifier"] = legacy_module
exec(
    compile(legacy_source, legacy_path, "exec", dont_inherit=True),
    legacy_module.__dict__,
    legacy_module.__dict__,
)

main_module = types.ModuleType("__main__")
main_module.__file__ = trainer_path
main_module.__cached__ = None
main_module.__package__ = None
main_module.__loader__ = None
main_module.__spec__ = None
sys.modules["__main__"] = main_module
exec(
    compile(trainer_source, trainer_path, "exec", dont_inherit=True),
    main_module.__dict__,
    main_module.__dict__,
)
'''

def stable_file_bytes(path, description, maximum_bytes=16 * 1024 * 1024):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise ValueError(f"{description} must be a bounded regular file")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise ValueError(f"{description} changed while its bytes were read")
    content = b"".join(chunks)
    if len(content) != before.st_size:
        raise ValueError(f"{description} read was incomplete")
    return content

def require_sha256(value, description):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} is not a lowercase SHA-256")
    return value

preflight_path = Path(sys.argv[1])
preflight = json.loads(
    stable_file_bytes(preflight_path, "training preflight", 32 * 1024 * 1024)
)
report_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
mode = sys.argv[4]
if mode not in {"dry-run", "train"}:
    raise ValueError("unsupported verified trainer mode")
config = preflight["training_config"]
manifests = {row["role"]: row["path"] for row in preflight["manifests"]}
code_root = Path(preflight["code_root"])
trainer_path = Path(preflight["trainer_path"])
legacy_path = code_root / "scripts" / "train_verifier.py"
if not code_root.is_absolute() or not trainer_path.is_absolute():
    raise ValueError("verified code/trainer paths must be absolute")
if trainer_path != code_root / "scripts" / "train_multitask_verifier.py":
    raise ValueError("preflight trainer path differs from the fixed trainer")
expected_trainer_sha256 = require_sha256(
    preflight["trainer_sha256"], "preflight trainer SHA-256"
)
if (
    preflight.get("dataset_consumption_contract", {}).get("trainer_sha256")
    != expected_trainer_sha256
):
    raise ValueError("preflight consumption contract trainer SHA mismatch")
inventory_bytes = stable_file_bytes(
    Path(preflight["code_inventory"]), "code inventory", 32 * 1024 * 1024
)
if hashlib.sha256(inventory_bytes).hexdigest() != require_sha256(
    preflight["code_inventory_sha256"], "preflight code inventory SHA-256"
):
    raise ValueError("code inventory differs from the preflight SHA-256")
inventory = json.loads(inventory_bytes)
if not isinstance(inventory, dict) or inventory.get("schema") != "v4_candidate_code_inventory.v1":
    raise ValueError("unsupported verified code inventory")
inventory_rows = inventory.get("files")
if not isinstance(inventory_rows, list):
    raise ValueError("verified code inventory files are absent")
inventory_by_path = {}
for row in inventory_rows:
    if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
        raise ValueError("verified code inventory row is malformed")
    relative = row["path"]
    if not isinstance(relative, str) or relative in inventory_by_path:
        raise ValueError("verified code inventory path is invalid or duplicated")
    inventory_by_path[relative] = row
trainer_row = inventory_by_path.get("scripts/train_multitask_verifier.py")
legacy_row = inventory_by_path.get("scripts/train_verifier.py")
if not isinstance(trainer_row, dict) or not isinstance(legacy_row, dict):
    raise ValueError("trainer or legacy dependency is absent from the code inventory")
if require_sha256(
    trainer_row.get("sha256"), "inventoried trainer SHA-256"
) != expected_trainer_sha256:
    raise ValueError("trainer preflight and code inventory SHA-256 differ")
expected_legacy_sha256 = require_sha256(
    legacy_row.get("sha256"), "inventoried legacy dependency SHA-256"
)
trainer_source = stable_file_bytes(trainer_path, "trainer source")
legacy_source = stable_file_bytes(legacy_path, "legacy trainer dependency")
if (
    len(trainer_source) != trainer_row.get("size")
    or hashlib.sha256(trainer_source).hexdigest() != expected_trainer_sha256
):
    raise ValueError("trainer source differs from its verified inventory bytes")
if (
    len(legacy_source) != legacy_row.get("size")
    or hashlib.sha256(legacy_source).hexdigest() != expected_legacy_sha256
):
    raise ValueError("legacy dependency differs from its verified inventory bytes")

archive_contract = {
    "schema": "v4_verified_trainer_archive.v1",
    "trainer_path": trainer_path.as_posix(),
    "trainer_sha256": expected_trainer_sha256,
    "legacy_path": legacy_path.as_posix(),
    "legacy_sha256": expected_legacy_sha256,
}
archive_entries = {
    "__main__.py": VERIFIED_WORKER_BOOTSTRAP.encode("utf-8"),
    "scripts/__init__.py": b"",
    "scripts/train_verifier.py": legacy_source,
    "trainer.py": trainer_source,
    "verified_contract.json": json.dumps(
        archive_contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"),
}
archive_buffer = io.BytesIO()
with zipfile.ZipFile(
    archive_buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=False
) as archive:
    for name in sorted(archive_entries):
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100444 << 16
        archive.writestr(info, archive_entries[name])
archive_bytes = archive_buffer.getvalue()
archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
bootstrap_sha256 = hashlib.sha256(archive_entries["__main__.py"]).hexdigest()

args = [
    "--manifest", manifests["train"],
    "--manifest", manifests["model_validation"],
    "--backbone", config["backbone"],
    "--size", str(config["input_size"]),
    "--epochs", str(config["epochs"]),
    "--patience", str(config["patience"]),
    "--batch", str(config["batch"]),
    "--workers", str(config["workers"]),
    "--lr", str(config["lr"]),
    "--backbone-lr", str(config["backbone_lr"]),
    "--head-lr", str(config["head_lr"]),
    "--label-smoothing", str(config["label_smoothing"]),
    "--class-weight-mode", config["class_weight_mode"],
    "--class-weight-beta", str(config["class_weight_beta"]),
    "--objectness-weight", str(config["objectness_weight"]),
    "--material-weight", str(config["material_weight"]),
    "--condition-weight", str(config["condition_weight"]),
    "--seed", str(config["seed"]),
    "--device", "cuda",
    "--dataset-snapshot-report-sha256",
    preflight["dataset_consumption_contract"]["dataset_snapshot_report_sha256"],
    "--dataset-snapshot-tree-sha256",
    preflight["dataset_consumption_contract"]["dataset_snapshot_tree_sha256"],
    "--manifest-payload-set-sha256",
    preflight["dataset_consumption_contract"]["manifest_payload_set_sha256"],
]
for head in config["condition_heads"]:
    args.extend(("--condition-head", head))
for origin, weight in sorted(config["origin_weights"].items()):
    args.extend(("--origin-weight", f"{origin}={weight}"))
env = os.environ.copy()
env["TORCH_HOME"] = preflight["torch_home"]
command = [
    sys.executable,
    "-c",
    VERIFIED_TRAINER_LOADER,
    archive_sha256,
    bootstrap_sha256,
    expected_trainer_sha256,
    trainer_path.as_posix(),
    expected_legacy_sha256,
    legacy_path.as_posix(),
    *args,
]
if mode == "dry-run":
    command.append("--dry-run")
    result = subprocess.run(
        command, env=env, input=archive_bytes, capture_output=True, check=False
    )
    if result.returncode != 0:
        os.write(2, result.stdout)
        os.write(2, result.stderr)
        raise SystemExit(result.returncode)
    value = json.loads(result.stdout.decode("utf-8"))
    if set(value) != {
        "ok", "mode", "seed", "manifest", "condition_heads",
        "class_weights", "sampling", "output_contract",
        "dataset_consumption_contract",
    }:
        raise ValueError("trainer dry-run response schema mismatch")
    if value.get("ok") is not True or value.get("mode") != "dry-run":
        raise ValueError("trainer dry-run did not return the strict success contract")
    if value.get("seed") != config["seed"]:
        raise ValueError("trainer dry-run seed mismatch")
    if value.get("condition_heads") != config["condition_heads"]:
        raise ValueError("trainer dry-run condition-head mismatch")
    if value.get("sampling") != preflight["sampling"]:
        raise ValueError("trainer dry-run sampling plan mismatch")
    if value.get("dataset_consumption_contract") != preflight.get(
        "dataset_consumption_contract"
    ):
        raise ValueError("trainer dry-run consumption contract mismatch")
    if value.get("output_contract", {}).get("version") != "multitask_verifier.v3":
        raise ValueError("trainer dry-run output contract mismatch")
    if value.get("output_contract", {}).get("output_order") != [
        "objectness", "material", "dent", "label", "foreign_material"
    ]:
        raise ValueError("trainer dry-run output order mismatch")
    class_weights = value.get("class_weights")
    if not isinstance(class_weights, dict):
        raise ValueError("trainer dry-run class weights are missing")
    if class_weights.get("mode") != config["class_weight_mode"]:
        raise ValueError("trainer dry-run class-weight mode mismatch")
    if class_weights.get("beta") != config["class_weight_beta"]:
        raise ValueError("trainer dry-run class-weight beta mismatch")
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(report_path)
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    os.link(temporary, report_path)
    temporary.unlink()
else:
    command.extend(("--output-dir", output_dir.as_posix()))
    raise SystemExit(
        subprocess.run(
            command, env=env, input=archive_bytes, check=False
        ).returncode
    )
PY
}
if ! run_verified_trainer dry-run
then
  fail "candidate trainer dry-run failed" 65
fi

verify_inputs || fail "bound inputs changed during trainer dry-run" 65
DRY_MARKER=$CONTROL/dry_run.sha256
temporary=$(mktemp "$CONTROL/.dry-run.XXXXXX") || fail "failed to stage dry-run marker"
sha256sum "$INPUT_MARKER" "$PREFLIGHT" "$DRY_RUN_REPORT" > "$temporary" || \
  fail "failed to hash dry-run evidence"
if ! ln "$temporary" "$DRY_MARKER" 2>/dev/null; then
  rm -f "$temporary"
  fail "refusing to overwrite dry-run marker" 73
fi
rm -f "$temporary"
sha256sum -c "$DRY_MARKER" >/dev/null 2>&1 || fail "dry-run marker verification failed" 65
verify_inputs || fail "bound inputs changed before candidate training" 65

if [ -e "$CANDIDATE" ] || [ -L "$CANDIDATE" ]; then
  fail "candidate output directory already exists" 73
fi
if ! mkdir "$CANDIDATE" 2>/dev/null; then
  fail "failed to create unique candidate output directory" 73
fi
CANDIDATE_IDENTITY=$CONTROL/candidate_dir_identity.json
if ! "$PYTHON_BIN" - "$RUN_DIR" "$CANDIDATE" "$CANDIDATE_IDENTITY" <<'PY'
import json
import os
import sys
from pathlib import Path

run_dir_arg, candidate_arg, identity_path = map(Path, sys.argv[1:])

def reject_symlink_components(path: Path, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path contains a symlink: {cursor}")

reject_symlink_components(run_dir_arg, "RUN_DIR")
reject_symlink_components(candidate_arg, "candidate output")
run_dir = run_dir_arg.resolve(strict=True)
candidate = candidate_arg.resolve(strict=True)
if candidate.parent != run_dir or candidate.name != "candidate":
    raise ValueError("candidate output must be the exact direct child of RUN_DIR")
if not candidate.is_dir() or candidate.is_symlink() or any(candidate.iterdir()):
    raise ValueError("candidate output must be a new empty non-symlink directory")
run_stat = run_dir.stat()
candidate_stat = candidate.stat()
payload = {
    "schema": "v4_candidate_output_directory_identity.v1",
    "run_dir": run_dir.as_posix(),
    "run_dev": run_stat.st_dev,
    "run_ino": run_stat.st_ino,
    "candidate_dir": candidate.as_posix(),
    "candidate_dev": candidate_stat.st_dev,
    "candidate_ino": candidate_stat.st_ino,
}
if identity_path.exists() or identity_path.is_symlink():
    raise FileExistsError(identity_path)
temporary = identity_path.with_name(f".{identity_path.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
os.link(temporary, identity_path)
temporary.unlink()
PY
then
  fail "candidate output directory identity publication failed" 65
fi

verify_candidate_dir() {
  "$PYTHON_BIN" - "$CANDIDATE_IDENTITY" <<'PY'
import json
import os
import sys
from pathlib import Path

identity_path = Path(sys.argv[1])
if identity_path.is_symlink() or not identity_path.is_file():
    raise ValueError("candidate identity must remain a regular file")
value = json.loads(identity_path.read_text(encoding="utf-8"))

def reject_symlink_components(path: Path, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path contains a symlink: {cursor}")

run_dir = Path(value["run_dir"])
candidate = Path(value["candidate_dir"])
reject_symlink_components(run_dir, "RUN_DIR")
reject_symlink_components(candidate, "candidate output")
run_dir = run_dir.resolve(strict=True)
candidate = candidate.resolve(strict=True)
if candidate.parent != run_dir or candidate.name != "candidate":
    raise ValueError("candidate output escaped RUN_DIR")
run_stat = run_dir.stat()
candidate_stat = candidate.stat()
if (run_stat.st_dev, run_stat.st_ino) != (value["run_dev"], value["run_ino"]):
    raise ValueError("RUN_DIR identity changed")
if (candidate_stat.st_dev, candidate_stat.st_ino) != (
    value["candidate_dev"], value["candidate_ino"]
):
    raise ValueError("candidate output directory identity changed")
PY
}

verify_candidate_dir || fail "candidate output directory changed before training" 65
if ! run_verified_trainer train
then
  fail "candidate-only verifier training failed"
fi

verify_inputs || fail "bound inputs changed during candidate training" 65
verify_candidate_dir || fail "candidate output directory changed during training" 65
sha256sum -c "$DRY_MARKER" >/dev/null 2>&1 || fail "dry-run evidence changed during training" 65

METADATA=$CANDIDATE/multitask_verifier_metadata.json
CHECKPOINT=$CANDIDATE/best_multitask_verifier.pt
ONNX=$CANDIDATE/multitask_verifier.onnx
OUTPUT_INVENTORY=$CONTROL/candidate_outputs.json
if ! "$PYTHON_BIN" - \
  "$PREFLIGHT" "$METADATA" "$CHECKPOINT" "$ONNX" "$OUTPUT_INVENTORY" \
  "$CANDIDATE_IDENTITY" <<'PY'
import hashlib
import io
import json
import os
import posixpath
import re
import socket
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torchvision
from torchvision import models

(
    preflight_path, metadata_path, checkpoint_path, onnx_path, inventory_path,
    candidate_identity_path,
) = map(Path, sys.argv[1:])
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
candidate_identity = json.loads(candidate_identity_path.read_text(encoding="utf-8"))
runtime_dependency_contract = {
    "onnx": str(onnx.__version__),
    "onnxruntime": str(ort.__version__),
    "onnxruntime_providers": sorted(ort.get_available_providers()),
    "torch": str(torch.__version__),
    "torchvision": str(torchvision.__version__),
}
if preflight.get("runtime_dependency_contract") != runtime_dependency_contract:
    raise ValueError("runtime dependencies changed after preflight")
if "CPUExecutionProvider" not in runtime_dependency_contract["onnxruntime_providers"]:
    raise ValueError("onnxruntime lacks required CPUExecutionProvider")
if sys.platform.startswith("linux"):
    current_hostname = socket.gethostname()
    current_cgroup_ids = sorted({
        match
        for path in (Path("/proc/self/cgroup"), Path("/proc/1/cgroup"))
        for match in re.findall(
            r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])",
            path.read_text(encoding="utf-8", errors="strict"),
        )
    })
    current_runtime_container_identity_contract = {
        "platform": "linux",
        "hostname": current_hostname,
        "cgroup_container_ids": current_cgroup_ids,
        "identity_evidence": (
            "cgroup_and_hostname" if current_cgroup_ids
            else "hostname_with_trusted_host_attestation_fallback"
        ),
    }
else:
    current_runtime_container_identity_contract = {
        "platform": sys.platform,
        "identity_evidence": "not_applicable_non_linux_test_host",
    }
if preflight.get(
    "runtime_container_identity_contract"
) != current_runtime_container_identity_contract:
    raise ValueError("container identity contract changed after preflight")

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def stable_bytes(path: Path, description: str) -> bytes:
    before = path.stat()
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        content = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = path.stat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
    )
    if not (
        identity(before) == identity(opened_before) ==
        identity(opened_after) == identity(after)
    ):
        raise RuntimeError(f"{description} changed while being read")
    return content


def collect_qnap_mapped_library_contract(snapshot_report):
    inventory_value = snapshot_report["inventory"]
    snapshot_root = Path(snapshot_report["snapshot_root"])
    tree_reports = {
        row["container_root"]: row for row in snapshot_report["trees"]
    }
    entries_by_root = {
        tree["container_root"]: {entry["path"]: entry for entry in tree["entries"]}
        for tree in inventory_value["trees"]
    }

    def terminal_relative(container_root, relative):
        entries = entries_by_root[container_root]
        cursor = relative
        visited = set()
        while True:
            if cursor in visited:
                raise ValueError("required QNAP mapped library contains a symlink cycle")
            visited.add(cursor)
            entry = entries[cursor]
            if entry["type"] == "file":
                return cursor, entry
            cursor = posixpath.normpath(
                posixpath.join(posixpath.dirname(cursor), entry["target"])
            )

    required_terminal_rows = []
    required_absolute_paths = set()
    inventoried_files = {}
    inventoried_basenames = set()
    for container_root, entries in entries_by_root.items():
        destination_root = (
            Path(tree_reports[container_root]["snapshot_root"])
            if container_root in tree_reports
            else snapshot_root / (
                "nvidia/lib" if container_root == "/qnap/nvidia/lib" else "cuda/lib64"
            )
        )
        for relative, entry in entries.items():
            inventoried_basenames.add(posixpath.basename(relative))
            if entry["type"] == "file":
                inventoried_files[(destination_root / relative).as_posix()] = (
                    container_root, relative, entry
                )
    for required in inventory_value["required_mapped_libraries"]:
        container_root = required["container_root"]
        terminal, entry = terminal_relative(container_root, required["path"])
        destination_root = (
            Path(tree_reports[container_root]["snapshot_root"])
            if container_root in tree_reports
            else snapshot_root / (
                "nvidia/lib" if container_root == "/qnap/nvidia/lib" else "cuda/lib64"
            )
        )
        absolute = (destination_root / terminal).as_posix()
        required_absolute_paths.add(absolute)
        required_terminal_rows.append({
            "container_root": container_root,
            "declared_path": required["path"],
            "terminal_path": terminal,
            "snapshot_path": absolute,
            "size": entry["size"],
            "sha256": entry["sha256"],
        })
    required_terminal_rows.sort(
        key=lambda row: (row["container_root"], row["declared_path"])
    )
    if not sys.platform.startswith("linux"):
        return {
            "status": "not_applicable_non_linux_test_host",
            "observer_role": "launcher_boundary_process_not_trainer_process",
            "platform": sys.platform,
            "required_terminal_files": required_terminal_rows,
            "mapped_files": [],
        }
    original_roots = ("/qnap/nvidia/lib", "/qnap/cuda/lib64")
    mapped_snapshot_paths = set()
    deleted_collisions = []
    outside_collisions = []
    for line in Path("/proc/self/maps").read_text(
        encoding="utf-8", errors="strict"
    ).splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        mapped_text = fields[5]
        deleted = mapped_text.endswith(" (deleted)")
        if deleted:
            mapped_text = mapped_text[:-10]
        basename = posixpath.basename(mapped_text)
        under_original = any(
            mapped_text == root or mapped_text.startswith(root + "/")
            for root in original_roots
        )
        under_snapshot = (
            mapped_text == snapshot_root.as_posix()
            or mapped_text.startswith(snapshot_root.as_posix() + "/")
        )
        collision = basename in inventoried_basenames
        if deleted and (under_original or under_snapshot or collision):
            deleted_collisions.append(fields[5])
        elif under_original or (collision and not under_snapshot):
            outside_collisions.append(mapped_text)
        elif under_snapshot:
            mapped_snapshot_paths.add(mapped_text)
    if deleted_collisions:
        raise ValueError(
            f"QNAP-related mapped libraries are deleted: {sorted(set(deleted_collisions))}"
        )
    if outside_collisions:
        raise ValueError(
            "QNAP-inventoried library basenames mapped outside the private snapshot: "
            f"{sorted(set(outside_collisions))}"
        )
    if not mapped_snapshot_paths:
        raise ValueError("no QNAP library was mapped from the private snapshot")
    missing_required = sorted(required_absolute_paths - mapped_snapshot_paths)
    if missing_required:
        raise ValueError(
            f"policy-required QNAP libraries were not mapped: {missing_required}"
        )
    rows = []
    for mapped_text in sorted(mapped_snapshot_paths):
        expected = inventoried_files.get(mapped_text)
        if expected is None:
            raise ValueError(f"mapped snapshot library is not inventoried: {mapped_text}")
        container_root, relative, entry = expected
        path = Path(mapped_text)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"mapped QNAP snapshot path is not regular: {path}")
        content = stable_bytes(path, "mapped QNAP snapshot library")
        value = path.stat(follow_symlinks=False)
        if len(content) != entry["size"] or hashlib.sha256(
            content
        ).hexdigest() != entry["sha256"]:
            raise ValueError(f"mapped QNAP snapshot bytes differ from inventory: {path}")
        rows.append({
            "container_root": container_root,
            "relative_path": relative,
            "path": mapped_text,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "dev": value.st_dev,
            "ino": value.st_ino,
        })
    return {
        "status": "qnap_snapshot_mappings_verified",
        "observer_role": "launcher_boundary_process_not_trainer_process",
        "platform": "linux",
        "required_terminal_files": required_terminal_rows,
        "mapped_files": rows,
    }

if sys.platform.startswith("linux"):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ValueError("CUDA became unavailable after candidate training")
    cuda_device = torch.device("cuda:0")
    cuda_properties = torch.cuda.get_device_properties(cuda_device)
    cuda_probe = torch.arange(4096, dtype=torch.float32, device=cuda_device)
    cuda_probe_result = float((cuda_probe.square().sum()).cpu().item())
    output_probe_model = models.mobilenet_v3_small(weights=None).eval().to(cuda_device)
    output_model_probe = torch.zeros(
        1, 3, 320, 320, dtype=torch.float32, device=cuda_device
    )
    with torch.no_grad():
        output_model_probe_result = output_probe_model(output_model_probe)
    if tuple(output_model_probe_result.shape) != (1, 1000):
        raise ValueError("post-training CUDA MobileNetV3 smoke shape mismatch")
    torch.cuda.synchronize(cuda_device)
    if not torch.isfinite(torch.tensor(cuda_probe_result)).item() or cuda_probe_result <= 0:
        raise ValueError("post-training CUDA smoke probe returned an invalid value")
    driver_version_bytes = stable_bytes(
        Path("/proc/driver/nvidia/version"), "live NVIDIA driver version"
    )
    current_cuda_runtime_contract = {
        "required": True,
        "device_count": torch.cuda.device_count(),
        "device_name": str(cuda_properties.name),
        "compute_capability": [cuda_properties.major, cuda_properties.minor],
        "total_memory_bytes": int(cuda_properties.total_memory),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_version_sha256": hashlib.sha256(driver_version_bytes).hexdigest(),
        "smoke_probe": "cuda:0 arange-square-sum-synchronize",
    }
    del (
        cuda_probe, output_probe_model, output_model_probe,
        output_model_probe_result,
    )
    torch.cuda.empty_cache()
else:
    current_cuda_runtime_contract = {
        "required": False,
        "platform": sys.platform,
        "smoke_probe": "not_applicable_non_linux_test_host",
    }
if preflight.get("cuda_runtime_contract") != current_cuda_runtime_contract:
    raise ValueError("CUDA runtime contract changed after preflight")

def reject_symlink_components(path: Path, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path contains a symlink: {cursor}")

for path, description in (
    (candidate_identity_path, "candidate identity"),
    (metadata_path, "candidate metadata"),
    (checkpoint_path, "candidate checkpoint"),
    (onnx_path, "candidate ONNX"),
):
    reject_symlink_components(path, description)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file")

candidate_root = metadata_path.parent.resolve(strict=True)
candidate_stat = candidate_root.stat()
if candidate_root.as_posix() != candidate_identity.get("candidate_dir"):
    raise ValueError("candidate output path differs from its recorded identity")
if (candidate_stat.st_dev, candidate_stat.st_ino) != (
    candidate_identity.get("candidate_dev"), candidate_identity.get("candidate_ino")
):
    raise ValueError("candidate output inode changed before verification")
if candidate_root.parent.as_posix() != candidate_identity.get("run_dir"):
    raise ValueError("candidate output is no longer contained by RUN_DIR")
expected = {metadata_path.resolve(), checkpoint_path.resolve(), onnx_path.resolve()}
candidate_symlinks = [path for path in candidate_root.rglob("*") if path.is_symlink()]
if candidate_symlinks:
    raise ValueError("candidate output tree must not contain symlinks")
actual = {
    path.resolve()
    for path in candidate_root.rglob("*")
    if path.is_file()
}
if actual != expected:
    raise ValueError("candidate output file set is not the exact three-file contract")
for path in expected:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"candidate output is missing, empty, or symlinked: {path}")
output_bytes = {
    metadata_path.resolve(): stable_bytes(metadata_path, "candidate metadata"),
    checkpoint_path.resolve(): stable_bytes(checkpoint_path, "candidate checkpoint"),
    onnx_path.resolve(): stable_bytes(onnx_path, "candidate ONNX"),
}
def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"candidate metadata contains duplicate key: {key}")
        result[key] = value
    return result

metadata = json.loads(
    output_bytes[metadata_path.resolve()].decode("utf-8"),
    object_pairs_hook=reject_duplicate_keys,
    parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"candidate metadata contains non-finite value: {token}")
    ),
)
expected_metadata_fields = {
    "format_version", "architecture", "candidate_only",
    "production_runtime_modified", "checkpoint", "onnx", "model_config",
    "classes", "material_classes", "objectness_classes",
    "condition_classes", "output_contract", "preprocessing",
    "manifest_contract", "manifest_summary", "training_config",
    "dataset_consumption_contract", "selection_contract", "best_epoch",
    "best_selection_score", "best_metrics",
}
if not isinstance(metadata, dict) or set(metadata) != expected_metadata_fields:
    raise ValueError("candidate metadata root schema mismatch")
if metadata.get("format_version") != 3:
    raise ValueError("candidate metadata format version mismatch")
if metadata.get("candidate_only") is not True:
    raise ValueError("metadata must declare candidate_only=true")
if metadata.get("production_runtime_modified") is not False:
    raise ValueError("metadata must declare production_runtime_modified=false")
if metadata.get("architecture") != "multitask_crop_verifier":
    raise ValueError("unexpected candidate architecture")
config = preflight["training_config"]
model_config = metadata.get("model_config", {})
if set(model_config) != {"backbone", "input_size", "condition_heads"}:
    raise ValueError("candidate model config schema mismatch")
if model_config.get("backbone") != config["backbone"] or model_config.get("input_size") != 320:
    raise ValueError("candidate model config differs from frozen training config")
if model_config.get("condition_heads") != ["dent", "label", "foreign_material"]:
    raise ValueError("candidate is missing an explicit condition head")
material_classes = [
    "can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass",
    "battery", "fluorescent",
]
objectness_classes = ["background", "material"]
expected_conditions = {
    "dent": ["not_dented", "dented"],
    "label": ["no_label", "has_label"],
    "foreign_material": ["no_foreign_material", "has_foreign_material"],
}
expected_outputs = [
    {
        "name": "objectness", "kind": "logits", "shape": ["batch", 2],
        "activation": "softmax", "class_names": objectness_classes,
        "class_ids": {"background": 0, "material": 1},
        "trained_on": "all proposal rows",
    },
    {
        "name": "material", "kind": "logits", "shape": ["batch", 9],
        "activation": "softmax", "class_names": material_classes,
        "class_ids": {name: index for index, name in enumerate(material_classes)},
        "valid_when": {
            "output": "objectness", "class_id": 1, "class_name": "material",
        },
        "trained_on": "positive material rows only; background is excluded from CE",
    },
]
for name in ("dent", "label", "foreign_material"):
    names = expected_conditions[name]
    expected_outputs.append({
        "name": name, "kind": "logits", "shape": ["batch", 2],
        "activation": "softmax", "class_names": names,
        "class_ids": {class_name: index for index, class_name in enumerate(names)},
        "valid_when": {
            "output": "objectness", "class_id": 1, "label_is_present": True,
        },
        "trained_on": "labeled positive material rows only",
    })
expected_output_contract = {
    "version": "multitask_verifier.v3",
    "output_order": ["objectness", "material", "dent", "label", "foreign_material"],
    "outputs": expected_outputs,
    "material_background_class_id": None,
    "decision_order": ["objectness", "material", "conditions"],
    "warning": "This v3 contract is not the legacy four-output production contract.",
}
output = metadata.get("output_contract", {})
if output != expected_output_contract:
    raise ValueError("candidate output contract differs from the exact v3 schema")
if output.get("version") != "multitask_verifier.v3":
    raise ValueError("unexpected output contract version")
if output.get("output_order") != ["objectness", "material", "dent", "label", "foreign_material"]:
    raise ValueError("candidate output order mismatch")
if metadata.get("objectness_classes") != objectness_classes:
    raise ValueError("candidate objectness classes mismatch")
if metadata.get("material_classes") != material_classes:
    raise ValueError("candidate material classes mismatch")
if metadata.get("classes") != material_classes:
    raise ValueError("candidate legacy class alias mismatch")
if metadata.get("condition_classes") != expected_conditions:
    raise ValueError("candidate condition class maps mismatch")
expected_preprocessing = {
    "color_space": "RGB",
    "resize": [320, 320],
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}
if metadata.get("preprocessing") != expected_preprocessing:
    raise ValueError("candidate preprocessing contract mismatch")
if metadata.get("checkpoint") != checkpoint_path.name or metadata.get("onnx") != onnx_path.name:
    raise ValueError("candidate metadata artifact names mismatch")
expected_manifest_contract = {
    "required_fields": [
        "filepath", "split", "source_id", "material", "category", "dent",
        "label", "foreign_material", "source_object_count", "sample_id",
        "role", "fold", "source_sha256", "image_sha256", "object_group",
        "capture_session", "origin",
    ],
    "lineage_fields": [
        "sample_id", "source_sha256", "object_group", "capture_session",
        "role", "fold",
    ],
    "allowed_roles": ["train", "model_validation", "calibration", "blind_test"],
    "optimization_role": "train",
    "checkpoint_selection_role": "model_validation",
    "excluded_roles": ["calibration", "blind_test"],
}
if metadata.get("manifest_contract") != expected_manifest_contract:
    raise ValueError("candidate manifest contract mismatch")
if metadata.get("dataset_consumption_contract") != preflight.get(
    "dataset_consumption_contract"
):
    raise ValueError("candidate metadata consumption contract mismatch")
summary = metadata.get("manifest_summary", {})
if not isinstance(summary, dict) or set(summary) != {
    "strict", "required_lineage_fields", "rows", "lineage_sha256",
    "payload_set_sha256",
    "input_manifests", "role_counts", "excluded_from_training_role_counts",
    "folds_by_role", "unique", "objectness_counts", "positive_material_counts",
}:
    raise ValueError("candidate manifest summary schema mismatch")
if summary.get("strict") is not True:
    raise ValueError("candidate manifest summary is not strict")
if summary.get("required_lineage_fields") != expected_manifest_contract["lineage_fields"]:
    raise ValueError("candidate manifest summary lineage fields mismatch")
if summary.get("payload_set_sha256") != preflight[
    "dataset_consumption_contract"
]["manifest_payload_set_sha256"]:
    raise ValueError("candidate manifest payload-set SHA mismatch")
declared_inputs = summary.get("input_manifests")
expected_inputs = [
    {"path": row["path"], "sha256": row["sha256"]}
    for row in preflight["manifests"]
]
if declared_inputs != expected_inputs:
    raise ValueError("candidate metadata manifest bindings mismatch")
training = metadata.get("training_config", {})
if not isinstance(training, dict) or set(training) != {
    "seed", "deterministic_algorithms", "smoke", "learning_rates",
    "label_smoothing", "class_weights", "task_weights", "sampling", "effective",
}:
    raise ValueError("candidate training config schema mismatch")
if training.get("deterministic_algorithms") is not True or training.get("smoke") is not False:
    raise ValueError("candidate training determinism/smoke contract mismatch")
effective = training.get("effective", {})
if not isinstance(effective, dict) or set(effective) != {
    "backbone", "size", "epochs", "patience", "batch", "workers",
    "max_train_batches", "max_validation_batches", "pretrained", "export_onnx",
}:
    raise ValueError("candidate effective training schema mismatch")
for field in ("epochs", "patience", "batch", "workers"):
    if effective.get(field) != config[field]:
        raise ValueError(f"candidate effective {field} mismatch")
if effective.get("backbone") != config["backbone"] or effective.get("size") != 320:
    raise ValueError("candidate effective backbone or size mismatch")
if effective.get("pretrained") is not True or effective.get("export_onnx") is not True:
    raise ValueError("candidate must use the pinned pretrained backbone and export ONNX")
if effective.get("max_train_batches") is not None or effective.get("max_validation_batches") is not None:
    raise ValueError("candidate training may not use smoke batch limits")
if training.get("seed") != config["seed"] or training.get("label_smoothing") != config["label_smoothing"]:
    raise ValueError("candidate seed or label smoothing mismatch")
if training.get("learning_rates") != {
    "base": config["lr"], "backbone": config["backbone_lr"], "heads": config["head_lr"]
}:
    raise ValueError("candidate learning-rate contract mismatch")
class_weights = training.get("class_weights")
if not isinstance(class_weights, dict) or set(class_weights) != {"mode", "beta", "values"}:
    raise ValueError("candidate class-weight schema mismatch")
if class_weights.get("mode") != config["class_weight_mode"] or class_weights.get("beta") != config["class_weight_beta"]:
    raise ValueError("candidate class-weight configuration mismatch")
if not isinstance(class_weights.get("values"), dict):
    raise ValueError("candidate class-weight values must be an object")
expected_task_weights = {
    "objectness": config["objectness_weight"],
    "material": config["material_weight"],
    "dent": config["condition_weight"],
    "label": config["condition_weight"],
    "foreign_material": config["condition_weight"],
}
if training.get("task_weights") != expected_task_weights:
    raise ValueError("candidate task weights mismatch")
if training.get("sampling") != preflight["sampling"]:
    raise ValueError("candidate sampling plan differs from dry-run/preflight")
selection_contract = metadata.get("selection_contract")
if selection_contract != {
    "metric": "mean balanced accuracy",
    "heads": ["objectness", "material", "dent", "label", "foreign_material"],
    "requires_every_class_in_validation": True,
}:
    raise ValueError("candidate checkpoint selection contract mismatch")
if type(metadata.get("best_epoch")) is not int or metadata["best_epoch"] < 1:
    raise ValueError("candidate best epoch is invalid")
if (
    isinstance(metadata.get("best_selection_score"), bool)
    or not isinstance(metadata.get("best_selection_score"), (int, float))
    or not torch.isfinite(torch.tensor(metadata["best_selection_score"])).item()
):
    raise ValueError("candidate best selection score is invalid")
if not isinstance(metadata.get("best_metrics"), dict):
    raise ValueError("candidate best metrics must be an object")

checkpoint = torch.load(
    io.BytesIO(output_bytes[checkpoint_path.resolve()]),
    map_location="cpu",
    weights_only=True,
)
if not isinstance(checkpoint, dict):
    raise ValueError("candidate checkpoint root must be a dictionary")
expected_checkpoint_fields = {
    "format_version", "architecture", "state_dict", "model_config", "backbone",
    "input_size", "classes", "material_classes", "objectness_classes",
    "condition_classes", "output_contract", "preprocessing", "manifest_contract",
    "manifest_summary", "dataset_consumption_contract", "training_config",
    "selection_contract", "epoch",
    "selection_score", "metrics",
}
if set(checkpoint) != expected_checkpoint_fields:
    raise ValueError("candidate checkpoint root schema mismatch")
if checkpoint.get("format_version") != 3 or checkpoint.get("architecture") != "multitask_crop_verifier":
    raise ValueError("candidate checkpoint format or architecture mismatch")
if checkpoint.get("model_config") != model_config:
    raise ValueError("candidate checkpoint and metadata model config differ")
if checkpoint.get("output_contract") != output:
    raise ValueError("candidate checkpoint and metadata output contract differ")
for field in (
    "classes", "material_classes", "objectness_classes", "condition_classes",
    "preprocessing", "manifest_contract", "manifest_summary", "training_config",
    "dataset_consumption_contract", "selection_contract",
):
    if checkpoint.get(field) != metadata.get(field):
        raise ValueError(f"candidate checkpoint and metadata differ: {field}")
if checkpoint.get("backbone") != config["backbone"] or checkpoint.get("input_size") != 320:
    raise ValueError("candidate checkpoint backbone/input size mismatch")
if checkpoint.get("epoch") != metadata.get("best_epoch"):
    raise ValueError("candidate checkpoint and metadata epoch differ")
if checkpoint.get("selection_score") != metadata.get("best_selection_score"):
    raise ValueError("candidate checkpoint and metadata selection score differ")
if checkpoint.get("metrics") != metadata.get("best_metrics"):
    raise ValueError("candidate checkpoint and metadata metrics differ")
state = checkpoint.get("state_dict")
if not isinstance(state, dict) or not state:
    raise ValueError("candidate checkpoint has no state_dict tensors")
for name, tensor in state.items():
    if not isinstance(name, str) or not name or not torch.is_tensor(tensor):
        raise ValueError("candidate state_dict entry is invalid")
    if tensor.device.type != "cpu" or not torch.isfinite(tensor).all().item():
        raise ValueError(f"candidate state tensor is non-CPU or non-finite: {name}")
required_head_widths = {
    "objectness_head.1": 2,
    "material_head.1": 9,
    "condition_heads.dent.1": 2,
    "condition_heads.label.1": 2,
    "condition_heads.foreign_material.1": 2,
}
feature_width = None
for prefix, output_width in required_head_widths.items():
    weight = state.get(f"{prefix}.weight")
    bias = state.get(f"{prefix}.bias")
    if not torch.is_tensor(weight) or weight.ndim != 2 or weight.shape[0] != output_width:
        raise ValueError(f"candidate checkpoint head weight mismatch: {prefix}")
    if not torch.is_tensor(bias) or tuple(bias.shape) != (output_width,):
        raise ValueError(f"candidate checkpoint head bias mismatch: {prefix}")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError(f"candidate checkpoint head dtype mismatch: {prefix}")
    if feature_width is None:
        feature_width = int(weight.shape[1])
    elif int(weight.shape[1]) != feature_width:
        raise ValueError("candidate checkpoint heads do not share one backbone width")
if feature_width is None or feature_width < 1:
    raise ValueError("candidate checkpoint head feature width is invalid")

class ExpectedMultitaskVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.objectness_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 2))
        self.material_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 9))
        self.condition_heads = nn.ModuleDict({
            name: nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 2))
            for name in ("dent", "label", "foreign_material")
        })

    def forward(self, image):
        features = self.backbone(image)
        return {
            "objectness": self.objectness_head(features),
            "material": self.material_head(features),
            **{
                name: self.condition_heads[name](features)
                for name in ("dent", "label", "foreign_material")
            },
        }

expected_model = ExpectedMultitaskVerifier()
expected_state = expected_model.state_dict()
if set(state) != set(expected_state):
    missing = sorted(set(expected_state) - set(state))
    unexpected = sorted(set(state) - set(expected_state))
    raise ValueError(
        f"candidate checkpoint is not a complete MobileNetV3-small verifier; "
        f"missing={missing[:5]} unexpected={unexpected[:5]}"
    )
for name, expected_tensor in expected_state.items():
    actual_tensor = state[name]
    if actual_tensor.shape != expected_tensor.shape:
        raise ValueError(f"candidate checkpoint tensor shape mismatch: {name}")
    if actual_tensor.dtype != expected_tensor.dtype:
        raise ValueError(f"candidate checkpoint tensor dtype mismatch: {name}")
expected_model.load_state_dict(state, strict=True)

model = onnx.load_model_from_string(output_bytes[onnx_path.resolve()])
onnx.checker.check_model(model)
if model.metadata_props or model.doc_string or model.graph.doc_string:
    raise ValueError("candidate ONNX contains unapproved metadata payload")
if model.functions or model.training_info or model.graph.sparse_initializer:
    raise ValueError("candidate ONNX contains unapproved auxiliary graph payload")
if [(entry.domain, entry.version) for entry in model.opset_import] != [("", 17)]:
    raise ValueError("candidate ONNX opset contract mismatch")
if any(node.doc_string for node in model.graph.node):
    raise ValueError("candidate ONNX node contains unapproved documentation payload")
if [value.name for value in model.graph.input] != ["img"]:
    raise ValueError("candidate ONNX input contract mismatch")
if [value.name for value in model.graph.output] != output["output_order"]:
    raise ValueError("candidate ONNX output names/order mismatch")

def tensor_contract(value):
    tensor_type = value.type.tensor_type
    if tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(f"candidate ONNX tensor is not float32: {value.name}")
    dimensions = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(str(dimension.dim_param))
        else:
            dimensions.append(None)
    return dimensions

input_dimensions = tensor_contract(model.graph.input[0])
if input_dimensions != ["batch", 3, 320, 320]:
    raise ValueError("candidate ONNX input dtype/shape contract mismatch")
expected_output_widths = {
    "objectness": 2, "material": 9, "dent": 2,
    "label": 2, "foreign_material": 2,
}
for value in model.graph.output:
    if tensor_contract(value) != ["batch", expected_output_widths[value.name]]:
        raise ValueError(f"candidate ONNX output dtype/shape mismatch: {value.name}")
for initializer in model.graph.initializer:
    if initializer.external_data or initializer.data_location == onnx.TensorProto.EXTERNAL:
        raise ValueError("candidate ONNX may not depend on external tensor data")
node_inputs = {name for node in model.graph.node for name in node.input if name}
unused_initializers = sorted(
    initializer.name
    for initializer in model.graph.initializer
    if initializer.name not in node_inputs
)
if unused_initializers:
    raise ValueError(
        f"candidate ONNX contains unused initializer payload: {unused_initializers[:5]}"
    )
reachable_tensors = {"img"}
for node in model.graph.node:
    if any(name in reachable_tensors for name in node.input):
        reachable_tensors.update(name for name in node.output if name)
unreachable_outputs = [
    value.name for value in model.graph.output
    if value.name not in reachable_tensors
]
if unreachable_outputs:
    raise ValueError(
        f"candidate ONNX outputs do not depend on img: {unreachable_outputs}"
    )
expected_model.eval()
torch.manual_seed(20260903)
parity_probe = torch.stack((
    torch.zeros(3, 320, 320, dtype=torch.float32),
    torch.randn(3, 320, 320, dtype=torch.float32),
))
with torch.no_grad():
    checkpoint_outputs = expected_model(parity_probe)
session = ort.InferenceSession(
    output_bytes[onnx_path.resolve()],
    providers=["CPUExecutionProvider"],
)
runtime_outputs = session.run(
    output["output_order"], {"img": parity_probe.numpy()}
)
for name, runtime_value in zip(output["output_order"], runtime_outputs, strict=True):
    expected_value = checkpoint_outputs[name]
    actual_value = torch.from_numpy(runtime_value)
    if actual_value.dtype != torch.float32 or actual_value.shape != expected_value.shape:
        raise ValueError(f"candidate ONNX/checkpoint parity shape mismatch: {name}")
    if not torch.allclose(actual_value, expected_value, rtol=1e-4, atol=1e-5):
        maximum_error = float((actual_value - expected_value).abs().max().item())
        raise ValueError(
            f"candidate ONNX does not represent checkpoint head {name}; "
            f"max_abs_error={maximum_error}"
        )

current_mapped_qnap_library_contract = collect_qnap_mapped_library_contract(
    preflight["qnap_library_snapshot"]
)
if preflight.get(
    "mapped_qnap_library_contract"
) != current_mapped_qnap_library_contract:
    raise ValueError("QNAP mapped-library provenance changed after candidate training")

rows = [
    {
        "path": path.name,
        "size": len(output_bytes[path]),
        "sha256": hashlib.sha256(output_bytes[path]).hexdigest(),
    }
    for path in sorted(expected, key=lambda value: value.name)
]
for path, content in output_bytes.items():
    if stable_bytes(path, f"candidate output final rehash {path.name}") != content:
        raise RuntimeError(f"candidate output changed during verification: {path.name}")
payload = {
    "schema": "v4_candidate_training_outputs.v1",
    "status": "candidate_outputs_verified",
    "candidate_only": True,
    "candidate_promotion_authorized": False,
    "production_deployment_authorized": False,
    "files": rows,
}
if inventory_path.exists() or inventory_path.is_symlink():
    raise FileExistsError(inventory_path)
temporary = inventory_path.with_name(f".{inventory_path.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
os.link(temporary, inventory_path)
temporary.unlink()
PY
then
  fail "candidate output contract verification failed" 65
fi
verify_inputs || fail "bound inputs or QNAP snapshot changed during output verification" 65

OUTPUT_MARKER=$CONTROL/candidate_outputs.sha256
temporary=$(mktemp "$CONTROL/.candidate-outputs.XXXXXX") || fail "failed to stage output marker"
sha256sum "$CANDIDATE_IDENTITY" "$METADATA" "$CHECKPOINT" "$ONNX" \
  "$OUTPUT_INVENTORY" > "$temporary" || \
  fail "failed to hash candidate outputs"
if ! ln "$temporary" "$OUTPUT_MARKER" 2>/dev/null; then
  rm -f "$temporary"
  fail "refusing to overwrite candidate output marker" 73
fi
rm -f "$temporary"
sha256sum -c "$OUTPUT_MARKER" >/dev/null 2>&1 || fail "candidate output marker verification failed" 65
verify_inputs || fail "bound inputs changed before ready publication" 65
verify_candidate_dir || fail "candidate output directory changed before ready publication" 65
sha256sum -c "$DRY_MARKER" >/dev/null 2>&1 || fail "dry-run evidence changed before ready publication" 65

READY=$CONTROL/candidate_training_ready.json
if ! "$PYTHON_BIN" - \
  "$READY" "$PREFLIGHT" "$INPUT_MARKER" "$DRY_MARKER" "$OUTPUT_MARKER" \
  "$OUTPUT_INVENTORY" "$METADATA" "$CONTAINER_IMAGE_ID" \
  "$CANDIDATE_IDENTITY" "$CHECKPOINT" "$ONNX" "$CANDIDATE" \
  "$QNAP_SNAPSHOT_REPORT" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ready, preflight, inputs, dry, outputs, inventory, metadata = map(Path, sys.argv[1:8])
image_id = sys.argv[8]
candidate_identity, checkpoint, onnx_path, candidate_root = map(Path, sys.argv[9:13])
qnap_snapshot_report = Path(sys.argv[13])
if ready.exists() or ready.is_symlink():
    raise FileExistsError(ready)

def stable_bytes(path: Path, description: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file")
    before = path.stat()
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        content = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = path.stat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
    )
    if not (
        identity(before) == identity(opened_before) ==
        identity(opened_after) == identity(after)
    ):
        raise RuntimeError(f"{description} changed while being read")
    return content

def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

if candidate_root.is_symlink() or not candidate_root.is_dir():
    raise ValueError("candidate root changed before ready publication")
expected_candidate_files = {
    metadata.resolve(), checkpoint.resolve(), onnx_path.resolve()
}
if any(path.is_symlink() for path in candidate_root.rglob("*")):
    raise ValueError("candidate output tree gained a symlink before ready publication")
actual_candidate_files = {
    path.resolve() for path in candidate_root.rglob("*") if path.is_file()
}
if actual_candidate_files != expected_candidate_files:
    raise ValueError("candidate output tree changed before ready publication")

bound_paths = (
    preflight, inputs, dry, outputs, inventory, metadata,
    candidate_identity, checkpoint, onnx_path, qnap_snapshot_report,
)
bound_bytes = {
    path.resolve(): stable_bytes(path, f"ready input {path.name}")
    for path in bound_paths
}
preflight_value = json.loads(bound_bytes[preflight.resolve()].decode("utf-8"))
dataset_snapshot_runtime = preflight_value.get("candidate_dataset_snapshot_runtime")
if not isinstance(dataset_snapshot_runtime, dict):
    raise ValueError("candidate dataset snapshot runtime contract is missing before ready")
dataset_consumption_contract = preflight_value.get("dataset_consumption_contract")
if not isinstance(dataset_consumption_contract, dict):
    raise ValueError("candidate dataset consumption contract is missing before ready")
dataset_consumption_contract_sha = digest(
    (json.dumps(
        dataset_consumption_contract, ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")
)
if preflight_value.get(
    "dataset_consumption_contract_sha256"
) != dataset_consumption_contract_sha:
    raise ValueError("candidate dataset consumption contract SHA mismatch before ready")
if (
    dataset_consumption_contract.get("dataset_snapshot_report_sha256")
    != dataset_snapshot_runtime.get("report_sha256")
    or dataset_consumption_contract.get("dataset_snapshot_tree_sha256")
    != dataset_snapshot_runtime.get("tree_sha256")
    or dataset_consumption_contract.get("trainer_sha256")
    != preflight_value.get("trainer_sha256")
):
    raise ValueError("candidate dataset consumption bindings differ before ready")
near_duplicate_audit = preflight_value.get("near_duplicate_audit")
if not isinstance(near_duplicate_audit, dict):
    raise ValueError("near-duplicate audit is missing before ready")
near_duplicate_audit_sha = digest(
    (json.dumps(
        near_duplicate_audit, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")
)
if preflight_value.get("near_duplicate_audit_sha256") != near_duplicate_audit_sha:
    raise ValueError("near-duplicate audit SHA mismatch before ready")
if (
    near_duplicate_audit.get("schema") != "v4_near_duplicate_leakage_audit.v1"
    or near_duplicate_audit.get("status") != "passed"
    or near_duplicate_audit.get("ok") is not True
    or near_duplicate_audit.get("summary", {}).get("blocking_multi_role_clusters") != 0
):
    raise ValueError("near-duplicate audit lost its passing contract before ready")
near_duplicate_bindings = near_duplicate_audit.get("bindings")
if not isinstance(near_duplicate_bindings, dict):
    raise ValueError("near-duplicate audit bindings are missing before ready")
near_protected_sources = near_duplicate_bindings.get("protected_sources")
near_protected_inventory = near_duplicate_bindings.get("protected_inventory")
if not isinstance(near_protected_sources, dict) or not isinstance(near_protected_inventory, dict):
    raise ValueError("near-duplicate protected bindings are missing before ready")
inventory_value = json.loads(bound_bytes[inventory.resolve()].decode("utf-8"))
if set(inventory_value) != {
    "schema", "status", "candidate_only", "candidate_promotion_authorized",
    "production_deployment_authorized", "files",
}:
    raise ValueError("candidate output inventory schema mismatch before ready")
if inventory_value.get("schema") != "v4_candidate_training_outputs.v1":
    raise ValueError("candidate output inventory version mismatch before ready")
if inventory_value.get("status") != "candidate_outputs_verified":
    raise ValueError("candidate output inventory status mismatch before ready")
if inventory_value.get("candidate_only") is not True:
    raise ValueError("candidate output inventory lost candidate_only")
if inventory_value.get("candidate_promotion_authorized") is not False:
    raise ValueError("candidate output inventory grants promotion authority")
if inventory_value.get("production_deployment_authorized") is not False:
    raise ValueError("candidate output inventory grants production authority")
expected_inventory_rows = sorted(
    (
        {
            "path": path.name,
            "size": len(bound_bytes[path.resolve()]),
            "sha256": digest(bound_bytes[path.resolve()]),
        }
        for path in (metadata, checkpoint, onnx_path)
    ),
    key=lambda row: row["path"],
)
if inventory_value.get("files") != expected_inventory_rows:
    raise ValueError("candidate bytes differ from candidate output inventory")

marker_rows = {}
for line in bound_bytes[outputs.resolve()].decode("utf-8").splitlines():
    # GNU sha256sum writes two spaces for text mode and `` *`` for binary
    # mode (MSYS selects binary mode for these files). Both are canonical
    # formats accepted by ``sha256sum -c``.
    match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
    if match is None:
        raise ValueError("candidate output marker is malformed")
    path = Path(match.group(2)).resolve(strict=True)
    if path in marker_rows:
        raise ValueError("candidate output marker contains a duplicate path")
    marker_rows[path] = match.group(1)
expected_marker_paths = {
    candidate_identity.resolve(), metadata.resolve(), checkpoint.resolve(),
    onnx_path.resolve(), inventory.resolve(),
}
if set(marker_rows) != expected_marker_paths:
    raise ValueError("candidate output marker path set mismatch before ready")
for path, declared in marker_rows.items():
    if declared != digest(bound_bytes[path]):
        raise ValueError("candidate output marker bytes mismatch before ready")

candidate_identity_value = json.loads(
    bound_bytes[candidate_identity.resolve()].decode("utf-8")
)
candidate_stat = candidate_root.resolve(strict=True).stat()
if candidate_identity_value.get("candidate_dir") != candidate_root.resolve().as_posix():
    raise ValueError("candidate output identity path mismatch before ready")
if (
    candidate_identity_value.get("candidate_dev"),
    candidate_identity_value.get("candidate_ino"),
) != (candidate_stat.st_dev, candidate_stat.st_ino):
    raise ValueError("candidate output identity inode mismatch before ready")

payload = {
    "schema": "v4_candidate_training_ready.v1",
    "status": "candidate_training_completed",
    "artifact_role": "offline_candidate_only_not_blind_promotion_or_deployment_authority",
    "candidate_only": True,
    "training_authority_consumed": True,
    "blind_test_authority": False,
    "candidate_promotion_authorized": False,
    "production_deployment_authorized": False,
    "pi_deployment_authorized": False,
    "spring_contract_modified": False,
    "requires_offline_judge": True,
    "requires_independent_blind_hardware_gate": True,
    "dataset_consumption_contract": dataset_consumption_contract,
    "container_image_id": image_id,
    "bindings": {
        "preflight_sha256": digest(bound_bytes[preflight.resolve()]),
        "input_marker_sha256": digest(bound_bytes[inputs.resolve()]),
        "dry_run_marker_sha256": digest(bound_bytes[dry.resolve()]),
        "output_marker_sha256": digest(bound_bytes[outputs.resolve()]),
        "output_inventory_sha256": digest(bound_bytes[inventory.resolve()]),
        "candidate_metadata_sha256": digest(bound_bytes[metadata.resolve()]),
        "qnap_library_snapshot_report_sha256": digest(
            bound_bytes[qnap_snapshot_report.resolve()]
        ),
        "candidate_dataset_snapshot_report_sha256": dataset_snapshot_runtime[
            "report_sha256"
        ],
        "candidate_dataset_snapshot_tree_sha256": dataset_snapshot_runtime[
            "tree_sha256"
        ],
        "dataset_consumption_contract_sha256": dataset_consumption_contract_sha,
        "trainer_sha256": dataset_consumption_contract["trainer_sha256"],
        "manifest_payload_set_sha256": dataset_consumption_contract[
            "manifest_payload_set_sha256"
        ],
        "candidate_near_duplicate_audit_sha256": near_duplicate_audit_sha,
        "protected_sources_sha256": near_protected_sources["file_sha256"],
        "protected_reference_inventory_sha256": near_protected_inventory["file_sha256"],
    },
}
temporary = ready.with_name(f".{ready.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
for path, content in bound_bytes.items():
    if stable_bytes(path, f"ready final rehash {path.name}") != content:
        raise RuntimeError(f"ready input changed before publication: {path}")
os.link(temporary, ready)
temporary.unlink()
PY
then
  fail "failed to publish candidate-only ready marker" 65
fi

# Ready publication is deliberately the final operation.  Consumers must also
# require absence of failed.txt and must never treat this as blind, promotion,
# Pi, Spring, or production authority.
terminal_state=1
exit 0
;;
*)
  printf '%s\n' \
    "unsupported direct launch: use the policy-bound Docker /usr/bin/env -i command" >&2
  exit 64
;;
esac
