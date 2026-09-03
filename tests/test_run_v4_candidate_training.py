from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "nas"
    / "run_v4_candidate_training.sh"
)
IMAGE_ID = "sha256:" + "a" * 64


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _embedded_raw_string(name: str) -> str:
    text = _text()
    marker = f"{name} = r'''\n"
    start = text.index(marker) + len(marker)
    end = text.index("\n'''", start)
    return text[start:end]


def _verified_archive(
    *, trainer_path: Path, trainer_source: bytes,
    legacy_path: Path, legacy_source: bytes,
) -> tuple[bytes, str, str, str]:
    bootstrap = _embedded_raw_string("VERIFIED_WORKER_BOOTSTRAP").encode("utf-8")
    trainer_sha = hashlib.sha256(trainer_source).hexdigest()
    legacy_sha = hashlib.sha256(legacy_source).hexdigest()
    contract = {
        "schema": "v4_verified_trainer_archive.v1",
        "trainer_path": trainer_path.as_posix(),
        "trainer_sha256": trainer_sha,
        "legacy_path": legacy_path.as_posix(),
        "legacy_sha256": legacy_sha,
    }
    entries = {
        "__main__.py": bootstrap,
        "scripts/__init__.py": b"",
        "scripts/train_verifier.py": legacy_source,
        "trainer.py": trainer_source,
        "verified_contract.json": json.dumps(
            contract, sort_keys=True, separators=(",", ":")
        ).encode(),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=False
    ) as archive:
        for entry_name in sorted(entries):
            info = zipfile.ZipInfo(entry_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100444 << 16
            archive.writestr(info, entries[entry_name])
    return (
        buffer.getvalue(),
        hashlib.sha256(bootstrap).hexdigest(),
        trainer_sha,
        legacy_sha,
    )


def _run_verified_loader(
    archive: bytes, bootstrap_sha: str, trainer_sha: str, trainer_path: Path,
    legacy_sha: str, legacy_path: Path, *trainer_args: str,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _embedded_raw_string("VERIFIED_TRAINER_LOADER"),
            hashlib.sha256(archive).hexdigest(),
            bootstrap_sha,
            trainer_sha,
            trainer_path.as_posix(),
            legacy_sha,
            legacy_path.as_posix(),
            *trainer_args,
        ],
        input=archive,
        capture_output=True,
        check=False,
    )


def _integration_bash(tmp_path: Path) -> str:
    candidates = [Path("C:/Program Files/Git/bin/bash.exe"), Path(shutil.which("bash") or "")]
    for candidate in candidates:
        if candidate.is_file() and subprocess.run(
            [str(candidate), "-c", 'test -d "$1"', "bash", tmp_path.as_posix()],
            check=False,
        ).returncode == 0:
            return str(candidate)
    pytest.skip("no bash can access the pytest temp path")


def _external_env_and_sh(tmp_path: Path) -> tuple[str, str]:
    candidates = [
        (
            Path("C:/Program Files/Git/usr/bin/env.exe"),
            Path("C:/Program Files/Git/usr/bin/sh.exe"),
        ),
        (Path(shutil.which("env") or ""), Path(shutil.which("sh") or "")),
    ]
    for env_executable, sh_executable in candidates:
        if not env_executable.is_file() or not sh_executable.is_file():
            continue
        probe = subprocess.run(
            [str(env_executable), "-i", "PATH=/usr/bin:/bin", str(sh_executable), "-c", 'test -d "$1"', "sh", tmp_path.as_posix()],
            check=False,
        )
        if probe.returncode == 0:
            return str(env_executable), str(sh_executable)
    pytest.skip("no external env -i and sh pair can access the pytest temp path")


def _minimum_env(tmp_path: Path) -> dict[str, str]:
    run_root = tmp_path / "run-root"
    run_root.mkdir()
    global_root = tmp_path / "global"
    code_root = global_root / "code"
    nas = code_root / "scripts" / "nas"
    nas.mkdir(parents=True)
    launcher = nas / SCRIPT.name
    launcher_text = SCRIPT.read_text(encoding="utf-8").replace(
        "PYTHON_BIN=/usr/local/bin/python3",
        f"PYTHON_BIN='{Path(sys.executable).as_posix()}'",
    )
    launcher.write_text(launcher_text, encoding="utf-8", newline="\n")
    trainer = code_root / "scripts" / "train_multitask_verifier.py"
    trainer.write_text("print('not reached')\n", encoding="utf-8")
    inputs = global_root / "inputs"
    inputs.mkdir()
    placeholder = inputs / "placeholder"
    placeholder.write_text("placeholder\n", encoding="utf-8")
    return {
        "RUN_ROOT": run_root.as_posix(),
        "RUN_DIR": (run_root / "candidate-run").as_posix(),
        "GLOBAL_ROOT": global_root.as_posix(),
        "CODE_ROOT": code_root.as_posix(),
        "AUTHORITY_JSON": placeholder.as_posix(),
        "AUTHORITY_MARKER": placeholder.as_posix(),
        "CODE_INVENTORY": placeholder.as_posix(),
        "TRAINING_CONFIG": placeholder.as_posix(),
        "HOST_LAUNCH_CONTRACT": placeholder.as_posix(),
        "PRETRAINED_BACKBONE": placeholder.as_posix(),
        "CONTAINER_IMAGE_ID": IMAGE_ID,
        "V4_CLEAN_REEXEC": "1",
    }


def test_shell_syntax_when_bash_is_available() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    subprocess.run([bash, "-n"], input=_text().encode(), check=True)


def test_verified_loader_executes_bound_bytes_with_direct_script_context(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    trainer_path = scripts / "train_multitask_verifier.py"
    legacy_path = scripts / "train_verifier.py"
    helper_path = scripts / "verified_sibling.py"
    output_path = tmp_path / "trusted.json"
    malicious_marker = tmp_path / "malicious.txt"
    legacy_source = b"LEGACY_TOKEN = 'sealed-legacy'\n"
    trainer_source = b"""from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import verified_sibling
from scripts.train_verifier import LEGACY_TOKEN

Path(sys.argv[1]).write_text(json.dumps({
    'file': __file__,
    'argv0': sys.argv[0],
    'argv': sys.argv[1:],
    'path0': sys.path[0],
    'sibling': verified_sibling.VALUE,
    'legacy': LEGACY_TOKEN,
    'torch_home': os.environ.get('TORCH_HOME'),
}), encoding='utf-8')
print('trusted-stdout')
sys.stderr.write('trusted-stderr\\n')
raise SystemExit(int(sys.argv[2]))
"""
    trainer_path.write_bytes(trainer_source)
    legacy_path.write_bytes(legacy_source)
    helper_path.write_text("VALUE = 'sibling-import-ok'\n", encoding="utf-8")
    archive, bootstrap_sha, trainer_sha, legacy_sha = _verified_archive(
        trainer_path=trainer_path,
        trainer_source=trainer_source,
        legacy_path=legacy_path,
        legacy_source=legacy_source,
    )
    trainer_path.write_text(
        "from pathlib import Path\n"
        f"Path({str(malicious_marker)!r}).write_text('MALICIOUS')\n"
        "raise SystemExit(91)\n",
        encoding="utf-8",
    )
    env_home = tmp_path / "torch-home"
    previous_torch_home = os.environ.get("TORCH_HOME")
    os.environ["TORCH_HOME"] = str(env_home)
    try:
        result = _run_verified_loader(
            archive,
            bootstrap_sha,
            trainer_sha,
            trainer_path,
            legacy_sha,
            legacy_path,
            output_path.as_posix(),
            "7",
        )
    finally:
        if previous_torch_home is None:
            os.environ.pop("TORCH_HOME", None)
        else:
            os.environ["TORCH_HOME"] = previous_torch_home
    assert result.returncode == 7, result.stdout + result.stderr
    assert result.stdout.splitlines() == [b"trusted-stdout"]
    assert result.stderr.splitlines() == [b"trusted-stderr"]
    assert not malicious_marker.exists()
    observed = json.loads(output_path.read_text(encoding="utf-8"))
    assert Path(observed["file"]) == trainer_path
    assert Path(observed["argv0"]) == trainer_path
    assert observed["argv"] == [output_path.as_posix(), "7"]
    assert Path(observed["path0"]) == scripts
    assert observed["sibling"] == "sibling-import-ok"
    assert observed["legacy"] == "sealed-legacy"
    assert Path(observed["torch_home"]) == env_home


def test_verified_loader_rejects_wrong_bound_sha_before_execution(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    trainer_path = scripts / "train_multitask_verifier.py"
    legacy_path = scripts / "train_verifier.py"
    marker = tmp_path / "executed.txt"
    trainer_source = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('EXECUTED')\n"
    ).encode()
    legacy_source = b"VALUE = 1\n"
    trainer_path.write_bytes(trainer_source)
    legacy_path.write_bytes(legacy_source)
    archive, bootstrap_sha, _trainer_sha, legacy_sha = _verified_archive(
        trainer_path=trainer_path,
        trainer_source=trainer_source,
        legacy_path=legacy_path,
        legacy_source=legacy_source,
    )
    result = _run_verified_loader(
        archive,
        bootstrap_sha,
        "0" * 64,
        trainer_path,
        legacy_sha,
        legacy_path,
    )
    assert result.returncode != 0
    assert not marker.exists()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="sealed memfd spawn-worker replay is a Linux/QNAP contract",
)
def test_linux_spawn_worker_uses_sealed_archive_after_path_swap(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    trainer_path = scripts / "train_multitask_verifier.py"
    legacy_path = scripts / "train_verifier.py"
    approved_marker = tmp_path / "approved-worker.txt"
    malicious_marker = tmp_path / "malicious-worker.txt"
    trainer_source = b"""from __future__ import annotations
import multiprocessing as mp
import sys
from pathlib import Path

def worker(path):
    Path(path).write_text('APPROVED', encoding='utf-8')

if __name__ == '__main__':
    malicious = (
        "from pathlib import Path\\n"
        f"Path({sys.argv[2]!r}).write_text('MALICIOUS', encoding='utf-8')\\n"
    )
    Path(__file__).write_text(malicious, encoding='utf-8')
    context = mp.get_context('spawn')
    process = context.Process(target=worker, args=(sys.argv[1],))
    process.start()
    process.join()
    print(context.get_start_method())
    raise SystemExit(process.exitcode)
"""
    legacy_source = b"VALUE = 1\n"
    trainer_path.write_bytes(trainer_source)
    legacy_path.write_bytes(legacy_source)
    archive, bootstrap_sha, trainer_sha, legacy_sha = _verified_archive(
        trainer_path=trainer_path,
        trainer_source=trainer_source,
        legacy_path=legacy_path,
        legacy_source=legacy_source,
    )
    result = _run_verified_loader(
        archive,
        bootstrap_sha,
        trainer_sha,
        trainer_path,
        legacy_sha,
        legacy_path,
        approved_marker.as_posix(),
        malicious_marker.as_posix(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == b"spawn\n"
    assert approved_marker.read_text(encoding="utf-8") == "APPROVED"
    assert not malicious_marker.exists()
    assert b"MALICIOUS" in trainer_path.read_bytes()


def test_authority_and_marker_contract_is_fail_closed() -> None:
    text = _text()
    for fragment in (
        '"schema": "v4_candidate_training_authority.v3"',
        '"artifact_role": "v4_candidate_training_input_authority_not_blind_or_deployment"',
        '"status": "candidate_training_inputs_ready"',
        '"candidate_only", "candidate_training_input_authorized", "training_authority"',
        '"diagnostic_only", "production_runtime_modified", "blind_test_authority"',
        '"candidate_promotion_authorized", "production_deployment_authorized"',
        '"pi_deployment_authorized", "spring_contract_modified"',
        'len(manifest_values) != 2',
        'expected_roles = {"train", "model_validation"}',
        'len(marker_rows) != 8',
        'training_authority.sha256 must bind exactly eight expected files',
        '"dataset_snapshot_report"',
        '"candidate_dataset_snapshot_sha256"',
        '"quality_exclusion_assembly_receipt_sha256"',
        '"dataset_snapshot_publish_receipt_sha256"',
        '"dataset_consumption_contract_sha256"',
        '"candidate_near_duplicate_audit_sha256"',
        '"protected_reference_inventory_sha256"',
        '"near_duplicate_audit"',
        'FORBIDDEN_DIAGNOSTIC_TOKENS',
    ):
        assert fragment in text
    assert 'require_exact_bool(authority.get(field), True' in text
    assert 'require_exact_bool(authority.get(field), False' in text


def test_near_duplicate_audit_is_fail_closed_and_bound_through_ready() -> None:
    text = _text()
    for fragment in (
        '"v4_near_duplicate_leakage_audit.v1"',
        '"candidate_dataset_separation_evidence_only"',
        '"id": "oneexpo_phash_rot4_v1"',
        '"threshold": 4',
        '"graph_edge_cap": 1000000',
        '"automatic_delete_or_relabel": False',
        'near-duplicate audit candidate manifest binding mismatch',
        'near-duplicate coverage complete',
        'near-duplicate protected source union count mismatch',
        'near-duplicate protected crop-view coverage mismatch',
        'near-duplicate candidate crop-view coverage mismatch',
        'near-duplicate candidate payload-set binding mismatch',
        'near-duplicate protected payload-set binding mismatch',
        'near-duplicate protected canonical union binding mismatch',
        'near-duplicate candidate crop entries differ from the exact manifest rows',
        'near-duplicate supplied edge array exceeds its graph edge cap',
        'near-duplicate audit omitted a forbidden cross-role edge',
        'near-duplicate audit edge set is incomplete',
        'near-duplicate audit edge evidence is incomplete',
        'near-duplicate cluster is multi-role or malformed',
        'near-duplicate cluster blocking',
        '"blocking_multi_role_clusters": 0',
        'near-duplicate auditor differs from the report/code inventory binding',
        '"runtime_code_sha256"',
        'runtime_code_fingerprint_sha256',
        'near-duplicate auditor executed-code fingerprint differs from the report',
        '"near_duplicate_audit": near_duplicate_audit',
        '"near_duplicate_audit_sha256": near_duplicate_audit_sha',
        '"candidate_near_duplicate_audit_sha256": near_duplicate_audit_sha',
        '"protected_sources_sha256": near_protected_sources["file_sha256"]',
        '"protected_reference_inventory_sha256": near_protected_inventory["file_sha256"]',
    ):
        assert fragment in text
    assert text.count('near_duplicate_audit_sha =') == 2
    assert text.count('near_duplicate_audit.get("status") != "passed"') == 2


def test_code_image_config_and_pretrained_bytes_are_pinned() -> None:
    text = _text()
    for fragment in (
        'bindings.get("container_image_id") != image_id',
        'artifact_entry("code_inventory"',
        'artifact_entry("training_config"',
        'artifact_entry("host_launch_contract"',
        'artifact_entry("pretrained_backbone"',
        'actual_code_paths - excluded_code_paths != inventory_paths',
        'CODE_ROOT must not contain symlinks',
        'mobilenet_v3_small-047dcff4.pth',
        'env["TORCH_HOME"] = preflight["torch_home"]',
        'config.get("pretrained"), True',
        'APPROVED_TRUSTED_POLICY_SHA256="UNCONFIGURED"',
        'git_bundled_code_sha256_pin',
        'dataset_content_inventory differs from manifest bytes',
        'dataset content inventory SHA binding mismatch',
    ):
        assert fragment in text


def test_host_launch_contract_is_exact_and_has_only_one_rw_mount() -> None:
    text = _text()
    assert text.index('case "${V4_CLEAN_REEXEC-}" in') < text.index(
        "set -eu"
    )
    for fragment in (
        '"network_mode": "none"',
        '"restart_policy": "no"',
        '"shm_size_bytes": 8589934592',
        '"privileged": False',
        '"device_requests": None',
        'len(devices) != 7',
        'RUN_ROOT must be the sole read-write host mount',
        'global Container mount must be read-only',
        'host launch command does not clean and reconstruct the environment',
        '"/usr/bin/env", "-i", f"PATH={CLEAN_CONTAINER_PATH}"',
        'BASH_EXPORTED_FUNCTION_PREFIX = "BASH_FUNC_"',
        'name.startswith(BASH_EXPORTED_FUNCTION_PREFIX)',
        'RUN_ROOT must be a dedicated empty per-container workspace',
        'expected_run_source = f"/share/Container/runs/{container_name}-workspace"',
        'raw_inspect_path',
        'raw_inspect_sha256',
        'raw docker inspect must contain exactly one inspect object',
        'raw_inspect.get("Id") != container_id',
        'raw_inspect.get("Image") != image_id',
        'raw_inspect.get("Name") != f"/{container_name}"',
        'raw docker inspect HostConfig.{field} mismatch',
        'raw_config.get("Cmd") != command',
        'raw_config.get("Hostname") != container_id[:12]',
        'raw docker inspect mounts differ from host launch contract',
        'live {description} tree contains writable mountpoints',
        'live RUN_ROOT mount is not read-write',
        'live /dev/shm is smaller than 8 GiB',
        'live network namespace contains an interface other than lo',
    ):
        assert fragment in text
    for device in (
        "/dev/nvidia0",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
        "/dev/nvidia-modeset",
        "/dev/nvidia-caps/nvidia-cap1",
        "/dev/nvidia-caps/nvidia-cap2",
    ):
        assert device in text


def test_runtime_dependencies_optimizer_and_sampling_fail_before_training() -> None:
    text = _text()
    training_boundary = text.index("DRY_RUN_REPORT=$CONTROL/training_dry_run.json")
    for fragment in (
        "import onnxruntime as ort",
        "import torchvision",
        "from torchvision import models",
        '"CPUExecutionProvider" not in runtime_providers',
        "models.mobilenet_v3_small(weights=None)",
        'raise ValueError("CUDA is unavailable before candidate training")',
        "cuda_probe.square().sum()",
        '"optimizer_runtime_contract": optimizer_runtime_contract',
        'pinned trainer scheduler T_max is not effective epochs',
        'pinned trainer scheduler eta_min differs from config',
        '"sampling": derived_sampling',
    ):
        assert text.index(fragment) < training_boundary


def test_qnap_native_libraries_are_policy_bound_snapshotted_and_reverified() -> None:
    text = _text()
    first_native_import = text.index("import onnx")
    bootstrap = text.index("QNAP library inventory/snapshot bootstrap failed")
    export_snapshot = text.index("LD_LIBRARY_PATH=$QNAP_SNAPSHOT_LIBRARY_PATH")
    assert bootstrap < export_snapshot < first_native_import
    assert "LD_LIBRARY_PATH=/qnap/nvidia/lib:/qnap/cuda/lib64" not in text
    for fragment in (
        '"schema", "snapshot_max_bytes", "trees", "required_mapped_libraries"',
        "QNAP library inventory entries must be path-sorted",
        "QNAP symlink target escapes its library tree",
        "special QNAP library entry is forbidden",
        "QNAP required mapped libraries must include libcuda.so.1",
        "QNAP required mapped libraries must include at least one library from each tree",
        "QNAP library tree contains writable mounts",
        "/dev/shm is not an exact tmpfs mount",
        "/dev/shm lacks free space for the policy-bound QNAP snapshot",
        "QNAP source library tree changed during snapshot creation",
        "QNAP snapshot file is not mode 0444",
        "verify_qnap_snapshot || return 1",
        "qnap_snapshot_report_path",
        '"qnap_library_snapshot": qnap_snapshot_report',
        '"qnap_library_snapshot_report_sha256"',
    ):
        assert fragment in text
    assert text.count("verify_qnap_snapshot || return 1") >= 2


def test_qnap_loaded_library_provenance_is_bound_before_and_after_training() -> None:
    text = _text()
    for fragment in (
        "def collect_qnap_mapped_library_contract(snapshot_report):",
        'Path("/proc/self/maps")',
        "QNAP-inventoried library basenames mapped outside the private snapshot",
        "policy-required QNAP libraries were not mapped",
        "mapped QNAP snapshot bytes differ from inventory",
        '"observer_role": "launcher_boundary_process_not_trainer_process"',
        '"mapped_qnap_library_contract": mapped_qnap_library_contract',
        "current_mapped_qnap_library_contract = collect_qnap_mapped_library_contract",
        "QNAP mapped-library provenance changed after candidate training",
    ):
        assert fragment in text


def test_dataset_snapshot_is_policy_bound_and_reverified_at_every_boundary() -> None:
    text = _text()
    for fragment in (
        '"schema") != "v4_candidate_dataset_snapshot.v2"',
        '"candidate_dataset_snapshot_sha256"',
        '"dataset_snapshot_publish_receipt_sha256"',
        'candidate dataset snapshot contains duplicate sample/path/SHA',
        'manifest rows do not consume the exact dataset snapshot object set',
        '"candidate_dataset_snapshot": dataset_snapshot_report',
        '"candidate_dataset_snapshot_runtime": dataset_snapshot_runtime_contract',
        'dataset snapshot publish receipt changed',
        'candidate_dataset_snapshot_report_sha256',
        'candidate_dataset_snapshot_tree_sha256',
        'dataset_consumption_contract_sha256',
        'manifest_payload_set_sha256',
    ):
        assert fragment in text
    assert text.count('verify_inputs || fail') >= 6
    assert text.count("def collect_qnap_mapped_library_contract(snapshot_report):") == 2
    assert text.count("collect_qnap_mapped_library_contract(") >= 5


def test_linux_output_verifier_imports_every_identity_module() -> None:
    text = _text()
    output_section = text[text.index("METADATA=$CANDIDATE/multitask_verifier_metadata.json"):]
    source = output_section.split("<<'PY'\n", 1)[1].split("\nPY\nthen", 1)[0]
    tree = ast.parse(source)
    imported = {
        alias.asname or alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert {"re", "socket"}.issubset(imported)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "socket"
        and node.func.attr == "gethostname"
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "re"
        and node.func.attr == "findall"
        for node in ast.walk(tree)
    )


def test_training_is_two_manifest_dry_run_then_candidate_only() -> None:
    text = _text()
    assert text.count('"--manifest", manifests["train"]') == 1
    assert text.count('"--manifest", manifests["model_validation"]') == 1
    assert text.count('"--condition-head", head') == 1
    assert text.count("run_verified_trainer dry-run") == 1
    assert text.count("run_verified_trainer train") == 1
    assert '[sys.executable, preflight["trainer_path"]' not in text
    for fragment in (
        "VERIFIED_WORKER_BOOTSTRAP = r'''",
        "VERIFIED_TRAINER_LOADER = r'''",
        "v4_verified_trainer_archive.v1",
        "os.memfd_create(",
        "fcntl.F_ADD_SEALS",
        "multiprocessing_spawn.get_preparation_data = verified_get_preparation_data",
        "popen_spawn_posix.Popen._launch = verified_launch",
        'data["init_main_from_path"] = archive_path',
        "compile(trainer_source, trainer_path, \"exec\", dont_inherit=True)",
        'sys.modules["__main__"] = main_module',
        'sys.path[0] = trainer_parent',
        'input=archive_bytes',
    ):
        assert fragment in text
    assert text.index('mode = sys.argv[4]') < text.index('DRY_MARKER=$CONTROL/dry_run.sha256')
    assert text.index('DRY_MARKER=$CONTROL/dry_run.sha256') < text.index(
        'candidate-only verifier training failed'
    )
    for fragment in (
        'metadata.get("candidate_only") is not True',
        'metadata.get("production_runtime_modified") is not False',
        '["objectness", "material", "dent", "label", "foreign_material"]',
        '"can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass"',
        'candidate output file set is not the exact three-file contract',
        'candidate metadata root schema mismatch',
        'candidate checkpoint root schema mismatch',
        'candidate preprocessing contract mismatch',
        'candidate checkpoint and metadata differ: {field}',
        'candidate output contract differs from the exact v3 schema',
        'runtime dependencies changed after preflight',
        'onnxruntime lacks required CPUExecutionProvider',
        'CUDA allocation/kernel smoke probe returned an invalid value',
        'CUDA runtime contract changed after preflight',
        'candidate ONNX contains unapproved metadata payload',
        'candidate ONNX contains unused initializer payload',
        'requires_independent_blind_hardware_gate',
        'production_deployment_authorized": False',
    ):
        assert fragment in text


def test_authoritative_external_env_i_blocks_shell_startup_injection(tmp_path: Path) -> None:
    env_executable, sh_executable = _external_env_and_sh(tmp_path)
    env = _minimum_env(tmp_path)
    wrapper = Path(env["CODE_ROOT"]) / "scripts" / "nas" / SCRIPT.name
    outside = tmp_path / "outside" / "candidate-run"
    env["RUN_DIR"] = outside.as_posix()
    startup_marker = tmp_path / "bash-env-ran"
    startup = tmp_path / "hostile-bash-env.sh"
    startup.write_text(
        f"/usr/bin/printf startup > '{startup_marker.as_posix()}'\n",
        encoding="utf-8",
    )
    hostile_parent = {
        **env,
        "BASH_ENV": startup.as_posix(),
        "SHELLOPTS": "xtrace",
        "PS4": f"$(/usr/bin/printf trace > '{startup_marker.as_posix()}') ",
        "BASH_FUNC_set%%": "() { /usr/bin/printf 'INJECTED_BEFORE_GUARD\\n'; }",
    }
    assignments = [
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "V4_CLEAN_REEXEC=1",
        *[
            f"{name}={env[name]}"
            for name in sorted(env)
            if name != "V4_CLEAN_REEXEC"
        ],
    ]
    result = subprocess.run(
        [
            env_executable, "-i", *assignments,
            sh_executable, wrapper.as_posix(),
        ],
        env=hostile_parent, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "INJECTED_BEFORE_GUARD" not in result.stdout + result.stderr
    assert not startup_marker.exists()
    assert "RUN_DIR path precheck failed; no directory was created" in result.stderr
    assert not outside.exists()


def test_wrapper_cannot_manage_or_deploy_external_state() -> None:
    text = _text().lower()
    for token in (
        "docker-compose", "systemctl", "service ", "kubectl", "ssh ",
        "scp ", "reboot", "shutdown", "git push", "curl ", "wget ", "api_key",
        "password", "credential",
    ):
        assert token not in text
    assert "\ndocker " not in text
    assert "production_deployment_authorized\": true" not in text
    assert "candidate_promotion_authorized\": true" not in text


def test_mutable_inputs_are_rehashed_at_every_training_boundary(tmp_path: Path) -> None:
    text = _text()
    assert 'sha256sum -c "$INPUT_MARKER"' in text
    assert text.count("verify_inputs || fail") >= 5
    payload = tmp_path / "mutable-input"
    payload.write_bytes(b"before\n")
    marker = tmp_path / "inputs.sha256"
    import hashlib
    marker.write_text(
        f"{hashlib.sha256(payload.read_bytes()).hexdigest()}  {payload.as_posix()}\n",
        encoding="utf-8",
    )
    payload.write_bytes(b"after\n")
    bash = _integration_bash(tmp_path)
    result = subprocess.run(
        [bash, "-c", 'sha256sum -c "$1" >/dev/null 2>&1', "bash", marker.as_posix()],
        check=False,
    )
    assert result.returncode != 0


def test_raw_inspect_is_bound_for_repeated_verification() -> None:
    text = _text()
    assert 'input_bytes[raw_inspect_path] = raw_inspect_content' in text
    assert 'policy_path, raw_inspect_path,' in text
    assert '"raw_inspect_sha256": raw_inspect_sha' in text
    assert '"bound_inputs": snapshot_rows' in text


def test_path_payload_and_candidate_identity_guards_are_present() -> None:
    text = _text()
    for fragment in (
        "GLOBAL_ROOT and RUN_ROOT must be fully disjoint",
        "manifest payload escapes the read-only global mount",
        "manifest payload changed",
        "candidate_dir_identity.json",
        "candidate output directory identity changed",
        "candidate output path differs from its recorded identity",
        "candidate output tree must not contain symlinks",
        "candidate output changed during verification",
        "io.BytesIO(output_bytes[checkpoint_path.resolve()])",
        '"condition_heads.foreign_material.1": 2',
        "candidate checkpoint heads do not share one backbone width",
        "onnx.checker.check_model(model)",
        'input_dimensions != ["batch", 3, 320, 320]',
        "candidate ONNX output dtype/shape mismatch",
        "candidate ONNX may not depend on external tensor data",
        "candidate checkpoint is not a complete MobileNetV3-small verifier",
        "candidate checkpoint tensor dtype mismatch",
        "candidate ONNX does not represent checkpoint head",
        "candidate bytes differ from candidate output inventory",
    ):
        assert fragment in text


def test_invalid_run_dir_is_rejected_before_any_directory_is_created(tmp_path: Path) -> None:
    bash = _integration_bash(tmp_path)
    env = _minimum_env(tmp_path)
    outside = tmp_path / "outside" / "candidate-run"
    env["RUN_DIR"] = outside.as_posix()
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert "no directory was created" in result.stderr
    assert not outside.exists()


def test_existing_run_dir_is_not_reused_or_modified(tmp_path: Path) -> None:
    bash = _integration_bash(tmp_path)
    env = _minimum_env(tmp_path)
    run_dir = Path(env["RUN_DIR"])
    run_dir.mkdir()
    sentinel = run_dir / "sentinel"
    sentinel.write_text("keep\n", encoding="utf-8")
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert not (run_dir / "control").exists()


def test_diagnostic_authority_fails_without_ready_marker(tmp_path: Path) -> None:
    bash = _integration_bash(tmp_path)
    env = _minimum_env(tmp_path)
    wrapper = Path(env["CODE_ROOT"]) / "scripts" / "nas" / SCRIPT.name
    authority = Path(env["AUTHORITY_JSON"])
    authority = authority.with_name("training_authority.json")
    authority.write_text(
        json.dumps(
            {
                "schema": "v4_candidate_training_authority.v1",
                "artifact_role": "v4_candidate_training_input_authority_not_blind_or_deployment",
                "status": "candidate_training_inputs_ready",
                "candidate_only": True,
                "candidate_training_input_authorized": True,
                "training_authority": True,
                "lineage_execution_authorized": True,
                "diagnostic_only": True,
                "production_runtime_modified": False,
                "blind_test_authority": False,
                "candidate_promotion_authorized": False,
                "production_deployment_authorized": False,
                "pi_deployment_authorized": False,
                "spring_contract_modified": False,
                "artifacts": {},
                "bindings": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    env["AUTHORITY_JSON"] = authority.as_posix()
    result = subprocess.run(
        [bash, wrapper.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    control = Path(env["RUN_DIR"]) / "control"
    assert (control / "failed.txt").is_file()
    assert not (control / "candidate_training_ready.json").exists()
