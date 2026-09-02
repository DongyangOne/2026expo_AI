from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "nas"
    / "run_v4_repro_selection_audit.sh"
)
SELECTION_CONTRACT = (
    "v4_repro_pilot_inputs."
    "gt_stratified_historical_observation_priority_blake2b.v3"
)
PILOT_ROLE = (
    "v4_batch1_reproducibility_pilot_inputs_diagnostic_only_"
    "not_training_blind_or_deployment_authority"
)
SEALED_NAMES = (
    "pilot_dataset.yaml",
    "selection_inventory.json",
    "train_pilot.txt",
    "validation_pilot.txt",
    "inputs.sha256",
    "input_ready.json",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _integration_bash(tmp_path: Path) -> str:
    candidates = [shutil.which("bash")]
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if git_bash.is_file():
        candidates.insert(0, str(git_bash))
    for bash in dict.fromkeys(item for item in candidates if item):
        if subprocess.run(
            [bash, "-c", 'test -d "$1"', "bash", tmp_path.as_posix()],
            check=False,
        ).returncode == 0:
            return bash
    pytest.skip("no bash can access the pytest temp path")


def _write_pilot_artifacts(
    directory: Path,
    *,
    inventory: dict[str, object],
    train: bytes,
    validation: bytes,
    yaml_content: bytes,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    contents = {
        "pilot_dataset.yaml": yaml_content,
        "selection_inventory.json": _json_bytes(inventory),
        "train_pilot.txt": train,
        "validation_pilot.txt": validation,
    }
    for name, content in contents.items():
        (directory / name).write_bytes(content)
    artifact_hashes = {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
    }
    marker = "".join(
        f"{artifact_hashes[name]}  {name}\n" for name in sorted(artifact_hashes)
    ).encode("ascii")
    (directory / "inputs.sha256").write_bytes(marker)
    ready = {
        "schema_version": 1,
        "artifact_role": PILOT_ROLE,
        "status": "pilot_inputs_ready",
        "selection_contract": SELECTION_CONTRACT,
        "seed": inventory["seed"],
        "selected_sources": sum(inventory["selected_counts"].values()),
        "selected_counts": inventory["selected_counts"],
        "full_quota_met": True,
        "bindings": {
            "inputs_marker_sha256": hashlib.sha256(marker).hexdigest(),
            "artifacts": artifact_hashes,
            "resolved_universe_sha256": inventory["bindings"][
                "resolved_universe_sha256"
            ],
        },
        "validator_authority": False,
        "training_authorized": False,
        "blind_test_authorized": False,
        "production_deployment_authorized": False,
    }
    (directory / "input_ready.json").write_bytes(_json_bytes(ready))


FAKE_BUILDER = r'''
import argparse, hashlib, json, os, shutil
from pathlib import Path

ROLE = "v4_batch1_reproducibility_pilot_inputs_diagnostic_only_not_training_blind_or_deployment_authority"
CONTRACT = "v4_repro_pilot_inputs.gt_stratified_historical_observation_priority_blake2b.v3"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--dataset-dir", required=True)
p.add_argument("--output-dir", required=True)
p.add_argument("--seed", required=True, type=int)
p.add_argument("--train-quota-per-stratum", required=True, type=int)
p.add_argument("--validation-quota-per-stratum", required=True, type=int)
p.add_argument("--old-manifest", required=True)
p.add_argument("--drift-report", required=True)
a = p.parse_args()
if (
    os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void"
    or os.environ.get("PRESERVE_CALLER_SENTINEL") != "preserved"
):
    raise RuntimeError("selector subprocess CPU or caller environment mismatch")
if Path(a.output_dir) != Path(a.output_dir).resolve():
    raise RuntimeError("selector output path was not normalized")
dataset = Path(a.dataset_dir)
if (dataset / "force-builder-failure").exists():
    raise RuntimeError("forced selector failure")
reference = dataset / "audit-reference"
expected = json.loads((reference / "invocation.json").read_text(encoding="utf-8"))
actual = {
    "data": Path(a.data).resolve().as_posix(),
    "dataset_dir": dataset.resolve().as_posix(),
    "seed": a.seed,
    "train_quota": a.train_quota_per_stratum,
    "validation_quota": a.validation_quota_per_stratum,
    "old_manifest": Path(a.old_manifest).resolve().as_posix(),
    "drift_report": Path(a.drift_report).resolve().as_posix(),
}
if actual != expected:
    raise RuntimeError(f"selector invocation mismatch: {actual!r} != {expected!r}")
output = Path(a.output_dir)
output.mkdir(parents=False, exist_ok=False)
for name in ("selection_inventory.json", "train_pilot.txt", "validation_pilot.txt"):
    shutil.copyfile(reference / name, output / name)
yaml_lines = [
    f'path: "{dataset.resolve().as_posix()}"',
    f'train: "{(output / "train_pilot.txt").resolve().as_posix()}"',
    f'val: "{(output / "validation_pilot.txt").resolve().as_posix()}"',
    "names:",
    "  0: can",
]
(output / "pilot_dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
artifacts = {
    name: output / name
    for name in (
        "pilot_dataset.yaml",
        "selection_inventory.json",
        "train_pilot.txt",
        "validation_pilot.txt",
    )
}
artifact_hashes = {name: sha(path) for name, path in artifacts.items()}
marker = "".join(
    f"{artifact_hashes[name]}  {name}\n" for name in sorted(artifact_hashes)
)
(output / "inputs.sha256").write_text(marker, encoding="ascii")
inventory = json.loads((output / "selection_inventory.json").read_text(encoding="utf-8"))
ready = {
    "schema_version": 1,
    "artifact_role": ROLE,
    "status": "pilot_inputs_ready",
    "selection_contract": CONTRACT,
    "seed": a.seed,
    "bindings": {
        "inputs_marker_sha256": sha(output / "inputs.sha256"),
        "artifacts": artifact_hashes,
        "resolved_universe_sha256": inventory["bindings"]["resolved_universe_sha256"],
    },
    "validator_authority": False,
    "training_authorized": False,
    "blind_test_authorized": False,
    "production_deployment_authorized": False,
}
(output / "input_ready.json").write_text(
    json.dumps(ready, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
extra_kind = os.environ.get("FAKE_RECOMPUTE_EXTRA")
if extra_kind == "file":
    (output / "unexpected.bin").write_bytes(b"unexpected")
elif extra_kind == "directory":
    (output / "unexpected-directory").mkdir()
mutation_target = os.environ.get("FAKE_MUTATE_AFTER_SELECTOR_TARGET")
if mutation_target:
    with Path(mutation_target).open("ab") as handle:
        handle.write(b"# mutated while selector ran\n")
'''


def _fixture(tmp_path: Path, *, mode: str = "success") -> dict[str, str]:
    code = tmp_path / "code"
    scripts = code / "scripts"
    nas = scripts / "nas"
    nas.mkdir(parents=True)
    (nas / SCRIPT.name).write_bytes(SCRIPT.read_bytes())
    builder = scripts / "build_v4_repro_pilot_inputs.py"
    builder.write_text(FAKE_BUILDER, encoding="utf-8", newline="\n")
    proposal = scripts / "prepare_proposal_verifier_dataset.py"
    proposal.write_text("CLASS_NAMES = ('can',)\n", encoding="utf-8", newline="\n")

    dataset = tmp_path / "dataset"
    reference = dataset / "audit-reference"
    reference.mkdir(parents=True)
    data = tmp_path / "dataset.yaml"
    data.write_text("path: dataset\ntrain: train/images\nval: val/images\n", encoding="utf-8")
    old_manifest = tmp_path / "historical.csv"
    old_manifest.write_text("source_id,split,category\n", encoding="utf-8")
    drift_report = tmp_path / "drift.json"
    drift_report.write_text('{"replay":{}}\n', encoding="utf-8")
    pilot = tmp_path / "pilot"
    train = b"/dataset/training/a.jpg\n/dataset/training/b.jpg\n"
    validation = b"/dataset/validation/a.jpg\n"
    selected_counts = {"training/can": 2, "validation/can": 1}
    inventory: dict[str, object] = {
        "schema_version": 1,
        "artifact_role": PILOT_ROLE,
        "selection_contract": SELECTION_CONTRACT,
        "status": "selection_complete_not_replay_validated",
        "seed": 20260901,
        "quota_per_stratum": {"training": 2, "validation": 1},
        "selected_counts": selected_counts,
        "full_quota_met": True,
        "selected_sources": [
            {"path": "/dataset/training/a.jpg", "rank": 1},
            {"path": "/dataset/training/b.jpg", "rank": 2},
            {"path": "/dataset/validation/a.jpg", "rank": 1},
        ],
        "bindings": {
            "data_path": data.resolve().as_posix(),
            "data_sha256": _sha(data),
            "dataset_dir": dataset.resolve().as_posix(),
            "selector_path": builder.resolve().as_posix(),
            "selector_sha256": _sha(builder),
            "proposal_generator_path": proposal.resolve().as_posix(),
            "proposal_generator_sha256": _sha(proposal),
            "resolved_universe_sha256": "a" * 64,
        },
        "historical_selection_evidence": {
            "used_for_selection_only": True,
            "old_manifest": {
                "path": old_manifest.resolve().as_posix(),
                "sha256": _sha(old_manifest),
                "rows": 0,
            },
            "drift_report": {
                "path": drift_report.resolve().as_posix(),
                "sha256": _sha(drift_report),
                "anchor_source_ids": 0,
            },
        },
        "authority": {
            "raw_generation_authorized": False,
            "validator_authority": False,
            "training_authorized": False,
            "blind_test_authorized": False,
            "production_deployment_authorized": False,
        },
    }
    if mode == "archived_code_paths":
        inventory["bindings"]["selector_path"] = (
            tmp_path / "original-code" / "scripts" / builder.name
        ).resolve().as_posix()
        inventory["bindings"]["proposal_generator_path"] = (
            tmp_path / "original-code" / "scripts" / proposal.name
        ).resolve().as_posix()
    reference_inventory = json.loads(json.dumps(inventory))
    (reference / "selection_inventory.json").write_bytes(
        _json_bytes(reference_inventory)
    )
    (reference / "train_pilot.txt").write_bytes(train)
    (reference / "validation_pilot.txt").write_bytes(validation)
    (reference / "invocation.json").write_text(
        json.dumps(
            {
                "data": data.resolve().as_posix(),
                "dataset_dir": dataset.resolve().as_posix(),
                "seed": 20260901,
                "train_quota": 2,
                "validation_quota": 1,
                "old_manifest": old_manifest.resolve().as_posix(),
                "drift_report": drift_report.resolve().as_posix(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if mode == "forged_non_top_k":
        inventory["selected_sources"][0]["path"] = "/dataset/training/forged.jpg"
        train = b"/dataset/training/forged.jpg\n/dataset/training/b.jpg\n"
    elif mode == "boolean_seed":
        inventory["seed"] = True
    _write_pilot_artifacts(
        pilot,
        inventory=inventory,
        train=train,
        validation=validation,
        yaml_content=b"path: /dataset\ntrain: train_pilot.txt\nval: validation_pilot.txt\n",
    )

    if mode == "data_tamper":
        data.write_text("tampered: true\n", encoding="utf-8")
    elif mode == "selector_tamper":
        with builder.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("# tampered after pilot binding\n")
    elif mode == "pilot_marker_tamper":
        with (pilot / "train_pilot.txt").open("ab") as handle:
            handle.write(b"/dataset/training/extra.jpg\n")
    elif mode == "builder_failure":
        (dataset / "force-builder-failure").write_text("1\n", encoding="utf-8")

    environment = {
        **os.environ,
        "AUDIT_DIR": (tmp_path / "selection-audit").as_posix(),
        "CODE_ROOT": code.as_posix(),
        "PILOT_INPUT_DIR": pilot.as_posix(),
        "PYTHON_BIN": Path(sys.executable).as_posix(),
        "PRESERVE_CALLER_SENTINEL": "preserved",
        "FIXTURE_DATASET_DIR": dataset.as_posix(),
    }
    mutation_targets = {
        "mutate_wrapper_after_selector": nas / SCRIPT.name,
        "mutate_selector_after_selector": builder,
        "mutate_proposal_after_selector": proposal,
        "mutate_data_after_selector": data,
        "mutate_old_manifest_after_selector": old_manifest,
        "mutate_drift_report_after_selector": drift_report,
        "mutate_pilot_yaml_after_selector": pilot / "pilot_dataset.yaml",
        "mutate_pilot_inventory_after_selector": pilot / "selection_inventory.json",
        "mutate_pilot_train_after_selector": pilot / "train_pilot.txt",
        "mutate_pilot_validation_after_selector": pilot / "validation_pilot.txt",
        "mutate_pilot_marker_after_selector": pilot / "inputs.sha256",
        "mutate_pilot_ready_after_selector": pilot / "input_ready.json",
    }
    if mode in mutation_targets:
        environment["FAKE_MUTATE_AFTER_SELECTOR_TARGET"] = mutation_targets[
            mode
        ].as_posix()
    elif mode == "extra_recompute_file":
        environment["FAKE_RECOMPUTE_EXTRA"] = "file"
    elif mode == "extra_recompute_directory":
        environment["FAKE_RECOMPUTE_EXTRA"] = "directory"
    return environment


def _run(
    tmp_path: Path, *, mode: str = "success"
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    env = _fixture(tmp_path, mode=mode)
    result = subprocess.run(
        [_integration_bash(tmp_path), SCRIPT.as_posix()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, env


def test_shell_syntax_when_bash_is_available() -> None:
    bash = _integration_bash(SCRIPT.parent)
    subprocess.run([bash, "-n", SCRIPT.as_posix()], check=True)


def test_wrapper_contract_is_cpu_only_immutable_and_fail_closed() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for name in ("AUDIT_DIR", "CODE_ROOT", "PILOT_INPUT_DIR"):
        assert name in text
    assert 'PYTHON_BIN=${PYTHON_BIN:-python3}' in text
    assert '"PYTHONDONTWRITEBYTECODE": "1"' in text
    assert '"CUDA_VISIBLE_DEVICES": ""' in text
    assert '"NVIDIA_VISIBLE_DEVICES": "void"' in text
    assert 'if [ -e "$AUDIT_DIR" ] || [ -L "$AUDIT_DIR" ]' in text
    assert "fresh selector output differs byte-for-byte" in text
    assert "builder_arg.is_symlink()" in text
    assert "proposal_arg.is_symlink()" in text
    assert '"status": "selection_audit_ready"' in text
    assert "selection_audit.sha256" in text
    for field in (
        '"raw_generation_authorized": False',
        '"validator_authority": False',
        '"judge_authority": False',
        '"training_authority": False',
        '"blind_test_authority": False',
        '"candidate_promotion_authorized": False',
        '"production_deployment_authorized": False',
    ):
        assert field in text
    lowered = text.lower()
    for token in (
        "docker ",
        "docker-compose",
        "systemctl",
        "service ",
        "kubectl",
        "ssh ",
        "scp ",
        "reboot",
        "shutdown",
        "password",
        "credential",
        "api_key",
        "token=",
    ):
        assert token not in lowered


def test_integration_success_seals_exact_external_contract(tmp_path: Path) -> None:
    result, env = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    audit = Path(env["AUDIT_DIR"])
    ready = json.loads(
        (audit / "selection_audit_ready.json").read_text(encoding="utf-8")
    )
    assert ready["schema_version"] == 1
    assert ready["status"] == "selection_audit_ready"
    assert ready["selection_contract"] == SELECTION_CONTRACT
    assert type(ready["seed"]) is int and ready["seed"] == 20260901
    assert set(ready["bindings"]["pilot_artifacts"]) == set(SEALED_NAMES)
    assert set(ready["bindings"]["recompute_artifacts"]) == set(SEALED_NAMES)
    for field in (
        "raw_generation_authorized",
        "validator_authority",
        "judge_authority",
        "training_authority",
        "blind_test_authority",
        "candidate_promotion_authorized",
        "production_deployment_authorized",
    ):
        assert ready[field] is False
    assert not (audit / "failed.txt").exists()
    evidence_path = audit / "selection_audit_evidence.json"
    evidence_content = evidence_path.read_bytes()
    evidence = json.loads(evidence_content.decode("utf-8"))
    assert evidence["status"] == "selection_recomputed_byte_exact"
    assert evidence["comparisons"] == {
        "selection_inventory_json_byte_exact": True,
        "train_pilot_txt_byte_exact": True,
        "validation_pilot_txt_byte_exact": True,
    }
    assert ready["bindings"]["selection_audit_evidence_sha256"] == (
        hashlib.sha256(evidence_content).hexdigest()
    )
    marker_path = audit / "selection_audit.sha256"
    marker_content = marker_path.read_bytes()
    assert ready["bindings"]["selection_audit_marker_sha256"] == (
        hashlib.sha256(marker_content).hexdigest()
    )
    assert ready["bindings"]["pilot_artifacts"] == evidence["bindings"][
        "pilot_artifacts"
    ]
    assert ready["bindings"]["recompute_artifacts"] == evidence["bindings"][
        "recompute_artifacts"
    ]
    marker_lines = marker_content.decode("utf-8").splitlines()
    assert len(marker_lines) == len(SEALED_NAMES) + 1
    assert all(
        "/recompute/" in line.replace("\\", "/")
        for line in marker_lines[: len(SEALED_NAMES)]
    )
    assert marker_lines[-1].replace("\\", "/").endswith(
        "/selection_audit_evidence.json"
    )
    marked_paths = {line.split("  ", 1)[1] for line in marker_lines}
    assert marked_paths == {
        *((audit / "recompute" / name).resolve().as_posix() for name in SEALED_NAMES),
        evidence_path.resolve().as_posix(),
    }
    bash = _integration_bash(tmp_path)
    checked = subprocess.run(
        [
            bash,
            "-c",
            'sha256sum -c "$1" >/dev/null',
            "bash",
            marker_path.as_posix(),
        ],
        check=False,
    )
    assert checked.returncode == 0
    assert (audit / "recompute" / "selection_inventory.json").read_bytes() == (
        Path(env["PILOT_INPUT_DIR"]) / "selection_inventory.json"
    ).read_bytes()
    assert not list(Path(env["CODE_ROOT"]).rglob("__pycache__"))


def test_integration_accepts_archived_code_path_when_bound_sha_is_exact(
    tmp_path: Path,
) -> None:
    result, env = _run(tmp_path, mode="archived_code_paths")
    assert result.returncode == 0, result.stderr
    ready = json.loads(
        (
            Path(env["AUDIT_DIR"]) / "selection_audit_ready.json"
        ).read_text(encoding="utf-8")
    )
    assert ready["bindings"]["selector_sha256"] == _sha(
        Path(env["CODE_ROOT"]) / "scripts" / "build_v4_repro_pilot_inputs.py"
    )


@pytest.mark.parametrize(
    "mode",
    (
        "forged_non_top_k",
        "boolean_seed",
        "data_tamper",
        "selector_tamper",
        "pilot_marker_tamper",
        "builder_failure",
        "extra_recompute_file",
        "extra_recompute_directory",
        "mutate_wrapper_after_selector",
        "mutate_selector_after_selector",
        "mutate_proposal_after_selector",
        "mutate_data_after_selector",
        "mutate_old_manifest_after_selector",
        "mutate_drift_report_after_selector",
        "mutate_pilot_yaml_after_selector",
        "mutate_pilot_inventory_after_selector",
        "mutate_pilot_train_after_selector",
        "mutate_pilot_validation_after_selector",
        "mutate_pilot_marker_after_selector",
        "mutate_pilot_ready_after_selector",
    ),
)
def test_integration_failures_publish_failure_only(tmp_path: Path, mode: str) -> None:
    result, env = _run(tmp_path, mode=mode)
    assert result.returncode != 0
    audit = Path(env["AUDIT_DIR"])
    assert (audit / "failed.txt").is_file()
    assert not (audit / "selection_audit_ready.json").exists()


def test_integration_refuses_reuse_without_overwriting_ready(tmp_path: Path) -> None:
    result, env = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    ready = Path(env["AUDIT_DIR"]) / "selection_audit_ready.json"
    before = ready.read_bytes()
    second = subprocess.run(
        [_integration_bash(tmp_path), SCRIPT.as_posix()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 73
    assert ready.read_bytes() == before


def test_rejects_relative_audit_dir_before_creation(tmp_path: Path) -> None:
    env = _fixture(tmp_path)
    env["AUDIT_DIR"] = "relative-selection-audit"
    target = tmp_path / env["AUDIT_DIR"]
    result = subprocess.run(
        [_integration_bash(tmp_path), SCRIPT.as_posix()],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert not target.exists()


def test_rejects_audit_path_escape_into_dataset_before_creation(
    tmp_path: Path,
) -> None:
    env = _fixture(tmp_path)
    target = Path(env["FIXTURE_DATASET_DIR"]) / "escaped-audit"
    env["AUDIT_DIR"] = target.as_posix()
    result = subprocess.run(
        [_integration_bash(tmp_path), SCRIPT.as_posix()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert not target.exists()


def test_normalizes_dotdot_audit_path_before_selector_subprocess(
    tmp_path: Path,
) -> None:
    env = _fixture(tmp_path)
    hop = tmp_path / "path-hop"
    hop.mkdir()
    target = tmp_path / "normalized-selection-audit"
    env["AUDIT_DIR"] = (hop / ".." / target.name).as_posix()
    result = subprocess.run(
        [_integration_bash(tmp_path), SCRIPT.as_posix()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (target / "selection_audit_ready.json").is_file()
