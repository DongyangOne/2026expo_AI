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
    "gt_stratified_historical_observation_priority_blake2b.v4"
)
AUDIT_CONTRACT = "v4_repro_selection_audit.cpu_only_byte_exact.v2"
QUALITY_CONTRACT = "v4_capture_quality_exclusions.sha256_reason_only.v1"
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


def _entries_sha(entries: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        (
            json.dumps(entries, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    ).hexdigest()


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
        "quality_exclusion": {
            key: value
            for key, value in inventory["quality_exclusion"].items()
            if key not in {"manifest_path", "matched_resolved_sources"}
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
CONTRACT = "v4_repro_pilot_inputs.gt_stratified_historical_observation_priority_blake2b.v4"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--dataset-dir", required=True)
p.add_argument("--output-dir", required=True)
p.add_argument("--quality-exclusion-manifest", required=True)
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
    "quality_exclusion_manifest": Path(a.quality_exclusion_manifest).resolve().as_posix(),
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
    "quality_exclusion": {
        key: value
        for key, value in inventory["quality_exclusion"].items()
        if key not in {"manifest_path", "matched_resolved_sources"}
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
    selected_files = {
        "training/a": dataset / "training" / "a.jpg",
        "training/b": dataset / "training" / "b.jpg",
        "validation/a": dataset / "validation" / "a.jpg",
    }
    for name, path in selected_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"selected-{name}".encode())
    excluded_source = dataset / "excluded" / "bad-vinyl.jpg"
    excluded_source.parent.mkdir(parents=True)
    excluded_source.write_bytes(b"bad-vinyl-capture")
    quality_entries = [
        {"source_sha256": _sha(excluded_source), "reason": "severe_frame_crop"}
    ]
    quality_manifest = tmp_path / "quality-exclusions.json"
    quality_manifest.write_bytes(
        _json_bytes(
            {
                "schema_version": 1,
                "artifact_role": (
                    "v4_capture_quality_exclusion_manifest_selection_only_"
                    "not_ground_truth_or_authority"
                ),
                "quality_exclusion_contract": QUALITY_CONTRACT,
                "status": "quality_exclusions_ready",
                "excluded_source_count": 1,
                "max_excluded_sources": 100,
                "reason_counts": {"severe_frame_crop": 1},
                "source_list_sha256": _entries_sha(quality_entries),
                "entries": quality_entries,
                "authority": {
                    "selection": False,
                    "ground_truth": False,
                    "replay": False,
                    "training": False,
                    "calibration": False,
                    "blind_test": False,
                    "deployment": False,
                },
            }
        )
    )
    if mode == "quality_canonical_hash_tamper":
        quality_value = json.loads(quality_manifest.read_text(encoding="utf-8"))
        quality_value["source_list_sha256"] = "f" * 64
        quality_manifest.write_bytes(_json_bytes(quality_value))
    elif mode == "quality_over_limit":
        quality_value = json.loads(quality_manifest.read_text(encoding="utf-8"))
        oversized_entries = [
            {
                "source_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                "reason": "severe_frame_crop",
            }
            for index in range(101)
        ]
        oversized_entries.sort(key=lambda row: row["source_sha256"])
        quality_value["entries"] = oversized_entries
        quality_value["excluded_source_count"] = 101
        quality_value["reason_counts"] = {"severe_frame_crop": 101}
        quality_value["source_list_sha256"] = _entries_sha(oversized_entries)
        quality_manifest.write_bytes(_json_bytes(quality_value))
    elif mode == "quality_duplicate_json_key":
        rendered = quality_manifest.read_text(encoding="utf-8")
        quality_manifest.write_text(
            '{"status":"forged",' + rendered[1:], encoding="utf-8"
        )
    elif mode in {
        "quality_authority_zero",
        "quality_schema_true",
        "quality_max_float",
        "quality_reason_count_bool",
    }:
        quality_value = json.loads(quality_manifest.read_text(encoding="utf-8"))
        if mode == "quality_authority_zero":
            quality_value["authority"]["training"] = 0
        elif mode == "quality_schema_true":
            quality_value["schema_version"] = True
        elif mode == "quality_max_float":
            quality_value["max_excluded_sources"] = 100.0
        else:
            quality_value["reason_counts"] = {"severe_frame_crop": True}
        quality_manifest.write_bytes(_json_bytes(quality_value))
    data = tmp_path / "dataset.yaml"
    data.write_text("path: dataset\ntrain: train/images\nval: val/images\n", encoding="utf-8")
    old_manifest = tmp_path / "historical.csv"
    old_manifest.write_text("source_id,split,category\n", encoding="utf-8")
    drift_report = tmp_path / "drift.json"
    drift_report.write_text('{"replay":{}}\n', encoding="utf-8")
    pilot = tmp_path / "pilot"
    train = (
        selected_files["training/a"].resolve().as_posix()
        + "\n"
        + selected_files["training/b"].resolve().as_posix()
        + "\n"
    ).encode()
    validation = (selected_files["validation/a"].resolve().as_posix() + "\n").encode()
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
            {
                "path": selected_files["training/a"].resolve().as_posix(),
                "split": "training",
                "source_sha256": _sha(selected_files["training/a"]),
                "rank": 1,
            },
            {
                "path": selected_files["training/b"].resolve().as_posix(),
                "split": "training",
                "source_sha256": _sha(selected_files["training/b"]),
                "rank": 2,
            },
            {
                "path": selected_files["validation/a"].resolve().as_posix(),
                "split": "validation",
                "source_sha256": _sha(selected_files["validation/a"]),
                "rank": 1,
            },
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
            "quality_exclusion_manifest_path": quality_manifest.resolve().as_posix(),
            "quality_exclusion_manifest_sha256": _sha(quality_manifest),
        },
        "quality_exclusion": {
            "required": True,
            "manifest_contract": QUALITY_CONTRACT,
            "manifest_path": quality_manifest.resolve().as_posix(),
            "manifest_sha256": _sha(quality_manifest),
            "source_list_sha256": _entries_sha(quality_entries),
            "excluded_source_count": 1,
            "max_excluded_sources": 100,
            "matched_resolved_sources": 1,
            "reason_counts": {"severe_frame_crop": 1},
            "selection_authority": False,
            "ground_truth_authority": False,
            "replay_authority": False,
            "training_authority": False,
            "calibration_authority": False,
            "blind_test_authority": False,
            "deployment_authority": False,
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
                "quality_exclusion_manifest": quality_manifest.resolve().as_posix(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if mode == "forged_non_top_k":
        forged = dataset / "training" / "forged.jpg"
        forged.write_bytes(b"forged")
        inventory["selected_sources"][0]["path"] = forged.resolve().as_posix()
        inventory["selected_sources"][0]["source_sha256"] = _sha(forged)
        train = (
            forged.resolve().as_posix()
            + "\n"
            + selected_files["training/b"].resolve().as_posix()
            + "\n"
        ).encode()
    elif mode == "selected_quality_excluded_source":
        inventory["selected_sources"][0]["path"] = excluded_source.resolve().as_posix()
        inventory["selected_sources"][0]["source_sha256"] = _sha(excluded_source)
        train = (
            excluded_source.resolve().as_posix()
            + "\n"
            + selected_files["training/b"].resolve().as_posix()
            + "\n"
        ).encode()
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
    elif mode == "pilot_ready_duplicate_key":
        ready_path = pilot / "input_ready.json"
        rendered = ready_path.read_text(encoding="utf-8")
        ready_path.write_text(
            '{"status":"forged",' + rendered[1:], encoding="utf-8"
        )
    elif mode == "builder_failure":
        (dataset / "force-builder-failure").write_text("1\n", encoding="utf-8")
    elif mode == "quality_manifest_tamper":
        with quality_manifest.open("ab") as handle:
            handle.write(b"\n")

    environment = {
        **os.environ,
        "AUDIT_DIR": (tmp_path / "selection-audit").as_posix(),
        "CODE_ROOT": code.as_posix(),
        "PILOT_INPUT_DIR": pilot.as_posix(),
        "QUALITY_EXCLUSION_MANIFEST": quality_manifest.as_posix(),
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
        "mutate_quality_manifest_after_selector": quality_manifest,
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
    for name in (
        "AUDIT_DIR",
        "CODE_ROOT",
        "PILOT_INPUT_DIR",
        "QUALITY_EXCLUSION_MANIFEST",
    ):
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
    assert ready["audit_contract"] == AUDIT_CONTRACT
    assert ready["selection_contract"] == SELECTION_CONTRACT
    assert type(ready["seed"]) is int and ready["seed"] == 20260901
    assert ready["quality_exclusion"]["excluded_source_count"] == 1
    assert ready[
        "quality_exclusion_dataset_membership_verified_by_selector_replay"
    ] is True
    assert ready["quality_exclusion"]["source_list_sha256"] == _entries_sha(
        [
            {
                "source_sha256": _sha(
                    Path(env["FIXTURE_DATASET_DIR"]) / "excluded" / "bad-vinyl.jpg"
                ),
                "reason": "severe_frame_crop",
            }
        ]
    )
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
    assert evidence["audit_contract"] == AUDIT_CONTRACT
    assert evidence["quality_exclusion"]["matched_resolved_sources"] == 1
    assert evidence[
        "quality_exclusion_dataset_membership_verified_by_selector_replay"
    ] is True
    assert evidence["bindings"]["quality_exclusion_manifest_sha256"] == _sha(
        Path(env["QUALITY_EXCLUSION_MANIFEST"])
    )
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
        "pilot_ready_duplicate_key",
        "builder_failure",
        "extra_recompute_file",
        "extra_recompute_directory",
        "quality_manifest_tamper",
        "quality_canonical_hash_tamper",
        "quality_over_limit",
        "quality_duplicate_json_key",
        "quality_authority_zero",
        "quality_schema_true",
        "quality_max_float",
        "quality_reason_count_bool",
        "selected_quality_excluded_source",
        "mutate_wrapper_after_selector",
        "mutate_selector_after_selector",
        "mutate_proposal_after_selector",
        "mutate_data_after_selector",
        "mutate_old_manifest_after_selector",
        "mutate_drift_report_after_selector",
        "mutate_quality_manifest_after_selector",
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


def test_rejects_quality_manifest_ancestor_symlink_before_audit_creation(
    tmp_path: Path,
) -> None:
    env = _fixture(tmp_path)
    original = Path(env["QUALITY_EXCLUSION_MANIFEST"])
    real_parent = tmp_path / "real-quality-input"
    real_parent.mkdir()
    copied = real_parent / original.name
    shutil.copyfile(original, copied)
    linked_parent = tmp_path / "linked-quality-input"
    try:
        os.symlink(real_parent, linked_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    env["QUALITY_EXCLUSION_MANIFEST"] = (linked_parent / copied.name).as_posix()
    audit = Path(env["AUDIT_DIR"])
    result = subprocess.run(
        [_integration_bash(tmp_path), SCRIPT.as_posix()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert not audit.exists()


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
