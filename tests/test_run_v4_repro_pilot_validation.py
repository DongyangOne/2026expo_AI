from __future__ import annotations

import base64
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
    / "run_v4_repro_pilot_validation.sh"
)


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _integration_bash(tmp_path: Path) -> str:
    candidates = [shutil.which("bash")]
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if git_bash.is_file():
        candidates.insert(0, str(git_bash))
    for bash in dict.fromkeys(item for item in candidates if item):
        if subprocess.run(
            [bash, "-c", 'test -d "$1"', "bash", tmp_path.as_posix()], check=False
        ).returncode == 0:
            return bash
    pytest.skip("no bash can access the pytest temp path")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sha_marker(path: Path, artifacts: list[Path]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "".join(
                f"{_sha(artifact)}  {artifact.as_posix()}\n" for artifact in artifacts
            )
        )


MATERIALS = (
    "can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass",
    "battery", "fluorescent",
)


@pytest.fixture(scope="module")
def cohort_base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("v4-pilot-cohort")
    strata = (*MATERIALS, "background")
    class_ids = {name: index for index, name in enumerate(MATERIALS)}
    rows = []
    for split, quota in (("training", 250), ("validation", 100)):
        for stratum in strata:
            for rank in range(1, quota + 1):
                source = root / "sources" / split / "images" / stratum / f"{rank:04d}.jpg"
                label = root / "sources" / split / "labels" / stratum / f"{rank:04d}.txt"
                source.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(f"source:{split}:{stratum}:{rank}".encode())
                label.write_text(
                    "" if stratum == "background" else f"{class_ids[stratum]} 0.5 0.5 0.2 0.2\n",
                    encoding="utf-8",
                )
                anchor = rank == 1 and stratum in {"can", "vinyl", "background"}
                rows.append(
                    {
                        "split": split,
                        "stratum": stratum,
                        "path": source.resolve().as_posix(),
                        "source_sha256": _sha(source),
                        "label_path": label.resolve().as_posix(),
                        "label_sha256": _sha(label),
                        "source_size": source.stat().st_size,
                        "label_size": label.stat().st_size,
                        "drift_anchor": anchor,
                        "selection_reason": (
                            "drift_anchor_priority" if anchor else "deterministic_blake2"
                        ),
                    }
                )
    old_manifest = root / "historical-manifest.csv"
    old_manifest.write_text(
        "source_id,split,category\n" + "b" * 64 + ",training,can\n",
        encoding="utf-8",
    )
    drift_report = root / "historical-drift.json"
    drift_report.write_text('{"replay":{}}\n', encoding="utf-8")
    return {
        "root": root,
        "selected_rows": rows,
        "old_manifest": old_manifest,
        "drift_report": drift_report,
    }


def _fixture(
    tmp_path: Path,
    *,
    mode: str = "success",
    cohort: dict[str, object] | None = None,
) -> dict[str, str]:
    code = tmp_path / "code"
    scripts = code / "scripts"
    nas = scripts / "nas"
    nas.mkdir(parents=True)
    (nas / SCRIPT.name).write_bytes(SCRIPT.read_bytes())
    (nas / "run_v4_reproducible_generation.sh").write_text(
        "#!/bin/sh\n# frozen batch=1 generation fixture\n", encoding="utf-8"
    )
    (scripts / "build_v4_repro_pilot_inputs.py").write_text("BUILDER = 1\n", encoding="utf-8")
    (scripts / "prepare_proposal_verifier_dataset.py").write_text("PREPARE = 1\n", encoding="utf-8")
    (scripts / "verifier_preprocessing_contract.py").write_text("CONTRACT = 1\n", encoding="utf-8")
    validator = scripts / "validate_v4_background_candidates.py"
    validator.write_text(
        r'''
import argparse,csv,hashlib,json,os
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

p=argparse.ArgumentParser()
p.add_argument('--input-manifest',required=True)
p.add_argument('--dataset-info',required=True)
p.add_argument('--detector-model',required=True)
p.add_argument('--inference-spec',required=True)
p.add_argument('--output-manifest',required=True)
p.add_argument('--output-report',required=True)
p.add_argument('--diagnostic-only',action='store_true')
a=p.parse_args()
if not a.diagnostic_only:
    raise SystemExit('diagnostic-only flag is required')
mode=os.environ.get('FAKE_VALIDATOR_MODE','success')
run='a' if Path(a.output_manifest).parent.name == 'validator-a' else 'b'
validation=Path(os.environ['VALIDATION_DIR'])
if mode == 'first_failure' and run == 'a':
    raise SystemExit(29)
if mode == 'ready_race' and run == 'a':
    (validation/'control'/'diagnostic_ready.json').write_text('{}\n',encoding='utf-8')
    raise SystemExit(31)
if mode == 'raw_mutation' and run == 'a':
    raw=Path(a.input_manifest).resolve()
    raw.write_bytes(raw.read_bytes()+b'changed')
validated=Path(a.output_manifest)
content=Path(a.input_manifest).read_text(encoding='utf-8')
if mode == 'ab_mismatch' and run == 'b':
    lines=content.splitlines(); lines[-1]=lines[-1]+',different'; content='\n'.join(lines)+'\n'
validated.write_text(content,encoding='utf-8')
with validated.open(encoding='utf-8',newline='') as handle:
    rows=sum(1 for _ in csv.DictReader(handle))
materials=('can','pet','paper','plastic','styrofoam','vinyl','glass','battery','fluorescent')
counts={**{'training/'+name:25 for name in materials},**{'validation/'+name:10 for name in materials},'training/background':100,'validation/background':50}
counts['training/background'] += rows - 465
if mode == 'coverage_shortage':
    counts['validation/vinyl']=9
    counts['validation/background']=51
provenance={
    'provider_kind':'frozen_yolo_runtime',
    'runtime_detector_executed':True,
    'runtime_top1_replayed':True,
    'provided_top1_predictions_matched':True,
    'proposal_class_confidence_bbox_matched':True,
    'confidence_abs_tolerance':1e-6,
    'bbox_abs_tolerance':1e-4,
    'production_or_blind_authority':False,
}
report={
    'schema_version':1,
    'artifact_role':'v4_runtime_replay_diagnostic_not_lineage_blind_or_deployment_authority',
    'ready_for_lineage_upgrade':mode == 'diagnostic_lineage_ready',
    'lineage_execution_authorized':mode == 'diagnostic_lineage_ready',
    'blind_test_eligible':False,
    'production_deployment_authorized':False,
    'rows':rows,
    'counts':counts,
    'contract':{'proposal_provenance':provenance},
    'bindings':{
        'input_manifest_sha256':sha(a.input_manifest),
        'dataset_info_sha256':sha(a.dataset_info),
        'detector_model_sha256':sha(a.detector_model),
        'inference_spec_sha256':sha(a.inference_spec),
        'validated_manifest_sha256':sha(validated),
    },
}
Path(a.output_report).write_text(json.dumps(report,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(report,sort_keys=True))
''',
        encoding="utf-8",
    )

    materials = MATERIALS
    strata = (*materials, "background")
    selected_counts = {
        **{f"training/{name}": 250 for name in strata},
        **{f"validation/{name}": 100 for name in strata},
    }
    pilot = tmp_path / "pilot-inputs"
    pilot.mkdir()
    if cohort is None:
        raise ValueError("integration cohort fixture is required")
    selected_rows = [
        {key: value for key, value in row.items() if key not in {"source_size", "label_size"}}
        for row in cohort["selected_rows"]
    ]
    if mode == "selected_source_tamper":
        tampered_source = tmp_path / "tampered" / "source.jpg"
        tampered_source.parent.mkdir()
        tampered_source.write_bytes(b"tampered-selected-source")
        selected_rows[0]["path"] = tampered_source.resolve().as_posix()
    if mode == "selected_label_tamper":
        tampered_label = tmp_path / "tampered" / "label.txt"
        tampered_label.parent.mkdir()
        tampered_label.write_bytes(b"tampered-selected-label")
        selected_rows[0]["label_path"] = tampered_label.resolve().as_posix()
    train = pilot / "train_pilot.txt"
    validation = pilot / "validation_pilot.txt"
    train.write_text(
        "\n".join(sorted(row["path"] for row in selected_rows if row["split"] == "training"))
        + "\n",
        encoding="utf-8",
    )
    validation.write_text(
        "\n".join(sorted(row["path"] for row in selected_rows if row["split"] == "validation"))
        + "\n",
        encoding="utf-8",
    )
    pilot_yaml = pilot / "pilot_dataset.yaml"
    pilot_yaml.write_text(
        "\n".join(
            [
                f'path: "{Path(cohort["root"]).joinpath("sources").resolve().as_posix()}"',
                f'train: "{train.resolve().as_posix()}"',
                f'val: "{validation.resolve().as_posix()}"',
                "names:",
                *(f"  {index}: {name}" for index, name in enumerate(materials)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    universe = "a" * 64
    old_manifest = Path(cohort["old_manifest"])
    drift_report = Path(cohort["drift_report"])
    anchors_selected = sum(row["drift_anchor"] for row in selected_rows)
    pilot_inventory = pilot / "selection_inventory.json"
    pilot_inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_role": (
                    "v4_batch1_reproducibility_pilot_inputs_diagnostic_only_"
                    "not_training_blind_or_deployment_authority"
                ),
                "selection_contract": "v4_repro_pilot_inputs.label_stratified_blake2b.v1",
                "status": "selection_complete_not_replay_validated",
                "quota_per_stratum": {"training": 250, "validation": 100},
                "classes": list(materials),
                "strata": list(strata),
                "selected_counts": selected_counts,
                "quota_shortages": {name: 0 for name in selected_counts},
                "full_quota_met": True,
                "selected_sources": selected_rows,
                "historical_selection_evidence": {
                    "used_for_selection_only": True,
                    "old_manifest": {
                        "path": old_manifest.resolve().as_posix(),
                        "sha256": _sha(old_manifest),
                        "rows": 1,
                    },
                    "drift_report": {
                        "path": drift_report.resolve().as_posix(),
                        "sha256": _sha(drift_report),
                        "anchor_source_ids": anchors_selected,
                    },
                    "anchors_selected": anchors_selected,
                    "anchors_priority_selected": anchors_selected,
                },
                "bindings": {
                    "resolved_universe_sha256": universe,
                    "dataset_dir": Path(cohort["root"]).joinpath("sources").resolve().as_posix(),
                    "selector_path": (scripts / "build_v4_repro_pilot_inputs.py")
                    .resolve()
                    .as_posix(),
                    "selector_sha256": _sha(scripts / "build_v4_repro_pilot_inputs.py"),
                    "proposal_generator_path": (
                        scripts / "prepare_proposal_verifier_dataset.py"
                    )
                    .resolve()
                    .as_posix(),
                    "proposal_generator_sha256": _sha(
                        scripts / "prepare_proposal_verifier_dataset.py"
                    ),
                },
                "authority": {
                    "raw_generation_authorized": False,
                    "validator_authority": False,
                    "training_authorized": False,
                    "blind_test_authorized": False,
                    "production_deployment_authorized": False,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    pilot_artifacts = {
        "pilot_dataset.yaml": pilot_yaml,
        "selection_inventory.json": pilot_inventory,
        "train_pilot.txt": train,
        "validation_pilot.txt": validation,
    }
    pilot_inputs = pilot / "inputs.sha256"
    with pilot_inputs.open("w", encoding="ascii", newline="\n") as handle:
        for name, artifact in sorted(pilot_artifacts.items()):
            handle.write(f"{_sha(artifact)}  {name}\n")
    pilot_ready = pilot / "input_ready.json"
    pilot_ready.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_role": (
                    "v4_batch1_reproducibility_pilot_inputs_diagnostic_only_"
                    "not_training_blind_or_deployment_authority"
                ),
                "selection_contract": "v4_repro_pilot_inputs.label_stratified_blake2b.v1",
                "status": "pilot_inputs_ready",
                "selected_sources": 3500,
                "selected_counts": selected_counts,
                "full_quota_met": True,
                "historical_selection_only": True,
                "bindings": {
                    "inputs_marker_sha256": _sha(pilot_inputs),
                    "artifacts": {name: _sha(path) for name, path in sorted(pilot_artifacts.items())},
                    "resolved_universe_sha256": universe,
                },
                "validator_authority": False,
                "training_authorized": False,
                "blind_test_authorized": False,
                "production_deployment_authorized": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    detector = tmp_path / "best.pt"
    detector.write_bytes(b"detector")
    spec = tmp_path / "inference.json"
    spec.write_text("{}\n", encoding="utf-8")
    generation = tmp_path / "generation"
    raw = generation / "raw"
    control = generation / "control"
    (raw / "training").mkdir(parents=True)
    (raw / "validation").mkdir()
    manifest = raw / "manifest.csv"
    emitted_count = 465 if mode == "low_emitted_coverage" else 3465
    emitted = list(selected_rows[:emitted_count])
    if mode == "missing_anchor":
        anchor_index = next(index for index, row in enumerate(emitted) if row["drift_anchor"])
        emitted.pop(anchor_index)
        emitted.append(selected_rows[emitted_count])
    if mode == "foreign_manifest_source":
        foreign = tmp_path / "foreign" / "source.jpg"
        foreign.parent.mkdir()
        foreign.write_bytes(b"foreign-source")
        emitted[0] = {
            **emitted[0],
            "path": foreign.resolve().as_posix(),
            "source_sha256": _sha(foreign),
        }
    manifest_lines = ["id,value,filepath,crop_bytes,source_path_b64,source_id,split"]
    current_crop_seed: Path | None = None
    for index, row in enumerate(emitted, start=1):
        crop = raw / row["split"] / "crops" / f"{index:05d}.jpg"
        crop.parent.mkdir(parents=True, exist_ok=True)
        if mode == "external_crop" and index == 1:
            external_crop = generation / "external.jpg"
            external_crop.write_bytes(b"sealed-pilot-crop")
            crop_value = "../external.jpg"
        elif current_crop_seed is None or (index - 1) % 512 == 0:
            crop.write_bytes(b"sealed-pilot-crop")
            current_crop_seed = crop
            crop_value = crop.relative_to(raw).as_posix()
        else:
            os.link(current_crop_seed, crop)
            crop_value = crop.relative_to(raw).as_posix()
        encoded = base64.urlsafe_b64encode(os.fsencode(row["path"])).decode("ascii")
        manifest_lines.append(
            f'{index},raw,{crop_value},{len(b"sealed-pilot-crop")},{encoded},'
            f'{row["source_sha256"]},{row["split"]}'
        )
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    info = raw / "dataset_info.json"
    data_binding = pilot_yaml.resolve()
    if mode == "pilot_link_mismatch":
        data_binding = tmp_path / "wrong-pilot.yaml"
        data_binding.write_text("wrong: true\n", encoding="utf-8")
    batch = 2 if mode == "wrong_batch" else 1
    info.write_text(
        json.dumps(
            {
                "model": detector.resolve().as_posix(),
                "data": data_binding.resolve().as_posix(),
                "dataset_dir": Path(cohort["root"]).joinpath("sources").resolve().as_posix(),
                "manifest": manifest.resolve().as_posix(),
                "written_crops": emitted_count,
                "inference": {"batch": batch, "imgsz": 640, "conf": 0.10, "nms_iou": 0.70},
                "assignment": {
                    "positive_iou_inclusive": 0.50,
                    "negative_iou_inclusive": 0.10,
                    "ambiguous_iou_skipped": True,
                },
                "proposal_policy": {
                    "selection_mode": "runtime-top1",
                    "background_policy": "strict-zero-intersection",
                    "background_gt_margin": 0.10,
                },
                "crop": {"size": 320, "padding": 0.08, "jpeg_quality": 92},
                "selection": {
                    "max_per_class": 10000,
                    "val_max_per_class": 2000,
                    "max_background": 10000,
                    "val_max_background": 2000,
                    "seed": 20260901,
                },
                "storage_guards": {"min_free_gb": 300.0, "max_output_gb": 30.0},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    control.mkdir()
    inventory_rows = []
    for artifact in sorted(
        (item for item in raw.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(raw).as_posix(),
    ):
        inventory_rows.append(
            {
                "path": artifact.relative_to(raw).as_posix(),
                "size": artifact.stat().st_size,
                "sha256": _sha(artifact),
            }
        )
    inventory = control / "raw_output_inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": raw.resolve().as_posix(),
                "file_count": len(inventory_rows),
                "files": inventory_rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    inputs = control / "inputs.sha256"
    outputs = control / "outputs.sha256"
    dataset_input_inventory = control / "dataset_input_inventory.json"
    dataset_artifacts = []
    for row in selected_rows:
        for kind, path_key, sha_key in (
            ("source", "path", "source_sha256"),
            ("label", "label_path", "label_sha256"),
        ):
            artifact = Path(row[path_key])
            dataset_artifacts.append(
                {
                    "kind": kind,
                    "split": row["split"],
                    "path": artifact.resolve().as_posix(),
                    "exists": True,
                    "size": artifact.stat().st_size,
                    "sha256": row[sha_key],
                }
            )
    dataset_artifacts.sort(key=lambda row: (row["split"], row["kind"], row["path"]))
    if mode == "generation_inventory_mismatch":
        dataset_artifacts[0]["sha256"] = "f" * 64
    dataset_input_inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "resolved_yolo_train_val_sources_and_label_sidecars_sha256.v1",
                "data_path": pilot_yaml.resolve().as_posix(),
                "dataset_dir": Path(cohort["root"]).joinpath("sources").resolve().as_posix(),
                "artifact_count": len(dataset_artifacts),
                "artifacts": dataset_artifacts,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    _write_sha_marker(
        inputs,
        [
            detector,
            pilot_yaml,
            scripts / "prepare_proposal_verifier_dataset.py",
            scripts / "verifier_preprocessing_contract.py",
            nas / "run_v4_reproducible_generation.sh",
            dataset_input_inventory,
        ],
    )
    if mode == "generation_wrapper_tamper":
        with (nas / "run_v4_reproducible_generation.sh").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write("# tampered after generation marker\n")
    _write_sha_marker(outputs, [manifest, info, inventory])
    ready = control / "raw_generation_ready.json"
    ready.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_role": "raw_v4_reproducible_generation_not_validation_or_promotion_authority",
                "status": "raw_generation_ready",
                "batch": 1,
                "seed": 20260901,
                "validator_authority": False,
                "judge_authority": False,
                "training_authority": False,
                "blind_test_authority": False,
                "production_deployment_authorized": False,
                "bindings": {
                    "input_marker_sha256": _sha(inputs),
                    "output_marker_sha256": _sha(outputs),
                    "manifest_sha256": _sha(manifest),
                    "dataset_info_sha256": _sha(info),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    python_bin = Path(sys.executable).as_posix()
    tamper_env: dict[str, str] = {}
    if mode == "same_size_crop_tamper":
        assert len(b"tamper-pilot-crop") == len(b"sealed-pilot-crop")
        launcher = tmp_path / "python-tamper-launcher.sh"
        launcher.write_text(
            """#!/usr/bin/env bash
set -eu
if [ "${FAKE_VALIDATOR_MODE-}" = "same_size_crop_tamper" ] \
  && [ "${1-}" = "-" ] \
  && [ "${2-}" = "create" ] \
  && [ "${3-}" = "${TAMPER_COHORT_BINDING-}" ] \
  && [ ! -e "$TAMPER_SENTINEL" ]; then
  printf '%s' 'tamper-pilot-crop' > "$TAMPER_CROP"
  : > "$TAMPER_SENTINEL"
fi
exec "$REAL_PYTHON" "$@"
""",
            encoding="utf-8",
            newline="\n",
        )
        launcher.chmod(0o755)
        python_bin = launcher.as_posix()
        tamper_env = {
            "REAL_PYTHON": Path(sys.executable).as_posix(),
            "TAMPER_CROP": (
                raw / emitted[0]["split"] / "crops" / "00001.jpg"
            ).as_posix(),
            "TAMPER_SENTINEL": (tmp_path / "same-size-crop-tampered").as_posix(),
            "TAMPER_COHORT_BINDING": (
                tmp_path / "validation-pilot" / "control" / "cohort_binding.json"
            ).as_posix(),
        }
    return {
        **os.environ,
        **tamper_env,
        "VALIDATION_DIR": (tmp_path / "validation-pilot").as_posix(),
        "CODE_ROOT": code.as_posix(),
        "GEN_DIR": generation.as_posix(),
        "PILOT_INPUT_DIR": pilot.as_posix(),
        "DETECTOR_MODEL": detector.as_posix(),
        "INFERENCE_SPEC": spec.as_posix(),
        "PYTHON_BIN": python_bin,
        "FAKE_VALIDATOR_MODE": mode,
    }


def _run(
    tmp_path: Path,
    cohort: dict[str, object],
    *,
    mode: str = "success",
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    bash = _integration_bash(tmp_path)
    env = _fixture(tmp_path, mode=mode, cohort=cohort)
    result = subprocess.run(
        [bash, SCRIPT.as_posix()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, env


def test_shell_syntax_when_bash_is_available() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    subprocess.run([bash, "-n"], input=_text().encode("utf-8"), check=True)


def test_wrapper_is_fail_closed_and_diagnostic_only() -> None:
    text = _text()
    for name in (
        "VALIDATION_DIR", "CODE_ROOT", "GEN_DIR", "PILOT_INPUT_DIR",
        "DETECTOR_MODEL", "INFERENCE_SPEC",
    ):
        assert name in text
    assert 'if ! mkdir "$VALIDATION_DIR"' in text
    assert "trap on_exit 0" in text
    assert "raw_generation_ready.json" in text
    assert "raw_output_inventory.json" in text
    assert "dataset_input_inventory.json" in text
    assert "selection_inventory.json" in text
    assert "build_v4_repro_pilot_inputs.py" in text
    assert "prepare_proposal_verifier_dataset.py" in text
    assert "verifier_preprocessing_contract.py" in text
    assert "run_v4_reproducible_generation.sh" in text
    assert "pilot_dataset.yaml" in text
    assert "train_pilot.txt" in text
    assert "validation_pilot.txt" in text
    assert '"training": 250, "validation": 100' in text
    assert 'ready.get("selected_sources") != 3500' in text
    assert '("inference", "batch"): 1' in text
    assert "generation input marker does not bind the exact pilot dependencies" in text
    assert "selected source or label current bytes differ from inventory" in text
    assert "generation dataset input inventory differs from selected current bytes" in text
    assert "raw manifest contains a source outside the selected pilot cohort" in text
    assert "raw manifest selected-source coverage is below 99 percent" in text
    assert "raw manifest omits one or more selected drift anchors" in text
    assert "raw manifest crop escapes the inventoried raw directory" in text
    assert 'crop_size, crop_sha = stable_artifact(crop, description="raw manifest crop")' in text
    assert "raw manifest crop current bytes differ from the raw output inventory" in text
    assert "raw output inventory is not exactly manifest, info, and emitted crops" in text
    assert text.count("verify_generation_contract") >= 5
    assert text.count("verify_raw_inventory") >= 2
    assert 'cmp -s "$A_MANIFEST" "$B_MANIFEST"' in text
    assert 'remove_ready' in text
    for field in (
        '"lineage_execution_authorized": False',
        '"judge_authority": False',
        '"training_authority": False',
        '"blind_test_authority": False',
        '"candidate_promotion_authorized": False',
        '"production_deployment_authorized": False',
    ):
        assert field in text


def test_wrapper_requires_exact_authoritative_replay_contract() -> None:
    text = _text()
    required = (
        "v4_runtime_replay_diagnostic_not_lineage_blind_or_deployment_authority",
        '"ready_for_lineage_upgrade"',
        '"lineage_execution_authorized"',
        '"provider_kind": "frozen_yolo_runtime"',
        '"runtime_detector_executed": True',
        '"runtime_top1_replayed": True',
        '"provided_top1_predictions_matched": True',
        '"proposal_class_confidence_bbox_matched": True',
        '"confidence_abs_tolerance": 1e-6',
        '"bbox_abs_tolerance": 1e-4',
        "actual_rows != rows",
        "raw_rows != rows",
        '"training/background": 100',
        '"validation/background": 50',
        "--diagnostic-only",
    )
    for fragment in required:
        assert fragment in text


def test_wrapper_has_no_external_control_or_credentials() -> None:
    text = _text().lower()
    for token in (
        "docker ", "docker-compose", "systemctl", "service ", "kubectl",
        "ssh ", "scp ", "reboot", "shutdown", "password", "credential",
        "api_key", "token=",
    ):
        assert token not in text


def test_integration_success_seals_two_exact_runs(
    tmp_path: Path, cohort_base: dict[str, object]
) -> None:
    result, env = _run(tmp_path, cohort_base)
    assert result.returncode == 0, result.stderr
    root = Path(env["VALIDATION_DIR"])
    control = root / "control"
    ready = json.loads((control / "diagnostic_ready.json").read_text(encoding="utf-8"))
    assert ready["status"] == "batch1_validator_ab_reproducibility_passed"
    assert ready["validator_runs"] == 2
    assert ready["validated_rows"] == 3465
    assert ready["selected_sources"] == 3500
    assert ready["emitted_unique_sources"] == 3465
    assert ready["minimum_emitted_sources"] == 3465
    assert ready["selected_source_coverage"] == pytest.approx(0.99)
    assert ready["selected_drift_anchors"] > 0
    assert ready["emitted_drift_anchors"] == ready["selected_drift_anchors"]
    assert ready["lineage_execution_authorized"] is False
    assert ready["training_authority"] is False
    assert ready["blind_test_authority"] is False
    assert ready["production_deployment_authorized"] is False
    assert not (control / "failed.txt").exists()
    assert (root / "validator-a" / "manifest.csv").is_symlink()
    assert (root / "validator-b" / "validation").is_symlink()
    assert (root / "validator-a" / "manifest.v4.validated.csv").read_bytes() == (
        root / "validator-b" / "manifest.v4.validated.csv"
    ).read_bytes()
    bash = _integration_bash(tmp_path)
    for marker in (
        "00_raw_generation.sha256", "01_validator_a.sha256",
        "02_validator_b.sha256", "03_reproducibility.sha256",
    ):
        checked = subprocess.run(
            [bash, "-c", 'sha256sum -c "$1" >/dev/null', "bash", (control / marker).as_posix()],
            check=False,
        )
        assert checked.returncode == 0, marker


@pytest.mark.parametrize(
    "mode",
    [
        "ab_mismatch",
        "first_failure",
        "raw_mutation",
        "pilot_link_mismatch",
        "wrong_batch",
        "generation_wrapper_tamper",
        "selected_source_tamper",
        "selected_label_tamper",
        "generation_inventory_mismatch",
        "foreign_manifest_source",
        "external_crop",
        "same_size_crop_tamper",
        "low_emitted_coverage",
        "missing_anchor",
        "coverage_shortage",
        "diagnostic_lineage_ready",
    ],
)
def test_integration_failure_modes_publish_failure_only(
    tmp_path: Path, cohort_base: dict[str, object], mode: str
) -> None:
    result, env = _run(tmp_path, cohort_base, mode=mode)
    assert result.returncode != 0
    control = Path(env["VALIDATION_DIR"]) / "control"
    assert (control / "failed.txt").is_file()
    assert not (control / "diagnostic_ready.json").exists()


def test_integration_failure_removes_racing_ready_marker(
    tmp_path: Path, cohort_base: dict[str, object]
) -> None:
    result, env = _run(tmp_path, cohort_base, mode="ready_race")
    assert result.returncode != 0
    control = Path(env["VALIDATION_DIR"]) / "control"
    assert (control / "failed.txt").is_file()
    assert not (control / "diagnostic_ready.json").exists()
