from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from scripts import build_v4_candidate_training_authority as quality_authority


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
                source_sha = _sha(source)
                anchor = rank == 1 and stratum in {"can", "vinyl", "background"}
                historical_observed = anchor or (
                    rank == 2 and stratum == "paper"
                )
                historical_categories = (
                    ["can"]
                    if rank == 2 and stratum == "paper"
                    else [stratum]
                    if historical_observed
                    else []
                )
                selection_reason = (
                    "current_explicit_empty_label"
                    if stratum == "background"
                    else "drift_anchor_priority"
                    if anchor
                    else "historical_observation_priority_blake2"
                    if historical_observed
                    else "deterministic_blake2"
                )
                rows.append(
                    {
                        "split": split,
                        "stratum": stratum,
                        "selection_stratum": stratum,
                        "current_gt_stratum": stratum,
                        "selection_cohort": "current_yolo_ground_truth",
                        "selection_rank_within_stratum": rank,
                        "selection_score_blake2b128": hashlib.blake2b(
                            f"20260901|{split}|{stratum}|{source_sha}".encode(),
                            digest_size=16,
                        ).hexdigest(),
                        "path": source.resolve().as_posix(),
                        "source_sha256": source_sha,
                        "label_path": label.resolve().as_posix(),
                        "label_sha256": _sha(label),
                        "source_size": source.stat().st_size,
                        "label_size": label.stat().st_size,
                        "drift_anchor": anchor,
                        "selection_reason": selection_reason,
                        "explicit_empty_label": stratum == "background",
                        "historical_background_probe_selection_only": False,
                        "gt_class_id": None if stratum == "background" else class_ids[stratum],
                        "gt_xywhn": None if stratum == "background" else [0.5, 0.5, 0.2, 0.2],
                        "historical_categories_selection_only": historical_categories,
                    }
                )
    tier_by_reason = {
        "current_explicit_empty_label": 0,
        "drift_anchor_priority": 0,
        "historical_observation_priority_blake2": 1,
        "historical_background_probe_blake2": 2,
        "deterministic_blake2": 2,
    }
    for split in ("training", "validation"):
        for stratum in strata:
            bucket = sorted(
                (
                    row
                    for row in rows
                    if row["split"] == split and row["stratum"] == stratum
                ),
                key=lambda row: (
                    tier_by_reason[row["selection_reason"]],
                    row["selection_score_blake2b128"],
                ),
            )
            for selection_rank, row in enumerate(bucket, start=1):
                row["selection_rank_within_stratum"] = selection_rank
    quality_source = (
        root / "sources" / "training" / "images" / "vinyl" / "quality-excluded.jpg"
    )
    quality_label = (
        root / "sources" / "training" / "labels" / "vinyl" / "quality-excluded.txt"
    )
    quality_source.write_bytes(b"severely-cropped-operational-frame-after-cutoff")
    quality_label.write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    anchor_rows = [row for row in rows if row["drift_anchor"]]
    old_manifest = root / "historical-manifest.csv"
    old_manifest.write_text(
        "source_id,split,category\n"
        + "".join(
            f"{row['source_sha256']},{row['split']},{category}\n"
            for row in rows
            for category in row["historical_categories_selection_only"]
        ),
        encoding="utf-8",
    )
    drift_report = root / "historical-drift.json"
    drift_report.write_text(
        json.dumps(
            {
                "replay": {
                    "hard_semantic_mismatch_examples": {
                        "fixture": [
                            {"source_id": row["source_sha256"]}
                            for row in anchor_rows
                        ]
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "root": root,
        "selected_rows": rows,
        "old_manifest": old_manifest,
        "drift_report": drift_report,
        "quality_source": quality_source,
        "quality_source_sha256": _sha(quality_source),
    }


def _fixture(
    tmp_path: Path,
    *,
    mode: str = "success",
    cohort: dict[str, object] | None = None,
    quality_bundle: tuple[Path, Path] | None = None,
) -> dict[str, str]:
    code = tmp_path / "code"
    scripts = code / "scripts"
    nas = scripts / "nas"
    nas.mkdir(parents=True)
    quality_validator = scripts / "operational_quality_assembly_contract.py"
    quality_validator.write_bytes((SCRIPT.parents[1] / quality_validator.name).read_bytes())
    for source in quality_authority._quality_assembly_code_paths().values():
        (scripts / source.name).write_bytes(source.read_bytes())
    (nas / SCRIPT.name).write_bytes(SCRIPT.read_bytes())
    audit_wrapper_source = SCRIPT.parent / "run_v4_repro_selection_audit.sh"
    (nas / audit_wrapper_source.name).write_bytes(audit_wrapper_source.read_bytes())
    (nas / "run_v4_reproducible_generation.sh").write_text(
        "#!/bin/sh\n# frozen batch=1 generation fixture\n", encoding="utf-8"
    )
    (scripts / "build_v4_repro_pilot_inputs.py").write_text(
        r'''
import argparse,hashlib,json,os,shutil
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

counter=Path(os.environ['FAKE_SELECTION_COUNTER'])
count=int(counter.read_text(encoding='ascii')) if counter.exists() else 0
count+=1
counter.write_text(str(count),encoding='ascii')
if count != 1:
    raise SystemExit('CPU selector was executed more than once')

p=argparse.ArgumentParser()
p.add_argument('--data',required=True)
p.add_argument('--dataset-dir',required=True)
p.add_argument('--output-dir',required=True)
p.add_argument('--seed',required=True,type=int)
p.add_argument('--train-quota-per-stratum',required=True)
p.add_argument('--validation-quota-per-stratum',required=True)
p.add_argument('--old-manifest')
p.add_argument('--drift-report')
p.add_argument('--quality-exclusion-manifest',required=True)
p.add_argument('--quality-exclusion-assembly-receipt',required=True)
a=p.parse_args()
reference=Path(os.environ['FAKE_SELECTION_RECOMPUTE_REFERENCE'])
output=Path(a.output_dir)
output.mkdir(parents=False,exist_ok=False)
for name in ('selection_inventory.json','train_pilot.txt','validation_pilot.txt'):
    shutil.copyfile(reference/name,output/name)
materials=('can','pet','paper','plastic','styrofoam','vinyl','glass','battery','fluorescent')
yaml_lines=[
    f'path: "{Path(a.dataset_dir).resolve().as_posix()}"',
    f'train: "{(output/"train_pilot.txt").resolve().as_posix()}"',
    f'val: "{(output/"validation_pilot.txt").resolve().as_posix()}"',
    'names:',
    *(f'  {index}: {name}' for index,name in enumerate(materials)),
]
(output/'pilot_dataset.yaml').write_text('\n'.join(yaml_lines)+'\n',encoding='utf-8')
artifacts={name:output/name for name in (
    'pilot_dataset.yaml','selection_inventory.json','train_pilot.txt','validation_pilot.txt'
)}
marker=''.join(f'{sha(path)}  {name}\n' for name,path in sorted(artifacts.items()))
(output/'inputs.sha256').write_text(marker,encoding='ascii')
ready=json.loads((reference/'input_ready.json').read_text(encoding='utf-8'))
ready['seed']=a.seed
ready['bindings']['inputs_marker_sha256']=sha(output/'inputs.sha256')
ready['bindings']['artifacts']={
    name:sha(path) for name,path in sorted(artifacts.items())
}
(output/'input_ready.json').write_text(
    json.dumps(ready,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8'
)
''',
        encoding="utf-8",
    )
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
if mode in {'quality_receipt_stage_mutation', 'quality_validator_stage_mutation'} and run == 'a':
    target = (
        Path(os.environ['QUALITY_EXCLUSION_ASSEMBLY_RECEIPT'])
        if mode == 'quality_receipt_stage_mutation'
        else Path(os.environ['CODE_ROOT'])/'scripts'/'operational_quality_assembly_contract.py'
    )
    before = target.stat()
    target.write_bytes(target.read_bytes()+b'\n')
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
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
    probe_modes = {"probe_replays_material", "forged_probe_membership"}
    probe_source_sha: str | None = None
    if mode in probe_modes:
        probe_index = next(
            index
            for index, row in enumerate(selected_rows)
            if row["split"] == "training"
            and row["stratum"] == "background"
            and not row["drift_anchor"]
        )
        source = tmp_path / "probe-source" / "training" / "images" / "probe.jpg"
        label = tmp_path / "probe-source" / "training" / "labels" / "probe.txt"
        source.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        source.write_bytes(b"current-material-source-selected-as-historical-background-probe")
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        probe_source_sha = _sha(source)
        selected_rows[probe_index] = {
            **selected_rows[probe_index],
            "path": source.resolve().as_posix(),
            "source_sha256": probe_source_sha,
            "label_path": label.resolve().as_posix(),
            "label_sha256": _sha(label),
            "current_gt_stratum": "can",
            "selection_cohort": "historical_background_probe",
            "selection_reason": "historical_background_probe_blake2",
            "selection_score_blake2b128": hashlib.blake2b(
                f"20260901|training|background|{probe_source_sha}".encode(),
                digest_size=16,
            ).hexdigest(),
            "explicit_empty_label": False,
            "historical_background_probe_selection_only": True,
            "gt_class_id": 0,
            "gt_xywhn": [0.5, 0.5, 0.2, 0.2],
            "historical_categories_selection_only": ["background"],
        }
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
    if mode == "forged_label_semantics":
        forged_index = next(
            index
            for index, row in enumerate(selected_rows)
            if row["stratum"] == "can"
        )
        forged_label = tmp_path / "forged" / "label.txt"
        forged_label.parent.mkdir()
        forged_label.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        selected_rows[forged_index]["label_path"] = forged_label.resolve().as_posix()
        selected_rows[forged_index]["label_sha256"] = _sha(forged_label)
    if mode == "forged_selection_blake_score":
        original_score = selected_rows[0]["selection_score_blake2b128"]
        selected_rows[0]["selection_score_blake2b128"] = (
            ("0" if original_score[0] != "0" else "1") + original_score[1:]
        )
    if mode == "forged_observation_priority_membership":
        forged_observed_row = next(
            row
            for row in selected_rows
            if row["stratum"] == "paper"
            and row["selection_reason"] == "deterministic_blake2"
        )
        forged_observed_row["selection_reason"] = (
            "historical_observation_priority_blake2"
        )
        forged_observed_row["historical_categories_selection_only"] = ["paper"]
    tier_by_reason = {
        "current_explicit_empty_label": 0,
        "drift_anchor_priority": 0,
        "historical_observation_priority_blake2": 1,
        "historical_background_probe_blake2": 2,
        "deterministic_blake2": 2,
    }
    for split in ("training", "validation"):
        for stratum in (*materials, "background"):
            bucket = sorted(
                (
                    row
                    for row in selected_rows
                    if row["split"] == split and row["stratum"] == stratum
                ),
                key=lambda row: (
                    (
                        1
                        if stratum == "background"
                        and row["selection_reason"] == "drift_anchor_priority"
                        else tier_by_reason[row["selection_reason"]]
                    ),
                    row["selection_score_blake2b128"],
                ),
            )
            for selection_rank, row in enumerate(bucket, start=1):
                row["selection_rank_within_stratum"] = selection_rank
    if mode == "forged_observation_priority_tier_order":
        observed_row = next(
            row
            for row in selected_rows
            if row["split"] == "training"
            and row["stratum"] == "paper"
            and row["selection_reason"]
            == "historical_observation_priority_blake2"
        )
        unseen_row = next(
            row
            for row in selected_rows
            if row["split"] == "training"
            and row["stratum"] == "paper"
            and row["selection_reason"] == "deterministic_blake2"
        )
        observed_row["selection_rank_within_stratum"], unseen_row[
            "selection_rank_within_stratum"
        ] = (
            unseen_row["selection_rank_within_stratum"],
            observed_row["selection_rank_within_stratum"],
        )
    forged_anchor_row: dict[str, object] | None = None
    if mode == "forged_drift_anchor":
        forged_anchor_row = next(
            row
            for row in selected_rows
            if row["split"] == "training"
            and row["stratum"] == "paper"
            and not row["drift_anchor"]
        )
        forged_anchor_row["drift_anchor"] = True
        forged_anchor_row["selection_reason"] = "drift_anchor_priority"
        forged_anchor_row["historical_categories_selection_only"] = ["paper"]
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
    source_data = tmp_path / "source-data.yaml"
    source_data.write_text(
        "\n".join(
            [
                f'path: "{Path(cohort["root"]).joinpath("sources").resolve().as_posix()}"',
                'train: "training/images"',
                'val: "validation/images"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    universe = "a" * 64
    quality_source_sha = str(cohort["quality_source_sha256"])
    if mode == "quality_selected_source":
        quality_source_sha = str(selected_rows[0]["source_sha256"])
    quality_reason = (
        "unknown_capture_quality_reason"
        if mode == "quality_unknown_reason"
        else "too_low_resolution"
    )
    quality_entries = (
        sorted(
            [
                {
                    "source_sha256": hashlib.sha256(
                        f"quality-over-cap:{index}".encode()
                    ).hexdigest(),
                    "reason": quality_reason,
                }
                for index in range(101)
            ],
            key=lambda row: row["source_sha256"],
        )
        if mode == "quality_over_maximum"
        else [{"source_sha256": quality_source_sha, "reason": quality_reason}]
    )
    quality_count = len(quality_entries)
    canonical_quality_bytes = (
        json.dumps(
            quality_entries,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    quality_root = tmp_path / "quality-assembly"
    quality_root.mkdir()
    quality_manifest = quality_root / "operational_quality_exclusions.json"
    quality_receipt = quality_root / "operational_quality_exclusion_assembly.json"
    quality_marker = quality_root / "assembly.sha256"
    quality_manifest_value = {
        "schema_version": 1,
        "artifact_role": (
            "v4_capture_quality_exclusion_manifest_selection_only_"
            "not_ground_truth_or_authority"
        ),
        "quality_exclusion_contract": (
            "v4_capture_quality_exclusions.sha256_reason_only.v1"
        ),
        "status": "quality_exclusions_ready",
        "excluded_source_count": quality_count,
        "max_excluded_sources": 100,
        "reason_counts": {quality_reason: quality_count},
        "source_list_sha256": hashlib.sha256(canonical_quality_bytes).hexdigest(),
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
    if mode == "quality_reason_counts_mismatch":
        quality_manifest_value["reason_counts"] = {quality_reason: 2}
    if mode == "quality_numeric_false_authority":
        quality_manifest_value["authority"]["selection"] = 0
    if quality_bundle is not None:
        quality_manifest_value = json.loads(quality_bundle[0].read_bytes())
    quality_manifest.write_bytes(quality_authority._canonical_json(quality_manifest_value))
    quality_count = quality_manifest_value["excluded_source_count"]
    quality_reason_counts = quality_manifest_value["reason_counts"]
    receipt_value = {
        "schema_version": 1,
        "assembly_schema": quality_authority.QUALITY_ASSEMBLY_SCHEMA,
        "artifact_role": quality_authority.QUALITY_ASSEMBLY_ROLE,
        "status": quality_authority.QUALITY_ASSEMBLY_STATUS,
        "assembly_mode": quality_authority.QUALITY_ASSEMBLY_MODE,
        "quality_exclusion_contract": quality_authority.QUALITY_CONTRACT,
        "operational_capture_cutoff_kst": quality_authority.OPERATIONAL_CUTOFF.isoformat(),
        "teacher_label_schema_version": quality_authority.QUALITY_ASSEMBLY_TEACHER_SCHEMA,
        "selected_source_count": quality_count,
        "reason_counts": quality_reason_counts,
        "quality_manifest_sha256": _sha(quality_manifest),
        "quality_source_list_sha256": quality_manifest_value["source_list_sha256"],
        "input_sha256": {name: hashlib.sha256(name.encode()).hexdigest() for name in quality_authority.QUALITY_ASSEMBLY_INPUT_SHA_FIELDS},
        "observed_code_sha256": quality_authority._quality_assembly_code_hashes(),
        "scope": {
            "teacher_subjective_quality_included": True,
            "objective_queue_quality_included": False,
            "objective_prepare_bundle_validated": True,
            "subjective_quality_source_count": quality_count,
            "objective_quality_source_count": 0,
            "paths_or_private_ids_exported": False,
            "trusted_policy_pinned": False,
            "executed_code_cryptographically_attested": False,
        },
        "authority": {name: False for name in quality_authority.FALSE_AUTHORITY_FIELDS},
    }
    if quality_bundle is not None:
        receipt_value = json.loads(quality_bundle[1].read_bytes())
    quality_receipt.write_bytes(quality_authority._canonical_json(receipt_value))
    quality_marker.write_bytes(quality_authority._quality_assembly_marker_bytes(
        manifest_content=quality_manifest.read_bytes(), receipt_content=quality_receipt.read_bytes()
    ))
    assembly_metadata = {
        "assembly_schema": receipt_value["assembly_schema"],
        "assembly_mode": receipt_value["assembly_mode"],
        "operational_capture_cutoff_kst": receipt_value["operational_capture_cutoff_kst"],
        "objective_prepare_bundle_validated": receipt_value["scope"]["objective_prepare_bundle_validated"],
        "assembly_receipt_path": quality_receipt.resolve().as_posix(),
        "assembly_receipt_sha256": _sha(quality_receipt),
        "assembly_marker_path": quality_marker.resolve().as_posix(),
        "assembly_marker_sha256": _sha(quality_marker),
        "assembly_validator_path": quality_validator.resolve().as_posix(),
        "assembly_validator_sha256": _sha(quality_validator),
    }
    assembly_digest_bindings = {
        "quality_exclusions_sha256": _sha(quality_manifest),
        "quality_exclusion_manifest_sha256": _sha(quality_manifest),
        "quality_exclusion_assembly_receipt_sha256": _sha(quality_receipt),
        "quality_exclusion_assembly_marker_sha256": _sha(quality_marker),
        "quality_assembly_validator_sha256": _sha(quality_validator),
    }
    assembly_path_bindings = {
        "quality_exclusion_manifest_path": quality_manifest.resolve().as_posix(),
        "quality_exclusion_assembly_receipt_path": quality_receipt.resolve().as_posix(),
        "quality_exclusion_assembly_marker_path": quality_marker.resolve().as_posix(),
        "quality_assembly_validator_path": quality_validator.resolve().as_posix(),
    }
    quality_false_fields = {
        "selection_authority": False,
        "ground_truth_authority": False,
        "replay_authority": False,
        "training_authority": False,
        "calibration_authority": False,
        "blind_test_authority": False,
        "deployment_authority": False,
    }
    inventory_quality = {
        "required": True,
        "manifest_contract": "v4_capture_quality_exclusions.sha256_reason_only.v1",
        "manifest_path": quality_manifest.resolve().as_posix(),
        "manifest_sha256": _sha(quality_manifest),
        "excluded_source_count": quality_count,
        "max_excluded_sources": 100,
        "matched_resolved_sources": quality_count,
        "reason_counts": quality_reason_counts,
        "source_list_sha256": quality_manifest_value["source_list_sha256"],
        **quality_false_fields,
        **assembly_metadata,
    }
    ready_quality = {
        key: value
        for key, value in inventory_quality.items()
        if key not in {"manifest_path", "assembly_receipt_path", "assembly_marker_path", "assembly_validator_path", "matched_resolved_sources"}
    }
    old_manifest = tmp_path / "historical-manifest.csv"
    old_manifest.write_bytes(Path(cohort["old_manifest"]).read_bytes())
    if probe_source_sha is not None and mode != "forged_probe_membership":
        with old_manifest.open("a", encoding="utf-8", newline="") as handle:
            handle.write(f"{probe_source_sha},training,background\n")
    if forged_anchor_row is not None:
        with old_manifest.open("a", encoding="utf-8", newline="") as handle:
            handle.write(
                f"{forged_anchor_row['source_sha256']},"
                f"{forged_anchor_row['split']},paper\n"
            )
    old_manifest_rows = sum(
        1 for _ in old_manifest.read_text(encoding="utf-8").splitlines()[1:]
    )
    if mode == "forged_historical_row_count":
        old_manifest_rows += 1
    drift_report = Path(cohort["drift_report"])
    anchors_selected = sum(row["drift_anchor"] for row in selected_rows)
    priority_anchors = sum(
        row["selection_reason"] == "drift_anchor_priority" for row in selected_rows
    )
    historical_observation_priorities = sum(
        row["selection_reason"] == "historical_observation_priority_blake2"
        for row in selected_rows
    )
    eligible_material_historical_observed_counts = dict(
        sorted(
            Counter(
                f"{row['split']}/{row['stratum']}"
                for row in selected_rows
                if row["stratum"] != "background"
                and row["historical_categories_selection_only"]
            ).items()
        )
    )
    for split in ("training", "validation"):
        for material in materials:
            eligible_material_historical_observed_counts.setdefault(
                f"{split}/{material}", 0
            )
    if mode == "forged_eligible_observation_count":
        eligible_material_historical_observed_counts["training/paper"] += 1
    pilot_seed: object = (
        -1
        if mode == "negative_selection_seed"
        else 20260902
        if mode == "forged_positive_pilot_seed"
        else 20260901
    )
    ready_seed: object = True if mode == "boolean_ready_seed" else pilot_seed
    selected_current_gt_counts = dict(
        sorted(
            Counter(
                f"{row['split']}/{row['current_gt_stratum']}"
                for row in selected_rows
            ).items()
        )
    )
    selected_cohort_counts = dict(
        sorted(
            Counter(
                f"{row['split']}/{row['selection_cohort']}"
                for row in selected_rows
            ).items()
        )
    )
    background_quota_composition = {}
    for split in ("training", "validation"):
        background_rows = [
            row
            for row in selected_rows
            if row["split"] == split and row["stratum"] == "background"
        ]
        background_quota_composition[split] = {
            "current_explicit_empty_label": sum(
                row["explicit_empty_label"] is True for row in background_rows
            ),
            "historical_background_probe": sum(
                row["historical_background_probe_selection_only"] is True
                for row in background_rows
            ),
            "total": len(background_rows),
        }
    pilot_inventory = pilot / "selection_inventory.json"
    pilot_inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_role": (
                    "v4_batch1_reproducibility_pilot_inputs_diagnostic_only_"
                    "not_training_blind_or_deployment_authority"
                ),
                "selection_contract": (
                    "v4_repro_pilot_inputs."
                    "gt_stratified_historical_observation_priority_blake2b.v4"
                ),
                "status": "selection_complete_not_replay_validated",
                "seed": pilot_seed,
                "quota_per_stratum": {"training": 250, "validation": 100},
                "classes": list(materials),
                "strata": list(strata),
                "source_contract": {
                    "explicit_label_file_required": True,
                    "background_prefers_current_explicit_empty_label": True,
                    "historical_background_probe_requires_current_single_object_label": True,
                    "historical_background_category_is_selection_only": True,
                    "historical_background_category_is_not_ground_truth": True,
                    "historical_observation_priority_is_selection_only": (
                        mode != "false_observation_source_contract"
                    ),
                    "current_batch1_replay_decides_emitted_category": True,
                    "material_requires_exactly_one_valid_yolo_label": True,
                    "multi_object_excluded": True,
                    "cross_split_content_duplicates_quarantined": True,
                    "same_split_conflicting_ground_truth_quarantined": True,
                    "capture_quality_exclusion_manifest_required": True,
                    "quality_excluded_sources_never_selected": True,
                    "object_dent_or_crush_is_not_a_capture_quality_exclusion": True,
                },
                "selected_counts": selected_counts,
                "eligible_counts": {
                    **selected_counts,
                    "training/background": (
                        selected_counts["training/background"] + 1
                        if mode == "forged_background_eligible_count"
                        else selected_counts["training/background"]
                    ),
                },
                "selected_current_gt_counts": selected_current_gt_counts,
                "selected_cohort_counts": selected_cohort_counts,
                "background_quota_composition": background_quota_composition,
                "quota_shortages": {name: 0 for name in selected_counts},
                "full_quota_met": True,
                "quality_exclusion": inventory_quality,
                "selected_sources": selected_rows,
                "historical_selection_evidence": {
                    "used_for_selection_only": True,
                    "ground_truth_authority": False,
                    "replay_validation_authority": False,
                    "background_category_authority": False,
                    "old_manifest": {
                        "path": old_manifest.resolve().as_posix(),
                        "sha256": _sha(old_manifest),
                        "rows": old_manifest_rows,
                    },
                    "drift_report": {
                        "path": drift_report.resolve().as_posix(),
                        "sha256": _sha(drift_report),
                        "anchor_source_ids": anchors_selected,
                    },
                    "anchors_selected": anchors_selected,
                    "anchors_priority_selected": priority_anchors,
                    "historical_observation_priority_selected": (
                        historical_observation_priorities
                        + (1 if mode == "forged_observation_priority_count" else 0)
                    ),
                    "eligible_material_historical_observed_counts": (
                        eligible_material_historical_observed_counts
                    ),
                    "eligible_current_explicit_empty_counts": {
                        split: background_quota_composition[split][
                            "current_explicit_empty_label"
                        ]
                        for split in ("training", "validation")
                    },
                    "eligible_historical_background_probe_counts": {
                        split: background_quota_composition[split][
                            "historical_background_probe"
                        ]
                        for split in ("training", "validation")
                    },
                },
                "bindings": {
                    "data_path": source_data.resolve().as_posix(),
                    **assembly_digest_bindings,
                    **assembly_path_bindings,
                    "data_sha256": _sha(source_data),
                    "resolved_universe_sha256": universe,
                    "dataset_dir": Path(cohort["root"]).joinpath("sources").resolve().as_posix(),
                    "selector_path": (
                        tmp_path / "archived-code" / "build_v4_repro_pilot_inputs.py"
                        if mode == "archived_code_paths"
                        else scripts / "build_v4_repro_pilot_inputs.py"
                    ).resolve().as_posix(),
                    "selector_sha256": _sha(scripts / "build_v4_repro_pilot_inputs.py"),
                    "proposal_generator_path": (
                        tmp_path
                        / "archived-code"
                        / "prepare_proposal_verifier_dataset.py"
                        if mode == "archived_code_paths"
                        else scripts / "prepare_proposal_verifier_dataset.py"
                    ).resolve().as_posix(),
                    "proposal_generator_sha256": _sha(
                        scripts / "prepare_proposal_verifier_dataset.py"
                    ),
                    "quality_exclusion_manifest_path": (
                        (tmp_path / "wrong-quality-exclusions.json").resolve().as_posix()
                        if mode == "quality_manifest_path_binding_mismatch"
                        else quality_manifest.resolve().as_posix()
                    ),
                    "quality_exclusion_manifest_sha256": (
                        "f" * 64
                        if mode == "quality_manifest_hash_binding_mismatch"
                        else _sha(quality_manifest)
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
                "selection_contract": (
                    "v4_repro_pilot_inputs."
                    "gt_stratified_historical_observation_priority_blake2b.v4"
                ),
                "status": "pilot_inputs_ready",
                "seed": ready_seed,
                "selected_sources": 3500,
                "selected_counts": selected_counts,
                "selected_current_gt_counts": selected_current_gt_counts,
                "selected_cohort_counts": selected_cohort_counts,
                "background_quota_composition": background_quota_composition,
                "full_quota_met": True,
                "historical_selection_only": True,
                "quality_exclusion": ready_quality,
                "bindings": {
                    "inputs_marker_sha256": _sha(pilot_inputs),
                    **assembly_digest_bindings,
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
    selection_reference = tmp_path / "selection-audit-reference"
    selection_reference.mkdir()
    for artifact in (pilot_inventory, train, validation, pilot_ready):
        shutil.copyfile(artifact, selection_reference / artifact.name)

    if mode == "forged_non_top_k_selection":
        victim_index, victim = max(
            (
                (index, row)
                for index, row in enumerate(selected_rows)
                if row["split"] == "training"
                and row["stratum"] == "paper"
                and row["selection_reason"] == "deterministic_blake2"
            ),
            key=lambda item: item[1]["selection_score_blake2b128"],
        )
        extra_dir = Path(cohort["root"]) / "sources" / "training"
        extra_source = extra_dir / "images" / "paper" / "non-top-k-extra.jpg"
        extra_label = extra_dir / "labels" / "paper" / "non-top-k-extra.txt"
        extra_label.write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        for attempt in range(1, 100_000):
            extra_source.write_bytes(f"non-top-k-extra:{attempt}".encode())
            extra_sha = _sha(extra_source)
            extra_score = hashlib.blake2b(
                f"20260901|training|paper|{extra_sha}".encode(), digest_size=16
            ).hexdigest()
            if extra_score > victim["selection_score_blake2b128"]:
                break
        else:  # pragma: no cover - astronomically unlikely fixture failure
            raise AssertionError("could not construct a higher-BLAKE eligible source")
        replacement = {
            **victim,
            "path": extra_source.resolve().as_posix(),
            "source_sha256": extra_sha,
            "label_path": extra_label.resolve().as_posix(),
            "label_sha256": _sha(extra_label),
            "selection_score_blake2b128": extra_score,
        }
        selected_rows[victim_index] = replacement
        train.write_text(
            "\n".join(
                sorted(row["path"] for row in selected_rows if row["split"] == "training")
            )
            + "\n",
            encoding="utf-8",
        )
        forged_inventory = json.loads(pilot_inventory.read_text(encoding="utf-8"))
        forged_inventory["selected_sources"] = selected_rows
        forged_inventory["eligible_counts"]["training/paper"] += 1
        pilot_inventory.write_text(
            json.dumps(forged_inventory, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        pilot_artifacts = {
            "pilot_dataset.yaml": pilot_yaml,
            "selection_inventory.json": pilot_inventory,
            "train_pilot.txt": train,
            "validation_pilot.txt": validation,
        }
        with pilot_inputs.open("w", encoding="ascii", newline="\n") as handle:
            for name, artifact in sorted(pilot_artifacts.items()):
                handle.write(f"{_sha(artifact)}  {name}\n")
        forged_ready = json.loads(pilot_ready.read_text(encoding="utf-8"))
        forged_ready["bindings"]["inputs_marker_sha256"] = _sha(pilot_inputs)
        forged_ready["bindings"]["artifacts"] = {
            name: _sha(path) for name, path in sorted(pilot_artifacts.items())
        }
        pilot_ready.write_text(
            json.dumps(forged_ready, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    selection_audit = tmp_path / "selection-audit"
    selection_recompute = selection_audit / "recompute"
    selection_recompute.mkdir(parents=True)
    recomputed_inventory = selection_recompute / "selection_inventory.json"
    recomputed_train = selection_recompute / "train_pilot.txt"
    recomputed_validation = selection_recompute / "validation_pilot.txt"
    shutil.copyfile(
        selection_reference / "selection_inventory.json", recomputed_inventory
    )
    shutil.copyfile(selection_reference / "train_pilot.txt", recomputed_train)
    shutil.copyfile(
        selection_reference / "validation_pilot.txt", recomputed_validation
    )
    recomputed_yaml = selection_recompute / "pilot_dataset.yaml"
    recomputed_yaml.write_text(
        "\n".join(
            [
                f'path: "{Path(cohort["root"]).joinpath("sources").resolve().as_posix()}"',
                f'train: "{recomputed_train.resolve().as_posix()}"',
                f'val: "{recomputed_validation.resolve().as_posix()}"',
                "names:",
                *(f"  {index}: {name}" for index, name in enumerate(materials)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    recomputed_input_artifacts = {
        "pilot_dataset.yaml": recomputed_yaml,
        "selection_inventory.json": recomputed_inventory,
        "train_pilot.txt": recomputed_train,
        "validation_pilot.txt": recomputed_validation,
    }
    recomputed_inputs = selection_recompute / "inputs.sha256"
    with recomputed_inputs.open("w", encoding="ascii", newline="\n") as handle:
        for name, artifact in sorted(recomputed_input_artifacts.items()):
            handle.write(f"{_sha(artifact)}  {name}\n")
    recomputed_ready = selection_recompute / "input_ready.json"
    recomputed_ready.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pilot_inputs_ready",
                "selection_contract": (
                    "v4_repro_pilot_inputs."
                    "gt_stratified_historical_observation_priority_blake2b.v4"
                ),
                "seed": 20260901,
                "quality_exclusion": ready_quality,
                "bindings": {
                    "inputs_marker_sha256": _sha(recomputed_inputs),
                    **assembly_digest_bindings,
                    "artifacts": {
                        name: _sha(path)
                        for name, path in sorted(recomputed_input_artifacts.items())
                    },
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    recompute_audit_artifacts = {
        "input_ready.json": recomputed_ready,
        "inputs.sha256": recomputed_inputs,
        "pilot_dataset.yaml": recomputed_yaml,
        "selection_inventory.json": recomputed_inventory,
        "train_pilot.txt": recomputed_train,
        "validation_pilot.txt": recomputed_validation,
    }
    audit_pilot_artifacts = {
        "input_ready.json": pilot_ready,
        "inputs.sha256": pilot_inputs,
        "pilot_dataset.yaml": pilot_yaml,
        "selection_inventory.json": pilot_inventory,
        "train_pilot.txt": train,
        "validation_pilot.txt": validation,
    }
    false_authority = {
        "raw_generation_authorized": False,
        "validator_authority": False,
        "judge_authority": False,
        "training_authority": False,
        "blind_test_authority": False,
        "candidate_promotion_authorized": False,
        "production_deployment_authorized": False,
    }
    selection_audit_evidence = selection_audit / "selection_audit_evidence.json"
    selection_audit_evidence_value = {
                "schema_version": 1,
                "artifact_role": (
                    "v4_repro_selection_audit_cpu_only_diagnostic_"
                    "not_generation_training_blind_or_deployment_authority"
                ),
                "audit_contract": "v4_repro_selection_audit.cpu_only_byte_exact.v2",
                "status": "selection_recomputed_byte_exact",
                "selection_contract": (
                    "v4_repro_pilot_inputs."
                    "gt_stratified_historical_observation_priority_blake2b.v4"
                ),
                "cpu_only": True,
                "seed": 20260901,
                "quota_per_stratum": {"training": 250, "validation": 100},
                "selected_sources": 3500,
                "quality_exclusion_dataset_membership_verified_by_selector_replay": True,
                "quality_exclusion": (
                    {**inventory_quality, "manifest_sha256": "f" * 64}
                    if mode == "quality_audit_evidence_mismatch"
                    else inventory_quality
                ),
                "comparisons": {
                    "selection_inventory_json_byte_exact": True,
                    "train_pilot_txt_byte_exact": True,
                    "validation_pilot_txt_byte_exact": True,
                },
                "bindings": {
                    "wrapper_path": (
                        nas / "run_v4_repro_selection_audit.sh"
                    ).resolve().as_posix(),
                    "wrapper_sha256": _sha(
                        nas / "run_v4_repro_selection_audit.sh"
                    ),
                    "selector_path": (
                        scripts / "build_v4_repro_pilot_inputs.py"
                    ).resolve().as_posix(),
                    "selector_sha256": _sha(
                        scripts / "build_v4_repro_pilot_inputs.py"
                    ),
                    "proposal_generator_path": (
                        scripts / "prepare_proposal_verifier_dataset.py"
                    ).resolve().as_posix(),
                    "proposal_generator_sha256": _sha(
                        scripts / "prepare_proposal_verifier_dataset.py"
                    ),
                    "data_path": source_data.resolve().as_posix(),
                    "data_sha256": _sha(source_data),
                    "dataset_dir": Path(cohort["root"]).joinpath(
                        "sources"
                    ).resolve().as_posix(),
                    "resolved_universe_sha256": universe,
                    "historical_manifest_path": old_manifest.resolve().as_posix(),
                    "historical_manifest_sha256": _sha(old_manifest),
                    "drift_report_path": drift_report.resolve().as_posix(),
                    "drift_report_sha256": _sha(drift_report),
                    **assembly_digest_bindings,
                    **assembly_path_bindings,
                    "quality_exclusion_manifest_path": quality_manifest.resolve().as_posix(),
                    "quality_exclusion_manifest_sha256": _sha(quality_manifest),
                    "pilot_artifacts": {
                        name: _sha(path)
                        for name, path in sorted(audit_pilot_artifacts.items())
                    },
                    "recompute_artifacts": {
                        name: _sha(path)
                        for name, path in sorted(recompute_audit_artifacts.items())
                    },
                },
                "authority": {
                    **false_authority,
                    "validator_authority": (
                        mode == "selection_audit_evidence_authority"
                    ),
                },
    }
    if mode == "selection_audit_wrapper_hash_mismatch":
        selection_audit_evidence_value["bindings"]["wrapper_sha256"] = "f" * 64
    if mode == "selection_audit_wrapper_hash_missing":
        selection_audit_evidence_value["bindings"].pop("wrapper_sha256")
    if mode == "quality_audit_evidence_membership_attestation_missing":
        selection_audit_evidence_value.pop(
            "quality_exclusion_dataset_membership_verified_by_selector_replay"
        )
    selection_audit_evidence.write_text(
        json.dumps(
            selection_audit_evidence_value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    selection_audit_marker = selection_audit / "selection_audit.sha256"
    _write_sha_marker(
        selection_audit_marker,
        [
            path
            for _, path in sorted(
                {
                    **recompute_audit_artifacts,
                    "selection_audit_evidence.json": selection_audit_evidence,
                }.items()
            )
        ],
    )
    selection_audit_ready = selection_audit / "selection_audit_ready.json"
    selection_audit_ready_value = {
        "schema_version": 1,
        "artifact_role": (
            "v4_repro_selection_audit_cpu_only_diagnostic_"
            "not_generation_training_blind_or_deployment_authority"
        ),
        "audit_contract": "v4_repro_selection_audit.cpu_only_byte_exact.v2",
        "status": "selection_audit_ready",
        "selection_contract": (
            "v4_repro_pilot_inputs."
            "gt_stratified_historical_observation_priority_blake2b.v4"
        ),
        "seed": 20260901,
        "cpu_only": True,
        "quota_per_stratum": {"training": 250, "validation": 100},
        "selected_sources": 3500,
        "quality_exclusion_dataset_membership_verified_by_selector_replay": True,
        "quality_exclusion": (
            {**ready_quality, "manifest_sha256": "f" * 64}
            if mode == "quality_audit_ready_mismatch"
            else ready_quality
        ),
        "byte_exact_artifacts": [
            "selection_inventory.json",
            "train_pilot.txt",
            "validation_pilot.txt",
        ],
        "bindings": {
            "selector_sha256": _sha(scripts / "build_v4_repro_pilot_inputs.py"),
            "selection_audit_marker_sha256": _sha(selection_audit_marker),
            "selection_audit_evidence_sha256": _sha(selection_audit_evidence),
            "resolved_universe_sha256": universe,
            **assembly_digest_bindings,
            **assembly_path_bindings,
            "quality_exclusion_manifest_path": quality_manifest.resolve().as_posix(),
            "quality_exclusion_manifest_sha256": _sha(quality_manifest),
            "pilot_artifacts": {
                name: _sha(path) for name, path in sorted(audit_pilot_artifacts.items())
            },
            "recompute_artifacts": {
                name: _sha(path)
                for name, path in sorted(recompute_audit_artifacts.items())
            },
        },
        **false_authority,
    }
    if mode == "selection_audit_selector_mismatch":
        selection_audit_ready_value["bindings"]["selector_sha256"] = "f" * 64
    if mode == "selection_audit_pilot_binding_mismatch":
        selection_audit_ready_value["bindings"]["pilot_artifacts"][
            "selection_inventory.json"
        ] = "f" * 64
    if mode == "selection_audit_authority":
        selection_audit_ready_value["validator_authority"] = True
    if mode == "quality_audit_ready_membership_attestation_missing":
        selection_audit_ready_value.pop(
            "quality_exclusion_dataset_membership_verified_by_selector_replay"
        )
    selection_audit_ready.write_text(
        json.dumps(selection_audit_ready_value, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    if mode == "selection_audit_marker_tamper":
        with selection_audit_marker.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{'f' * 64}  {(tmp_path / 'foreign-audit.txt').as_posix()}\n")
    if mode == "selection_audit_extra_root_file":
        (selection_audit / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    if mode == "selection_audit_extra_recompute_directory":
        (selection_recompute / "unexpected-directory").mkdir()
    if mode == "selection_audit_wrapper_file_missing":
        (nas / "run_v4_repro_selection_audit.sh").unlink()

    selection_counter = tmp_path / "selection-selector-count.txt"
    selection_audit_for_validation = selection_audit
    if mode == "success":
        real_selection_audit = tmp_path / "real-selection-audit"
        audit_env = {
            **os.environ,
            "AUDIT_DIR": real_selection_audit.as_posix(),
            "CODE_ROOT": code.as_posix(),
            "PILOT_INPUT_DIR": pilot.as_posix(),
            "QUALITY_EXCLUSION_MANIFEST": quality_manifest.as_posix(),
            "QUALITY_EXCLUSION_ASSEMBLY_RECEIPT": quality_receipt.as_posix(),
            "PYTHON_BIN": Path(sys.executable).as_posix(),
            "FAKE_SELECTION_RECOMPUTE_REFERENCE": selection_reference.as_posix(),
            "FAKE_SELECTION_COUNTER": selection_counter.as_posix(),
        }
        audit_result = subprocess.run(
            [
                _integration_bash(tmp_path),
                (nas / "run_v4_repro_selection_audit.sh").as_posix(),
            ],
            env=audit_env,
            text=True,
            capture_output=True,
            check=False,
        )
        if audit_result.returncode != 0:
            raise AssertionError(
                "real selection audit fixture failed:\n"
                f"stdout={audit_result.stdout}\nstderr={audit_result.stderr}"
            )
        selection_audit_for_validation = real_selection_audit

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
    manifest_lines = [
        "id,value,filepath,crop_bytes,source_path_b64,source_id,split,category"
    ]
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
        replay_category = row["stratum"]
        if (
            mode == "probe_replays_material"
            and row["selection_cohort"] == "historical_background_probe"
        ):
            replay_category = "can"
        manifest_lines.append(
            f'{index},raw,{crop_value},{len(b"sealed-pilot-crop")},{encoded},'
            f'{row["source_sha256"]},{row["split"]},{replay_category}'
        )
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    info = raw / "dataset_info.json"
    data_binding = pilot_yaml.resolve()
    if mode == "pilot_link_mismatch":
        data_binding = tmp_path / "wrong-pilot.yaml"
        data_binding.write_text("wrong: true\n", encoding="utf-8")
    batch = 2 if mode == "wrong_batch" else 1
    generation_seed = (
        20260902 if mode == "pilot_generation_seed_mismatch" else 20260901
    )
    dataset_selection_seed: object = (
        20260901.0 if mode == "float_generation_selection_seed" else generation_seed
    )
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
                    "seed": dataset_selection_seed,
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
                "seed": generation_seed,
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
        "SELECTION_AUDIT_DIR": selection_audit_for_validation.as_posix(),
        "QUALITY_EXCLUSION_MANIFEST": quality_manifest.as_posix(),
        "QUALITY_EXCLUSION_ASSEMBLY_RECEIPT": quality_receipt.as_posix(),
        "DETECTOR_MODEL": detector.as_posix(),
        "INFERENCE_SPEC": spec.as_posix(),
        "PYTHON_BIN": python_bin,
        "FAKE_SELECTION_RECOMPUTE_REFERENCE": selection_reference.as_posix(),
        "FAKE_SELECTION_COUNTER": selection_counter.as_posix(),
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
        [bash, (Path(env["CODE_ROOT"]) / "scripts" / "nas" / SCRIPT.name).as_posix()],
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
        "SELECTION_AUDIT_DIR",
        "QUALITY_EXCLUSION_MANIFEST",
        "QUALITY_EXCLUSION_ASSEMBLY_RECEIPT",
        "DETECTOR_MODEL", "INFERENCE_SPEC",
    ):
        assert name in text
    assert 'if ! mkdir "$VALIDATION_DIR"' in text
    assert "trap on_exit 0" in text
    assert "raw_generation_ready.json" in text
    assert "raw_output_inventory.json" in text
    assert "dataset_input_inventory.json" in text
    assert "selection_inventory.json" in text
    assert "selection_audit_ready.json" in text
    assert "selection_audit.sha256" in text
    assert "build_v4_repro_pilot_inputs.py" in text
    assert "prepare_proposal_verifier_dataset.py" in text
    assert "verifier_preprocessing_contract.py" in text
    assert "run_v4_reproducible_generation.sh" in text
    assert "run_v4_repro_selection_audit.sh" in text
    assert "pilot_dataset.yaml" in text
    assert "train_pilot.txt" in text
    assert "validation_pilot.txt" in text
    assert '"training": 250, "validation": 100' in text
    assert 'ready.get("selected_sources") != 3500' in text
    assert '("inference", "batch"): 1' in text
    assert "generation input marker does not bind the exact pilot dependencies" in text
    assert "selected source or label current bytes differ from inventory" in text
    assert "generation dataset input inventory differs from selected current bytes" in text
    assert "selection audit inventory differs from pilot input" in text
    assert "selection audit marker does not bind the exact seven audit artifacts" in text
    assert "selection audit evidence comparisons are invalid" in text
    assert "selection audit evidence wrapper hash mismatch" in text
    assert "selection audit root file set is not exact" in text
    assert "selection audit recompute file set is not exact" in text
    assert (
        "quality_exclusion_dataset_membership_verified_by_selector_replay"
        in text
    )
    assert "lacks exact quality exclusion dataset membership attestation" in text
    assert "pilot selection contains a quality-excluded source SHA" in text
    assert "raw manifest contains a quality-excluded source SHA" in text
    assert "quality exclusion reason is not allowlisted" in text
    assert "quality exclusion source-list canonical hash mismatch" in text
    assert "quality exclusion manifest entries must contain 1..100 rows" in text
    for reason in (
        "severe_frame_crop",
        "person_occlusion_or_dominance",
        "excessive_background_or_multi_object",
        "unreadable_boundary",
        "too_low_resolution",
        "extreme_exposure",
    ):
        assert f'"{reason}"' in text
    assert "subprocess.run" not in text
    assert "raw manifest contains a source outside the selected pilot cohort" in text
    assert "raw manifest selected-source coverage is below 99 percent" in text
    assert "raw manifest omits one or more selected drift anchors" in text
    assert "pilot background probe lacks bound historical membership" in text
    assert "raw manifest omits one or more selected background probes" in text
    assert "historical_background_probe_replay_result_assumed" in text
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


def test_rejects_quality_manifest_ancestor_symlink(
    tmp_path: Path, cohort_base: dict[str, object]
) -> None:
    env = _fixture(
        tmp_path,
        mode="quality_ancestor_symlink_preflight",
        cohort=cohort_base,
    )
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
    result = subprocess.run(
        [_integration_bash(tmp_path), SCRIPT.as_posix()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    control = Path(env["VALIDATION_DIR"]) / "control"
    assert (control / "failed.txt").is_file()
    assert not (control / "diagnostic_ready.json").exists()


def test_integration_success_seals_two_exact_runs(
    tmp_path: Path, cohort_base: dict[str, object]
) -> None:
    result, env = _run(tmp_path, cohort_base)
    assert result.returncode == 0, result.stderr
    assert Path(env["FAKE_SELECTION_COUNTER"]).read_text(encoding="ascii") == "1"
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
    assert ready["quality_exclusion_required"] is True
    assert ready["quality_excluded_sources"] == 1
    assert ready["quality_exclusion_dataset_membership_verified"] is True
    assert ready[
        "quality_exclusion_membership_attested_by_cpu_selector_audit"
    ] is True
    assert ready["quality_exclusion_absent_from_selection_and_raw"] is True
    assert ready["lineage_execution_authorized"] is False
    assert ready["training_authority"] is False
    assert ready["blind_test_authority"] is False
    assert ready["production_deployment_authorized"] is False
    selection = json.loads(
        (Path(env["PILOT_INPUT_DIR"]) / "selection_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    historical_material = next(
        row
        for row in selection["selected_sources"]
        if row["split"] == "training"
        and row["current_gt_stratum"] == "paper"
        and row["selection_reason"]
        == "historical_observation_priority_blake2"
    )
    assert historical_material["historical_categories_selection_only"] == ["can"]
    assert historical_material["selection_stratum"] == "paper"
    assert not (control / "failed.txt").exists()
    audit = Path(env["SELECTION_AUDIT_DIR"])
    assert (audit / "selection_audit.sha256").is_file()
    assert (audit / "recompute" / "selection_inventory.json").read_bytes() == (
        Path(env["PILOT_INPUT_DIR"]) / "selection_inventory.json"
    ).read_bytes()
    assert ready["bindings"]["selection_audit_ready_sha256"] == _sha(
        audit / "selection_audit_ready.json"
    )
    assert ready["bindings"]["selection_audit_marker_sha256"] == _sha(
        audit / "selection_audit.sha256"
    )
    assert ready["bindings"]["selection_audit_evidence_sha256"] == _sha(
        audit / "selection_audit_evidence.json"
    )
    assert ready["bindings"]["quality_exclusion_manifest_sha256"] == _sha(
        Path(env["QUALITY_EXCLUSION_MANIFEST"])
    )
    quality_fields = {
        "quality_exclusions_sha256": _sha(Path(env["QUALITY_EXCLUSION_MANIFEST"])),
        "quality_exclusion_manifest_sha256": _sha(Path(env["QUALITY_EXCLUSION_MANIFEST"])),
        "quality_exclusion_assembly_receipt_sha256": _sha(Path(env["QUALITY_EXCLUSION_ASSEMBLY_RECEIPT"])),
        "quality_exclusion_assembly_marker_sha256": _sha(Path(env["QUALITY_EXCLUSION_ASSEMBLY_RECEIPT"]).parent / "assembly.sha256"),
        "quality_assembly_validator_sha256": _sha(Path(env["CODE_ROOT"]) / "scripts" / "operational_quality_assembly_contract.py"),
    }
    for name in ("cohort_binding.json", "reproducibility_comparison.json", "diagnostic_ready.json"):
        payload = json.loads((control / name).read_bytes())
        assert {key: payload["bindings"][key] for key in quality_fields} == quality_fields
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


def test_integration_historical_background_probe_does_not_force_replay_category(
    tmp_path: Path, cohort_base: dict[str, object]
) -> None:
    result, env = _run(tmp_path, cohort_base, mode="probe_replays_material")
    assert result.returncode == 0, result.stderr
    ready = json.loads(
        (
            Path(env["VALIDATION_DIR"])
            / "control"
            / "diagnostic_ready.json"
        ).read_text(encoding="utf-8")
    )
    assert ready["selected_background_probes"] == 1
    assert ready["emitted_background_probes"] == 1
    assert ready["background_probe_replay_categories"] == {"training/can": 1}
    assert ready["historical_background_probe_is_selection_only"] is True
    assert ready["historical_background_probe_replay_result_assumed"] is False
    assert ready["production_deployment_authorized"] is False


def test_integration_accepts_archived_pilot_code_paths_with_exact_hashes(
    tmp_path: Path, cohort_base: dict[str, object]
) -> None:
    result, env = _run(tmp_path, cohort_base, mode="archived_code_paths")
    assert result.returncode == 0, result.stderr
    ready = json.loads(
        (
            Path(env["VALIDATION_DIR"])
            / "control"
            / "diagnostic_ready.json"
        ).read_text(encoding="utf-8")
    )
    assert ready["status"] == "batch1_validator_ab_reproducibility_passed"
    assert ready["production_deployment_authorized"] is False


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
        "forged_label_semantics",
        "forged_selection_blake_score",
        "forged_observation_priority_membership",
        "forged_observation_priority_count",
        "forged_observation_priority_tier_order",
        "forged_eligible_observation_count",
        "false_observation_source_contract",
        "negative_selection_seed",
        "boolean_ready_seed",
        "forged_positive_pilot_seed",
        "pilot_generation_seed_mismatch",
        "float_generation_selection_seed",
        "forged_background_eligible_count",
        "forged_non_top_k_selection",
        "selection_audit_marker_tamper",
        "selection_audit_selector_mismatch",
        "selection_audit_pilot_binding_mismatch",
        "selection_audit_authority",
        "selection_audit_evidence_authority",
        "selection_audit_wrapper_hash_mismatch",
        "selection_audit_wrapper_hash_missing",
        "selection_audit_wrapper_file_missing",
        "selection_audit_extra_root_file",
        "selection_audit_extra_recompute_directory",
        "quality_unknown_reason",
        "quality_reason_counts_mismatch",
        "quality_numeric_false_authority",
        "quality_over_maximum",
        "quality_audit_ready_membership_attestation_missing",
        "quality_audit_evidence_membership_attestation_missing",
        "quality_selected_source",
        "quality_manifest_path_binding_mismatch",
        "quality_manifest_hash_binding_mismatch",
        "quality_audit_evidence_mismatch",
        "quality_audit_ready_mismatch",
        "quality_receipt_stage_mutation",
        "quality_validator_stage_mutation",
        "generation_inventory_mismatch",
        "foreign_manifest_source",
        "external_crop",
        "same_size_crop_tamper",
        "low_emitted_coverage",
        "missing_anchor",
        "coverage_shortage",
        "diagnostic_lineage_ready",
        "forged_probe_membership",
        "forged_historical_row_count",
        "forged_drift_anchor",
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
    if mode in {"quality_receipt_stage_mutation", "quality_validator_stage_mutation"}:
        assert (Path(env["VALIDATION_DIR"]) / "validator-a" / "manifest.v4.validation.json").is_file()


def test_integration_failure_removes_racing_ready_marker(
    tmp_path: Path, cohort_base: dict[str, object]
) -> None:
    result, env = _run(tmp_path, cohort_base, mode="ready_race")
    assert result.returncode != 0
    control = Path(env["VALIDATION_DIR"]) / "control"
    assert (control / "failed.txt").is_file()
    assert not (control / "diagnostic_ready.json").exists()


@pytest.mark.parametrize("mutation", [
    "missing_receipt_env", "naked_manifest", "legacy_mode", "later_cutoff",
    "receipt_manifest_mismatch", "receipt_schema_bool", "receipt_scope_int",
    "receipt_authority_int", "receipt_extra_key", "extra_file", "marker",
    "validator_bytes", "pilot_receipt_hash", "pilot_quality_hash",
    "audit_receipt_hash", "audit_quality_hash", "audit_schema_bool",
    "receipt_symlink", "manifest_ancestor_symlink",
])
def test_quality_assembly_contract_rejects_tampered_chain(
    tmp_path: Path, cohort_base: dict[str, object], mutation: str
) -> None:
    env = _fixture(tmp_path, mode="assembly_adversarial", cohort=cohort_base)
    receipt = Path(env["QUALITY_EXCLUSION_ASSEMBLY_RECEIPT"])
    manifest = Path(env["QUALITY_EXCLUSION_MANIFEST"])
    marker = receipt.parent / "assembly.sha256"
    if mutation == "missing_receipt_env":
        del env["QUALITY_EXCLUSION_ASSEMBLY_RECEIPT"]
    elif mutation == "naked_manifest":
        naked = tmp_path / "generic-quality.json"
        naked.write_bytes(manifest.read_bytes())
        env["QUALITY_EXCLUSION_MANIFEST"] = naked.as_posix()
    elif mutation in {"legacy_mode", "later_cutoff", "receipt_manifest_mismatch", "receipt_schema_bool", "receipt_scope_int", "receipt_authority_int", "receipt_extra_key"}:
        value = json.loads(receipt.read_bytes())
        if mutation == "legacy_mode":
            value["assembly_mode"] = "legacy"
        elif mutation == "later_cutoff":
            value["operational_capture_cutoff_kst"] = "2026-08-02T00:00:00+09:00"
        elif mutation == "receipt_manifest_mismatch":
            value["quality_manifest_sha256"] = "a" * 64
        elif mutation == "receipt_schema_bool":
            value["schema_version"] = True
        elif mutation == "receipt_scope_int":
            value["scope"]["objective_prepare_bundle_validated"] = 1
        elif mutation == "receipt_authority_int":
            value["authority"]["selection"] = 0
        else:
            value["extra"] = False
        receipt.write_bytes(quality_authority._canonical_json(value))
        marker.write_bytes(quality_authority._quality_assembly_marker_bytes(
            manifest_content=manifest.read_bytes(), receipt_content=receipt.read_bytes()
        ))
    elif mutation == "extra_file":
        (receipt.parent / "unexpected.json").write_text("{}")
    elif mutation == "marker":
        marker.write_bytes(marker.read_bytes() + b"\n")
    elif mutation == "validator_bytes":
        path = Path(env["CODE_ROOT"]) / "scripts" / "operational_quality_assembly_contract.py"
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation.startswith("pilot_"):
        path = Path(env["PILOT_INPUT_DIR"]) / "input_ready.json"
        value = json.loads(path.read_bytes())
        key = "quality_exclusion_assembly_receipt_sha256" if mutation == "pilot_receipt_hash" else "quality_exclusions_sha256"
        value["bindings"][key] = "b" * 64
        path.write_bytes(quality_authority._canonical_json(value))
    elif mutation.startswith("audit_"):
        path = Path(env["SELECTION_AUDIT_DIR"]) / "selection_audit_ready.json"
        value = json.loads(path.read_bytes())
        if mutation == "audit_schema_bool":
            value["schema_version"] = True
        else:
            key = "quality_exclusion_assembly_receipt_sha256" if mutation == "audit_receipt_hash" else "quality_exclusions_sha256"
            value["bindings"][key] = "b" * 64
        path.write_bytes(quality_authority._canonical_json(value))
    elif mutation == "receipt_symlink":
        target = tmp_path / "external-quality" / "external-receipt.json"
        target.parent.mkdir()
        target.write_bytes(receipt.read_bytes())
        receipt.unlink()
        try:
            receipt.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation is unavailable")
    else:
        linked = tmp_path / "quality-alias"
        try:
            linked.symlink_to(receipt.parent, target_is_directory=True)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        env["QUALITY_EXCLUSION_MANIFEST"] = (linked / manifest.name).as_posix()
    result = subprocess.run(
        [_integration_bash(tmp_path), (Path(env["CODE_ROOT"]) / "scripts" / "nas" / SCRIPT.name).as_posix()],
        env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0, result.stdout
    if mutation == "receipt_symlink":
        assert "QUALITY_EXCLUSION_ASSEMBLY_RECEIPT must be an absolute regular file" in result.stderr
    control = Path(env["VALIDATION_DIR"]) / "control"
    assert not (control / "diagnostic_ready.json").exists()
    if mutation != "missing_receipt_env":
        assert (control / "failed.txt").is_file()


@pytest.mark.parametrize("mutation", ["receipt", "marker", "validator", "extra_file"])
def test_quality_assembly_terminal_toctou_removes_ready(
    tmp_path: Path, cohort_base: dict[str, object], mutation: str
) -> None:
    env = _fixture(tmp_path, mode="assembly_terminal_toctou", cohort=cohort_base)
    receipt = Path(env["QUALITY_EXCLUSION_ASSEMBLY_RECEIPT"])
    target = {
        "receipt": receipt,
        "marker": receipt.parent / "assembly.sha256",
        "validator": Path(env["CODE_ROOT"]) / "scripts" / "operational_quality_assembly_contract.py",
        "extra_file": receipt.parent / "unexpected.json",
    }[mutation]
    launcher = tmp_path / "terminal-tamper.sh"
    launcher.write_text(
        '#!/usr/bin/env bash\nset -eu\n'
        'if [ "${1-}" = "-" ] && [ "${2-}" = "$TERMINAL_READY" ]; then\n'
        '  "$REAL_PYTHON" "$@"\n'
        '  test -f "$TERMINAL_READY"\n'
        '  printf "\\n" >> "$TERMINAL_TARGET"\n'
        '  printf "mutated-after-ready\\n" > "$TERMINAL_SENTINEL"\n'
        '  exit 0\n'
        'fi\nexec "$REAL_PYTHON" "$@"\n',
        encoding="utf-8", newline="\n",
    )
    launcher.chmod(0o755)
    env.update({
        "PYTHON_BIN": launcher.as_posix(),
        "REAL_PYTHON": Path(sys.executable).as_posix(),
        "TERMINAL_READY": (Path(env["VALIDATION_DIR"]) / "control" / "diagnostic_ready.json").as_posix(),
        "TERMINAL_TARGET": target.as_posix(),
        "TERMINAL_SENTINEL": (tmp_path / "terminal-mutation-reached.txt").as_posix(),
    })
    result = subprocess.run(
        [_integration_bash(tmp_path), (Path(env["CODE_ROOT"]) / "scripts" / "nas" / SCRIPT.name).as_posix()],
        env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0, result.stdout
    assert Path(env["TERMINAL_SENTINEL"]).is_file(), result.stderr
    control = Path(env["VALIDATION_DIR"]) / "control"
    assert (control / "failed.txt").is_file()
    assert not (control / "diagnostic_ready.json").exists()


def test_validation_output_cannot_contaminate_quality_bundle(
    tmp_path: Path, cohort_base: dict[str, object]
) -> None:
    env = _fixture(tmp_path, mode="assembly_nested_output", cohort=cohort_base)
    root = Path(env["QUALITY_EXCLUSION_ASSEMBLY_RECEIPT"]).parent
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    env["VALIDATION_DIR"] = (root / "nested-validation").as_posix()
    result = subprocess.run(
        [_integration_bash(tmp_path), (Path(env["CODE_ROOT"]) / "scripts" / "nas" / SCRIPT.name).as_posix()],
        env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 64
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
