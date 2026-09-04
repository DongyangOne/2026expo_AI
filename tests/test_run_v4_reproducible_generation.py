from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "nas"
    / "run_v4_reproducible_generation.sh"
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


def _generation_fixture(
    tmp_path: Path, *, fail: bool = False, audited: bool = False,
    binding_fault: str | None = None,
) -> dict[str, str]:
    code = tmp_path / "code"
    scripts = code / "scripts"
    scripts.mkdir(parents=True)
    nas = scripts / "nas"
    nas.mkdir()
    (nas / SCRIPT.name).write_bytes(SCRIPT.read_bytes())
    generator = scripts / "prepare_proposal_verifier_dataset.py"
    generator.write_text(
        """
import argparse,csv,hashlib,json
from pathlib import Path
def _label_path(source):
    parts=list(Path(source).parts); index=parts.index('images'); parts[index]='labels'
    return Path(*parts).with_suffix('.txt')
def resolve_split_images(data_path, dataset_dir):
    root=Path(dataset_dir)
    return {
        'training': sorted((root/'images'/'train').glob('*.jpg')),
        'validation': sorted((root/'images'/'val').glob('*.jpg')),
    }
def main():
    %s
    p=argparse.ArgumentParser()
    for n in ('model','data','dataset_dir','output_dir','device','batch','imgsz','conf','nms_iou','positive_iou','negative_iou','crop_size','padding','jpeg_quality','proposal_selection','background_policy','background_gt_margin','max_per_class','val_max_per_class','max_background','val_max_background','seed','min_free_gb','max_output_gb'):
        p.add_argument('--'+n.replace('_','-'), required=True)
    for n in ('audited_aihub_report','audited_aihub_report_sha256','audited_aihub_cohort','aihub_origin'):
        p.add_argument('--'+n.replace('_','-'))
    p.add_argument('--audited-aihub-diagnostic',action='store_true')
    a=p.parse_args(); out=Path(a.output_dir); out.mkdir(); manifest=out/'manifest.csv'
    (out/'generator_args.json').write_text(json.dumps(vars(a)),encoding='utf-8')
    with manifest.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['filepath']); w.writeheader(); w.writerow({'filepath':'crop.jpg'})
    (out/'crop.jpg').write_bytes(b'crop')
    info={'model':a.model,'data':a.data,'dataset_dir':a.dataset_dir,'manifest':str(manifest),'written_crops':1,
    'inference':{'batch':1,'imgsz':640,'conf':.10,'nms_iou':.70},
    'assignment':{'positive_iou_inclusive':.50,'negative_iou_inclusive':.10,'ambiguous_iou_skipped':True},
    'proposal_policy':{'selection_mode':'runtime-top1','background_policy':'strict-zero-intersection','background_gt_margin':.10},
    'crop':{'size':320,'padding':.08,'jpeg_quality':92},
    'selection':{'max_per_class':10000,'val_max_per_class':2000,'max_background':10000,'val_max_background':2000,'seed':int(a.seed)},
    'storage_guards':{'min_free_gb':300.0,'max_output_gb':30.0}}
    if a.audited_aihub_report:
        info['audited_aihub_snapshot']={
            'report_path':Path(a.audited_aihub_report).resolve().as_posix(),
            'report_sha256':a.audited_aihub_report_sha256,
            'cohort_path':Path(a.audited_aihub_cohort).resolve().as_posix(),
            'cohort_sha256':hashlib.sha256(Path(a.audited_aihub_cohort).read_bytes()).hexdigest(),
            'require_full_cohort':not a.audited_aihub_diagnostic,
        }
    fault=%r
    if fault == 'missing':
        info.pop('audited_aihub_snapshot',None)
    elif fault == 'report_sha':
        info['audited_aihub_snapshot']['report_sha256']='0'*64
    elif fault == 'cohort_sha':
        info['audited_aihub_snapshot']['cohort_sha256']='0'*64
    elif fault == 'integer_full':
        info['audited_aihub_snapshot']['require_full_cohort']=1
    elif fault == 'unexpected':
        info['audited_aihub_snapshot']={'unexpected':True}
    (out/'dataset_info.json').write_text(json.dumps(info)+'\\n',encoding='utf-8')
if __name__ == '__main__':
    main()
""" % ("raise SystemExit(29)" if fail else "pass", binding_fault)
        ,
        encoding="utf-8",
    )
    (scripts / "verifier_preprocessing_contract.py").write_text("CONTRACT=1\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        path = dataset / "images" / split / "sample.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(split.encode())
        label = dataset / "labels" / split / "sample.txt"
        label.parent.mkdir(parents=True)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    model = tmp_path / "model.pt"; model.write_bytes(b"model")
    data = tmp_path / "data.yaml"; data.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    env = {
        "GEN_DIR": (tmp_path / "generation").as_posix(),
        "CODE_ROOT": code.as_posix(),
        "MODEL_PATH": model.as_posix(),
        "DATA_PATH": data.as_posix(),
        "DATASET_DIR": dataset.as_posix(),
        "PYTHON_BIN": Path(sys.executable).as_posix(),
    }
    if audited:
        # Deliberate doubles: this fixture tests shell/CLI/hash plumbing, not the
        # audited snapshot loader, image inference or training eligibility.
        for name in ("audited_aihub_snapshot.py", "audit_aihub_original_annotations.py",
                     "materialize_audited_aihub_sources.py"):
            (scripts / name).write_text("CONTRACT=1\n", encoding="utf-8")
        report = tmp_path / "audited_report.json"
        report.write_text('{"snapshot_only":true}\n', encoding="utf-8")
        cohort = tmp_path / "audited_cohort.json"
        cohort.write_text('{"records":[]}\n', encoding="utf-8")
        env.update({
            "AUDITED_AIHUB_REPORT": report.as_posix(),
            "AUDITED_AIHUB_REPORT_SHA256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "AUDITED_AIHUB_COHORT": cohort.as_posix(),
        })
    return env


def test_shell_syntax_when_bash_is_available() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    subprocess.run([bash, "-n"], input=_text().encode("utf-8"), check=True)


def test_generation_uses_frozen_batch_one_contract() -> None:
    text = _text()
    required = (
        "--batch 1",
        "--imgsz 640",
        "--conf 0.10",
        "--nms-iou 0.70",
        "--positive-iou 0.50",
        "--negative-iou 0.10",
        "--crop-size 320",
        "--padding 0.08",
        "--jpeg-quality 92",
        "--proposal-selection runtime-top1",
        "--background-policy strict-zero-intersection",
        "--background-gt-margin 0.10",
        "--max-per-class 10000",
        "--val-max-per-class 2000",
        "--max-background 10000",
        "--val-max-background 2000",
        "--min-free-gb 300",
        "--max-output-gb 30",
    )
    for fragment in required:
        assert fragment in text
    assert "SEED=${SEED:-20260901}" in text
    assert "--batch 12" not in text


def test_generation_directory_and_markers_are_exclusive_and_hash_bound() -> None:
    text = _text()
    assert "eval " not in text
    assert 'printenv "$1"' in text
    assert 'if ! mkdir "$GEN_DIR"' in text
    assert "refusing to reuse immutable GEN_DIR" in text
    assert "trap on_exit 0" in text
    assert 'mktemp "$CONTROL/.failed.XXXXXX"' in text
    assert 'mktemp "$CONTROL/.inputs.XXXXXX"' in text
    assert 'mktemp "$CONTROL/.outputs.XXXXXX"' in text
    assert 'ln "$temporary" "$INPUT_MARKER"' in text
    assert 'ln "$temporary" "$OUTPUT_MARKER"' in text
    assert '"$WRAPPER"' in text
    assert "sha256sum -c" in text
    assert 'inventory_generation_sources "$DATASET_INPUT_INVENTORY"' in text
    assert 'inventory_generation_sources "$DATASET_INPUT_INVENTORY_END"' in text
    assert 'cmp -s "$DATASET_INPUT_INVENTORY" "$DATASET_INPUT_INVENTORY_END"' in text
    assert text.count('verify_inventory "$RAW_DIR" "$OUTPUT_INVENTORY"') == 2
    assert "inventory path escapes root" in text


def test_input_inventory_covers_resolved_external_splits_and_label_sidecars() -> None:
    text = _text()
    inventory = text[text.index("inventory_generation_sources()") : text.index("DATASET_INPUT_INVENTORY=")]
    assert "resolve_split_images(data_path, dataset_dir)" in inventory
    assert "_label_path(source)" in inventory
    assert 'for split in ("training", "validation")' in inventory
    assert '"contract": "resolved_yolo_train_val_sources_and_label_sidecars_sha256.v1"' in inventory
    assert '"kind": "unresolved_label_path"' in inventory
    assert "before.st_mtime_ns" in inventory


def test_dataset_info_is_independently_checked_before_ready() -> None:
    text = _text()
    checks = (
        'Path(info.get(field, "")).resolve() != expected',
        '("inference", "batch"): 1',
        '("proposal_policy", "selection_mode"): "runtime-top1"',
        '("proposal_policy", "background_policy"): "strict-zero-intersection"',
        '("proposal_policy", "background_gt_margin"): 0.10',
        'written <= 0',
        'rows <= 0 or rows != written',
    )
    for fragment in checks:
        assert fragment in text
    assert text.index("generated dataset contract verification failed") < text.index(
        "raw_generation_ready.json"
    )


def test_failure_cannot_create_ready_marker() -> None:
    text = _text()
    ready_assignment = text.index("READY=$CONTROL/raw_generation_ready.json")
    generation_failure = text.index('fail "raw proposal generation failed"')
    contract_failure = text.index('fail "generated dataset contract verification failed"')
    input_rehash_failure = text.index('fail "model, data, or code changed during generation"')
    output_rehash_failure = text.index('fail "generated outputs changed before ready marker"')
    assert generation_failure < ready_assignment
    assert contract_failure < ready_assignment
    assert input_rehash_failure < ready_assignment
    assert output_rehash_failure < ready_assignment
    assert "validator_authority\": False" in text
    assert "judge_authority\": False" in text
    assert "training_authority\": False" in text
    assert "blind_test_authority\": False" in text
    assert "production_deployment_authorized\": False" in text


def test_wrapper_has_no_external_control_or_credentials() -> None:
    text = _text().lower()
    forbidden = (
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
    )
    for token in forbidden:
        assert token not in text


def test_only_raw_generation_ready_is_published() -> None:
    text = _text()
    assert "raw_generation_ready.json" in text
    assert "candidate_ready" not in text
    assert "offline_gate" not in text
    assert "train_multitask_verifier.py" not in text
    assert "evaluate_v4_candidate_judge.py" not in text


def test_integration_fake_generation_publishes_raw_ready(tmp_path: Path) -> None:
    bash = _integration_bash(tmp_path)
    env = _generation_fixture(tmp_path)
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    control = Path(env["GEN_DIR"]) / "control"
    assert (control / "raw_generation_ready.json").is_file()
    assert not (control / "failed.txt").exists()
    received = json.loads((control.parent / "raw" / "generator_args.json").read_text())
    assert received["audited_aihub_report"] is None
    assert received["audited_aihub_cohort"] is None
    assert received["aihub_origin"] is None
    assert received["audited_aihub_diagnostic"] is False
    for marker in ("inputs.sha256", "outputs.sha256"):
        checked = subprocess.run(
            [bash, "-c", 'sha256sum -c "$1" >/dev/null', "bash", (control / marker).as_posix()],
            check=False,
        )
        assert checked.returncode == 0, marker


def test_integration_fake_generator_failure_leaves_only_failure(tmp_path: Path) -> None:
    bash = _integration_bash(tmp_path)
    env = _generation_fixture(tmp_path, fail=True)
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    control = Path(env["GEN_DIR"]) / "control"
    assert (control / "failed.txt").is_file()
    assert not (control / "raw_generation_ready.json").exists()
    text = _text()
    publication = text.index("os.link(temporary, ready)", text.index("READY=$CONTROL/raw_generation_ready.json"))
    assert text.index("terminal_state=1", publication) > publication
    assert text.index("exit 0", publication) > publication


@pytest.mark.parametrize("diagnostic", [False, True])
def test_mock_audited_generation_pins_five_extra_inputs_and_passes_canonical_origin(
    tmp_path: Path, diagnostic: bool,
) -> None:
    bash = _integration_bash(tmp_path)
    env = _generation_fixture(tmp_path, audited=True)
    env["AUDITED_AIHUB_DIAGNOSTIC"] = "1" if diagnostic else "0"
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    root = Path(env["GEN_DIR"])
    control = root / "control"
    assert (control / "raw_generation_ready.json").is_file()
    assert not (control / "failed.txt").exists()
    received = json.loads((root / "raw" / "generator_args.json").read_text())
    assert received["aihub_origin"] == "aihub_original_annotation_v1"
    assert received["audited_aihub_diagnostic"] is diagnostic
    assert received["audited_aihub_report"] == env["AUDITED_AIHUB_REPORT"]
    assert received["audited_aihub_report_sha256"] == env["AUDITED_AIHUB_REPORT_SHA256"]
    assert received["audited_aihub_cohort"] == env["AUDITED_AIHUB_COHORT"]
    info = json.loads((root / "raw" / "dataset_info.json").read_text())
    binding = info["audited_aihub_snapshot"]
    assert binding == {
        "report_path": Path(env["AUDITED_AIHUB_REPORT"]).resolve().as_posix(),
        "report_sha256": env["AUDITED_AIHUB_REPORT_SHA256"],
        "cohort_path": Path(env["AUDITED_AIHUB_COHORT"]).resolve().as_posix(),
        "cohort_sha256": hashlib.sha256(Path(env["AUDITED_AIHUB_COHORT"]).read_bytes()).hexdigest(),
        "require_full_cohort": not diagnostic,
    }
    input_lines = (control / "inputs.sha256").read_text().splitlines()
    assert len(input_lines) == 11  # six existing pins plus the five audited inputs
    expected_paths = [Path(env["AUDITED_AIHUB_REPORT"]), Path(env["AUDITED_AIHUB_COHORT"])]
    expected_paths += [Path(env["CODE_ROOT"]) / "scripts" / name for name in (
        "audited_aihub_snapshot.py", "audit_aihub_original_annotations.py",
        "materialize_audited_aihub_sources.py",
    )]
    for path in expected_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert any(line.startswith(digest + " ") and line.endswith(path.as_posix()) for line in input_lines)
    verified = subprocess.run(
        [bash, "-c", 'sha256sum -c "$1" >/dev/null', "bash", (control / "inputs.sha256").as_posix()],
        check=False,
    )
    assert verified.returncode == 0
    ready = json.loads((control / "raw_generation_ready.json").read_text())
    for authority in ("validator_authority", "judge_authority", "training_authority",
                      "blind_test_authority", "production_deployment_authorized"):
        assert ready[authority] is False


@pytest.mark.parametrize("missing", [
    "AUDITED_AIHUB_REPORT", "AUDITED_AIHUB_REPORT_SHA256", "AUDITED_AIHUB_COHORT",
])
def test_mock_partial_audited_inputs_fail_before_generation(tmp_path: Path, missing: str) -> None:
    bash = _integration_bash(tmp_path)
    env = _generation_fixture(tmp_path, audited=True)
    del env[missing]
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    control = Path(env["GEN_DIR"]) / "control"
    assert "must be supplied together" in (control / "failed.txt").read_text()
    assert not (control / "raw_generation_ready.json").exists()
    assert not (control.parent / "raw").exists()


def test_mock_wrong_audited_report_sha_fails_before_generation(tmp_path: Path) -> None:
    bash = _integration_bash(tmp_path)
    env = _generation_fixture(tmp_path, audited=True)
    env["AUDITED_AIHUB_REPORT_SHA256"] = "0" * 64
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    control = Path(env["GEN_DIR"]) / "control"
    assert "report SHA mismatch" in (control / "failed.txt").read_text()
    assert not (control / "raw_generation_ready.json").exists()
    assert not (control.parent / "raw").exists()


def test_mock_diagnostic_flag_without_audited_inputs_is_not_accepted(tmp_path: Path) -> None:
    bash = _integration_bash(tmp_path)
    env = _generation_fixture(tmp_path)
    env["AUDITED_AIHUB_DIAGNOSTIC"] = "1"
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    control = Path(env["GEN_DIR"]) / "control"
    assert "diagnostic requires report, SHA, and cohort" in (control / "failed.txt").read_text()
    assert not (control / "raw_generation_ready.json").exists()
    assert not (control.parent / "raw").exists()


@pytest.mark.parametrize("fault", ["missing", "report_sha", "cohort_sha", "integer_full"])
def test_mock_generated_audited_binding_must_match_launch_inputs(tmp_path: Path, fault: str) -> None:
    bash = _integration_bash(tmp_path)
    env = _generation_fixture(tmp_path, audited=True, binding_fault=fault)
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    control = Path(env["GEN_DIR"]) / "control"
    assert (control.parent / "raw" / "generator_args.json").exists()
    assert "generated dataset contract verification failed" in (control / "failed.txt").read_text()
    assert not (control / "raw_generation_ready.json").exists()


def test_mock_unrequested_audited_binding_is_rejected(tmp_path: Path) -> None:
    bash = _integration_bash(tmp_path)
    env = _generation_fixture(tmp_path, binding_fault="unexpected")
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    control = Path(env["GEN_DIR"]) / "control"
    assert "generated dataset contract verification failed" in (control / "failed.txt").read_text()
    assert not (control / "raw_generation_ready.json").exists()
