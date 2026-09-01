from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "nas"
    / "run_v4_independent_judge_gate.sh"
)


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _integration_bash(tmp_path: Path) -> str:
    candidates = [shutil.which("bash")]
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if git_bash.is_file():
        candidates.insert(0, str(git_bash))
    for bash in dict.fromkeys(item for item in candidates if item):
        probe = subprocess.run(
            [bash, "-c", 'test -d "$1"', "bash", tmp_path.as_posix()],
            check=False,
        )
        if probe.returncode == 0:
            return bash
    pytest.skip("no bash can access the pytest temp path")


def _write_fake(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _fake_judge_fixture(tmp_path: Path, *, validator_fails: bool = False) -> dict[str, str]:
    code = tmp_path / "code"
    scripts = code / "scripts"
    validator_body = (
        "raise SystemExit(23)\n"
        if validator_fails
        else """
import argparse
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('input_manifest','dataset_info','detector_model','inference_spec','output_manifest','output_report'):
    p.add_argument('--'+n.replace('_','-'), required=True)
a=p.parse_args()
Path(a.output_manifest).write_text('validated\\n', encoding='utf-8')
Path(a.output_report).write_text('{\"ready_for_lineage_upgrade\":true}\\n', encoding='utf-8')
"""
    )
    _write_fake(scripts / "validate_v4_background_candidates.py", validator_body)
    _write_fake(scripts / "prepare_proposal_verifier_dataset.py", "# dependency\n")
    _write_fake(scripts / "verifier_preprocessing_contract.py", "# dependency\n")
    _write_fake(
        scripts / "upgrade_proposal_manifest_lineage.py",
        """
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('input','validator_report','validator_report_sha256','output_csv','output_jsonl','lineage_json','rejections_json','quarantine_validation_near_phash_distance','origin'):
    p.add_argument('--'+n.replace('_','-'), required=True)
a=p.parse_args(); fields=['filepath','role','split','material','category','crop_object_count']
row={'filepath':'crop.jpg','role':'model_validation','split':'validation','material':'9','category':'background','crop_object_count':'0'}
with Path(a.output_csv).open('x',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator='\\n'); w.writeheader(); w.writerow(row)
Path(a.output_jsonl).write_text(json.dumps(row)+'\\n',encoding='utf-8')
Path(a.lineage_json).write_text('{}\\n',encoding='utf-8')
Path(a.rejections_json).write_text('{\"rejections\":[]}\\n',encoding='utf-8')
""",
    )
    _write_fake(
        scripts / "replay_v4_candidate_metrics.py",
        """
import argparse
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('manifest','verifier_onnx','verifier_metadata','inference_spec','output_jsonl','output_attestation'):
    p.add_argument('--'+n.replace('_','-'), required=True)
a=p.parse_args(); Path(a.output_jsonl).write_text('{}\\n',encoding='utf-8'); Path(a.output_attestation).write_text('{}\\n',encoding='utf-8')
""",
    )
    _write_fake(
        scripts / "run_independent_visual_judges.py",
        """
import argparse
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--input-manifest'); p.add_argument('--judge-spec',action='append'); p.add_argument('--output-jsonl'); p.add_argument('--output-report')
a=p.parse_args(); Path(a.output_jsonl).write_text('{}\\n',encoding='utf-8'); Path(a.output_report).write_text('{}\\n',encoding='utf-8')
""",
    )
    _write_fake(
        scripts / "evaluate_v4_candidate_judge.py",
        "APPROVED_TRUSTED_POLICY_SHA256 = 'UNCONFIGURED'\nraise RuntimeError('final gate must not run')\n",
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    values: dict[str, str] = {
        "RUN_DIR": (tmp_path / "run").as_posix(),
        "CODE_ROOT": code.as_posix(),
        "RAW_MANIFEST": (artifacts / "manifest.csv").as_posix(),
        "DATASET_INFO": (artifacts / "dataset_info.json").as_posix(),
        "DETECTOR_MODEL": (artifacts / "detector.pt").as_posix(),
        "INFERENCE_SPEC": (artifacts / "spec.json").as_posix(),
        "VALIDATED_MANIFEST": (artifacts / "validated.csv").as_posix(),
        "VALIDATOR_REPORT": (artifacts / "validation.json").as_posix(),
        "CANDIDATE_ONNX": (artifacts / "candidate.onnx").as_posix(),
        "CANDIDATE_METADATA": (artifacts / "candidate.json").as_posix(),
        "BASELINE_ONNX": (artifacts / "baseline.onnx").as_posix(),
        "BASELINE_METADATA": (artifacts / "baseline.json").as_posix(),
        "JUDGE_SPEC_1": (artifacts / "judge1.json").as_posix(),
        "JUDGE_SPEC_2": (artifacts / "judge2.json").as_posix(),
        "PYTHON_BIN": Path(sys.executable).as_posix(),
    }
    for name, value in values.items():
        if name in {"RUN_DIR", "CODE_ROOT", "VALIDATED_MANIFEST", "VALIDATOR_REPORT", "PYTHON_BIN"}:
            continue
        Path(value).write_text(f"{name}\n", encoding="utf-8")
    return values


def test_shell_syntax_when_bash_is_available() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    subprocess.run([bash, "-n"], input=_text().encode("utf-8"), check=True)


def test_pipeline_order_and_stage_hash_checkpoints() -> None:
    text = _text()
    commands = (
        "validate_v4_background_candidates.py",
        "upgrade_proposal_manifest_lineage.py",
        "replay_v4_candidate_metrics.py",
        "run_independent_visual_judges.py",
        "evaluate_v4_candidate_judge.py",
    )
    positions = [text.index(command, text.index("# The validator requires")) for command in commands]
    assert positions == sorted(positions)

    markers = (
        "00_inputs.sha256",
        "01_validator.sha256",
        "02_lineage.sha256",
        "03_replays.sha256",
        "04_visual.sha256",
        "05_evidence_prepared.sha256",
        "06_offline_gate.sha256",
    )
    assert [text.index(marker) for marker in markers] == sorted(
        text.index(marker) for marker in markers
    )
    assert "sha256sum -c" in text
    assert 'ln "$temporary" "$marker"' in text
    assert text.count('verify_marker "$CONTROL/00_inputs.sha256"') >= 7


def test_input_marker_binds_validator_and_replay_python_dependencies() -> None:
    text = _text()
    marker = text[
        text.index('write_marker "$CONTROL/00_inputs.sha256"') :
        text.index('verify_marker "$CONTROL/00_inputs.sha256"')
    ]
    for dependency in (
        "prepare_proposal_verifier_dataset.py",
        "verifier_preprocessing_contract.py",
    ):
        assert dependency in marker
        assert text.index(dependency) < text.index(
            'write_marker "$CONTROL/00_inputs.sha256"'
        )


def test_run_directory_and_artifacts_are_exclusive() -> None:
    text = _text()
    assert "eval " not in text
    assert 'printenv "$1"' in text
    assert 'if ! mkdir "$RUN_DIR"' in text
    assert "refusing to reuse immutable RUN_DIR" in text
    assert "refusing to overwrite validator outputs" in text
    assert 'target.exists() or target.is_symlink()' in text
    assert 'temporary.open("x"' in text
    assert "os.link(temporary, target)" in text


def test_unconfigured_policy_stops_after_evidence_without_final_gate() -> None:
    text = _text()
    branch = text[text.index('if [ "$policy_pin" = "UNCONFIGURED" ]') :]
    exit_position = branch.index("exit 78")
    final_gate_position = branch.index(
        '"$PYTHON_BIN" "$CODE_ROOT/scripts/evaluate_v4_candidate_judge.py"'
    )
    assert exit_position < final_gate_position
    assert "awaiting_policy_pin.sha256" in branch[:exit_position]
    assert "FINAL_READY" not in branch[:exit_position]
    assert "terminal_state=1" in branch[:exit_position]


def test_wrapper_has_no_service_or_deployment_operations() -> None:
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
        "production model",
        "pi model",
    )
    for token in forbidden:
        assert token not in text


def test_visual_projection_is_exact_strict_validation_background() -> None:
    text = _text()
    for condition in (
        'row.get("role") == "model_validation"',
        'row.get("split") == "validation"',
        'row.get("material") == "9"',
        'row.get("category") == "background"',
        'row.get("crop_object_count") == "0"',
    ):
        assert condition in text
    assert "strict manifest has no validation background rows" in text


def test_final_gate_uses_both_replays_visual_pair_and_frozen_policy() -> None:
    text = _text()
    required = (
        '--replay-predictions "$CANDIDATE_PREDICTIONS"',
        '--replay-attestation "$CANDIDATE_ATTESTATION"',
        '--baseline-replay-predictions "$BASELINE_PREDICTIONS"',
        '--baseline-replay-attestation "$BASELINE_ATTESTATION"',
        '--trusted-policy "$TRUSTED_POLICY"',
        '--visual-judge-report "$VISUAL_REPORT"',
        '--visual-judge-evidence "$VISUAL_EVIDENCE"',
    )
    for fragment in required:
        assert fragment in text
    assert "trusted policy bytes differ from code pin" in text


def test_integration_unconfigured_prepares_hash_bound_evidence_only(tmp_path: Path) -> None:
    bash = _integration_bash(tmp_path)
    env = _fake_judge_fixture(tmp_path)
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 78, result.stderr
    control = Path(env["RUN_DIR"]) / "control"
    assert (control / "awaiting_policy_pin.sha256").is_file()
    assert not (control / "failed.txt").exists()
    assert not (control / "offline_gate_ready.json").exists()
    assert not (Path(env["RUN_DIR"]) / "final" / "v4_candidate_judge_ready.txt").exists()
    for marker in (
        "01_validator.sha256",
        "02_lineage.sha256",
        "03_replays.sha256",
        "04_visual.sha256",
        "05_evidence_prepared.sha256",
        "awaiting_policy_pin.sha256",
    ):
        checked = subprocess.run(
            [bash, "-c", 'sha256sum -c "$1" >/dev/null', "bash", (control / marker).as_posix()],
            check=False,
        )
        assert checked.returncode == 0, marker


def test_integration_validator_failure_is_fail_closed(tmp_path: Path) -> None:
    bash = _integration_bash(tmp_path)
    env = _fake_judge_fixture(tmp_path, validator_fails=True)
    result = subprocess.run(
        [bash, SCRIPT.as_posix()], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    control = Path(env["RUN_DIR"]) / "control"
    assert (control / "failed.txt").is_file()
    assert not (control / "awaiting_policy_pin.sha256").exists()
    assert not (control / "offline_gate_ready.json").exists()
    assert not (Path(env["RUN_DIR"]) / "final" / "v4_candidate_judge_ready.txt").exists()


def test_final_wrapper_ready_is_last_and_binds_complete_evidence_chain() -> None:
    text = _text()
    assert "mktemp \"$CONTROL/.failed.XXXXXX\"" in text
    assert "mktemp \"$CONTROL/.marker.XXXXXX\"" in text
    assert "trap on_exit 0" in text
    assert 'mv "$EVALUATOR_READY" "$SEALED_EVALUATOR_READY"' in text
    chain = text[text.index('write_marker "$CONTROL/06_offline_gate.sha256"') :]
    for marker in (
        "00_inputs.sha256",
        "01_validator.sha256",
        "02_lineage.sha256",
        "03_replays.sha256",
        "04_visual.sha256",
        "05_evidence_prepared.sha256",
    ):
        assert marker in chain
    assert "offline_gate_ready.json" in chain
    assert "production_deployment_authorized\": False" in chain
    assert "requires_independent_blind_hardware_evidence\": True" in chain
    publication = chain.index('os.link(temporary, ready)')
    assert chain.index("terminal_state=1", publication) > publication
    assert chain.index("exit 0", publication) > publication
