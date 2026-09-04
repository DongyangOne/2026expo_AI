"""Bridge launch contract; no Docker, NAS, or live-process actions are performed."""
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/nas/continue_original_audit_to_cohort_20260904.sh"


def create_args():
    source = SCRIPT.read_text(encoding="utf-8")
    command = source.split('if ! bounded_command 60 "$DOCKER" create ', 1)[1].split("; then", 1)[0]
    return shlex.split(command.replace("\\\n", " "))


def values(args, flag):
    return [args[i + 1] for i, value in enumerate(args) if value == flag]


def test_host_owned_private_control_gets_only_required_dac_capability():
    args = create_args()
    assert values(args, "--cap-drop") == ["ALL"]
    assert values(args, "--cap-add") == ["DAC_OVERRIDE"]
    assert values(args, "--security-opt") == ["no-new-privileges"]
    assert "--privileged" not in args


def test_permission_fix_preserves_input_mount_and_resource_isolation():
    args = create_args()
    assert values(args, "-v") == ["$ROOT:/app:ro", "$CONTROL:$app_control:rw"]
    assert "--read-only" in args
    assert values(args, "--network") == ["none"]
    assert values(args, "--runtime") == ["runc"]
    assert values(args, "--cpus") == ["2"]
    assert values(args, "--memory") == ["3g"]
    assert values(args, "--memory-swap") == ["3g"]
    assert "--rm" not in args


def test_producer_is_waited_by_id_and_never_stopped_or_restarted():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"$DOCKER" wait "$producer_id"' in source
    assert 'inspect_state "$producer_id"' in source
    for operation in ("stop", "restart", "kill", "rm"):
        assert f'"$DOCKER" {operation} ' not in source
    assert 'kill -TERM "$bound_pid"' in source
    assert 'kill -KILL "$bound_pid"' in source


def test_posix_shell_syntax():
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    shell = str(git_bash) if git_bash.is_file() else shutil.which("sh")
    if shell is None:
        pytest.skip("POSIX shell is unavailable")
    subprocess.run([shell, "-n", str(SCRIPT)], check=True, capture_output=True, timeout=10)
