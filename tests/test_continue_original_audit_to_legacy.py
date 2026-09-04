"""One-shot legacy bridge checks with a fake Docker CLI; no NAS or GPU actions."""
from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/nas/continue_original_audit_to_legacy_20260904.sh"


def shell(tmp_path):
    choices = [Path("C:/Program Files/Git/bin/bash.exe"), shutil.which("bash"), shutil.which("sh")]
    for candidate in choices:
        if candidate and Path(candidate).is_file():
            if subprocess.run([str(candidate), "-c", 'test -d "$1"', "sh", tmp_path.as_posix()],
                              check=False, capture_output=True).returncode == 0:
                return str(candidate)
    pytest.skip("no POSIX shell can access the temporary directory")


def values(args, flag):
    return [args[i + 1] for i, arg in enumerate(args) if arg == flag]


def bridge(tmp_path):
    """Replace only host paths and content pins; run the actual shell branches."""
    source = SCRIPT.read_text(encoding="utf-8")
    root = tmp_path / "Container"
    job = root / "operational_refresh_80bf78a_20260904_101000"
    code = job / "legacy_link_code_20260904"
    control = job / "legacy_full_continuation_v1_20260904"
    out = root / "legacy_aihub_link_full_v1_20260904"
    job.mkdir(parents=True)
    producer = re.search(r"^PRODUCER=(.+)$", source, re.M).group(1)
    image = re.search(r"^IMAGE=(.+)$", source, re.M).group(1)
    pinned_files = []
    for symbolic, digest in re.findall(r'pinned "([^"\n]+)" ([0-9a-f]{64})', source):
        target = Path(symbolic.replace("$CODE", code.as_posix()).replace("$JOB", job.as_posix())
                      .replace("$ROOT", root.as_posix()))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((target.name + " fixture\n").encode())
        source = source.replace(digest, hashlib.sha256(target.read_bytes()).hexdigest())
        pinned_files.append(target)
    assert len(pinned_files) == 5
    original = job / "original_annotation_full_v2_20260904/result/report.json"
    original.parent.mkdir(parents=True)
    original.write_text('{"fixture":true}\n', encoding="utf-8")
    fake = tmp_path / "docker-fake.sh"
    fake.write_text("""#!/bin/sh
set -eu
printf '%s\\t' "$@" >> "$CALL_LOG"
printf '\\n' >> "$CALL_LOG"
case "$1" in
  inspect)
    if [ ! -f "$INSPECT_SEEN" ]; then
      : > "$INSPECT_SEEN"
      printf '%s\\n' "$BEFORE"
    else
      printf '%s\\n' "$AFTER"
    fi ;;
  wait)
    if [ -n "${MUTATE_DURING_WAIT:-}" ]; then printf changed >> "$MUTATE_DURING_WAIT"; fi
    if [ -n "${CREATE_OUTPUT_DURING_WAIT:-}" ]; then mkdir "$CREATE_OUTPUT_DURING_WAIT"; fi
    if [ "${WAIT_FAIL:-0}" = 1 ]; then exit 9; fi
    printf '0\\n' ;;
  run) printf '%s\\n' "$CHILD_ID" ;;
  *) exit 93 ;;
esac
""", encoding="utf-8", newline="\n")
    fake.chmod(0o700)
    source = re.sub(r"^ROOT=.+$", "ROOT=" + shlex.quote(root.as_posix()), source, flags=re.M)
    source = re.sub(r"^DOCKER=.+$", "DOCKER=" + shlex.quote(fake.as_posix()), source, flags=re.M)
    script = tmp_path / "bridge.sh"
    script.write_text(source, encoding="utf-8", newline="\n")
    log = tmp_path / "calls.tsv"
    env = dict(os.environ, CALL_LOG=log.as_posix(), INSPECT_SEEN=(tmp_path / "inspected").as_posix(),
               BEFORE=f"{producer} running 0 false {image}", AFTER=f"{producer} exited 0 false {image}",
               CHILD_ID="a" * 64)
    return {"script": script, "shell": shell(tmp_path), "env": env, "log": log,
            "control": control, "out": out, "root": root, "job": job,
            "pinned_files": pinned_files, "producer": producer, "image": image, "original": original}


def invoke(case):
    return subprocess.run([case["shell"], str(case["script"])], env=case["env"],
                          capture_output=True, text=True, encoding="utf-8", timeout=15, check=False)


def calls(case):
    if not case["log"].exists():
        return []
    return [line.rstrip("\t").split("\t") for line in case["log"].read_text(encoding="utf-8").splitlines()]


def not_dispatched(case):
    assert not any(args[0] == "run" for args in calls(case))
    assert not (case["control"] / "dispatched.txt").exists()


def test_posix_shell_syntax(tmp_path):
    subprocess.run([shell(tmp_path), "-n", str(SCRIPT)], check=True, capture_output=True, timeout=10)


def test_launch_is_cpu_only_read_only_and_full_legacy_scope(tmp_path):
    case = bridge(tmp_path)
    assert invoke(case).returncode == 0
    args = next(args for args in calls(case) if args[0] == "run")
    for flag, expected in (("--runtime", "runc"), ("--network", "none"), ("--cpus", "1"),
                           ("--memory", "2g"), ("--memory-swap", "2g"), ("--pids-limit", "128"),
                           ("--cap-drop", "ALL"), ("--cap-add", "DAC_OVERRIDE"),
                           ("--security-opt", "no-new-privileges"), ("--max-per-kind", "0")):
        assert values(args, flag) == [expected]
    assert values(args, "-v") == [case["root"].as_posix() + ":/app:ro",
                                   case["out"].as_posix() + ":/app/legacy_aihub_link_full_v1_20260904:rw"]
    assert "--read-only" in args and "-d" in args
    assert not {"--gpus", "--device", "--privileged", "--rm"}.intersection(args)
    assert not any("docker.sock" in arg or "nvidia" in arg.casefold() for arg in args)
    assert values(args, "--entrypoint") == ["python3"]
    assert case["image"] in args


@pytest.mark.parametrize("initial", ["running", "exited"])
def test_exact_producer_once_and_dispatch_is_not_completion(tmp_path, initial):
    case = bridge(tmp_path)
    case["env"]["BEFORE"] = f'{case["producer"]} {initial} 0 false {case["image"]}'
    assert invoke(case).returncode == 0
    observed = calls(case)
    assert [a[0] for a in observed] == (["inspect", "wait", "inspect", "run"] if initial == "running"
                                       else ["inspect", "inspect", "run"])
    assert all(a[-1] == case["producer"] for a in observed if a[0] in ("inspect", "wait"))
    assert (case["control"] / "audit_container_id.txt").read_text().strip() == "a" * 64
    assert "dispatched" in (case["control"] / "dispatched.txt").read_text()
    assert not (case["out"] / "result/report.json").exists()
    assert not list(case["control"].glob("*ready*"))
    assert not list(case["control"].glob("*complete*"))
    before = case["log"].read_bytes()
    assert invoke(case).returncode != 0
    assert case["log"].read_bytes() == before  # Exclusive control directory prevents a second dispatch.


@pytest.mark.parametrize("field,bad", [(0, "b" * 64), (1, "paused"), (4, "sha256:" + "0" * 64)])
def test_wrong_initial_producer_identity_image_or_state_refused(tmp_path, field, bad):
    case = bridge(tmp_path)
    state = case["env"]["BEFORE"].split()
    state[field] = bad
    case["env"]["BEFORE"] = " ".join(state)
    assert invoke(case).returncode != 0
    not_dispatched(case)
    assert (case["control"] / "failed.txt").exists()
    assert all(a[0] == "inspect" for a in calls(case))


@pytest.mark.parametrize("field,bad", [(0, "b" * 64), (1, "running"), (2, "1"),
                                     (3, "true"), (4, "sha256:" + "0" * 64)])
def test_terminal_identity_exited_zero_non_oom_and_image_are_all_required(tmp_path, field, bad):
    case = bridge(tmp_path)
    state = case["env"]["AFTER"].split()
    state[field] = bad
    case["env"]["AFTER"] = " ".join(state)
    assert invoke(case).returncode != 0
    not_dispatched(case)
    assert (case["control"] / "failed.txt").exists()


@pytest.mark.parametrize("when", ["before", "during_wait"])
def test_pinned_input_mutation_is_refused_before_dispatch(tmp_path, when):
    case = bridge(tmp_path)
    path = case["pinned_files"][0]
    if when == "before":
        path.write_bytes(b"changed")
    else:
        case["env"]["MUTATE_DURING_WAIT"] = path.as_posix()
    assert invoke(case).returncode != 0
    not_dispatched(case)
    assert (case["control"] / "failed.txt").exists()
    assert not case["out"].exists()


@pytest.mark.parametrize("when", ["before", "during_wait"])
def test_output_reuse_is_refused_even_if_it_appears_while_waiting(tmp_path, when):
    case = bridge(tmp_path)
    if when == "before":
        case["out"].mkdir()
    else:
        case["env"]["CREATE_OUTPUT_DURING_WAIT"] = case["out"].as_posix()
    assert invoke(case).returncode != 0
    not_dispatched(case)
    assert case["out"].is_dir() and not list(case["out"].iterdir())


@pytest.mark.parametrize("fault", ["wait_transport", "missing_report", "failed_report"])
def test_unconfirmed_or_failed_original_work_never_dispatches(tmp_path, fault):
    case = bridge(tmp_path)
    if fault == "wait_transport":
        case["env"]["WAIT_FAIL"] = "1"
    elif fault == "missing_report":
        case["original"].unlink()
    else:
        (case["original"].parent / "failed.json").write_text("{}")
    assert invoke(case).returncode != 0
    not_dispatched(case)
    marker = "observation_error.txt" if fault == "wait_transport" else "failed.txt"
    assert (case["control"] / marker).exists()
    assert not any(a[0] in {"stop", "restart", "rm", "kill"} for a in calls(case))
