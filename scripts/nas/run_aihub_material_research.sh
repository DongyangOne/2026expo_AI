#!/bin/sh
# Run only inside a reviewed, pinned, network-none research container.
# Inputs/code/TORCH_HOME are read-only; RUN is writable, with disposable /tmp.
# Not a candidate-gate bypass, resume command, state model, or deployment tool.
# Required plain env: CODE_ROOT, RUN, RAW_SOURCE_ROOT, TORCH_HOME, AIHUB_ORIGIN.
# Every input in INPUTS below requires ENV and ENV_SHA256. Every CODE entry
# requires KEY_SHA256. WRAPPER_SHA256 pins this file. No hashes are invented.
set -eu
exec python3 - "$0" <<'BOOT'
import hashlib, os, pathlib, subprocess, sys, tempfile
wrapper = pathlib.Path(sys.argv[1]).absolute()
raw = wrapper.read_bytes()
if hashlib.sha256(raw).hexdigest() != os.environ["WRAPPER_SHA256"]:
    raise SystemExit("wrapper SHA256 mismatch")
body = raw.split(b"\n: <<'PY_WORKER'\n", 1)[1].rsplit(b"\nPY_WORKER", 1)[0]
with tempfile.TemporaryDirectory(prefix="aihub-research-") as tmp:
    worker = pathlib.Path(tmp) / "worker.py"
    with worker.open("xb") as stream: stream.write(body)
    result = subprocess.run([sys.executable, str(worker), str(wrapper)], check=False)
    raise SystemExit(result.returncode)
BOOT
: <<'PY_WORKER'
import contextlib, csv, hashlib, importlib, json, os, re, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

INPUTS = "TRAIN_MANIFEST VALIDATION_MANIFEST LINEAGE_MANIFEST ORIGINAL_REPORT COHORT MATERIALIZER_REPORT GENERATION_INFO REPLAY_REPORT LINEAGE_REPORT PROTECTED_AUDIT PROTECTED_INVENTORY DETECTOR_MODEL INFERENCE_SPEC PRETRAINED LICENSE_EVIDENCE".split()
CODE = dict(TRAINER="train_multitask_verifier.py", LEGACY_TRAINER="train_verifier.py", READER="audited_aihub_snapshot.py", MATERIALIZER="materialize_audited_aihub_sources.py", ORIGINAL_AUDITOR="audit_aihub_original_annotations.py", DATASET_AUDITOR="audit_verifier_dataset.py", NEAR_AUDITOR="audit_v4_near_duplicate_leakage.py", LINEAGE_UPGRADER="upgrade_proposal_manifest_lineage.py", REFERENCE_READER="assemble_research_protected_references.py", RESEARCH_AUDITOR="audit_research_reference_leakage.py")
FALSE_AUTHORITY = dict(research_only=True, formal_protected_coverage=False, training_authorized=False, promotion_authorized=False, deployment_authorized=False, blind_test_authorized=False, blind_authorized=False, final_state_candidate=False)
pins, run, owned = {}, None, None

def require(ok, message):
    if not ok: raise ValueError(message)

def path(value):
    p = Path(value)
    require(p.is_absolute() and ".." not in p.parts, "absolute traversal-free paths required")
    require(not any(x.is_symlink() for x in (p, *p.parents)), "symlink path forbidden")
    return p

def digest(p):
    p = path(str(p)); before = p.stat()
    require(p.is_file(), "regular file required")
    h = hashlib.sha256()
    with p.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
        closed = os.fstat(stream.fileno())
    after = path(str(p)).stat()
    # Windows stat/fstat disagree on ctime semantics; compare like with like.
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) + (() if os.name == "nt" else (s.st_ctime_ns,))
    require(identity(before) == identity(opened) == identity(closed) == identity(after)
            and before.st_ctime_ns == after.st_ctime_ns and opened.st_ctime_ns == closed.st_ctime_ns, "input changed while hashing")
    return h.hexdigest()

def pin(p, expected):
    require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected), "explicit lowercase SHA256 required")
    require(digest(p) == expected, "input/code SHA256 mismatch")
    require(p not in pins or pins[p] == expected, "conflicting input pins")
    require(run is None or not (run.is_relative_to(p.parent) or p.is_relative_to(run)), "RUN overlaps a bound input tree")
    pins[p] = expected
    return p

def read(p): return json.loads(p.read_text(encoding="utf-8"))
def stamp(): return datetime.now(timezone.utc).isoformat()
def mark(name, values):
    path(str(run))
    require((run.stat().st_dev, run.stat().st_ino) == owned, "run directory ownership changed")
    with (run / name).open("x", encoding="utf-8") as stream:
        json.dump(dict(time_utc=stamp(), **values, **FALSE_AUTHORITY), stream, sort_keys=True); stream.write("\n")

def rows(p):
    with p.open(encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))

def row_key(row, parent):
    value = dict(row); raw = path(str((parent / row["filepath"]) if not Path(row["filepath"]).is_absolute() else row["filepath"]))
    value["filepath"] = raw.as_posix()
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def exact(left, right):
    return type(left) is type(right) and (left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
        if isinstance(left, dict) else len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
        if isinstance(left, list) else left == right)

def run_research():
    global run, owned
    try:
        root, run = path(os.environ["CODE_ROOT"]), path(os.environ["RUN"])
        raw_root, cache = path(os.environ["RAW_SOURCE_ROOT"]), path(os.environ["TORCH_HOME"])
        files = {key: path(os.environ[key]) for key in INPUTS}
        require(not run.exists(), "fresh RUN required; resume/overwrite unsupported")
        require(all(not run.is_relative_to(p) for p in [root, raw_root, cache, *(p.parent for p in files.values())]), "RUN overlaps an input tree")
        pin(path(str(Path(sys.argv[1]).absolute())), os.environ["WRAPPER_SHA256"])
        require(not any((p.parent / "failed.json").exists() for p in files.values()), "an input has a failure marker")
        for key, name in CODE.items(): pin(root / "scripts" / name, os.environ[key + "_SHA256"])
        require(not (root / "scripts/__init__.py").exists(), "expected frozen namespace package layout")
        require(files["PRETRAINED"] == cache / "hub/checkpoints/mobilenet_v3_small-047dcff4.pth", "pretrained file must occupy the actual TORCH_HOME cache path")
        sys.path.insert(0, str(root))
        trainer = importlib.import_module("scripts.train_multitask_verifier")
        require(Path(trainer.__file__) == root / "scripts" / CODE["TRAINER"], "trainer import escaped frozen code root")
        guard = trainer.eager_initialize_cuda_context()  # Same process through hashes, dry-run and all epochs.
        for key, p in files.items(): pin(p, os.environ[key + "_SHA256"])
        require(pins[files["PRETRAINED"]] == "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f", "wrong pretrained backbone")
        near = importlib.import_module("scripts.audit_v4_near_duplicate_leakage")
        dataset_auditor = importlib.import_module("scripts.audit_verifier_dataset")
        reader = importlib.import_module("scripts.audited_aihub_snapshot")
        upgrader = importlib.import_module("scripts.upgrade_proposal_manifest_lineage")
        reference_reader = importlib.import_module("scripts.assemble_research_protected_references")
        research_auditor = importlib.import_module("scripts.audit_research_reference_leakage")
        for name in CODE.values(): require(Path(importlib.import_module("scripts." + name[:-3]).__file__) == root / "scripts" / name, "import escaped frozen code root")
        data = {key: read(files[key]) for key in ("ORIGINAL_REPORT", "COHORT", "GENERATION_INFO", "REPLAY_REPORT", "LINEAGE_REPORT", "PROTECTED_AUDIT")}
        original, cohort, info, replay, lineage, separation = (data[k] for k in data)
        require(original.get("schema") == "aihub_original_annotation_audit_v1" and original.get("selected") == len(original["records"]) == sum(original["manifest_counts"].values()), "complete original audit required")
        require(cohort.get("schema") == "aihub_original_cohort_v1" and cohort.get("full_cohort") is True and cohort.get("training_authorized") is False, "full non-authorizing cohort required")
        require("unresolved_legacy_source_links" not in cohort.get("pending_checks", []), "unresolved legacy source links")
        require(cohort.get("legacy_exclusion_evidence", {}).get("unresolved_legacy_references") == 0, "completed actual legacy exclusion evidence required")
        for binding in cohort["metadata_bindings"]: pin(path(binding["path"]), binding["sha256"])
        require(any(Path(b["path"]) == files["ORIGINAL_REPORT"] and b["sha256"] == pins[files["ORIGINAL_REPORT"]] for b in cohort["metadata_bindings"]), "cohort does not bind original audit")
        snapshot = reader.load_audited_aihub_snapshot(files["MATERIALIZER_REPORT"], pins[files["MATERIALIZER_REPORT"]], cohort_path=files["COHORT"], require_full_cohort=True)
        require(info.get("audited_aihub_snapshot") == snapshot.binding(), "generation snapshot binding mismatch")
        require(replay.get("artifact_role") == "v4_development_candidates_not_blind_or_deployment_authority" and replay.get("ready_for_lineage_upgrade") is True, "non-diagnostic actual runtime replay required")
        require(replay["bindings"]["dataset_info_sha256"] == pins[files["GENERATION_INFO"]] and replay["bindings"]["inference_spec_sha256"] == pins[files["INFERENCE_SPEC"]], "replay input binding mismatch")
        require(replay["bindings"]["detector_model_sha256"] == pins[files["DETECTOR_MODEL"]], "replay detector model mismatch")
        require(lineage.get("builder") == "scripts/upgrade_proposal_manifest_lineage.py" and lineage.get("dry_run") is False and lineage.get("blind_test_eligible") is False, "completed strict lineage required")
        require(lineage["outputs"]["csv"] == dict(path=files["LINEAGE_MANIFEST"].as_posix(), sha256=pins[files["LINEAGE_MANIFEST"]]), "lineage output mismatch")
        require(len(lineage["inputs"]) == len(lineage["validator_reports"]) == 1, "one frozen AIHub replay input required")
        require(lineage["validator_reports"][0]["sha256"] == pins[files["REPLAY_REPORT"]] and lineage["inputs"][0]["sha256"] == replay["bindings"]["validated_manifest_sha256"], "lineage/replay mismatch")
        pin(path(lineage["inputs"][0]["path"]), lineage["inputs"][0]["sha256"])
        upgrader._load_validator_report(files["REPLAY_REPORT"], expected_report_sha256=pins[files["REPLAY_REPORT"]], validated_manifest_sha256=lineage["inputs"][0]["sha256"], validated_manifest_rows=len(rows(Path(lineage["inputs"][0]["path"]))))
        require(lineage["near_phash_quarantine"]["enabled"] is True and lineage["near_phash_quarantine"]["distance"] == 4, "existing pHash quarantine required; do not invent physical groups")
        combined = rows(files["LINEAGE_MANIFEST"]); partitions = {role: rows(files[key]) for role, key in (("train", "TRAIN_MANIFEST"), ("model_validation", "VALIDATION_MANIFEST"))}
        require(Counter(row_key(r, files["LINEAGE_MANIFEST"].parent) for r in combined) == Counter(row_key(r, files[key].parent) for role, key in (("train", "TRAIN_MANIFEST"), ("model_validation", "VALIDATION_MANIFEST")) for r in partitions[role]), "role partition changed rows beyond equivalent crop paths")
        for role, items in partitions.items():
            require(items and all(r["role"] == r["fold"] == role and r["origin"] == os.environ["AIHUB_ORIGIN"] for r in items), "non-AIHub or incorrect role/fold")
            for row in items:
                source = path(row["source_filepath"])
                require(all(row.get(k) == v for k, v in snapshot.metadata_for(source).items()) and snapshot.split_for(source) == row["split"], "original source linkage mismatch")
                require(all(row.get(k) == "-1" for k in ("dent", "label", "foreign_material")), "research must not invent condition labels")
        manifest_shas = {role: pins[files[key]] for role, key in (("train", "TRAIN_MANIFEST"), ("model_validation", "VALIDATION_MANIFEST"))}
        require(separation.get("schema") == research_auditor.SCHEMA and separation.get("status") == "passed"
                and separation.get("ok") is True and all(separation.get(k) is v for k, v in research_auditor.AUTHORITY.items()), "passed non-authorizing research v2 audit required")
        ready_path = files["PROTECTED_AUDIT"].parent / "research_audit_ready.json"
        require(not (ready_path.parent / "failed.json").exists() and not (ready_path.parent / "blocked.json").exists(), "research audit failed or blocked")
        pin(ready_path, digest(ready_path))
        ready = read(ready_path)
        require(exact(ready, dict(schema=research_auditor.SCHEMA, status="passed", report_sha256=pins[files["PROTECTED_AUDIT"]], **research_auditor.AUTHORITY)), "research ready/report binding mismatch")
        references = reference_reader.load_reference_inventory(files["PROTECTED_INVENTORY"], pins[files["PROTECTED_INVENTORY"]],
            expected_model_sha256=pins[files["DETECTOR_MODEL"]], expected_spec_sha256=pins[files["INFERENCE_SPEC"]])
        require(all(not run.is_relative_to(p.parent) and not p.is_relative_to(run) for p in references.input_paths), "RUN overlaps protected reference evidence")
        protected = references.records
        expected_files = {role: dict(path=files[key].as_posix(), sha256=pins[files[key]]) for role, key in
            (("train", "TRAIN_MANIFEST"), ("model_validation", "VALIDATION_MANIFEST"), ("lineage_manifest", "LINEAGE_MANIFEST"),
             ("lineage_report", "LINEAGE_REPORT"), ("replay_report", "REPLAY_REPORT"), ("reference_inventory", "PROTECTED_INVENTORY"),
             ("detector_model", "DETECTOR_MODEL"), ("inference_spec", "INFERENCE_SPEC"))}
        expected_code = {name: pins[root / "scripts" / name] for name in research_auditor.CODE_FILES}
        sb = separation["bindings"]
        require(exact(sb["candidate_manifest_sha256"], manifest_shas) and exact(sb["files"], expected_files)
                and exact(sb["code_sha256"], expected_code) and exact(sb["reference_inventory"], references.binding()), "research audit inputs/code/reference binding mismatch")
        require(exact(separation["algorithm"], dict(id=near.ALGORITHM_ID, threshold=4, shared_auditor=near._auditor_binding()))
                and type(separation["summary"].get("blocking_multi_role_clusters")) is int
                and separation["summary"]["blocking_multi_role_clusters"] == 0, "research audit algorithm or result mismatch")
        memberships = {r["source_sha256"]: r["roles"] for r in references.inventory["sources"]}
        candidate_assets = []
        for role, key in (("train", "TRAIN_MANIFEST"), ("model_validation", "VALIDATION_MANIFEST")):
            assets, actual_sha = near._load_candidate_manifest(role, files[key], max_bytes=research_auditor.MAX_METADATA_BYTES, allow_absolute_crop_paths=True)
            require(actual_sha == manifest_shas[role], "candidate manifest changed")
            candidate_assets.extend(assets)
        candidate_counts = Counter((a.role, a.source_sha256, a.view_kind) for a in candidate_assets)
        require(all(candidate_counts[(role, source_sha, kind)] == 1 for role, source_sha in
            {(a.role, a.source_sha256) for a in candidate_assets} for kind in near.VIEW_KINDS), "candidate source/crop coverage mismatch")
        for asset in candidate_assets: pin(path(str(asset.path)), asset.image_sha256)
        require(near._payload_set_sha(candidate_assets) == sb["candidate_payload_set_sha256"]
                and near._payload_set_sha(protected) == sb["protected_payload_set_sha256"], "actual candidate/protected payload set mismatch")
        expected_coverage = dict(candidate_assets=len(candidate_assets), protected_assets=len(protected), protected_sources=len(memberships),
            protected_crops=sum(r.view_kind == "crop" for r in protected), candidate_rows={r: len(v) for r, v in partitions.items()},
            supplied_reference_inventory_complete=True, formal_protected_coverage=False)
        require(exact(separation["coverage"], expected_coverage) and exact(separation["reference_memberships"], memberships), "research audit coverage/membership mismatch")
        run.mkdir(parents=True, exist_ok=False); owned = (run.stat().st_dev, run.stat().st_ino)
        manifests = [files["TRAIN_MANIFEST"], files["VALIDATION_MANIFEST"]]
        audit = dataset_auditor.audit_manifests(manifests, require_masked_status=True, require_single_object=True, require_source_references=True, phash_distance=4, fail_on_near_phash=True)
        require(audit["ok"] is True, "strict current-image/class/role audit failed")
        mark("dataset_audit.json", audit)
        def recheck():
            for p, expected in pins.items():
                require(not (p.parent / "failed.json").exists() and digest(p) == expected, "frozen input/code failed or changed")
            require(not (ready_path.parent / "blocked.json").exists(), "research audit became blocked")
            snapshot.recheck(); references.recheck()
            for p, expected in pins.items(): require(digest(p) == expected, "frozen input/code changed during evidence recheck")
        recheck()
        args = ["--manifest", str(manifests[0]), "--manifest", str(manifests[1]), "--backbone", "mobilenet_v3_small", "--size", "320", "--epochs", "100", "--patience", "15", "--batch", "64", "--workers", "2", "--lr", "0.0003", "--backbone-lr", "0.00003", "--head-lr", "0.0003", "--label-smoothing", "0.05", "--class-weight-mode", "effective-number", "--class-weight-beta", "0.999", "--objectness-weight", "1", "--material-weight", "2", "--no-condition-heads", "--seed", "20260827", "--device", "cuda"]
        with (run / "dry_run.json").open("x", encoding="utf-8") as log, contextlib.redirect_stdout(log): require(trainer.main(args + ["--dry-run"]) == 0, "trainer dry-run failed")
        require(read(run / "dry_run.json")["condition_heads"] == [], "unexpected condition heads")
        recheck(); mark("running.json", dict(scope="AIHub material/objectness research only", inputs={k: pins[p] for k, p in files.items()},
            license_evidence=dict(path=files["LICENSE_EVIDENCE"].as_posix(), sha256=pins[files["LICENSE_EVIDENCE"]], license_rights_independently_determined=False),
            protected_reference_binding=references.binding(), pending_checks=cohort.get("pending_checks", [])))
        data.clear(); del original, cohort, info, replay, lineage, separation, combined, partitions
        with (run / "train.log").open("x", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log): require(trainer.main(args + ["--output-dir", str(run / "artifacts")]) == 0, "trainer failed")
        recheck()
        artifacts = run / "artifacts"; metadata = read(artifacts / trainer.METADATA_NAME)
        require(metadata["model_config"]["condition_heads"] == [] and metadata["candidate_only"] is True and metadata["production_runtime_modified"] is False, "unexpected training outputs")
        outputs = {name: digest(artifacts / name) for name in (trainer.CHECKPOINT_NAME, trainer.METADATA_NAME, trainer.ONNX_NAME)}
        mark("research_complete.json", dict(scope="training completed, not accuracy improvement or deployment approval", outputs=outputs))
        recheck()  # A later failed marker dominates any just-published completion.
    except BaseException as error:
        if owned is not None:
            try: mark("failed.json", dict(stage="research_one_shot", exception_type=type(error).__name__))
            except Exception: pass
        print(json.dumps(dict(status="failed", exception_type=type(error).__name__, **FALSE_AUTHORITY)), file=sys.stderr)
        raise

if __name__ == "__main__":
    run_research()
PY_WORKER
