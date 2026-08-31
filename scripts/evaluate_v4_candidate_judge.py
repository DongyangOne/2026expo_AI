"""Fail-closed, hash-bound gate for an offline v4 verifier candidate.

This command deliberately does not tune any threshold.  It requires immutable
candidate and baseline replay bundles produced by actual ONNX inference, then
independently recomputes their metrics from sample-level logits.  It also checks
the training metadata, the exact strict manifests named by that metadata, and
blind-to-label visual-judge evidence for every ``model_validation`` background
crop.  The visual judges have veto-only
authority: they may quarantine a suspicious background crop, but they can never
change ground truth, repair a candidate prediction, or relax a metric gate.

A frozen policy bundled at a fixed repository-relative path and pinned by a
code-reviewed SHA-256 pins the approved baseline, strict manifests and lineage,
visual prompt/runner, and every visual judge model/server identity.  The visual
inputs are the canonical report and evidence JSONL written by
``run_independent_visual_judges.py``.  The gate recomputes their
report/evidence/vote/raw-response hashes and exact coverage; only unanimous
``background`` passes.  Truth, prediction, confidence, and logit fields
anywhere in a canonical raw response fail closed.

The report and pass marker are opened with exclusive-create semantics.  A run
can never overwrite earlier evidence.  A pass marker is an *offline judge gate*
only; it is explicitly not a production or hardware deployment approval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_v4_candidate_metrics import replay_validation


SCHEMA_VERSION = 1
TOOL_NAME = "scripts/evaluate_v4_candidate_judge.py"
REPORT_NAME = "v4_candidate_judge_report.json"
READY_MARKER_NAME = "v4_candidate_judge_ready.txt"
SHA256_LENGTH = 64
TRUSTED_POLICY_SCHEMA = "v4_candidate_judge_trusted_policy.v1"
TRUSTED_POLICY_RELATIVE_PATH = Path(
    "configs/v4_candidate_judge_trusted_policy.json"
)
APPROVED_TRUSTED_POLICY_SHA256 = "UNCONFIGURED"
TRUST_ROOT_METHOD = "git_bundled_code_sha256_pin"
UNCONFIGURED_TRUST_ROOT = "UNCONFIGURED"
VISUAL_REPORT_SCHEMA = "independent_visual_judge_report.v1"
VISUAL_EVIDENCE_SCHEMA = "independent_visual_judge_evidence.v1"
VISUAL_VOTE_SCHEMA = "independent_visual_judge_vote.v1"
VISUAL_CANONICAL_JSON_CONTRACT = (
    "utf8_sorted_keys_compact_separators_trailing_newline.v1"
)
VISUAL_EVIDENCE_PAIR_CONTRACT = (
    "votes_share_pair_id_and_report_pins_exact_jsonl_sha256.v1"
)
VISUAL_SERVER_DIGEST_CONTRACT = (
    "ollama_api_tags_digest_equals_sha256_of_local_oci_tag_manifest_bytes.v1"
)

BACKGROUND_MATERIAL_ID = 9
BACKGROUND_CLASS_NAME = "background"
EXPECTED_MATERIAL_CLASSES = (
    "can",
    "pet",
    "paper",
    "plastic",
    "styrofoam",
    "vinyl",
    "glass",
    "battery",
    "fluorescent",
)
TRAIN_ROLE = "train"
VALIDATION_ROLE = "model_validation"
KNOWN_ROLES = {TRAIN_ROLE, VALIDATION_ROLE, "calibration", "blind_test"}
ROLE_TO_SPLIT = {TRAIN_ROLE: "training", VALIDATION_ROLE: "validation"}

LINEAGE_FIELDS = (
    "sample_id",
    "source_sha256",
    "image_sha256",
    "object_group",
    "capture_session",
)
STRICT_FIELDS = (
    "filepath",
    "split",
    "source_id",
    "material",
    "category",
    "dent",
    "label",
    "foreign_material",
    "source_object_count",
    "sample_id",
    "source_sha256",
    "image_sha256",
    "object_group",
    "capture_session",
    "role",
    "fold",
    "origin",
)
FORBIDDEN_VISUAL_FIELDS = {
    "candidate_prediction",
    "candidate_predictions",
    "candidate_logits",
    "candidate_confidence",
    "truth",
    "ground_truth",
    "expected_truth",
    "category",
    "material",
    "label",
}
FORBIDDEN_RAW_RESPONSE_KEYS = {
    "truth",
    "ground_truth",
    "expected_truth",
    "candidate_prediction",
    "candidate_predictions",
    "prediction",
    "predictions",
    "candidate_confidence",
    "confidence",
    "candidate_logits",
    "logits",
}
FORBIDDEN_RAW_KEY_TOKENS = {
    "request",
    "requests",
    "input",
    "inputs",
    "prompt",
    "messagehistory",
    "messages",
    "image",
    "images",
    "base64",
    "truth",
    "groundtruth",
    "prediction",
    "candidateprediction",
    "confidence",
    "candidateconfidence",
    "sourcepath",
    "sourcepathb64",
    "croppath",
}
FORBIDDEN_RAW_VALUE_TOKENS = (
    "groundtruth",
    "truth",
    "candidateprediction",
    "prediction",
    "candidateconfidence",
    "confidence",
    "sourcepathb64",
)


@dataclass(frozen=True)
class GateThresholds:
    """Frozen default gates; CLI overrides are recorded, never learned."""

    min_background_support: int = 200
    min_background_recall: float = 0.90
    min_material_objectness_recall: float = 0.95
    min_objectness_balanced_accuracy: float = 0.925
    min_material_balanced_accuracy: float = 0.90
    min_each_material_recall: float = 0.85
    max_baseline_recall_drop: float = 0.01

    def validate(self) -> None:
        if isinstance(self.min_background_support, bool) or self.min_background_support < 1:
            raise ValueError("min_background_support must be a positive integer")
        for name, value in asdict(self).items():
            if name == "min_background_support":
                continue
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        frozen = type(self)()
        minimum_fields = (
            "min_background_support",
            "min_background_recall",
            "min_material_objectness_recall",
            "min_objectness_balanced_accuracy",
            "min_material_balanced_accuracy",
            "min_each_material_recall",
        )
        for name in minimum_fields:
            if getattr(self, name) < getattr(frozen, name):
                raise ValueError(
                    f"{name} cannot weaken the frozen default "
                    f"{getattr(frozen, name)}"
                )
        if self.max_baseline_recall_drop > frozen.max_baseline_recall_drop:
            raise ValueError(
                "max_baseline_recall_drop cannot weaken the frozen default "
                f"{frozen.max_baseline_recall_drop}"
            )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return (
        len(text) == SHA256_LENGTH
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    )


def _canonical_json(value: object, *, pretty: bool = False) -> bytes:
    separators = None if pretty else (",", ":")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=separators,
        )
        + "\n"
    ).encode("utf-8")


def _artifact(path: Path, *, kind: str) -> dict[str, object]:
    resolved = path.resolve(strict=False)
    if not path.is_file():
        return {
            "kind": kind,
            "path": str(resolved),
            "exists": False,
            "sha256": None,
            "bytes": None,
        }
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "exists": True,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _snapshot_artifact(
    path: Path, *, kind: str, destination: Path
) -> tuple[dict[str, object], Path]:
    """Read one input once and materialize the exact bytes used by the gate."""

    resolved = path.resolve(strict=False)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return (
            {
                "kind": kind,
                "path": str(resolved),
                "exists": False,
                "sha256": None,
                "bytes": None,
            },
            destination,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return (
        {
            "kind": kind,
            "path": str(resolved),
            "exists": True,
            "sha256": _sha256_bytes(raw),
            "bytes": len(raw),
        },
        destination,
    )


def _snapshot_validation_images(
    original_manifests: Sequence[Path],
    snapshot_manifests: Sequence[Path],
    *,
    snapshot_root: Path,
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    """Freeze only model-validation crop bytes used by runtime inference."""

    snapshots: dict[str, Path] = {}
    originals: dict[str, dict[str, object]] = {}
    cached: dict[str, tuple[str, Path]] = {}
    image_root = snapshot_root / "validation-images"
    for manifest_index, (original_manifest, snapshot_manifest) in enumerate(
        zip(original_manifests, snapshot_manifests, strict=True)
    ):
        if not snapshot_manifest.is_file():
            continue
        raw = snapshot_manifest.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"{original_manifest}: strict manifest must be UTF-8"
            ) from error
        reader = csv.DictReader(io.StringIO(text, newline=""))
        for row_index, row in enumerate(reader, start=2):
            if str(row.get("role", "")).strip() != VALIDATION_ROLE:
                continue
            filepath = str(row.get("filepath", "")).strip()
            declared_sha = str(row.get("image_sha256", "")).strip().casefold()
            if not filepath or not _is_sha256(declared_sha):
                raise ValueError(
                    f"{original_manifest}:{row_index}: invalid validation image identity"
                )
            relative = Path(filepath)
            original_image = (
                relative
                if relative.is_absolute()
                else original_manifest.parent / relative
            ).resolve()
            snapshot_lookup = (
                relative
                if relative.is_absolute()
                else snapshot_manifest.parent / relative
            ).resolve()
            original_key = str(original_image)
            cached_value = cached.get(original_key)
            if cached_value is None:
                try:
                    image_bytes = original_image.read_bytes()
                except FileNotFoundError as error:
                    raise ValueError(
                        f"{original_manifest}:{row_index}: validation image is missing"
                    ) from error
                actual_sha = _sha256_bytes(image_bytes)
                if actual_sha != declared_sha:
                    raise ValueError(
                        f"{original_manifest}:{row_index}: validation image SHA-256 mismatch"
                    )
                suffix = original_image.suffix or ".bin"
                destination = (
                    image_root / f"{len(cached):08d}-{manifest_index}{suffix}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(image_bytes)
                cached_value = (actual_sha, destination)
                cached[original_key] = cached_value
                originals[original_key] = {
                    "path": original_key,
                    "sha256": actual_sha,
                    "bytes": len(image_bytes),
                }
            elif cached_value[0] != declared_sha:
                raise ValueError(
                    f"{original_manifest}:{row_index}: conflicting validation image SHA-256"
                )
            snapshots[str(snapshot_lookup)] = cached_value[1]
    inventory = sorted(originals.values(), key=lambda item: str(item["path"]))
    return snapshots, inventory


def _load_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as error:
        raise ValueError(f"missing {description}: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _required_sha(value: object, *, field: str) -> str:
    digest = _required_text(value, field=field).casefold()
    if not _is_sha256(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _canonical_family(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    canonical = re.sub(r"[^a-z0-9]+", "", text.casefold())
    if not canonical:
        raise ValueError(f"{field} must contain an ASCII letter or digit")
    return canonical


def _audit_trusted_policy_trust_root(
    path: Path,
    *,
    actual_sha256: object | None = None,
) -> dict[str, object]:
    """Bind the policy to the repository path and code-reviewed digest pin."""

    relative = TRUSTED_POLICY_RELATIVE_PATH
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("trusted policy code path must be a safe relative path")
    repository = REPO_ROOT.resolve(strict=False)
    expected_path = (repository / relative).resolve(strict=False)
    if expected_path == repository or not expected_path.is_relative_to(repository):
        raise ValueError("trusted policy code path must stay inside the repository")
    supplied_path = path.resolve(strict=False)
    if supplied_path != expected_path:
        raise ValueError(
            "trusted policy path differs from the repository-pinned trust root"
        )

    approved_sha256 = APPROVED_TRUSTED_POLICY_SHA256.strip().casefold()
    if approved_sha256 == UNCONFIGURED_TRUST_ROOT.casefold():
        raise ValueError("trusted policy trust root is UNCONFIGURED")
    if not _is_sha256(approved_sha256):
        raise ValueError("approved trusted policy code pin is not a SHA-256 digest")

    verified = False
    if actual_sha256 is not None:
        actual = str(actual_sha256).strip().casefold()
        if not _is_sha256(actual):
            raise ValueError("trusted policy snapshot SHA-256 is invalid")
        if actual != approved_sha256:
            raise ValueError(
                "trusted policy SHA-256 differs from the repository-pinned trust root"
            )
        verified = True
    return {
        "trust_root_method": TRUST_ROOT_METHOD,
        "verified": verified,
        "repository_relative_policy_path": relative.as_posix(),
        "approved_policy_sha256": approved_sha256,
        "actual_policy_sha256": (
            str(actual_sha256).strip().casefold()
            if actual_sha256 is not None
            else None
        ),
    }


def load_trusted_policy(path: Path) -> tuple[dict[str, object], str]:
    """Load and strictly normalize the external frozen promotion trust root."""

    raw = path.read_bytes()
    try:
        policy = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("trusted policy must be a UTF-8 JSON object") from error
    if not isinstance(policy, Mapping):
        raise ValueError("trusted policy must be a JSON object")
    if policy.get("schema_version") != 1:
        raise ValueError("trusted policy schema_version must be 1")
    if policy.get("policy_schema") != TRUSTED_POLICY_SCHEMA:
        raise ValueError("trusted policy schema is unsupported")
    if policy.get("artifact_role") != "trusted_frozen_v4_candidate_judge_policy":
        raise ValueError("trusted policy artifact_role is invalid")
    if policy.get("frozen") is not True:
        raise ValueError("trusted policy must declare frozen=true")

    baseline = policy.get("approved_baseline")
    strict = policy.get("strict_validation")
    visual = policy.get("visual_judges")
    if not isinstance(baseline, Mapping):
        raise ValueError("trusted policy approved_baseline must be an object")
    if not isinstance(strict, Mapping):
        raise ValueError("trusted policy strict_validation must be an object")
    if not isinstance(visual, Mapping):
        raise ValueError("trusted policy visual_judges must be an object")

    manifest_hashes_raw = strict.get("manifest_sha256")
    if not isinstance(manifest_hashes_raw, list) or not manifest_hashes_raw:
        raise ValueError("trusted policy manifest_sha256 must be a non-empty array")
    manifest_hashes = [
        _required_sha(value, field=f"strict_validation.manifest_sha256[{index}]")
        for index, value in enumerate(manifest_hashes_raw)
    ]
    if len(manifest_hashes) != len(set(manifest_hashes)):
        raise ValueError("trusted policy strict manifest hashes must be distinct")

    judges_raw = visual.get("judges")
    if not isinstance(judges_raw, list) or len(judges_raw) < 2:
        raise ValueError("trusted policy requires at least two visual judges")
    judges: list[dict[str, object]] = []
    for index, raw_judge in enumerate(judges_raw):
        if not isinstance(raw_judge, Mapping):
            raise ValueError(f"trusted policy visual judge {index} must be an object")
        server_families_raw = raw_judge.get("server_model_families")
        if not isinstance(server_families_raw, list) or not server_families_raw:
            raise ValueError(
                f"trusted policy visual judge {index} server families are missing"
            )
        server_families = [
            _canonical_family(
                value, field=f"visual_judges.judges[{index}].family"
            )
            for value in server_families_raw
        ]
        if len(server_families) != len(set(server_families)):
            raise ValueError("trusted policy server model families must be distinct")
        judge = {
            "judge_id": _required_text(
                raw_judge.get("judge_id"), field=f"visual_judges.judges[{index}].judge_id"
            ),
            "model_family": _canonical_family(
                raw_judge.get("model_family"),
                field=f"visual_judges.judges[{index}].model_family",
            ),
            "ollama_model": _required_text(
                raw_judge.get("ollama_model"),
                field=f"visual_judges.judges[{index}].ollama_model",
            ),
            "model_manifest_sha256": _required_sha(
                raw_judge.get("model_manifest_sha256"),
                field=f"visual_judges.judges[{index}].model_manifest_sha256",
            ),
            "model_weight_layer_sha256": _required_sha(
                raw_judge.get("model_weight_layer_sha256"),
                field=(
                    f"visual_judges.judges[{index}].model_weight_layer_sha256"
                ),
            ),
            "model_config_sha256": _required_sha(
                raw_judge.get("model_config_sha256"),
                field=f"visual_judges.judges[{index}].model_config_sha256",
            ),
            "server_model_digest": _required_sha(
                raw_judge.get("server_model_digest"),
                field=f"visual_judges.judges[{index}].server_model_digest",
            ),
            "server_model_families": server_families,
        }
        if "qwen" in str(judge["model_family"]) or any(
            "qwen" in family for family in server_families
        ):
            raise ValueError("trusted policy visual judges must be non-Qwen")
        if judge["server_model_digest"] != judge["model_manifest_sha256"]:
            raise ValueError("trusted policy server digest must equal model manifest SHA")
        judges.append(judge)
    for field in (
        "judge_id",
        "model_family",
        "ollama_model",
        "model_manifest_sha256",
        "model_weight_layer_sha256",
        "model_config_sha256",
        "server_model_digest",
    ):
        values = [str(judge[field]).casefold() for judge in judges]
        if len(values) != len(set(values)):
            raise ValueError(f"trusted policy visual judge {field} values must be distinct")

    normalized: dict[str, object] = {
        "schema_version": 1,
        "policy_schema": TRUSTED_POLICY_SCHEMA,
        "artifact_role": "trusted_frozen_v4_candidate_judge_policy",
        "frozen": True,
        "approved_baseline": {
            "model_sha256": _required_sha(
                baseline.get("model_sha256"), field="approved_baseline.model_sha256"
            ),
            "metadata_sha256": _required_sha(
                baseline.get("metadata_sha256"),
                field="approved_baseline.metadata_sha256",
            ),
        },
        "strict_validation": {
            "manifest_sha256": manifest_hashes,
            "lineage_sha256": _required_sha(
                strict.get("lineage_sha256"),
                field="strict_validation.lineage_sha256",
            ),
        },
        "visual_judges": {
            "input_manifest_sha256": _required_sha(
                visual.get("input_manifest_sha256"),
                field="visual_judges.input_manifest_sha256",
            ),
            "prompt_sha256": _required_sha(
                visual.get("prompt_sha256"), field="visual_judges.prompt_sha256"
            ),
            "runner_script_sha256": _required_sha(
                visual.get("runner_script_sha256"),
                field="visual_judges.runner_script_sha256",
            ),
            "judges": judges,
        },
    }
    return normalized, _sha256_bytes(raw)


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _probability(value: object, *, field: str) -> float:
    number = _number(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be between zero and one")
    return number


def _support(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a non-negative integer") from error
    if integer < 0 or str(value).strip() not in {str(integer), f"{integer}.0"}:
        raise ValueError(f"{field} must be a non-negative integer")
    return integer


def _metric_head(
    raw: object,
    *,
    field: str,
    class_count: int,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field} must be an object")
    support_raw = raw.get("support")
    recall_raw = raw.get("per_class_recall")
    if not isinstance(support_raw, list) or len(support_raw) != class_count:
        raise ValueError(f"{field}.support must contain {class_count} values")
    if not isinstance(recall_raw, list) or len(recall_raw) != class_count:
        raise ValueError(f"{field}.per_class_recall must contain {class_count} values")
    support = [
        _support(value, field=f"{field}.support[{index}]")
        for index, value in enumerate(support_raw)
    ]
    recalls: list[float] = []
    for index, value in enumerate(recall_raw):
        if support[index] <= 0:
            raise ValueError(f"{field} class {index} has no evaluation support")
        recalls.append(
            _probability(value, field=f"{field}.per_class_recall[{index}]")
        )
    balanced = _probability(
        raw.get("balanced_accuracy"), field=f"{field}.balanced_accuracy"
    )
    calculated_balanced = sum(recalls) / len(recalls)
    if not math.isclose(balanced, calculated_balanced, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"{field}.balanced_accuracy differs from the per-class recall mean"
        )
    count = _support(raw.get("count"), field=f"{field}.count")
    if count != sum(support):
        raise ValueError(f"{field}.count differs from support sum")
    return {
        "count": count,
        "support": support,
        "per_class_recall": recalls,
        "balanced_accuracy": balanced,
    }


def _validation_metrics(
    metadata: Mapping[str, Any],
    *,
    replayed_validation: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    if replayed_validation is None:
        best_metrics = metadata.get("best_metrics")
        if not isinstance(best_metrics, Mapping):
            raise ValueError("metadata is missing best_metrics")
        validation = best_metrics.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError("metadata is missing best_metrics.validation")
    else:
        validation = replayed_validation
    objectness_classes = metadata.get("objectness_classes")
    material_classes = metadata.get("material_classes")
    if objectness_classes != [BACKGROUND_CLASS_NAME, "material"]:
        raise ValueError("metadata objectness_classes must be [background, material]")
    if material_classes != list(EXPECTED_MATERIAL_CLASSES):
        raise ValueError("metadata material classes differ from the frozen nine-class tuple")
    objectness = _metric_head(
        validation.get("objectness"), field="validation.objectness", class_count=2
    )
    material = _metric_head(
        validation.get("material"), field="validation.material", class_count=9
    )
    if material["count"] != objectness["support"][1]:  # type: ignore[index]
        raise ValueError(
            "validation material count differs from material objectness support"
        )
    return {
        "objectness_classes": list(objectness_classes),
        "material_classes": [str(value) for value in material_classes],
        "objectness": objectness,
        "material": material,
    }


def audit_candidate_metrics(
    metadata: Mapping[str, Any],
    baseline_metadata: Mapping[str, Any],
    thresholds: GateThresholds,
    *,
    replayed_validation: Mapping[str, Any],
    baseline_replayed_validation: Mapping[str, Any] | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Validate fixed metadata metrics and return evidence plus gate failures."""

    problems: list[str] = []
    if metadata.get("candidate_only") is not True:
        problems.append("metadata does not declare candidate_only=true")
    if metadata.get("production_runtime_modified") is not False:
        problems.append("metadata does not declare production_runtime_modified=false")
    metrics = _validation_metrics(
        metadata, replayed_validation=replayed_validation
    )
    metadata_claim_matches_replay = False
    try:
        metadata_claim_matches_replay = _validation_metrics(metadata) == metrics
    except ValueError:
        metadata_claim_matches_replay = False
    if not metadata_claim_matches_replay:
        problems.append("metadata best_metrics differ from independently replayed metrics")
    objectness = metrics["objectness"]
    material = metrics["material"]
    bg_support, positive_support = objectness["support"]  # type: ignore[index]
    bg_recall, positive_recall = objectness["per_class_recall"]  # type: ignore[index]
    objectness_balanced = float(objectness["balanced_accuracy"])  # type: ignore[arg-type]
    material_balanced = float(material["balanced_accuracy"])  # type: ignore[arg-type]
    material_recalls = list(material["per_class_recall"])  # type: ignore[arg-type]

    collapse = "none"
    if math.isclose(bg_recall, 1.0) and math.isclose(positive_recall, 0.0):
        collapse = "all_background"
        problems.append("objectness collapsed to all-background predictions")
    elif math.isclose(bg_recall, 0.0) and math.isclose(positive_recall, 1.0):
        collapse = "all_material"
        problems.append("objectness collapsed to all-material predictions")
    if bg_recall == 0.0 and collapse != "all_material":
        problems.append("objectness has zero background recall")
    if positive_recall == 0.0 and collapse != "all_background":
        problems.append("objectness has zero material recall")

    if bg_support < thresholds.min_background_support:
        problems.append(
            f"background support {bg_support} < {thresholds.min_background_support}"
        )
    if positive_support <= 0:
        problems.append("material objectness support is zero")
    if bg_recall + 1e-12 < thresholds.min_background_recall:
        problems.append(
            f"background recall {bg_recall:.6f} < {thresholds.min_background_recall:.6f}"
        )
    if positive_recall + 1e-12 < thresholds.min_material_objectness_recall:
        problems.append(
            "material objectness recall "
            f"{positive_recall:.6f} < {thresholds.min_material_objectness_recall:.6f}"
        )
    if objectness_balanced + 1e-12 < thresholds.min_objectness_balanced_accuracy:
        problems.append(
            "objectness balanced accuracy "
            f"{objectness_balanced:.6f} < "
            f"{thresholds.min_objectness_balanced_accuracy:.6f}"
        )
    if material_balanced + 1e-12 < thresholds.min_material_balanced_accuracy:
        problems.append(
            "material balanced accuracy "
            f"{material_balanced:.6f} < {thresholds.min_material_balanced_accuracy:.6f}"
        )
    material_classes = list(metrics["material_classes"])
    for name, recall in zip(material_classes, material_recalls, strict=True):
        if recall + 1e-12 < thresholds.min_each_material_recall:
            problems.append(
                f"material recall {name} {recall:.6f} < "
                f"{thresholds.min_each_material_recall:.6f}"
            )

    regressions: list[dict[str, object]] = []
    baseline_evidence: dict[str, object] = {
        "supplied": baseline_replayed_validation is not None
    }
    if baseline_replayed_validation is not None:
        baseline_metrics = _validation_metrics(
            baseline_metadata, replayed_validation=baseline_replayed_validation
        )
        baseline_metadata_claim_matches_replay = False
        try:
            baseline_metadata_claim_matches_replay = (
                _validation_metrics(baseline_metadata) == baseline_metrics
            )
        except ValueError:
            baseline_metadata_claim_matches_replay = False
        if not baseline_metadata_claim_matches_replay:
            problems.append(
                "baseline metadata best_metrics differ from independently "
                "replayed metrics"
            )
        comparisons: list[tuple[str, float, float]] = []
        for index, class_name in enumerate(metrics["objectness_classes"]):
            comparisons.append(
                (
                    f"objectness/{class_name}",
                    float(baseline_metrics["objectness"]["per_class_recall"][index]),  # type: ignore[index]
                    float(objectness["per_class_recall"][index]),  # type: ignore[index]
                )
            )
        for index, class_name in enumerate(material_classes):
            comparisons.append(
                (
                    f"material/{class_name}",
                    float(baseline_metrics["material"]["per_class_recall"][index]),  # type: ignore[index]
                    float(material_recalls[index]),
                )
            )
        for metric_name, baseline_value, candidate_value in comparisons:
            drop = baseline_value - candidate_value
            if drop > thresholds.max_baseline_recall_drop + 1e-12:
                regression = {
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "drop": drop,
                }
                regressions.append(regression)
                problems.append(
                    f"baseline recall regression {metric_name}: {drop:.6f} > "
                    f"{thresholds.max_baseline_recall_drop:.6f}"
                )
        baseline_evidence.update(
            {
                "max_allowed_recall_drop": thresholds.max_baseline_recall_drop,
                "metadata_claim_matches_replay": (
                    baseline_metadata_claim_matches_replay
                ),
                "regressions": regressions,
            }
        )

    evidence = {
        "passed": not problems,
        "collapse": collapse,
        "metric_authority": "independent_onnx_replay",
        "metadata_claim_matches_replay": metadata_claim_matches_replay,
        "thresholds": asdict(thresholds),
        "validation": metrics,
        "baseline_non_regression": baseline_evidence,
    }
    return evidence, problems


def _read_strict_manifests(
    manifest_paths: Sequence[Path],
    *,
    material_classes: Sequence[str],
) -> list[dict[str, str]]:
    if not manifest_paths:
        raise ValueError("at least one strict CSV manifest is required")
    rows: list[dict[str, str]] = []
    for path in manifest_paths:
        if path.suffix.casefold() != ".csv":
            raise ValueError(f"strict manifest must be CSV: {path}")
        try:
            file = path.open("r", encoding="utf-8-sig", newline="")
        except FileNotFoundError as error:
            raise ValueError(f"missing strict manifest: {path}") from error
        with file:
            reader = csv.DictReader(file)
            fields = set(reader.fieldnames or ())
            missing = sorted(set(STRICT_FIELDS) - fields)
            if missing:
                raise ValueError(f"{path}: missing strict fields {missing}")
            for line_number, raw in enumerate(reader, start=2):
                location = f"{path}:{line_number}"
                row = {name: str(value or "").strip() for name, value in raw.items()}
                for field in (*LINEAGE_FIELDS, "role", "fold", "origin"):
                    if not row.get(field):
                        raise ValueError(f"{location}: missing {field}")
                for field in ("source_sha256", "image_sha256"):
                    row[field] = row[field].casefold()
                    if not _is_sha256(row[field]):
                        raise ValueError(f"{location}: invalid {field}")
                role = row["role"]
                if role not in KNOWN_ROLES:
                    raise ValueError(f"{location}: unsupported role {role!r}")
                if role in ROLE_TO_SPLIT and row["split"] != ROLE_TO_SPLIT[role]:
                    raise ValueError(f"{location}: split is inconsistent with role")
                try:
                    material = int(row["material"])
                except ValueError as error:
                    raise ValueError(f"{location}: material must be an integer") from error
                if not 0 <= material <= BACKGROUND_MATERIAL_ID:
                    raise ValueError(f"{location}: material must be between 0 and 9")
                expected_category = (
                    BACKGROUND_CLASS_NAME
                    if material == BACKGROUND_MATERIAL_ID
                    else str(material_classes[material])
                )
                if row["category"] != expected_category:
                    raise ValueError(
                        f"{location}: category does not match material class contract"
                    )
                try:
                    source_object_count = int(row["source_object_count"])
                except ValueError as error:
                    raise ValueError(
                        f"{location}: source_object_count must be an integer"
                    ) from error
                if source_object_count not in {0, 1}:
                    raise ValueError(
                        f"{location}: source_object_count must be zero or one"
                    )
                expected_crop_object_count = (
                    0 if material == BACKGROUND_MATERIAL_ID else 1
                )
                raw_crop_count = row.get("crop_object_count", "")
                if raw_crop_count:
                    try:
                        crop_object_count = int(raw_crop_count)
                    except ValueError as error:
                        raise ValueError(
                            f"{location}: crop_object_count must be an integer"
                        ) from error
                else:
                    # Legacy manifests omitted crop_object_count and are safe
                    # only when the source count itself is unambiguous.
                    crop_object_count = expected_crop_object_count
                    if source_object_count != expected_crop_object_count:
                        raise ValueError(
                            f"{location}: crop_object_count is required for a "
                            "hard-negative background"
                        )
                if crop_object_count != expected_crop_object_count:
                    raise ValueError(
                        f"{location}: crop_object_count must be "
                        f"{expected_crop_object_count} for material={material}"
                    )
                if crop_object_count > source_object_count:
                    raise ValueError(
                        f"{location}: crop_object_count exceeds source_object_count"
                    )
                row["material"] = str(material)
                row["crop_object_count"] = str(crop_object_count)
                row["_manifest"] = str(path.resolve())
                row["_line"] = str(line_number)
                rows.append(row)
    if not rows:
        raise ValueError("strict manifests contain no rows")
    return rows


def _lineage_digest(rows: Sequence[Mapping[str, str]]) -> str:
    ordered = [
        {
            "sample_id": row["sample_id"],
            "source_sha256": row["source_sha256"],
            "object_group": row["object_group"],
            "capture_session": row["capture_session"],
            "role": row["role"],
            "fold": row["fold"],
            "image_sha256": row["image_sha256"],
            "material": row["material"],
        }
        for row in sorted(rows, key=lambda item: item["sample_id"])
    ]
    return _sha256_bytes(
        json.dumps(
            ordered, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def audit_lineage(
    metadata: Mapping[str, Any],
    manifest_paths: Sequence[Path],
    *,
    manifest_artifacts: Sequence[Mapping[str, object]],
) -> tuple[
    dict[str, object],
    list[dict[str, str]],
    list[dict[str, str]],
    list[str],
]:
    """Independently parse strict manifests and reject any role leakage."""

    problems: list[str] = []
    material_classes = metadata.get("material_classes")
    if not isinstance(material_classes, list) or len(material_classes) != 9:
        raise ValueError("metadata material class contract is unavailable for lineage audit")
    rows = _read_strict_manifests(
        manifest_paths, material_classes=[str(value) for value in material_classes]
    )
    role_counts = Counter(row["role"] for row in rows)
    for required_role in (TRAIN_ROLE, VALIDATION_ROLE):
        if role_counts[required_role] <= 0:
            problems.append(f"strict manifests contain no {required_role} rows")

    duplicate_counts: dict[str, int] = {}
    for field in ("sample_id", "image_sha256"):
        counts = Counter(row[field] for row in rows)
        duplicates = [value for value, count in counts.items() if count > 1]
        duplicate_counts[field] = len(duplicates)
        if duplicates:
            problems.append(f"duplicate strict identity {field}: {len(duplicates)}")

    cross_role_counts: dict[str, int] = {}
    for field in LINEAGE_FIELDS:
        roles_by_value: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            roles_by_value[row[field]].add(row["role"])
        crossing = [value for value, roles in roles_by_value.items() if len(roles) > 1]
        cross_role_counts[field] = len(crossing)
        if crossing:
            problems.append(f"{field} crosses role partitions: {len(crossing)}")

    calculated_lineage_sha = _lineage_digest(rows)
    summary = metadata.get("manifest_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("metadata is missing manifest_summary")
    if summary.get("strict") is not True:
        problems.append("metadata manifest_summary.strict is not true")
    if summary.get("lineage_sha256") != calculated_lineage_sha:
        problems.append("metadata lineage_sha256 differs from supplied strict manifests")
    if summary.get("rows") != len(rows):
        problems.append("metadata manifest row count differs from supplied strict manifests")
    metadata_role_counts = summary.get("role_counts")
    if not isinstance(metadata_role_counts, Mapping) or {
        str(key): int(value) for key, value in metadata_role_counts.items()
    } != dict(sorted(role_counts.items())):
        problems.append("metadata role_counts differ from supplied strict manifests")

    supplied_hashes = Counter(
        str(artifact["sha256"])
        for artifact in manifest_artifacts
        if artifact.get("exists") is True
    )
    metadata_inputs = summary.get("input_manifests")
    metadata_hashes: Counter[str] = Counter()
    if isinstance(metadata_inputs, list):
        for index, item in enumerate(metadata_inputs):
            if not isinstance(item, Mapping) or not _is_sha256(item.get("sha256")):
                problems.append(f"metadata input_manifests[{index}] lacks a valid SHA-256")
                continue
            metadata_hashes[str(item["sha256"])] += 1
    else:
        problems.append("metadata input_manifests is not an array")
    if metadata_hashes != supplied_hashes:
        problems.append("metadata input manifest hashes differ from supplied manifests")

    validation_background = [
        {
            "sample_id": row["sample_id"],
            "source_sha256": row["source_sha256"],
            "image_sha256": row["image_sha256"],
        }
        for row in rows
        if row["role"] == VALIDATION_ROLE
        and row["material"] == str(BACKGROUND_MATERIAL_ID)
    ]
    validation_background.sort(key=lambda row: row["sample_id"])
    validation_rows = [row for row in rows if row["role"] == VALIDATION_ROLE]
    validation_samples = [
        {
            "sample_id": row["sample_id"],
            "source_sha256": row["source_sha256"],
            "image_sha256": row["image_sha256"],
            "object_group": row["object_group"],
            "capture_session": row["capture_session"],
            "fold": row["fold"],
            "truth_objectness": (
                0 if row["material"] == str(BACKGROUND_MATERIAL_ID) else 1
            ),
            "truth_material": (
                "" if row["material"] == str(BACKGROUND_MATERIAL_ID) else row["material"]
            ),
        }
        for row in validation_rows
    ]
    validation_samples.sort(key=lambda row: row["sample_id"])
    validation_objectness_support = [
        sum(row["material"] == str(BACKGROUND_MATERIAL_ID) for row in validation_rows),
        sum(row["material"] != str(BACKGROUND_MATERIAL_ID) for row in validation_rows),
    ]
    validation_material_support = [
        sum(row["material"] == str(class_id) for row in validation_rows)
        for class_id in range(9)
    ]
    evidence = {
        "passed": not problems,
        "row_count": len(rows),
        "role_counts": dict(sorted(role_counts.items())),
        "calculated_lineage_sha256": calculated_lineage_sha,
        "metadata_lineage_sha256": summary.get("lineage_sha256"),
        "duplicate_identity_counts": duplicate_counts,
        "cross_role_identity_counts": cross_role_counts,
        "validation_background_samples": len(validation_background),
        "validation_objectness_support": validation_objectness_support,
        "validation_material_support": validation_material_support,
        "manifest_hashes_match_metadata": metadata_hashes == supplied_hashes,
    }
    return evidence, validation_background, validation_samples, problems


def _read_replay_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid replay JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: replay row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError("replay predictions are empty")
    return rows


def audit_trusted_policy_bindings(
    policy: Mapping[str, Any],
    *,
    policy_sha256: str,
    baseline_model_path: Path,
    baseline_metadata_path: Path,
    manifest_artifacts: Sequence[Mapping[str, object]],
    calculated_lineage_sha256: str,
) -> tuple[dict[str, object], list[str]]:
    """Bind all caller-supplied baseline/data identities to the frozen policy."""

    problems: list[str] = []
    baseline = policy["approved_baseline"]
    strict = policy["strict_validation"]
    actual_baseline_model = (
        _sha256_file(baseline_model_path) if baseline_model_path.is_file() else None
    )
    actual_baseline_metadata = (
        _sha256_file(baseline_metadata_path)
        if baseline_metadata_path.is_file()
        else None
    )
    if actual_baseline_model != baseline["model_sha256"]:
        problems.append("baseline model SHA-256 differs from trusted policy")
    if actual_baseline_metadata != baseline["metadata_sha256"]:
        problems.append("baseline metadata SHA-256 differs from trusted policy")
    actual_manifest_hashes = sorted(
        str(artifact["sha256"])
        for artifact in manifest_artifacts
        if artifact.get("exists") is True
    )
    policy_manifest_hashes = sorted(str(value) for value in strict["manifest_sha256"])
    if actual_manifest_hashes != policy_manifest_hashes:
        problems.append("strict manifest hashes differ from trusted policy")
    if calculated_lineage_sha256 != strict["lineage_sha256"]:
        problems.append("strict manifest lineage differs from trusted policy")
    evidence = {
        "passed": not problems,
        "trusted_policy_sha256": policy_sha256,
        "baseline_model_sha256": actual_baseline_model,
        "baseline_metadata_sha256": actual_baseline_metadata,
        "strict_manifest_sha256": actual_manifest_hashes,
        "strict_lineage_sha256": calculated_lineage_sha256,
    }
    return evidence, problems


def _independent_confusion_metrics(confusion: Sequence[Sequence[int]]) -> dict[str, object]:
    classes = len(confusion)
    if classes < 2 or any(len(row) != classes for row in confusion):
        raise ValueError("replay confusion matrix must be square")
    support = [sum(int(value) for value in row) for row in confusion]
    if any(value <= 0 for value in support):
        raise ValueError("every replay class must have positive support")
    recalls = [confusion[index][index] / support[index] for index in range(classes)]
    count = sum(support)
    return {
        "count": count,
        "support": support,
        "per_class_recall": recalls,
        "balanced_accuracy": sum(recalls) / classes,
        "accuracy": sum(confusion[index][index] for index in range(classes)) / count,
        "confusion": [list(row) for row in confusion],
    }


def _argmax_logits(value: object, *, classes: int, field: str) -> int:
    if not isinstance(value, list) or len(value) != classes:
        raise ValueError(f"{field} must contain {classes} logits")
    logits = [_number(item, field=field) for item in value]
    maximum = max(logits)
    if logits.count(maximum) != 1:
        raise ValueError(f"{field} has an unstable top-1 tie")
    return logits.index(maximum)


def audit_replay_evidence(
    *,
    predictions_path: Path,
    attestation_path: Path,
    model_path: Path,
    metadata_path: Path,
    inference_spec_path: Path,
    manifest_artifacts: Sequence[Mapping[str, object]],
    expected_validation: Sequence[Mapping[str, str]],
    calculated_lineage_sha256: str,
    metadata: Mapping[str, Any],
    runtime_replay_predictions_sha256: str,
) -> tuple[dict[str, object], Mapping[str, Any], list[str]]:
    """Recompute replay metrics without trusting metadata or attestation metrics."""

    problems: list[str] = []
    for description, path in (
        ("replay predictions", predictions_path),
        ("replay attestation", attestation_path),
        ("candidate model", model_path),
        ("inference spec", inference_spec_path),
    ):
        if not path.is_file():
            return {"passed": False}, {}, [f"{description} is missing"]
    attestation = _load_json(attestation_path, description="replay attestation")
    if attestation.get("schema_version") != 1:
        problems.append("replay attestation has unsupported schema_version")
    if attestation.get("evidence_schema") != "v4_candidate_validation_replay.v1":
        problems.append("replay attestation has unsupported evidence_schema")
    if attestation.get("evaluation_role") != VALIDATION_ROLE:
        problems.append("replay attestation role is not model_validation")
    if attestation.get("thresholds_applied") is not False:
        problems.append("replay evidence applied a threshold")
    if attestation.get("production_deployment_authorized") is not False:
        problems.append("replay evidence claims production authorization")
    if attestation.get("custom_session_factory_used") is not False:
        problems.append("replay evidence used a non-authoritative custom session factory")
    if (
        attestation.get("artifact_snapshot_contract")
        != "read_bytes_hash_before_use_and_hash_after.v1"
    ):
        problems.append("replay evidence lacks the frozen artifact snapshot contract")
    if attestation.get("runtime_artifact_hashes_match_snapshots") is not True:
        problems.append("replay runtime artifact hashes were not snapshot verified")

    bindings = {
        "predictions_sha256": _sha256_file(predictions_path),
        "model_sha256": _sha256_file(model_path),
        "verifier_metadata_sha256": _sha256_file(metadata_path),
        "inference_spec_sha256": _sha256_file(inference_spec_path),
    }
    for field, actual in bindings.items():
        if attestation.get(field) != actual:
            problems.append(f"replay attestation {field} mismatch")
    runtime_hashes = attestation.get("runtime_artifact_hashes")
    expected_runtime_hashes = {
        "model": bindings["model_sha256"],
        "metadata": bindings["verifier_metadata_sha256"],
        "spec": bindings["inference_spec_sha256"],
    }
    if runtime_hashes != expected_runtime_hashes:
        problems.append("replay runtime artifact hashes differ from final bindings")
    if not _is_sha256(runtime_replay_predictions_sha256):
        problems.append("independent runtime ONNX replay is unavailable")
    elif bindings["predictions_sha256"] != runtime_replay_predictions_sha256:
        problems.append(
            "provided replay predictions differ from independent runtime ONNX replay"
        )
    supplied_manifest_hashes = Counter(
        str(item["sha256"]) for item in manifest_artifacts if item.get("exists") is True
    )
    replay_manifest_hashes: Counter[str] = Counter()
    raw_manifest_artifacts = attestation.get("manifest_artifacts")
    if not isinstance(raw_manifest_artifacts, list):
        problems.append("replay attestation manifest_artifacts is not an array")
    else:
        for item in raw_manifest_artifacts:
            if not isinstance(item, Mapping) or not _is_sha256(item.get("sha256")):
                problems.append("replay attestation contains invalid manifest hash")
                continue
            replay_manifest_hashes[str(item["sha256"])] += 1
    if replay_manifest_hashes != supplied_manifest_hashes:
        problems.append("replay manifest hashes differ from strict manifests")
    if attestation.get("manifest_lineage_sha256") != calculated_lineage_sha256:
        problems.append("replay lineage SHA-256 differs from strict manifests")
    if attestation.get("objectness_classes") != ["background", "material"]:
        problems.append("replay objectness class order is invalid")
    if metadata.get("material_classes") != list(EXPECTED_MATERIAL_CLASSES):
        problems.append("replay metadata material classes differ from frozen tuple")
    if attestation.get("material_classes") != list(EXPECTED_MATERIAL_CLASSES):
        problems.append("replay material class order differs from frozen tuple")

    raw_rows = _read_replay_jsonl(predictions_path)
    expected = {row["sample_id"]: row for row in expected_validation}
    if len(raw_rows) != len(expected):
        problems.append("replay prediction count differs from validation manifest")
    seen: set[str] = set()
    objectness_confusion = [[0, 0], [0, 0]]
    material_confusion = [[0 for _ in range(9)] for _ in range(9)]
    for index, raw in enumerate(raw_rows, start=1):
        sample_id = str(raw.get("sample_id", "")).strip()
        if sample_id in seen:
            problems.append(f"duplicate replay sample_id: {sample_id}")
            continue
        seen.add(sample_id)
        expected_row = expected.get(sample_id)
        if expected_row is None:
            problems.append(f"unexpected replay sample_id: {sample_id}")
            continue
        for field in (
            "source_sha256",
            "image_sha256",
            "object_group",
            "capture_session",
            "fold",
        ):
            if str(raw.get(field, "")) != expected_row[field]:
                problems.append(f"replay sample {sample_id} has mismatched {field}")
        if raw.get("role") != VALIDATION_ROLE:
            problems.append(f"replay sample {sample_id} has invalid role")
        if not _is_sha256(raw.get("input_tensor_sha256")):
            problems.append(f"replay sample {sample_id} has invalid tensor SHA-256")
        truth_objectness = expected_row["truth_objectness"]
        truth_material = (
            None if expected_row["truth_material"] == "" else int(expected_row["truth_material"])
        )
        if raw.get("truth_objectness") != truth_objectness:
            problems.append(f"replay sample {sample_id} changed objectness truth")
        if raw.get("truth_material") != truth_material:
            problems.append(f"replay sample {sample_id} changed material truth")
        predicted_objectness = _argmax_logits(
            raw.get("objectness_logits"), classes=2, field=f"row {index} objectness"
        )
        predicted_material = _argmax_logits(
            raw.get("material_logits"), classes=9, field=f"row {index} material"
        )
        if raw.get("predicted_objectness") != predicted_objectness:
            problems.append(f"replay sample {sample_id} objectness prediction mismatch")
        if raw.get("predicted_material_head") != predicted_material:
            problems.append(f"replay sample {sample_id} material prediction mismatch")
        expected_cascade = predicted_material if predicted_objectness == 1 else None
        if raw.get("cascaded_material") != expected_cascade:
            problems.append(f"replay sample {sample_id} cascaded prediction mismatch")
        objectness_confusion[int(truth_objectness)][predicted_objectness] += 1
        if truth_material is not None:
            material_confusion[truth_material][predicted_material] += 1
    missing = sorted(set(expected) - seen)
    if missing:
        problems.append(f"replay is missing {len(missing)} validation samples")

    replayed_metrics = {
        "objectness": _independent_confusion_metrics(objectness_confusion),
        "material": _independent_confusion_metrics(material_confusion),
    }
    attested_metrics = attestation.get("metrics")
    if attested_metrics != replayed_metrics:
        problems.append("replay attestation metrics differ from independent recomputation")
    if attestation.get("prediction_count") != len(raw_rows):
        problems.append("replay attestation prediction_count mismatch")
    evidence = {
        "passed": not problems,
        "prediction_count": len(raw_rows),
        "artifact_binding": bindings,
        "runtime_replay_predictions_sha256": runtime_replay_predictions_sha256,
        "runtime_replay_matches_provided": (
            bindings["predictions_sha256"] == runtime_replay_predictions_sha256
        ),
        "metrics": replayed_metrics,
        "metrics_recomputed_independently": True,
        "sample_set_exact": not missing and len(raw_rows) == len(expected),
    }
    return evidence, replayed_metrics, problems


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        file = path.open("r", encoding="utf-8-sig")
    except FileNotFoundError as error:
        raise ValueError(f"missing visual judge evidence: {path}") from error
    with file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"visual judge evidence line {line_number} is invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"visual judge evidence line {line_number} must be an object"
                )
            rows.append(value)
    if not rows:
        raise ValueError("visual judge evidence is empty")
    return rows


def audit_visual_judges(
    config_path: Path,
    evidence_path: Path,
    validation_background: Sequence[Mapping[str, str]],
) -> tuple[dict[str, object], list[str]]:
    """Require independent hidden-input judges and treat every doubt as a veto."""

    problems: list[str] = []
    if not config_path.is_file():
        return {
            "passed": False,
            "authority": "diagnostic_veto_only",
            "reason": "visual judge config missing",
        }, ["visual judge config is missing"]
    if not evidence_path.is_file():
        return {
            "passed": False,
            "authority": "diagnostic_veto_only",
            "reason": "visual judge evidence missing",
        }, ["visual judge evidence is missing"]

    config = _load_json(config_path, description="visual judge config")
    if config.get("schema_version") != SCHEMA_VERSION:
        problems.append("visual judge config has unsupported schema_version")
    if config.get("authority") != "diagnostic_veto_only":
        problems.append("visual judge authority must be diagnostic_veto_only")
    if config.get("candidate_prediction_visible") is not False:
        problems.append("candidate predictions were not hidden from visual judges")
    if config.get("ground_truth_visible") is not False:
        problems.append("ground truth was not hidden from visual judges")
    prompt_sha = config.get("prompt_sha256")
    if not _is_sha256(prompt_sha):
        problems.append("visual judge config prompt_sha256 is invalid")
        prompt_sha = ""
    prompt_path_raw = str(config.get("prompt_path", "")).strip()
    prompt_path = Path(prompt_path_raw) if prompt_path_raw else Path()
    if prompt_path_raw and not prompt_path.is_absolute():
        prompt_path = config_path.parent / prompt_path
    if not prompt_path_raw or not prompt_path.is_file():
        problems.append("visual judge prompt artifact is missing")
    elif _sha256_file(prompt_path) != prompt_sha:
        problems.append("visual judge prompt artifact SHA-256 mismatch")

    runner_sha = str(config.get("runner_manifest_sha256", "")).strip().casefold()
    runner_path_raw = str(config.get("runner_manifest_path", "")).strip()
    runner_path = Path(runner_path_raw) if runner_path_raw else Path()
    if runner_path_raw and not runner_path.is_absolute():
        runner_path = config_path.parent / runner_path
    if not _is_sha256(runner_sha) or not runner_path_raw or not runner_path.is_file():
        problems.append("visual judge runner manifest is missing or invalid")
    elif _sha256_file(runner_path) != runner_sha:
        problems.append("visual judge runner manifest SHA-256 mismatch")
    else:
        runner = _load_json(runner_path, description="visual judge runner manifest")
        expected_runner_contract = {
            "schema_version": 1,
            "authority": "diagnostic_veto_only",
            "candidate_prediction_visible": False,
            "ground_truth_visible": False,
            "prompt_sha256": prompt_sha,
            "raw_response_capture": True,
            "immutable_outputs": True,
        }
        for field, expected_value in expected_runner_contract.items():
            if runner.get(field) != expected_value:
                problems.append(f"visual judge runner manifest has invalid {field}")

    teacher_raw = config.get("teacher_model_families")
    if not isinstance(teacher_raw, list) or not teacher_raw:
        problems.append("teacher_model_families must be a non-empty array")
        teacher_families: set[str] = set()
    else:
        teacher_families = {
            str(value).strip().casefold() for value in teacher_raw if str(value).strip()
        }
        if len(teacher_families) != len(teacher_raw):
            problems.append("teacher_model_families must be non-empty and unique")

    models_raw = config.get("judge_models")
    model_by_family: dict[str, dict[str, str]] = {}
    if not isinstance(models_raw, list) or len(models_raw) < 2:
        problems.append("at least two visual judge models are required")
    else:
        seen_model_ids: set[str] = set()
        seen_model_shas: set[str] = set()
        seen_manifest_paths: set[str] = set()
        for index, raw in enumerate(models_raw):
            if not isinstance(raw, Mapping):
                problems.append(f"judge_models[{index}] must be an object")
                continue
            family = str(raw.get("model_family", "")).strip().casefold()
            model_id = str(raw.get("model_id", "")).strip()
            model_sha = str(raw.get("model_sha256", "")).strip().casefold()
            raw_manifest_path = str(raw.get("model_manifest_path", "")).strip()
            if (
                not family
                or not model_id
                or not _is_sha256(model_sha)
                or not raw_manifest_path
            ):
                problems.append(f"judge_models[{index}] has invalid model identity")
                continue
            manifest_path = Path(raw_manifest_path)
            if not manifest_path.is_absolute():
                manifest_path = config_path.parent / manifest_path
            manifest_path = manifest_path.resolve(strict=False)
            if not manifest_path.is_file():
                problems.append(
                    f"judge_models[{index}] model manifest does not exist"
                )
                continue
            actual_model_sha = _sha256_file(manifest_path)
            if actual_model_sha != model_sha:
                problems.append(
                    f"judge_models[{index}] model manifest SHA-256 mismatch"
                )
                continue
            if family in model_by_family:
                problems.append("visual judge model families must be distinct")
                continue
            if model_id.casefold() in seen_model_ids:
                problems.append("visual judge model IDs must be distinct")
                continue
            normalized_manifest_path = str(manifest_path).casefold()
            if model_sha in seen_model_shas:
                problems.append("visual judge model manifest hashes must be distinct")
                continue
            if normalized_manifest_path in seen_manifest_paths:
                problems.append("visual judge model manifest paths must be distinct")
                continue
            if family in teacher_families:
                problems.append(f"visual judge family {family} is a teacher family")
            seen_model_ids.add(model_id.casefold())
            seen_model_shas.add(model_sha)
            seen_manifest_paths.add(normalized_manifest_path)
            model_by_family[family] = {
                "model_family": family,
                "model_id": model_id,
                "model_sha256": model_sha,
                "model_manifest_path": str(manifest_path),
            }
    if len(model_by_family) < 2:
        problems.append("fewer than two valid distinct visual judge families remain")

    evidence_rows = _load_jsonl(evidence_path)
    expected = {row["sample_id"]: dict(row) for row in validation_background}
    votes: dict[str, dict[str, str]] = defaultdict(dict)
    vetoes: list[dict[str, str]] = []
    invalid_vote_count = 0
    seen_raw_responses: set[str] = set()
    seen_raw_response_hashes: set[str] = set()
    for line_number, raw in enumerate(evidence_rows, start=1):
        location = f"visual judge evidence line {line_number}"
        forbidden = sorted(FORBIDDEN_VISUAL_FIELDS & set(raw))
        if forbidden:
            problems.append(f"{location} exposes forbidden fields {forbidden}")
            invalid_vote_count += 1
            continue
        if raw.get("candidate_prediction_visible") is not False:
            problems.append(f"{location} did not hide candidate prediction")
            invalid_vote_count += 1
            continue
        if raw.get("ground_truth_visible") is not False:
            problems.append(f"{location} did not hide ground truth")
            invalid_vote_count += 1
            continue
        if raw.get("prompt_sha256") != prompt_sha:
            problems.append(f"{location} prompt SHA-256 differs from config")
            invalid_vote_count += 1
            continue
        if raw.get("runner_manifest_sha256") != runner_sha:
            problems.append(f"{location} runner manifest SHA-256 differs from config")
            invalid_vote_count += 1
            continue
        raw_response_path_value = str(raw.get("raw_response_path", "")).strip()
        raw_response_sha = str(raw.get("raw_response_sha256", "")).strip().casefold()
        raw_response_path = (
            Path(raw_response_path_value) if raw_response_path_value else Path()
        )
        if raw_response_path_value and not raw_response_path.is_absolute():
            raw_response_path = config_path.parent / raw_response_path
        normalized_response_path = str(raw_response_path.resolve(strict=False)).casefold()
        if (
            not raw_response_path_value
            or not _is_sha256(raw_response_sha)
            or not raw_response_path.is_file()
        ):
            problems.append(f"{location} raw response artifact is missing or invalid")
            invalid_vote_count += 1
            continue
        if _sha256_file(raw_response_path) != raw_response_sha:
            problems.append(f"{location} raw response SHA-256 mismatch")
            invalid_vote_count += 1
            continue
        try:
            raw_response_payload = json.loads(
                raw_response_path.read_text(encoding="utf-8-sig")
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            problems.append(f"{location} raw response is not valid JSON: {error}")
            invalid_vote_count += 1
            continue
        if not isinstance(raw_response_payload, Mapping):
            problems.append(f"{location} raw response must be an object")
            invalid_vote_count += 1
            continue
        if normalized_response_path in seen_raw_responses:
            problems.append(f"{location} reuses a raw response artifact")
            invalid_vote_count += 1
            continue
        if raw_response_sha in seen_raw_response_hashes:
            problems.append(f"{location} reuses raw response content")
            invalid_vote_count += 1
            continue
        seen_raw_responses.add(normalized_response_path)
        seen_raw_response_hashes.add(raw_response_sha)
        sample_id = str(raw.get("sample_id", "")).strip()
        if sample_id not in expected:
            problems.append(f"{location} references an unexpected sample")
            invalid_vote_count += 1
            continue
        for field in ("source_sha256", "image_sha256"):
            if str(raw.get(field, "")).strip().casefold() != expected[sample_id][field]:
                problems.append(f"{location} has mismatched {field}")
                invalid_vote_count += 1
                break
        else:
            family = str(raw.get("model_family", "")).strip().casefold()
            configured = model_by_family.get(family)
            if configured is None:
                problems.append(f"{location} uses an unconfigured model family")
                invalid_vote_count += 1
                continue
            if (
                str(raw.get("model_id", "")).strip() != configured["model_id"]
                or str(raw.get("model_sha256", "")).strip().casefold()
                != configured["model_sha256"]
            ):
                problems.append(f"{location} model identity differs from config")
                invalid_vote_count += 1
                continue
            if family in votes[sample_id]:
                problems.append(f"{location} duplicates a family vote for {sample_id}")
                invalid_vote_count += 1
                continue
            verdict = str(raw.get("verdict", "")).strip().casefold()
            if verdict not in {"background", "ambiguous", "material"}:
                problems.append(f"{location} has an invalid verdict")
                invalid_vote_count += 1
                continue
            expected_raw_response = {
                "schema_version": 1,
                "sample_id": sample_id,
                "source_sha256": expected[sample_id]["source_sha256"],
                "image_sha256": expected[sample_id]["image_sha256"],
                "model_family": family,
                "model_id": configured["model_id"],
                "model_sha256": configured["model_sha256"],
                "prompt_sha256": prompt_sha,
                "runner_manifest_sha256": runner_sha,
                "candidate_prediction_visible": False,
                "ground_truth_visible": False,
                "verdict": verdict,
            }
            unexpected_raw_fields = sorted(
                set(raw_response_payload) - set(expected_raw_response)
            )
            if unexpected_raw_fields:
                problems.append(
                    f"{location} raw response exposes forbidden fields "
                    f"{unexpected_raw_fields}"
                )
                invalid_vote_count += 1
                continue
            if any(
                raw_response_payload.get(field) != expected_value
                for field, expected_value in expected_raw_response.items()
            ):
                problems.append(f"{location} differs from the raw judge response")
                invalid_vote_count += 1
                continue
            votes[sample_id][family] = verdict
            if verdict != "background":
                vetoes.append(
                    {
                        "sample_id": sample_id,
                        "model_family": family,
                        "verdict": verdict,
                    }
                )

    configured_families = set(model_by_family)
    coverage_failures = 0
    for sample_id in sorted(expected):
        sample_votes = votes.get(sample_id, {})
        if set(sample_votes) != configured_families:
            coverage_failures += 1
            problems.append(
                f"visual judge votes for {sample_id} are not exactly one per configured family"
            )
        if len(sample_votes) < 2:
            problems.append(f"visual judge votes for {sample_id} are fewer than two")
    if vetoes:
        problems.append(f"visual judges vetoed {len(vetoes)} background votes")

    evidence = {
        "passed": not problems,
        "authority": "diagnostic_veto_only",
        "candidate_prediction_visible": False,
        "ground_truth_visible": False,
        "configured_model_families": sorted(model_by_family),
        "teacher_model_families": sorted(teacher_families),
        "models_per_sample_required": len(model_by_family),
        "validation_background_samples_required": len(expected),
        "samples_with_votes": len(votes),
        "coverage_failures": coverage_failures,
        "invalid_vote_count": invalid_vote_count,
        "veto_count": len(vetoes),
        "vetoes": vetoes,
        "truth_relabels": 0,
        "threshold_changes": 0,
    }
    return evidence, problems


def _forbidden_raw_response_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().casefold()
            token = re.sub(r"[^a-z0-9]+", "", normalized)
            if (
                normalized in FORBIDDEN_RAW_RESPONSE_KEYS
                or token in FORBIDDEN_RAW_KEY_TOKENS
                or token.endswith(("truth", "prediction", "confidence"))
                or token.startswith(("request", "image"))
            ):
                found.add(normalized)
            found.update(_forbidden_raw_response_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_forbidden_raw_response_keys(nested))
    elif isinstance(value, str):
        compact = re.sub(r"[^a-z0-9]+", "", value.casefold())
        for token in FORBIDDEN_RAW_VALUE_TOKENS:
            if token in compact:
                found.add(f"value:{token}")
        if len(value) >= 128 and re.fullmatch(r"[A-Za-z0-9+/=\r\n]+", value):
            found.add("value:base64_payload")
    return found


def _families_compatible(left: str, right: str) -> bool:
    return left == right or left.startswith(right) or right.startswith(left)


def _response_families(
    response: Mapping[str, Any], *, location: str
) -> tuple[str, ...]:
    details = response.get("details")
    if not isinstance(details, Mapping):
        raise ValueError(f"{location} is missing details")
    raw_values: list[object] = []
    for field in ("family", "families"):
        value = details.get(field)
        if isinstance(value, str):
            raw_values.append(value)
        elif isinstance(value, list):
            raw_values.extend(value)
        elif value is not None:
            raise ValueError(f"{location}.details.{field} is invalid")
    if not raw_values:
        raise ValueError(f"{location} has no model family")
    families = tuple(
        sorted(
            {
                _canonical_family(value, field=f"{location}.details.family")
                for value in raw_values
            }
        )
    )
    if any("qwen" in family for family in families):
        raise ValueError(f"{location} identifies a Qwen/project teacher family")
    return families


def _audit_tags_response(
    response: object,
    *,
    approved: Mapping[str, Any],
    location: str,
) -> dict[str, object]:
    if not isinstance(response, Mapping):
        raise ValueError(f"{location} must be an object")
    models = response.get("models")
    if not isinstance(models, list):
        raise ValueError(f"{location} is missing models")
    tag = str(approved["ollama_model"])
    matches = [
        value
        for value in models
        if isinstance(value, Mapping)
        and (value.get("name") == tag or value.get("model") == tag)
    ]
    if len(matches) != 1:
        raise ValueError(f"{location} does not contain exactly one approved model tag")
    selected = matches[0]
    if selected.get("name") != tag or selected.get("model") != tag:
        raise ValueError(f"{location} model tag identity is inconsistent")
    raw_digest = _required_text(selected.get("digest"), field=f"{location}.digest")
    digest = raw_digest.casefold()
    if digest.startswith("sha256:"):
        digest = digest.split(":", 1)[1]
    if not _is_sha256(digest):
        raise ValueError(f"{location} digest is invalid")
    if digest != approved["model_manifest_sha256"]:
        raise ValueError(f"{location} digest differs from trusted policy")
    families = _response_families(selected, location=location)
    declared = str(approved["model_family"])
    if not all(_families_compatible(family, declared) for family in families):
        raise ValueError(f"{location} family differs from trusted policy")
    return {
        "sha256": _sha256_bytes(_canonical_json(response)),
        "model_digest": digest,
        "families": list(families),
    }


def _audit_show_response(
    response: object,
    *,
    approved: Mapping[str, Any],
    location: str,
) -> dict[str, object]:
    if not isinstance(response, Mapping):
        raise ValueError(f"{location} must be an object")
    families = _response_families(response, location=location)
    declared = str(approved["model_family"])
    if not all(_families_compatible(family, declared) for family in families):
        raise ValueError(f"{location} family differs from trusted policy")
    capabilities_raw = response.get("capabilities")
    if not isinstance(capabilities_raw, list) or not capabilities_raw:
        raise ValueError(f"{location} capabilities are missing")
    capabilities = sorted(
        {
            _required_text(value, field=f"{location}.capabilities").casefold()
            for value in capabilities_raw
        }
    )
    if "vision" not in capabilities:
        raise ValueError(f"{location} lacks vision capability")
    return {
        "sha256": _sha256_bytes(_canonical_json(response)),
        "families": list(families),
        "capabilities": capabilities,
    }


def audit_independent_visual_report(
    report_path: Path,
    evidence_path: Path,
    validation_background: Sequence[Mapping[str, str]],
    *,
    policy: Mapping[str, Any],
) -> tuple[dict[str, object], list[str]]:
    """Revalidate the independent runner's full report and canonical raw votes."""

    problems: list[str] = []
    if not report_path.is_file():
        return {"passed": False}, ["independent visual judge report is missing"]
    if not evidence_path.is_file():
        return {"passed": False}, ["independent visual judge evidence is missing"]
    report = _load_json(report_path, description="independent visual judge report")
    report_bytes = report_path.read_bytes()
    if report_bytes != _canonical_json(report):
        problems.append("independent visual judge report is not canonical JSON")
    required_report_fields = {
        "schema_version",
        "report_schema",
        "evidence_schema",
        "evidence_schema_version",
        "canonical_json_contract",
        "evidence_pair_contract",
        "evidence_pair_id",
        "evidence_pair_seed",
        "postflight_identity_set",
        "postflight_identity_set_sha256",
        "official_ollama_http",
        "authoritative_evidence",
        "artifact_pair",
        "artifact_role",
        "authority",
        "input_manifest_sha256",
        "prompt_version",
        "prompt_sha256",
        "runner_script_sha256",
        "server_digest_contract",
        "row_count",
        "judge_count",
        "vote_count",
        "expected_vote_count",
        "coverage",
        "candidate_metadata_exposed_to_prompt",
        "raw_response_content_stored",
        "request_content_stored",
        "image_content_stored",
        "evidence_jsonl_sha256",
        "evidence_jsonl_line_count",
        "canonical_raw_response_sha256_by_vote",
        "judges",
        "verdict_counts_by_judge",
        "results_jsonl_sha256",
        "generated_by",
    }
    if set(report) != required_report_fields:
        problems.append("independent visual report fields differ from frozen schema")
    evidence_bytes = evidence_path.read_bytes()
    try:
        evidence_text = evidence_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("independent visual evidence must be UTF-8") from error
    raw_lines = evidence_text.splitlines(keepends=True)
    if not raw_lines or any(not line.strip() for line in raw_lines):
        raise ValueError("independent visual evidence contains blank or no rows")
    votes: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        try:
            vote = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"independent visual evidence line {line_number} is invalid JSON"
            ) from error
        if not isinstance(vote, dict):
            raise ValueError(
                f"independent visual evidence line {line_number} must be an object"
            )
        if raw_line.encode("utf-8") != _canonical_json(vote):
            problems.append(
                f"independent visual evidence line {line_number} is not canonical JSON"
            )
        votes.append(vote)

    visual_policy = policy["visual_judges"]
    policy_judges = {
        str(judge["judge_id"]): judge for judge in visual_policy["judges"]
    }
    expected_samples = {
        row["sample_id"]: {
            "source_sha256": row["source_sha256"],
            "crop_sha256": row["image_sha256"],
        }
        for row in validation_background
    }
    expected_vote_keys = {
        (sample_id, judge_id)
        for sample_id in expected_samples
        for judge_id in policy_judges
    }
    evidence_sha = _sha256_bytes(evidence_bytes)
    exact_report_contract = {
        "schema_version": 1,
        "report_schema": VISUAL_REPORT_SCHEMA,
        "artifact_role": "diagnostic_veto_only_not_promotion_authority",
        "evidence_schema": VISUAL_EVIDENCE_SCHEMA,
        "evidence_schema_version": 1,
        "canonical_json_contract": VISUAL_CANONICAL_JSON_CONTRACT,
        "evidence_pair_contract": VISUAL_EVIDENCE_PAIR_CONTRACT,
        "official_ollama_http": True,
        "authoritative_evidence": True,
        "input_manifest_sha256": visual_policy["input_manifest_sha256"],
        "prompt_version": "independent_background_material_judge.v1",
        "prompt_sha256": visual_policy["prompt_sha256"],
        "runner_script_sha256": visual_policy["runner_script_sha256"],
        "server_digest_contract": VISUAL_SERVER_DIGEST_CONTRACT,
        "row_count": len(expected_samples),
        "judge_count": len(policy_judges),
        "vote_count": len(expected_vote_keys),
        "expected_vote_count": len(expected_vote_keys),
        "candidate_metadata_exposed_to_prompt": False,
        "raw_response_content_stored": True,
        "request_content_stored": False,
        "image_content_stored": False,
        "evidence_jsonl_sha256": evidence_sha,
        "results_jsonl_sha256": evidence_sha,
        "evidence_jsonl_line_count": len(raw_lines),
        "generated_by": "scripts/run_independent_visual_judges.py",
    }
    for field, expected_value in exact_report_contract.items():
        if report.get(field) != expected_value:
            problems.append(f"independent visual report has invalid {field}")
    expected_authority = {
        "promotion_authority": False,
        "ground_truth_authority": False,
        "may_relabel_truth": False,
        "may_tune_thresholds": False,
        "allowed_actions": ["diagnostic", "veto", "request_more_evidence"],
    }
    if report.get("authority") != expected_authority:
        problems.append("independent visual report authority contract is invalid")
    if report.get("coverage") != {"every_judge_exactly_one_vote_per_row": True}:
        problems.append("independent visual report coverage contract is invalid")
    if report.get("artifact_pair") != {
        "complete": True,
        "required_members": ["evidence_jsonl", "report_json"],
        "report_pins_exact_evidence_jsonl": True,
    }:
        problems.append("independent visual report artifact-pair contract is invalid")
    evidence_pair_id = report.get("evidence_pair_id")
    if not _is_sha256(evidence_pair_id):
        problems.append("independent visual report evidence_pair_id is invalid")

    report_judges_raw = report.get("judges")
    report_judges: dict[str, Mapping[str, Any]] = {}
    required_report_judge_fields = {
        "judge_id",
        "model_family",
        "ollama_model",
        "model_manifest_sha256",
        "model_weight_layer_sha256",
        "model_config_sha256",
        "model_config_families",
        "judge_spec_sha256",
        "server_model_digest",
        "server_model_families",
        "server_capabilities",
        "server_tags_response_sha256",
        "server_show_response_sha256",
        "preflight_tags_response",
        "preflight_tags_response_sha256",
        "preflight_show_response",
        "preflight_show_response_sha256",
        "postflight_tags_response",
        "postflight_tags_response_sha256",
        "postflight_show_response",
        "postflight_show_response_sha256",
        "postflight_identity_matches_preflight",
        "prompt_sha256",
        "runner_script_sha256",
    }
    if not isinstance(report_judges_raw, list):
        problems.append("independent visual report judges must be an array")
    else:
        for index, raw_judge in enumerate(report_judges_raw):
            if not isinstance(raw_judge, Mapping):
                problems.append(f"independent visual report judge {index} is invalid")
                continue
            if set(raw_judge) != required_report_judge_fields:
                problems.append(
                    f"independent visual report judge {index} fields differ from frozen schema"
                )
            judge_id = str(raw_judge.get("judge_id", ""))
            if not judge_id or judge_id in report_judges:
                problems.append("independent visual report judge IDs must be unique")
                continue
            report_judges[judge_id] = raw_judge
    if set(report_judges) != set(policy_judges):
        problems.append("independent visual report judges differ from trusted policy")
    parsed_report_identities: dict[str, dict[str, object]] = {}
    for judge_id, approved in policy_judges.items():
        actual = report_judges.get(judge_id)
        if actual is None:
            continue
        expected_identity = {
            "model_family": approved["model_family"],
            "ollama_model": approved["ollama_model"],
            "model_manifest_sha256": approved["model_manifest_sha256"],
            "model_weight_layer_sha256": approved["model_weight_layer_sha256"],
            "model_config_sha256": approved["model_config_sha256"],
            "server_model_digest": approved["server_model_digest"],
            "server_model_families": approved["server_model_families"],
            "prompt_sha256": visual_policy["prompt_sha256"],
            "runner_script_sha256": visual_policy["runner_script_sha256"],
        }
        for field, expected_value in expected_identity.items():
            actual_value = actual.get(field)
            if field == "model_family" and isinstance(actual_value, str):
                actual_value = re.sub(r"[^a-z0-9]+", "", actual_value.casefold())
            if field == "server_model_families" and isinstance(actual_value, list):
                actual_value = [str(value).casefold() for value in actual_value]
            if actual_value != expected_value:
                problems.append(
                    f"independent visual report judge {judge_id} has invalid {field}"
                )
        capabilities = actual.get("server_capabilities")
        if not isinstance(capabilities, list) or "vision" not in {
            str(value).casefold() for value in capabilities
        }:
            problems.append(
                f"independent visual report judge {judge_id} lacks vision capability"
            )
        for field in (
            "server_tags_response_sha256",
            "server_show_response_sha256",
        ):
            if not _is_sha256(actual.get(field)):
                problems.append(
                    f"independent visual report judge {judge_id} has invalid {field}"
                )
        try:
            preflight_tags = _audit_tags_response(
                actual.get("preflight_tags_response"),
                approved=approved,
                location=f"visual report judge {judge_id} preflight /api/tags",
            )
            preflight_show = _audit_show_response(
                actual.get("preflight_show_response"),
                approved=approved,
                location=f"visual report judge {judge_id} preflight /api/show",
            )
            postflight_tags = _audit_tags_response(
                actual.get("postflight_tags_response"),
                approved=approved,
                location=f"visual report judge {judge_id} postflight /api/tags",
            )
            postflight_show = _audit_show_response(
                actual.get("postflight_show_response"),
                approved=approved,
                location=f"visual report judge {judge_id} postflight /api/show",
            )
            for phase, parsed in (
                ("preflight_tags", preflight_tags),
                ("preflight_show", preflight_show),
                ("postflight_tags", postflight_tags),
                ("postflight_show", postflight_show),
            ):
                if actual.get(f"{phase}_response_sha256") != parsed["sha256"]:
                    problems.append(
                        f"independent visual report judge {judge_id} {phase} hash mismatch"
                    )
            if actual.get("server_tags_response_sha256") != preflight_tags["sha256"]:
                problems.append(
                    f"independent visual report judge {judge_id} server tags hash mismatch"
                )
            if actual.get("server_show_response_sha256") != preflight_show["sha256"]:
                problems.append(
                    f"independent visual report judge {judge_id} server show hash mismatch"
                )
            if preflight_tags["model_digest"] != postflight_tags["model_digest"]:
                problems.append(
                    f"independent visual report judge {judge_id} model digest changed"
                )
            if preflight_tags["families"] != postflight_tags["families"]:
                problems.append(
                    f"independent visual report judge {judge_id} tag family changed"
                )
            if preflight_show["families"] != postflight_show["families"]:
                problems.append(
                    f"independent visual report judge {judge_id} show family changed"
                )
            if preflight_show["capabilities"] != postflight_show["capabilities"]:
                problems.append(
                    f"independent visual report judge {judge_id} capabilities changed"
                )
            combined_families = sorted(
                set(preflight_tags["families"]) | set(preflight_show["families"])
            )
            if combined_families != sorted(approved["server_model_families"]):
                problems.append(
                    f"independent visual report judge {judge_id} server families differ from policy"
                )
            if actual.get("server_model_families") != combined_families:
                problems.append(
                    f"independent visual report judge {judge_id} summarized families are invalid"
                )
            if actual.get("server_capabilities") != preflight_show["capabilities"]:
                problems.append(
                    f"independent visual report judge {judge_id} summarized capabilities are invalid"
                )
            if actual.get("postflight_identity_matches_preflight") is not True:
                problems.append(
                    f"independent visual report judge {judge_id} lacks postflight identity proof"
                )
            parsed_report_identities[judge_id] = {
                "preflight_tag_families": preflight_tags["families"],
                "preflight_capabilities": preflight_show["capabilities"],
                "model_digest": preflight_tags["model_digest"],
            }
        except ValueError as error:
            problems.append(str(error))

    expected_postflight_identity_set = [
        {
            "judge_id": raw_judge.get("judge_id"),
            "postflight_tags_response_sha256": raw_judge.get(
                "postflight_tags_response_sha256"
            ),
            "postflight_show_response_sha256": raw_judge.get(
                "postflight_show_response_sha256"
            ),
        }
        for raw_judge in (
            report_judges_raw if isinstance(report_judges_raw, list) else []
        )
        if isinstance(raw_judge, Mapping)
    ]
    expected_postflight_identity_sha = _sha256_bytes(
        _canonical_json(expected_postflight_identity_set)
    )
    if report.get("postflight_identity_set") != expected_postflight_identity_set:
        problems.append("independent visual report postflight identity set is invalid")
    if (
        report.get("postflight_identity_set_sha256")
        != expected_postflight_identity_sha
    ):
        problems.append("independent visual report postflight identity hash mismatch")

    expected_pair_seed = {
        "evidence_schema": VISUAL_EVIDENCE_SCHEMA,
        "evidence_schema_version": 1,
        "evidence_pair_contract": VISUAL_EVIDENCE_PAIR_CONTRACT,
        "input_manifest_sha256": visual_policy["input_manifest_sha256"],
        "prompt_sha256": visual_policy["prompt_sha256"],
        "runner_script_sha256": visual_policy["runner_script_sha256"],
        "official_ollama_http": True,
        "authoritative_evidence": True,
        "judges": [],
        "postflight_identity_set": expected_postflight_identity_set,
    }
    for raw_judge in report_judges_raw if isinstance(report_judges_raw, list) else []:
        if not isinstance(raw_judge, Mapping):
            continue
        judge_spec_sha = raw_judge.get("judge_spec_sha256")
        if not _is_sha256(judge_spec_sha):
            problems.append("independent visual report judge_spec_sha256 is invalid")
        expected_pair_seed["judges"].append(
            {
                "judge_id": raw_judge.get("judge_id"),
                "judge_spec_sha256": judge_spec_sha,
                "model_manifest_sha256": raw_judge.get(
                    "model_manifest_sha256"
                ),
                "model_weight_layer_sha256": raw_judge.get(
                    "model_weight_layer_sha256"
                ),
                "model_config_sha256": raw_judge.get("model_config_sha256"),
                "preflight_tags_response_sha256": raw_judge.get(
                    "preflight_tags_response_sha256"
                ),
                "preflight_show_response_sha256": raw_judge.get(
                    "preflight_show_response_sha256"
                ),
            }
        )
    if report.get("evidence_pair_seed") != expected_pair_seed:
        problems.append("independent visual report evidence-pair seed is invalid")
    expected_pair_id = _sha256_bytes(_canonical_json(expected_pair_seed))
    if evidence_pair_id != expected_pair_id:
        problems.append("independent visual report evidence-pair ID mismatch")

    required_vote_fields = {
        "schema_version",
        "vote_schema",
        "evidence_schema",
        "evidence_schema_version",
        "canonical_json_contract",
        "evidence_pair_contract",
        "evidence_pair_id",
        "official_ollama_http",
        "authoritative_evidence",
        "postflight_identity_set_sha256",
        "judge_id",
        "model_family",
        "ollama_model",
        "sample_id",
        "verdict",
        "prompt_sha256",
        "runner_script_sha256",
        "model_manifest_sha256",
        "model_weight_layer_sha256",
        "model_config_sha256",
        "server_model_digest",
        "server_model_families",
        "server_capabilities",
        "server_digest_contract",
        "server_tags_response_sha256",
        "server_show_response_sha256",
        "postchat_tags_response",
        "postchat_tags_response_sha256",
        "postchat_server_model_digest",
        "postchat_server_model_families",
        "canonical_raw_response",
        "canonical_raw_response_sha256",
        "source_sha256",
        "crop_sha256",
        "vote_binding_sha256",
    }
    seen: set[tuple[str, str]] = set()
    vetoes: list[dict[str, str]] = []
    raw_hash_inventory: list[dict[str, str]] = []
    verdict_counts: dict[str, dict[str, int]] = {
        judge_id: {verdict: 0 for verdict in ("ambiguous", "background", "material")}
        for judge_id in policy_judges
    }
    for line_number, vote in enumerate(votes, start=1):
        location = f"independent visual evidence line {line_number}"
        if set(vote) != required_vote_fields:
            problems.append(f"{location} fields differ from the frozen vote schema")
        vote_payload = {
            key: value for key, value in vote.items() if key != "vote_binding_sha256"
        }
        if vote.get("vote_binding_sha256") != _sha256_bytes(
            _canonical_json(vote_payload)
        ):
            problems.append(f"{location} vote binding SHA-256 mismatch")
        fixed_vote_contract = {
            "schema_version": 1,
            "vote_schema": VISUAL_VOTE_SCHEMA,
            "evidence_schema": VISUAL_EVIDENCE_SCHEMA,
            "evidence_schema_version": 1,
            "canonical_json_contract": VISUAL_CANONICAL_JSON_CONTRACT,
            "evidence_pair_contract": VISUAL_EVIDENCE_PAIR_CONTRACT,
            "evidence_pair_id": evidence_pair_id,
            "official_ollama_http": True,
            "authoritative_evidence": True,
            "postflight_identity_set_sha256": expected_postflight_identity_sha,
            "prompt_sha256": visual_policy["prompt_sha256"],
            "runner_script_sha256": visual_policy["runner_script_sha256"],
            "server_digest_contract": VISUAL_SERVER_DIGEST_CONTRACT,
        }
        for field, expected_value in fixed_vote_contract.items():
            if vote.get(field) != expected_value:
                problems.append(f"{location} has invalid {field}")
        sample_id = str(vote.get("sample_id", ""))
        judge_id = str(vote.get("judge_id", ""))
        key = (sample_id, judge_id)
        if key in seen:
            problems.append(f"{location} duplicates sample/judge vote")
        seen.add(key)
        expected_sample = expected_samples.get(sample_id)
        approved = policy_judges.get(judge_id)
        if expected_sample is None or approved is None:
            problems.append(f"{location} references an unapproved sample or judge")
            continue
        for field in ("source_sha256", "crop_sha256"):
            if vote.get(field) != expected_sample[field]:
                problems.append(f"{location} has invalid {field}")
        expected_vote_identity = {
            "model_family": approved["model_family"],
            "ollama_model": approved["ollama_model"],
            "model_manifest_sha256": approved["model_manifest_sha256"],
            "model_weight_layer_sha256": approved["model_weight_layer_sha256"],
            "model_config_sha256": approved["model_config_sha256"],
            "server_model_digest": approved["server_model_digest"],
            "server_model_families": approved["server_model_families"],
        }
        for field, expected_value in expected_vote_identity.items():
            actual_value = vote.get(field)
            if field == "model_family" and isinstance(actual_value, str):
                actual_value = re.sub(r"[^a-z0-9]+", "", actual_value.casefold())
            if field == "server_model_families" and isinstance(actual_value, list):
                actual_value = [str(value).casefold() for value in actual_value]
            if actual_value != expected_value:
                problems.append(f"{location} has invalid {field}")
        capabilities = vote.get("server_capabilities")
        if not isinstance(capabilities, list) or "vision" not in {
            str(value).casefold() for value in capabilities
        }:
            problems.append(f"{location} lacks vision capability")
        report_judge = report_judges.get(judge_id)
        if report_judge is not None:
            for field in (
                "server_capabilities",
                "server_tags_response_sha256",
                "server_show_response_sha256",
            ):
                if vote.get(field) != report_judge.get(field):
                    problems.append(
                        f"{location} {field} differs from the visual report"
                    )
        try:
            postchat_tags = _audit_tags_response(
                vote.get("postchat_tags_response"),
                approved=approved,
                location=f"{location} post-chat /api/tags",
            )
            if vote.get("postchat_tags_response_sha256") != postchat_tags["sha256"]:
                problems.append(f"{location} post-chat tags hash mismatch")
            if vote.get("postchat_server_model_digest") != postchat_tags[
                "model_digest"
            ]:
                problems.append(f"{location} post-chat model digest mismatch")
            if vote.get("postchat_server_model_families") != postchat_tags[
                "families"
            ]:
                problems.append(f"{location} post-chat model families mismatch")
            report_identity = parsed_report_identities.get(judge_id)
            if report_identity is None:
                problems.append(f"{location} has no validated report identity")
            else:
                if postchat_tags["model_digest"] != report_identity["model_digest"]:
                    problems.append(f"{location} model digest changed after chat")
                if postchat_tags["families"] != report_identity[
                    "preflight_tag_families"
                ]:
                    problems.append(f"{location} model family changed after chat")
        except ValueError as error:
            problems.append(str(error))
        raw_response = vote.get("canonical_raw_response")
        if not isinstance(raw_response, Mapping):
            problems.append(f"{location} canonical raw response is not an object")
            continue
        forbidden = sorted(_forbidden_raw_response_keys(raw_response))
        if forbidden:
            problems.append(
                f"{location} canonical raw response exposes forbidden fields {forbidden}"
            )
        raw_sha = _sha256_bytes(_canonical_json(raw_response))
        if vote.get("canonical_raw_response_sha256") != raw_sha:
            problems.append(f"{location} canonical raw response SHA-256 mismatch")
        verdict = str(vote.get("verdict", "")).casefold()
        if verdict not in {"background", "material", "ambiguous"}:
            problems.append(f"{location} verdict is invalid")
            continue
        if raw_response.get("model") != approved["ollama_model"]:
            problems.append(f"{location} raw response model differs from policy")
        message = raw_response.get("message")
        content_value: object = None
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                try:
                    content_value = json.loads(content)
                except json.JSONDecodeError:
                    content_value = None
        if content_value != {"verdict": verdict}:
            problems.append(f"{location} raw response content differs from verdict")
        verdict_counts[judge_id][verdict] += 1
        raw_hash_inventory.append(
            {
                "sample_id": sample_id,
                "judge_id": judge_id,
                "canonical_raw_response_sha256": raw_sha,
            }
        )
        if verdict != "background":
            vetoes.append(
                {"sample_id": sample_id, "judge_id": judge_id, "verdict": verdict}
            )
    if seen != expected_vote_keys or len(votes) != len(expected_vote_keys):
        problems.append("independent visual evidence vote coverage is not exact")
    if report.get("canonical_raw_response_sha256_by_vote") != raw_hash_inventory:
        problems.append("independent visual report raw response inventory mismatch")
    if report.get("verdict_counts_by_judge") != verdict_counts:
        problems.append("independent visual report verdict counts mismatch")
    if vetoes:
        problems.append(f"independent visual judges vetoed {len(vetoes)} votes")
    evidence = {
        "passed": not problems,
        "authority": "diagnostic_veto_only",
        "report_sha256": _sha256_bytes(report_bytes),
        "evidence_sha256": evidence_sha,
        "vote_count": len(votes),
        "expected_vote_count": len(expected_vote_keys),
        "judge_count": len(policy_judges),
        "background_sample_count": len(expected_samples),
        "veto_count": len(vetoes),
        "vetoes": vetoes,
        "truth_relabels": 0,
        "threshold_changes": 0,
    }
    return evidence, problems


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as file:
            file.write(content)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}") from error


def _run_independent_runtime_replay(
    *,
    manifest_paths: Sequence[Path],
    model_path: Path,
    metadata_path: Path,
    inference_spec_path: Path,
    session_factory: Callable[[Path], Any] | None,
    validation_image_snapshots: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Actually execute ONNX and return a transient, hash-bound replay summary."""

    with tempfile.TemporaryDirectory(prefix="v4-runtime-replay-") as raw_directory:
        directory = Path(raw_directory)
        predictions_path = directory / "predictions.jsonl"
        attestation_path = directory / "attestation.json"
        attestation = replay_validation(
            manifest_paths=manifest_paths,
            verifier_onnx=model_path,
            verifier_metadata=metadata_path,
            inference_spec=inference_spec_path,
            output_jsonl=predictions_path,
            output_attestation=attestation_path,
            session_factory=session_factory,
            validation_image_snapshots=validation_image_snapshots,
        )
        predictions_sha256 = _sha256_file(predictions_path)
        if attestation.get("predictions_sha256") != predictions_sha256:
            raise ValueError("runtime replay attestation prediction hash mismatch")
        custom_factory_used = session_factory is not None
        if attestation.get("custom_session_factory_used") is not custom_factory_used:
            raise ValueError("runtime replay custom-session declaration mismatch")
        expected_runtime_hashes = {
            "model": _sha256_file(model_path),
            "metadata": _sha256_file(metadata_path),
            "spec": _sha256_file(inference_spec_path),
        }
        if attestation.get("runtime_artifact_hashes") != expected_runtime_hashes:
            raise ValueError("runtime replay artifact hashes differ from supplied files")
        if attestation.get("runtime_artifact_hashes_match_snapshots") is not True:
            raise ValueError("runtime replay did not verify artifact snapshots")
        return {
            "passed": True,
            "actual_onnx_inference": not custom_factory_used,
            "authoritative": not custom_factory_used,
            "custom_session_factory_used": custom_factory_used,
            "predictions_sha256": predictions_sha256,
            "prediction_count": attestation.get("prediction_count"),
            "model_sha256": attestation.get("model_sha256"),
            "runtime_artifact_hashes": attestation.get("runtime_artifact_hashes"),
            "metrics": attestation.get("metrics"),
        }


def evaluate_v4_candidate(
    *,
    metadata_path: Path,
    manifest_paths: Sequence[Path],
    candidate_onnx_path: Path,
    inference_spec_path: Path,
    replay_predictions_path: Path,
    replay_attestation_path: Path,
    baseline_metadata_path: Path,
    baseline_onnx_path: Path,
    baseline_replay_predictions_path: Path,
    baseline_replay_attestation_path: Path,
    trusted_policy_path: Path,
    visual_judge_report_path: Path,
    visual_judge_evidence_path: Path,
    output_dir: Path,
    thresholds: GateThresholds = GateThresholds(),
    report_name: str = REPORT_NAME,
    ready_marker_name: str = READY_MARKER_NAME,
    replay_session_factory: Callable[[Path], Any] | None = None,
) -> dict[str, object]:
    """Evaluate all fixed gates, write one immutable report, and mark only passes."""

    thresholds.validate()
    original_trusted_policy_path = trusted_policy_path
    trust_root_evidence = _audit_trusted_policy_trust_root(
        original_trusted_policy_path
    )
    for field, name in (
        ("report_name", report_name),
        ("ready_marker_name", ready_marker_name),
    ):
        if Path(name).name != name or Path(name).is_absolute():
            raise ValueError(f"{field} must be a basename")
    resolved_output = output_dir.resolve(strict=False)
    report_path = output_dir / report_name
    ready_path = output_dir / ready_marker_name
    if (
        report_path.resolve(strict=False).parent != resolved_output
        or ready_path.resolve(strict=False).parent != resolved_output
    ):
        raise ValueError("report and ready marker must stay inside output_dir")
    if report_path.resolve(strict=False) == ready_path.resolve(strict=False):
        raise ValueError("report and ready marker paths must differ")
    for path in (report_path, ready_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")

    problems: list[str] = []
    snapshot_holder = tempfile.TemporaryDirectory(prefix="v4-judge-inputs-")
    snapshot_root = Path(snapshot_holder.name)
    artifacts: list[dict[str, object]] = []
    snapshot_by_kind: dict[str, Path] = {}
    fixed_inputs = (
        (
            metadata_path,
            "candidate_metadata",
            snapshot_root / "candidate" / metadata_path.name,
        ),
        (
            candidate_onnx_path,
            "candidate_onnx",
            snapshot_root / "candidate" / candidate_onnx_path.name,
        ),
        (
            inference_spec_path,
            "inference_spec",
            snapshot_root / "shared" / inference_spec_path.name,
        ),
        (
            replay_predictions_path,
            "candidate_replay_predictions",
            snapshot_root / "candidate" / replay_predictions_path.name,
        ),
        (
            replay_attestation_path,
            "candidate_replay_attestation",
            snapshot_root / "candidate" / replay_attestation_path.name,
        ),
        (
            baseline_metadata_path,
            "baseline_metadata",
            snapshot_root / "baseline" / baseline_metadata_path.name,
        ),
        (
            baseline_onnx_path,
            "baseline_onnx",
            snapshot_root / "baseline" / baseline_onnx_path.name,
        ),
        (
            baseline_replay_predictions_path,
            "baseline_replay_predictions",
            snapshot_root / "baseline" / baseline_replay_predictions_path.name,
        ),
        (
            baseline_replay_attestation_path,
            "baseline_replay_attestation",
            snapshot_root / "baseline" / baseline_replay_attestation_path.name,
        ),
        (
            trusted_policy_path,
            "trusted_policy",
            snapshot_root / "policy" / trusted_policy_path.name,
        ),
        (
            visual_judge_report_path,
            "visual_judge_report",
            snapshot_root / "visual" / visual_judge_report_path.name,
        ),
        (
            visual_judge_evidence_path,
            "visual_judge_evidence",
            snapshot_root / "visual" / visual_judge_evidence_path.name,
        ),
    )
    for original, kind, destination in fixed_inputs:
        artifact, snapshot = _snapshot_artifact(
            original, kind=kind, destination=destination
        )
        artifacts.append(artifact)
        snapshot_by_kind[kind] = snapshot

    original_manifest_paths = [Path(path) for path in manifest_paths]
    snapshot_manifest_paths: list[Path] = []
    for index, original in enumerate(original_manifest_paths):
        artifact, snapshot = _snapshot_artifact(
            original,
            kind="strict_manifest",
            destination=(
                snapshot_root / "manifests" / str(index) / original.name
            ),
        )
        artifacts.append(artifact)
        snapshot_manifest_paths.append(snapshot)
    policy_artifact = next(
        artifact for artifact in artifacts if artifact["kind"] == "trusted_policy"
    )
    trust_root_evidence = _audit_trusted_policy_trust_root(
        original_trusted_policy_path,
        actual_sha256=policy_artifact["sha256"],
    )
    validation_image_snapshots: dict[str, Path] = {}
    validation_image_inventory: list[dict[str, object]] = []
    try:
        (
            validation_image_snapshots,
            validation_image_inventory,
        ) = _snapshot_validation_images(
            original_manifest_paths,
            snapshot_manifest_paths,
            snapshot_root=snapshot_root,
        )
    except (OSError, ValueError) as error:
        problems.append(f"validation image snapshot failed: {error}")

    metadata_path = snapshot_by_kind["candidate_metadata"]
    candidate_onnx_path = snapshot_by_kind["candidate_onnx"]
    inference_spec_path = snapshot_by_kind["inference_spec"]
    replay_predictions_path = snapshot_by_kind["candidate_replay_predictions"]
    replay_attestation_path = snapshot_by_kind["candidate_replay_attestation"]
    baseline_metadata_path = snapshot_by_kind["baseline_metadata"]
    baseline_onnx_path = snapshot_by_kind["baseline_onnx"]
    baseline_replay_predictions_path = snapshot_by_kind[
        "baseline_replay_predictions"
    ]
    baseline_replay_attestation_path = snapshot_by_kind[
        "baseline_replay_attestation"
    ]
    trusted_policy_path = snapshot_by_kind["trusted_policy"]
    visual_judge_report_path = snapshot_by_kind["visual_judge_report"]
    visual_judge_evidence_path = snapshot_by_kind["visual_judge_evidence"]
    manifest_paths = snapshot_manifest_paths

    metric_evidence: dict[str, object] = {"passed": False}
    lineage_evidence: dict[str, object] = {"passed": False}
    candidate_replay_evidence: dict[str, object] = {"passed": False}
    baseline_replay_evidence: dict[str, object] = {"passed": False}
    visual_evidence: dict[str, object] = {"passed": False}
    trusted_policy_evidence: dict[str, object] = {"passed": False}
    metadata: Mapping[str, Any] = {}
    baseline_metadata: Mapping[str, Any] = {}
    baseline_identity_evidence: dict[str, object] = {"passed": False}
    trusted_policy: Mapping[str, Any] = {}
    trusted_policy_sha256 = ""

    try:
        trusted_policy, trusted_policy_sha256 = load_trusted_policy(
            trusted_policy_path
        )
        if policy_artifact["sha256"] != trusted_policy_sha256:
            problems.append("trusted policy changed while it was being loaded")
    except (OSError, ValueError) as error:
        problems.append(f"trusted policy audit failed: {error}")
    if replay_session_factory is not None:
        problems.append(
            "custom replay session factory is test-only and cannot create a ready marker"
        )

    try:
        metadata = _load_json(metadata_path, description="candidate metadata")
        baseline_metadata = _load_json(
            baseline_metadata_path, description="baseline metadata"
        )
    except (OSError, ValueError) as error:
        problems.append(f"metadata audit failed: {error}")

    candidate_model_artifact = next(
        artifact for artifact in artifacts if artifact["kind"] == "candidate_onnx"
    )
    baseline_model_artifact = next(
        artifact for artifact in artifacts if artifact["kind"] == "baseline_onnx"
    )
    if (
        candidate_model_artifact["exists"] is True
        and baseline_model_artifact["exists"] is True
    ):
        models_are_distinct = (
            candidate_model_artifact["sha256"] != baseline_model_artifact["sha256"]
        )
        baseline_identity_evidence = {
            "passed": models_are_distinct,
            "candidate_model_sha256": candidate_model_artifact["sha256"],
            "baseline_model_sha256": baseline_model_artifact["sha256"],
            "models_are_distinct": models_are_distinct,
            "baseline_replay_required": True,
        }
        if not models_are_distinct:
            problems.append("baseline model must be distinct from the candidate model")

    validation_background: list[dict[str, str]] = []
    validation_samples: list[dict[str, str]] = []
    manifest_artifacts = [
        artifact for artifact in artifacts if artifact["kind"] == "strict_manifest"
    ]
    try:
        (
            lineage_evidence,
            validation_background,
            validation_samples,
            lineage_problems,
        ) = audit_lineage(metadata, manifest_paths, manifest_artifacts=manifest_artifacts)
        problems.extend(lineage_problems)
    except (OSError, ValueError) as error:
        problems.append(f"lineage audit failed: {error}")

    if trusted_policy:
        try:
            trusted_policy_evidence, trusted_policy_problems = (
                audit_trusted_policy_bindings(
                    trusted_policy,
                    policy_sha256=trusted_policy_sha256,
                    baseline_model_path=baseline_onnx_path,
                    baseline_metadata_path=baseline_metadata_path,
                    manifest_artifacts=manifest_artifacts,
                    calculated_lineage_sha256=str(
                        lineage_evidence.get("calculated_lineage_sha256", "")
                    ),
                )
            )
            trusted_policy_evidence.update(
                {
                    "trust_root_method": trust_root_evidence[
                        "trust_root_method"
                    ],
                    "verified": trust_root_evidence["verified"],
                    "repository_relative_policy_path": trust_root_evidence[
                        "repository_relative_policy_path"
                    ],
                    "approved_policy_sha256": trust_root_evidence[
                        "approved_policy_sha256"
                    ],
                }
            )
            problems.extend(trusted_policy_problems)
        except (OSError, ValueError, KeyError) as error:
            problems.append(f"trusted policy binding failed: {error}")

    candidate_replayed_metrics: Mapping[str, Any] = {}
    baseline_replayed_metrics: Mapping[str, Any] = {}
    candidate_runtime_replay: dict[str, object] = {"passed": False}
    baseline_runtime_replay: dict[str, object] = {"passed": False}
    try:
        candidate_runtime_replay = _run_independent_runtime_replay(
            manifest_paths=manifest_paths,
            model_path=candidate_onnx_path,
            metadata_path=metadata_path,
            inference_spec_path=inference_spec_path,
            session_factory=replay_session_factory,
            validation_image_snapshots=validation_image_snapshots,
        )
    except Exception as error:
        problems.append(f"candidate runtime ONNX replay failed: {error}")
    try:
        baseline_runtime_replay = _run_independent_runtime_replay(
            manifest_paths=manifest_paths,
            model_path=baseline_onnx_path,
            metadata_path=baseline_metadata_path,
            inference_spec_path=inference_spec_path,
            session_factory=replay_session_factory,
            validation_image_snapshots=validation_image_snapshots,
        )
    except Exception as error:
        problems.append(f"baseline runtime ONNX replay failed: {error}")
    try:
        (
            candidate_replay_evidence,
            candidate_replayed_metrics,
            replay_problems,
        ) = audit_replay_evidence(
            predictions_path=replay_predictions_path,
            attestation_path=replay_attestation_path,
            model_path=candidate_onnx_path,
            metadata_path=metadata_path,
            inference_spec_path=inference_spec_path,
            manifest_artifacts=manifest_artifacts,
            expected_validation=validation_samples,
            calculated_lineage_sha256=str(
                lineage_evidence.get("calculated_lineage_sha256", "")
            ),
            metadata=metadata,
            runtime_replay_predictions_sha256=str(
                candidate_runtime_replay.get("predictions_sha256", "")
            ),
        )
        candidate_replay_evidence["runtime_replay"] = candidate_runtime_replay
        problems.extend(f"candidate replay: {problem}" for problem in replay_problems)
    except (OSError, ValueError) as error:
        problems.append(f"candidate replay audit failed: {error}")
    try:
        (
            baseline_replay_evidence,
            baseline_replayed_metrics,
            baseline_replay_problems,
        ) = audit_replay_evidence(
            predictions_path=baseline_replay_predictions_path,
            attestation_path=baseline_replay_attestation_path,
            model_path=baseline_onnx_path,
            metadata_path=baseline_metadata_path,
            inference_spec_path=inference_spec_path,
            manifest_artifacts=manifest_artifacts,
            expected_validation=validation_samples,
            calculated_lineage_sha256=str(
                lineage_evidence.get("calculated_lineage_sha256", "")
            ),
            metadata=baseline_metadata,
            runtime_replay_predictions_sha256=str(
                baseline_runtime_replay.get("predictions_sha256", "")
            ),
        )
        baseline_replay_evidence["runtime_replay"] = baseline_runtime_replay
        problems.extend(
            f"baseline replay: {problem}" for problem in baseline_replay_problems
        )
    except (OSError, ValueError) as error:
        problems.append(f"baseline replay audit failed: {error}")

    try:
        metric_evidence, metric_problems = audit_candidate_metrics(
            metadata,
            baseline_metadata,
            thresholds,
            replayed_validation=candidate_replayed_metrics,
            baseline_replayed_validation=baseline_replayed_metrics,
        )
        problems.extend(metric_problems)
        support_matches = (
            metric_evidence["validation"]["objectness"]["support"]
            == lineage_evidence.get("validation_objectness_support")
            and metric_evidence["validation"]["material"]["support"]
            == lineage_evidence.get("validation_material_support")
        )
        lineage_evidence["metric_support_matches"] = support_matches
        if not support_matches:
            lineage_evidence["passed"] = False
            problems.append("replayed metric support differs from strict manifests")
    except (OSError, ValueError, KeyError) as error:
        problems.append(f"metric audit failed: {error}")

    if trusted_policy:
        try:
            visual_evidence, visual_problems = audit_independent_visual_report(
                visual_judge_report_path,
                visual_judge_evidence_path,
                validation_background,
                policy=trusted_policy,
            )
            problems.extend(visual_problems)
        except (OSError, ValueError, KeyError) as error:
            problems.append(f"visual judge audit failed: {error}")

    missing_artifacts = [
        artifact["kind"] for artifact in artifacts if artifact["exists"] is False
    ]
    for kind in missing_artifacts:
        problem = f"required input artifact is missing: {kind}"
        if problem not in problems:
            problems.append(problem)

    for artifact in artifacts:
        if artifact["exists"] is not True:
            continue
        artifact_path = Path(str(artifact["path"]))
        try:
            unchanged = (
                artifact_path.is_file()
                and _sha256_file(artifact_path) == artifact["sha256"]
            )
        except OSError:
            unchanged = False
        if not unchanged:
            problems.append(
                f"input artifact changed during judge evaluation: {artifact['kind']}"
            )
    for image in validation_image_inventory:
        original_image = Path(str(image["path"]))
        try:
            unchanged = (
                original_image.is_file()
                and _sha256_file(original_image) == image["sha256"]
            )
        except OSError:
            unchanged = False
        if not unchanged:
            problems.append(
                "validation image changed during judge evaluation: "
                f"{original_image}"
            )

    # Stable, concise evidence: preserve first occurrence and never hide a failure.
    problems = list(dict.fromkeys(problems))
    passed = not problems
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "status": "passed" if passed else "rejected",
        "candidate_ready": passed,
        "trusted_policy_sha256": trusted_policy_sha256 or None,
        "trust_root_method": trust_root_evidence["trust_root_method"],
        "trust_root_verified": trust_root_evidence["verified"],
        "production_deployment_authorized": False,
        "requires_independent_blind_hardware_evidence": True,
        "authority_contract": {
            "visual_judges": "diagnostic_veto_only",
            "truth_relabeling_allowed": False,
            "threshold_tuning_allowed": False,
            "production_promotion_allowed": False,
        },
        "input_artifacts": artifacts,
        "input_snapshot": {
            "contract": "single_read_immutable_temp_copy_with_final_rehash.v1",
            "fixed_artifact_count": len(artifacts),
            "validation_image_count": len(validation_image_inventory),
            "validation_image_inventory_sha256": _sha256_bytes(
                _canonical_json(validation_image_inventory)
            ),
        },
        "gates": {
            "candidate_replay": candidate_replay_evidence,
            "baseline_replay": baseline_replay_evidence,
            "baseline_identity": baseline_identity_evidence,
            "trusted_policy": trusted_policy_evidence,
            "candidate_metrics": metric_evidence,
            "strict_lineage": lineage_evidence,
            "visual_judges": visual_evidence,
        },
        "problems": problems,
    }
    report["attestation_sha256"] = _sha256_bytes(_canonical_json(report))
    report_bytes = _canonical_json(report, pretty=True)
    _write_exclusive(report_path, report_bytes)
    snapshot_holder.cleanup()

    if passed:
        marker = {
            "schema_version": SCHEMA_VERSION,
            "status": "offline_judge_gate_passed",
            "report": str(report_path.resolve()),
            "report_sha256": _sha256_bytes(report_bytes),
            "trusted_policy_sha256": trusted_policy_sha256,
            "trust_root_method": trust_root_evidence["trust_root_method"],
            "trust_root_verified": trust_root_evidence["verified"],
            "production_deployment_authorized": False,
            "requires_independent_blind_hardware_evidence": True,
        }
        _write_exclusive(ready_path, _canonical_json(marker, pretty=True))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--manifest", required=True, action="append", type=Path)
    parser.add_argument("--candidate-onnx", required=True, type=Path)
    parser.add_argument("--inference-spec", required=True, type=Path)
    parser.add_argument("--replay-predictions", required=True, type=Path)
    parser.add_argument("--replay-attestation", required=True, type=Path)
    parser.add_argument("--baseline-metadata", required=True, type=Path)
    parser.add_argument("--baseline-onnx", required=True, type=Path)
    parser.add_argument("--baseline-replay-predictions", required=True, type=Path)
    parser.add_argument("--baseline-replay-attestation", required=True, type=Path)
    parser.add_argument("--trusted-policy", required=True, type=Path)
    parser.add_argument("--visual-judge-report", required=True, type=Path)
    parser.add_argument("--visual-judge-evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-name", default=REPORT_NAME)
    parser.add_argument("--ready-marker-name", default=READY_MARKER_NAME)
    parser.add_argument("--min-background-support", type=int, default=200)
    parser.add_argument("--min-background-recall", type=float, default=0.90)
    parser.add_argument(
        "--min-material-objectness-recall", type=float, default=0.95
    )
    parser.add_argument(
        "--min-objectness-balanced-accuracy", type=float, default=0.925
    )
    parser.add_argument("--min-material-balanced-accuracy", type=float, default=0.90)
    parser.add_argument("--min-each-material-recall", type=float, default=0.85)
    parser.add_argument("--max-baseline-recall-drop", type=float, default=0.01)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    thresholds = GateThresholds(
        min_background_support=args.min_background_support,
        min_background_recall=args.min_background_recall,
        min_material_objectness_recall=args.min_material_objectness_recall,
        min_objectness_balanced_accuracy=args.min_objectness_balanced_accuracy,
        min_material_balanced_accuracy=args.min_material_balanced_accuracy,
        min_each_material_recall=args.min_each_material_recall,
        max_baseline_recall_drop=args.max_baseline_recall_drop,
    )
    try:
        report = evaluate_v4_candidate(
            metadata_path=args.metadata,
            manifest_paths=args.manifest,
            candidate_onnx_path=args.candidate_onnx,
            inference_spec_path=args.inference_spec,
            replay_predictions_path=args.replay_predictions,
            replay_attestation_path=args.replay_attestation,
            baseline_metadata_path=args.baseline_metadata,
            baseline_onnx_path=args.baseline_onnx,
            baseline_replay_predictions_path=args.baseline_replay_predictions,
            baseline_replay_attestation_path=args.baseline_replay_attestation,
            trusted_policy_path=args.trusted_policy,
            visual_judge_report_path=args.visual_judge_report,
            visual_judge_evidence_path=args.visual_judge_evidence,
            output_dir=args.output_dir,
            thresholds=thresholds,
            report_name=args.report_name,
            ready_marker_name=args.ready_marker_name,
        )
    except (OSError, ValueError) as error:
        print(f"judge error: {error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0 if report["candidate_ready"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
