"""Run independent visual judges on v4 validation background crops.

This command is deliberately *diagnostic and veto only*.  A visual judge vote
is never ground truth and can never authorize candidate promotion, relabel a
sample, or tune a threshold.  Each judge sees only two images, in a fixed
order: the original source frame and the runtime-shaped crop.  Candidate
predictions, labels, truth fields, and confidence values are never included in
the Ollama prompt or request payload.

Judge definitions are supplied as JSON files.  Every definition declares the
SHA-256 of a separate, real model-manifest file.  The runner binds that local
manifest and its immutable config blob to the model actually served by Ollama
through ``/api/tags`` and ``/api/show`` before sending an image.  At least two
distinct non-Qwen model families (and distinct model-manifest bytes) are
required.  The canonical chat response object is retained as gate evidence,
but request bodies and image bytes are never written to the artifacts.  The
result and report destinations are exclusive-create artifacts and are never
overwritten.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
VOTE_SCHEMA = "independent_visual_judge_vote.v1"
REPORT_SCHEMA = "independent_visual_judge_report.v1"
EVIDENCE_SCHEMA = "independent_visual_judge_evidence.v1"
EVIDENCE_SCHEMA_VERSION = 1
CANONICAL_JSON_CONTRACT = (
    "utf8_sorted_keys_compact_separators_trailing_newline.v1"
)
EVIDENCE_PAIR_CONTRACT = (
    "votes_share_pair_id_and_report_pins_exact_jsonl_sha256.v1"
)
PROMPT_VERSION = "independent_background_material_judge.v1"
SERVER_DIGEST_CONTRACT = (
    "ollama_api_tags_digest_equals_sha256_of_local_oci_tag_manifest_bytes.v1"
)
PROMPT = (
    "You are an independent visual safety judge. Two images are attached in "
    "this exact order: (1) the full original scene and (2) the runtime proposal "
    "crop. Decide whether the crop contains a real disposal or recyclable "
    "material object, rather than only empty-scene background, imaging artifact, "
    "reflection, shadow, or fixture. Return exactly one JSON object matching the "
    "provided schema. Use verdict=background when it is only background/artifact, "
    "verdict=material when a real material object is present, and "
    "verdict=ambiguous whenever the evidence is insufficient."
)
ALLOWED_VERDICTS = frozenset({"background", "material", "ambiguous"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_TEACHER_FAMILIES = ("qwen",)

REQUIRED_ROW_FIELDS = frozenset(
    {
        "sample_id",
        "role",
        "source_path_b64",
        "filepath",
        "source_sha256",
        "material",
        "category",
        "source_object_count",
    }
)


@dataclass(frozen=True)
class JudgeSpec:
    judge_id: str
    model_family: str
    ollama_model: str
    ollama_url: str
    model_manifest_path: Path
    model_manifest_sha256: str
    model_weight_layer_sha256: str
    model_config_path: Path
    model_config_sha256: str
    model_config_families: tuple[str, ...]
    spec_path: Path
    spec_sha256: str


@dataclass(frozen=True)
class BackgroundRow:
    sample_id: str
    source_path: Path
    crop_path: Path
    source_sha256: str
    crop_sha256: str


@dataclass(frozen=True)
class ServerBinding:
    model_digest: str
    tag_model_families: tuple[str, ...]
    show_model_families: tuple[str, ...]
    model_families: tuple[str, ...]
    capabilities: tuple[str, ...]
    tags_response: dict[str, Any]
    show_response: dict[str, Any]
    tags_response_sha256: str
    show_response_sha256: str


@dataclass(frozen=True)
class TagsEvidence:
    model_digest: str
    model_families: tuple[str, ...]
    response: dict[str, Any]
    response_sha256: str


@dataclass(frozen=True)
class ShowEvidence:
    model_families: tuple[str, ...]
    capabilities: tuple[str, ...]
    response: dict[str, Any]
    response_sha256: str


ApiClient = Callable[
    [JudgeSpec, str, str, Mapping[str, Any] | None, float], Mapping[str, Any]
]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _declared_sha256(value: object, *, field: str) -> str:
    normalized = _nonempty(value, field=field).casefold()
    if not SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    return normalized


def _canonical_family(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "", value.casefold())
    if not result:
        raise ValueError("model_family must contain an ASCII letter or digit")
    return result


def _families_compatible(left: str, right: str) -> bool:
    return left == right or left.startswith(right) or right.startswith(left)


def _family_is_blocked(value: str, blocked_families: Sequence[str]) -> bool:
    return any(
        blocked in value or value in blocked for blocked in blocked_families
    )


def _blocked_families(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _canonical_family(value)
            for value in (*DEFAULT_TEACHER_FAMILIES, *values)
        )
    )


def _load_json_object(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} must be a UTF-8 JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object: {path}")
    return value, raw


def _resolve_from(base: Path, value: object, *, field: str) -> Path:
    path = Path(_nonempty(value, field=field))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _validate_ollama_url(value: object) -> str:
    url = _nonempty(value, field="ollama_url").rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("ollama_url must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("ollama_url must not contain a query or fragment")
    return url


def _manifest_digest(value: object, *, field: str) -> str:
    digest = _nonempty(value, field=field).casefold()
    if not digest.startswith("sha256:"):
        raise ValueError(f"{field} must be a sha256: digest")
    return _declared_sha256(digest.split(":", 1)[1], field=field)


def _string_values(value: object, *, field: str) -> list[str]:
    if isinstance(value, str):
        return [_nonempty(value, field=field)]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return [_nonempty(item, field=field) for item in value]
    raise ValueError(f"{field} must be a non-empty string or string array")


def _model_config_families(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Read architecture identity from the immutable Ollama config blob.

    Ollama/registry config producers use either singular or plural family
    fields.  ``model_type`` is accepted as the architecture identity only when
    a family field is absent.  Requiring identity inside the blob prevents a
    judge spec from renaming a Qwen model to an unrelated family.
    """

    family_values: list[str] = []
    if "model_family" in config:
        family_values.extend(
            _string_values(
                config["model_family"], field="model_config.model_family"
            )
        )
    if "model_families" in config:
        family_values.extend(
            _string_values(config["model_families"], field="model_config.model_families")
        )
    if family_values:
        return tuple(
            dict.fromkeys(_canonical_family(value) for value in family_values)
        )
    if "model_type" in config:
        fallback_values = _string_values(
            config["model_type"], field="model_config.model_type"
        )
        return tuple(
            dict.fromkeys(_canonical_family(value) for value in fallback_values)
        )
    raise ValueError(
        "model config must declare model_family/model_families or model_type"
    )


def _config_architecture_families(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Read architecture-bearing config fields independently of size-like type.

    Ollama uses ``model_type`` for parameter sizes on some models (for example
    Gemma4 ``25.8B``), so it is not a family identity when explicit family
    fields exist.  Renderer, parser, and architecture fields remain useful
    immutable evidence against a family-name disguise.
    """

    values: list[str] = []
    for key in ("renderer", "parser", "architecture", "general.architecture"):
        if key in config:
            values.extend(
                _string_values(config[key], field=f"model_config.{key}")
            )
    general = config.get("general")
    if general is not None:
        if not isinstance(general, Mapping):
            raise ValueError("model_config.general must be an object")
        if "architecture" in general:
            values.extend(
                _string_values(
                    general["architecture"],
                    field="model_config.general.architecture",
                )
            )
    return tuple(dict.fromkeys(_canonical_family(value) for value in values))


def _model_type_tokens(config: Mapping[str, Any]) -> tuple[str, ...]:
    if "model_type" not in config:
        return ()
    return tuple(
        dict.fromkeys(
            _canonical_family(value)
            for value in _string_values(
                config["model_type"], field="model_config.model_type"
            )
        )
    )


def _validate_model_artifacts(
    *,
    value: Mapping[str, Any],
    spec_path: Path,
    judge_id: str,
    declared_family: str,
    blocked_families: Sequence[str],
) -> tuple[Path, str, str, Path, str, tuple[str, ...]]:
    model_manifest_path = _resolve_from(
        spec_path.parent,
        value.get("model_manifest_path"),
        field="model_manifest_path",
    )
    if not model_manifest_path.is_file():
        raise FileNotFoundError(
            f"judge {judge_id!r} Ollama tag manifest does not exist: "
            f"{model_manifest_path}"
        )
    manifest_raw = model_manifest_path.read_bytes()
    declared_manifest_sha = _declared_sha256(
        value.get("model_manifest_sha256"), field="model_manifest_sha256"
    )
    actual_manifest_sha = _sha256_bytes(manifest_raw)
    if actual_manifest_sha != declared_manifest_sha:
        raise ValueError(f"model manifest SHA-256 mismatch for judge {judge_id!r}")
    try:
        manifest_value = json.loads(manifest_raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"judge {judge_id!r} Ollama tag manifest must be a UTF-8 JSON object: "
            f"{model_manifest_path}"
        ) from error
    if not isinstance(manifest_value, dict):
        raise ValueError(
            f"judge {judge_id!r} Ollama tag manifest must be a JSON object"
        )
    if manifest_value.get("schemaVersion") != 2:
        raise ValueError(
            f"judge {judge_id!r} model manifest must be an Ollama/OCI schemaVersion=2 manifest"
        )
    config_descriptor = manifest_value.get("config")
    layers = manifest_value.get("layers")
    if not isinstance(config_descriptor, Mapping):
        raise ValueError(f"judge {judge_id!r} model manifest is missing config")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"judge {judge_id!r} model manifest must contain model layers")
    weight_layer_digests: list[str] = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise ValueError(f"judge {judge_id!r} model manifest layer {index} is invalid")
        layer_digest = _manifest_digest(
            layer.get("digest"), field=f"model_manifest.layers[{index}].digest"
        )
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            weight_layer_digests.append(layer_digest)
    if len(weight_layer_digests) != 1:
        raise ValueError(
            f"judge {judge_id!r} model manifest must contain exactly one Ollama model weight layer"
        )

    model_config_path = _resolve_from(
        spec_path.parent,
        value.get("model_config_path"),
        field="model_config_path",
    )
    if not model_config_path.is_file():
        raise FileNotFoundError(
            f"judge {judge_id!r} model config blob does not exist: {model_config_path}"
        )
    config_raw = model_config_path.read_bytes()
    declared_config_sha = _declared_sha256(
        value.get("model_config_sha256"), field="model_config_sha256"
    )
    actual_config_sha = _sha256_bytes(config_raw)
    if actual_config_sha != declared_config_sha:
        raise ValueError(f"model config SHA-256 mismatch for judge {judge_id!r}")
    try:
        config_value = json.loads(config_raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"judge {judge_id!r} model config blob must be a UTF-8 JSON object"
        ) from error
    if not isinstance(config_value, dict):
        raise ValueError(f"judge {judge_id!r} model config blob must be a JSON object")
    manifest_config_sha = _manifest_digest(
        config_descriptor.get("digest"), field="model_manifest.config.digest"
    )
    if manifest_config_sha != actual_config_sha:
        raise ValueError(
            f"model manifest config digest does not match config blob for judge {judge_id!r}"
        )

    actual_families = _model_config_families(config_value)
    architecture_families = _config_architecture_families(config_value)
    model_type_tokens = _model_type_tokens(config_value)
    declared_canonical = _canonical_family(declared_family)
    compatible_families = tuple(
        _families_compatible(actual, declared_canonical)
        for actual in actual_families
    )
    if not any(compatible_families):
        raise ValueError(
            f"judge {judge_id!r} declared model_family does not match immutable model config"
        )
    immutable_identity_tokens = tuple(
        dict.fromkeys(
            (*actual_families, *architecture_families, *model_type_tokens)
        )
    )
    if any(
        _family_is_blocked(actual, blocked_families)
        for actual in immutable_identity_tokens
    ):
        raise ValueError(
            f"judge {judge_id!r} immutable model config identifies a Qwen/project teacher family"
        )
    # Explicit singular/plural family fields must all describe the declared
    # family.  Size-like model_type values are intentionally not compared here.
    if not all(compatible_families):
        raise ValueError(
            f"judge {judge_id!r} immutable model config has conflicting family/type identities"
        )
    if any(
        not _families_compatible(architecture, declared_canonical)
        or not any(
            _families_compatible(architecture, family)
            for family in actual_families
        )
        for architecture in architecture_families
    ):
        raise ValueError(
            f"judge {judge_id!r} immutable model config has conflicting architecture identities"
        )
    return (
        model_manifest_path,
        actual_manifest_sha,
        weight_layer_digests[0],
        model_config_path,
        actual_config_sha,
        actual_families,
    )


def load_judge_specs(
    paths: Sequence[Path],
    *,
    teacher_model_families: Sequence[str] = DEFAULT_TEACHER_FAMILIES,
) -> list[JudgeSpec]:
    """Load and validate independent judge definitions before any HTTP call."""

    if len(paths) < 2:
        raise ValueError("at least two independent judge specs are required")
    blocked = _blocked_families(teacher_model_families)
    specs: list[JudgeSpec] = []
    for raw_path in paths:
        spec_path = raw_path.resolve()
        value, raw = _load_json_object(spec_path, field="judge spec")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported judge spec schema_version: {spec_path}")
        judge_id = _nonempty(value.get("judge_id"), field="judge_id")
        family = _nonempty(value.get("model_family"), field="model_family")
        canonical_family = _canonical_family(family)
        if any(
            teacher_family in canonical_family or canonical_family in teacher_family
            for teacher_family in blocked
        ):
            raise ValueError(
                f"judge {judge_id!r} model_family must be independent of the "
                "Qwen/project teacher family"
            )
        (
            model_manifest_path,
            actual_manifest_sha,
            model_weight_layer_sha256,
            model_config_path,
            actual_config_sha,
            actual_config_families,
        ) = _validate_model_artifacts(
            value=value,
            spec_path=spec_path,
            judge_id=judge_id,
            declared_family=family,
            blocked_families=blocked,
        )
        specs.append(
            JudgeSpec(
                judge_id=judge_id,
                model_family=family,
                ollama_model=_nonempty(
                    value.get("ollama_model"), field="ollama_model"
                ),
                ollama_url=_validate_ollama_url(value.get("ollama_url")),
                model_manifest_path=model_manifest_path,
                model_manifest_sha256=actual_manifest_sha,
                model_weight_layer_sha256=model_weight_layer_sha256,
                model_config_path=model_config_path,
                model_config_sha256=actual_config_sha,
                model_config_families=actual_config_families,
                spec_path=spec_path,
                spec_sha256=_sha256_bytes(raw),
            )
        )

    judge_ids = [spec.judge_id.casefold() for spec in specs]
    if len(set(judge_ids)) != len(judge_ids):
        raise ValueError("judge_id values must be unique")
    families = [_canonical_family(spec.model_family) for spec in specs]
    if len(set(families)) != len(families):
        raise ValueError("every judge must use a distinct model_family")
    manifest_hashes = [spec.model_manifest_sha256 for spec in specs]
    if len(set(manifest_hashes)) != len(manifest_hashes):
        raise ValueError("every judge must use distinct model-manifest bytes")
    weight_layer_hashes = [spec.model_weight_layer_sha256 for spec in specs]
    if len(set(weight_layer_hashes)) != len(weight_layer_hashes):
        raise ValueError("every judge must use a distinct OCI model weight layer")
    config_hashes = [spec.model_config_sha256 for spec in specs]
    if len(set(config_hashes)) != len(config_hashes):
        raise ValueError("every judge must use distinct model-config bytes")
    model_names = [spec.ollama_model.casefold() for spec in specs]
    if len(set(model_names)) != len(model_names):
        raise ValueError("every judge must use a distinct Ollama model")
    return sorted(specs, key=lambda spec: spec.judge_id.casefold())


def _read_manifest_rows(path: Path) -> tuple[list[dict[str, str]], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"input manifest does not exist: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("input manifest must be UTF-8") from error
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if not reader.fieldnames:
            raise ValueError("input CSV has no header")
        fieldnames = [str(field).strip() for field in reader.fieldnames]
        if any(not field for field in fieldnames) or len(fieldnames) != len(
            set(fieldnames)
        ):
            raise ValueError("input CSV has empty or duplicate columns")
        rows = []
        for raw_row in reader:
            if None in raw_row:
                raise ValueError("input CSV row has unnamed extra columns")
            rows.append(
                {
                    str(key).strip(): "" if value is None else str(value).strip()
                    for key, value in raw_row.items()
                }
            )
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid input JSON at line {number}: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"input JSONL line {number} must be an object")
            rows.append(
                {
                    str(key).strip(): "" if item is None else str(item).strip()
                    for key, item in value.items()
                }
            )
    else:
        raise ValueError("input manifest must be CSV or JSONL")
    if not rows:
        raise ValueError("input manifest is empty")
    return rows, raw


def _decode_source_path(value: str, *, base: Path, location: str) -> Path:
    encoded = _nonempty(value, field=f"{location}.source_path_b64")
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        decoded = base64.b64decode(
            padded.encode("ascii"), altchars=b"-_", validate=True
        )
    except (UnicodeEncodeError, ValueError) as error:
        raise ValueError(f"{location}.source_path_b64 is not valid URL-safe base64") from error
    if not decoded or b"\x00" in decoded:
        raise ValueError(f"{location}.source_path_b64 decodes to an invalid path")
    path = Path(os.fsdecode(decoded))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _parse_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(numeric)


def _optional_crop_sha(row: Mapping[str, str], *, location: str) -> str | None:
    values = []
    for field in ("crop_sha256", "image_sha256"):
        if str(row.get(field, "")).strip():
            values.append(_declared_sha256(row[field], field=f"{location}.{field}"))
    if len(set(values)) > 1:
        raise ValueError(f"{location} has conflicting crop SHA-256 declarations")
    return values[0] if values else None


def load_background_rows(path: Path) -> tuple[list[BackgroundRow], str]:
    """Load strict model-validation background rows and verify image bytes."""

    manifest_path = path.resolve()
    raw_rows, manifest_bytes = _read_manifest_rows(manifest_path)
    parsed: list[BackgroundRow] = []
    seen_samples: set[str] = set()
    for index, row in enumerate(raw_rows, start=1):
        location = f"row {index}"
        missing = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing:
            raise ValueError(f"{location} is missing required fields: {missing}")
        sample_id = _nonempty(row["sample_id"], field=f"{location}.sample_id")
        sample_key = sample_id.casefold()
        if sample_key in seen_samples:
            raise ValueError(f"duplicate sample_id in input manifest: {sample_id!r}")
        seen_samples.add(sample_key)
        if row["role"].strip().casefold() != "model_validation":
            raise ValueError(f"{location}.role must be model_validation")
        if str(row.get("split", "validation")).strip().casefold() != "validation":
            raise ValueError(f"{location}.split must be validation")
        if _parse_integer(row["material"], field=f"{location}.material") != 9:
            raise ValueError(f"{location} must be a background material row")
        if row["category"].strip().casefold() != "background":
            raise ValueError(f"{location}.category must be background")
        source_object_count = _parse_integer(
            row["source_object_count"], field=f"{location}.source_object_count"
        )
        if source_object_count not in {0, 1}:
            raise ValueError(f"{location}.source_object_count must be zero or one")
        raw_crop_object_count = str(row.get("crop_object_count", "")).strip()
        if not raw_crop_object_count:
            if source_object_count == 1:
                raise ValueError(
                    f"{location}.crop_object_count is required when "
                    "source_object_count is one"
                )
            crop_object_count = 0
        else:
            crop_object_count = _parse_integer(
                raw_crop_object_count, field=f"{location}.crop_object_count"
            )
        if crop_object_count != 0:
            raise ValueError(f"{location}.crop_object_count must be zero")

        source_path = _decode_source_path(
            row["source_path_b64"], base=manifest_path.parent, location=location
        )
        crop_path = _resolve_from(
            manifest_path.parent, row["filepath"], field=f"{location}.filepath"
        )
        for field, image_path in (("source", source_path), ("crop", crop_path)):
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{location} {field} image does not exist: {image_path}"
                )
            if image_path.stat().st_size <= 0:
                raise ValueError(f"{location} {field} image is empty")
        declared_source_sha = _declared_sha256(
            row["source_sha256"], field=f"{location}.source_sha256"
        )
        actual_source_sha = _sha256_file(source_path)
        if actual_source_sha != declared_source_sha:
            raise ValueError(f"{location}.source_sha256 does not match source bytes")
        actual_crop_sha = _sha256_file(crop_path)
        declared_crop_sha = _optional_crop_sha(row, location=location)
        if declared_crop_sha is not None and actual_crop_sha != declared_crop_sha:
            raise ValueError(f"{location} crop SHA-256 does not match crop bytes")
        parsed.append(
            BackgroundRow(
                sample_id=sample_id,
                source_path=source_path,
                crop_path=crop_path,
                source_sha256=actual_source_sha,
                crop_sha256=actual_crop_sha,
            )
        )
    return parsed, _sha256_bytes(manifest_bytes)


def _read_verified_image(path: Path, *, expected_sha256: str, field: str) -> bytes:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"{field} image is empty")
    if _sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"{field} image changed after manifest preflight")
    return raw


def build_ollama_payload(
    spec: JudgeSpec,
    *,
    source_bytes: bytes,
    crop_bytes: bytes,
) -> dict[str, Any]:
    """Create an image-only request with no candidate or truth metadata."""

    return {
        "model": spec.ollama_model,
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": sorted(ALLOWED_VERDICTS),
                }
            },
            "required": ["verdict"],
            "additionalProperties": False,
        },
        "options": {"temperature": 0},
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [
                    base64.b64encode(source_bytes).decode("ascii"),
                    base64.b64encode(crop_bytes).decode("ascii"),
                ],
            }
        ],
    }


def _ollama_api_request(
    spec: JudgeSpec,
    method: str,
    endpoint: str,
    payload: Mapping[str, Any] | None,
    timeout: float,
) -> Mapping[str, Any]:
    """Call one official Ollama endpoint and require one JSON object."""

    if (method, endpoint) not in {
        ("GET", "/api/tags"),
        ("POST", "/api/show"),
        ("POST", "/api/chat"),
    }:
        raise ValueError(f"unsupported Ollama API call: {method} {endpoint}")
    if method == "GET" and payload is not None:
        raise ValueError("GET Ollama API calls must not have a request body")
    if method == "POST" and payload is None:
        raise ValueError("POST Ollama API calls require a request body")
    data = (
        None
        if payload is None
        else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    request = urllib.request.Request(
        spec.ollama_url + endpoint,
        data=data,
        headers={
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if data is not None else {}),
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"judge {spec.judge_id!r} Ollama {endpoint} HTTP error {error.code}"
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            f"judge {spec.judge_id!r} Ollama {endpoint} request failed"
        ) from error
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"judge {spec.judge_id!r} returned invalid JSON from {endpoint}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"judge {spec.judge_id!r} {endpoint} response must be an object"
        )
    return value


def _canonical_response_snapshot(
    response: Mapping[str, Any], *, judge_id: str, endpoint: str
) -> tuple[dict[str, Any], str]:
    """Freeze the exact JSON value used for validation and hash its canonical form."""

    if not isinstance(response, Mapping):
        raise ValueError(f"judge {judge_id!r} {endpoint} response must be an object")
    try:
        canonical = _canonical_json_bytes(dict(response))
        snapshot = json.loads(canonical)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"judge {judge_id!r} {endpoint} response is not canonical JSON"
        ) from error
    if not isinstance(snapshot, dict):
        raise ValueError(f"judge {judge_id!r} {endpoint} response must be an object")
    return snapshot, _sha256_bytes(canonical)


def _server_digest(value: object, *, field: str) -> str:
    digest = _nonempty(value, field=field).casefold()
    if digest.startswith("sha256:"):
        digest = digest.split(":", 1)[1]
    return _declared_sha256(digest, field=field)


def _server_details_families(
    response: Mapping[str, Any], *, judge_id: str, endpoint: str
) -> tuple[str, ...]:
    details = response.get("details")
    if not isinstance(details, Mapping):
        raise ValueError(
            f"judge {judge_id!r} {endpoint} response is missing details"
        )
    values: list[str] = []
    if "family" in details:
        values.extend(
            _string_values(details["family"], field=f"{endpoint}.details.family")
        )
    if "families" in details:
        values.extend(
            _string_values(
                details["families"], field=f"{endpoint}.details.families"
            )
        )
    if not values:
        raise ValueError(
            f"judge {judge_id!r} {endpoint} details must declare family/families"
        )
    return tuple(dict.fromkeys(_canonical_family(value) for value in values))


def _validate_server_families(
    spec: JudgeSpec,
    families: Sequence[str],
    *,
    endpoint: str,
    blocked_families: Sequence[str],
) -> None:
    declared = _canonical_family(spec.model_family)
    for family in families:
        if _family_is_blocked(family, blocked_families):
            raise ValueError(
                f"judge {spec.judge_id!r} {endpoint} identifies a Qwen/project teacher family"
            )
        if not _families_compatible(family, declared) or not any(
            _families_compatible(family, config_family)
            for config_family in spec.model_config_families
        ):
            raise ValueError(
                f"judge {spec.judge_id!r} {endpoint} family does not match "
                "the immutable local model config"
            )


def _family_sets_compatible(
    left: Sequence[str], right: Sequence[str]
) -> bool:
    return all(any(_families_compatible(item, other) for other in right) for item in left) and all(
        any(_families_compatible(item, other) for other in left) for item in right
    )


def _validate_tags_evidence(
    spec: JudgeSpec,
    *,
    raw_response: Mapping[str, Any],
    blocked_families: Sequence[str],
) -> TagsEvidence:
    tags, tags_sha256 = _canonical_response_snapshot(
        raw_response, judge_id=spec.judge_id, endpoint="/api/tags"
    )
    models = tags.get("models")
    if not isinstance(models, list):
        raise ValueError(
            f"judge {spec.judge_id!r} /api/tags response is missing models"
        )
    exact_matches = [
        item
        for item in models
        if isinstance(item, Mapping)
        and (item.get("name") == spec.ollama_model or item.get("model") == spec.ollama_model)
    ]
    if len(exact_matches) != 1:
        raise ValueError(
            f"judge {spec.judge_id!r} /api/tags must contain exactly one exact model tag"
        )
    tag_entry = exact_matches[0]
    if (
        tag_entry.get("name") != spec.ollama_model
        or tag_entry.get("model") != spec.ollama_model
    ):
        raise ValueError(
            f"judge {spec.judge_id!r} /api/tags exact model tag identity is inconsistent"
        )
    model_digest = _server_digest(
        tag_entry.get("digest"), field="/api/tags.models[].digest"
    )
    if model_digest != spec.model_manifest_sha256:
        raise ValueError(
            f"judge {spec.judge_id!r} server model digest does not match local tag manifest SHA-256"
        )
    tag_families = _server_details_families(
        tag_entry, judge_id=spec.judge_id, endpoint="/api/tags"
    )
    _validate_server_families(
        spec,
        tag_families,
        endpoint="/api/tags",
        blocked_families=blocked_families,
    )
    return TagsEvidence(
        model_digest=model_digest,
        model_families=tuple(sorted(tag_families)),
        response=tags,
        response_sha256=tags_sha256,
    )


def _validate_show_evidence(
    spec: JudgeSpec,
    *,
    raw_response: Mapping[str, Any],
    blocked_families: Sequence[str],
) -> ShowEvidence:
    show, show_sha256 = _canonical_response_snapshot(
        raw_response, judge_id=spec.judge_id, endpoint="/api/show"
    )
    show_families = _server_details_families(
        show, judge_id=spec.judge_id, endpoint="/api/show"
    )
    _validate_server_families(
        spec,
        show_families,
        endpoint="/api/show",
        blocked_families=blocked_families,
    )
    capabilities_raw = show.get("capabilities")
    if not isinstance(capabilities_raw, list) or not capabilities_raw or not all(
        isinstance(value, str) and value.strip() for value in capabilities_raw
    ):
        raise ValueError(
            f"judge {spec.judge_id!r} /api/show capabilities must be a non-empty string array"
        )
    capabilities = tuple(
        sorted(dict.fromkeys(value.strip().casefold() for value in capabilities_raw))
    )
    if "vision" not in capabilities:
        raise ValueError(
            f"judge {spec.judge_id!r} /api/show model must have the vision capability"
        )
    return ShowEvidence(
        model_families=tuple(sorted(show_families)),
        capabilities=capabilities,
        response=show,
        response_sha256=show_sha256,
    )


def _preflight_ollama_model(
    spec: JudgeSpec,
    *,
    client: ApiClient,
    timeout: float,
    blocked_families: Sequence[str],
) -> ServerBinding:
    """Bind a local immutable model identity to the actual serving endpoint."""

    tags = _validate_tags_evidence(
        spec,
        raw_response=client(spec, "GET", "/api/tags", None, timeout),
        blocked_families=blocked_families,
    )

    show = _validate_show_evidence(
        spec,
        raw_response=client(
            spec,
            "POST",
            "/api/show",
            {"model": spec.ollama_model},
            timeout,
        ),
        blocked_families=blocked_families,
    )
    if not _family_sets_compatible(tags.model_families, show.model_families):
        raise ValueError(
            f"judge {spec.judge_id!r} /api/tags and /api/show model families disagree"
        )
    return ServerBinding(
        model_digest=tags.model_digest,
        tag_model_families=tags.model_families,
        show_model_families=show.model_families,
        model_families=tuple(
            sorted(set((*tags.model_families, *show.model_families)))
        ),
        capabilities=show.capabilities,
        tags_response=tags.response,
        show_response=show.response,
        tags_response_sha256=tags.response_sha256,
        show_response_sha256=show.response_sha256,
    )


def _require_same_server_identity(
    preflight: ServerBinding,
    later: ServerBinding,
    *,
    judge_id: str,
    stage: str,
) -> None:
    if (
        later.model_digest != preflight.model_digest
        or later.tag_model_families != preflight.tag_model_families
        or later.show_model_families != preflight.show_model_families
        or later.capabilities != preflight.capabilities
    ):
        raise ValueError(
            f"judge {judge_id!r} Ollama identity changed during {stage}"
        )


def _require_same_tag_identity(
    preflight: ServerBinding,
    later: TagsEvidence,
    *,
    judge_id: str,
) -> None:
    if (
        later.model_digest != preflight.model_digest
        or later.model_families != preflight.tag_model_families
    ):
        raise ValueError(
            f"judge {judge_id!r} Ollama tag identity changed after /api/chat"
        )


FORBIDDEN_RESPONSE_KEY_TOKENS = frozenset(
    {
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
)
FORBIDDEN_RESPONSE_VALUE_TOKENS = (
    "groundtruth",
    "truth",
    "candidateprediction",
    "prediction",
    "candidateconfidence",
    "confidence",
    "sourcepathb64",
)


def _reject_forbidden_chat_response(
    response: Mapping[str, Any], *, forbidden_echo_values: Sequence[str]
) -> None:
    """Reject request/image echoes and metadata that would contaminate evidence."""

    echo_values = tuple(value for value in forbidden_echo_values if value)

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("Ollama response object keys must be strings")
                key_token = re.sub(r"[^a-z0-9]+", "", key.casefold())
                if (
                    key_token in FORBIDDEN_RESPONSE_KEY_TOKENS
                    or key_token.endswith(("truth", "prediction", "confidence"))
                    or key_token.startswith(("request", "image"))
                ):
                    raise ValueError(
                        f"Ollama chat response contains forbidden evidence key at {path}.{key}"
                    )
                visit(item, f"{path}.{key}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, str):
            compact = re.sub(r"[^a-z0-9]+", "", value.casefold())
            if any(token in compact for token in FORBIDDEN_RESPONSE_VALUE_TOKENS):
                raise ValueError(
                    f"Ollama chat response contains forbidden metadata text at {path}"
                )
            if any(echo in value for echo in echo_values):
                raise ValueError(
                    f"Ollama chat response echoes request or image content at {path}"
                )

    visit(response, "response")


def parse_verdict(response: Mapping[str, Any], *, spec: JudgeSpec) -> str:
    judge_id = spec.judge_id
    if not isinstance(response, Mapping):
        raise ValueError(f"judge {judge_id!r} response must be an object")
    response_model = response.get("model")
    if response_model != spec.ollama_model:
        raise ValueError(
            f"judge {judge_id!r} /api/chat response model does not match the requested tag"
        )
    if _family_is_blocked(
        _canonical_family(response_model), DEFAULT_TEACHER_FAMILIES
    ):
        raise ValueError(
            f"judge {judge_id!r} /api/chat response identifies a Qwen/project teacher model"
        )
    message = response.get("message")
    if not isinstance(message, Mapping):
        raise ValueError(f"judge {judge_id!r} response is missing message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"judge {judge_id!r} response is missing message.content")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"judge {judge_id!r} content must be one strict JSON object"
        ) from error
    if not isinstance(value, dict) or set(value) != {"verdict"}:
        raise ValueError(
            f"judge {judge_id!r} content must contain only the verdict field"
        )
    verdict = value["verdict"]
    if not isinstance(verdict, str) or verdict not in ALLOWED_VERDICTS:
        raise ValueError(
            f"judge {judge_id!r} verdict must be background, material, or ambiguous"
        )
    return verdict


def _vote(
    spec: JudgeSpec,
    server_binding: ServerBinding,
    row: BackgroundRow,
    verdict: str,
    *,
    postchat_tags: TagsEvidence,
    canonical_raw_response: Mapping[str, Any],
    canonical_raw_response_sha256: str,
    runner_script_sha256: str,
    evidence_pair_id: str,
    official_ollama_http: bool,
    authoritative_evidence: bool,
) -> dict[str, Any]:
    raw_response_bytes = _canonical_json_bytes(dict(canonical_raw_response))
    actual_raw_response_sha256 = _sha256_bytes(raw_response_bytes)
    if actual_raw_response_sha256 != _declared_sha256(
        canonical_raw_response_sha256,
        field="canonical_raw_response_sha256",
    ):
        raise ValueError("canonical raw response SHA-256 mismatch")
    stored_raw_response = json.loads(raw_response_bytes)
    postchat_tags_bytes = _canonical_json_bytes(postchat_tags.response)
    actual_postchat_tags_sha256 = _sha256_bytes(postchat_tags_bytes)
    if actual_postchat_tags_sha256 != postchat_tags.response_sha256:
        raise ValueError("post-chat tags response SHA-256 mismatch")
    vote_payload = {
        "schema_version": SCHEMA_VERSION,
        "vote_schema": VOTE_SCHEMA,
        "evidence_schema": EVIDENCE_SCHEMA,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
        "evidence_pair_contract": EVIDENCE_PAIR_CONTRACT,
        "evidence_pair_id": evidence_pair_id,
        "official_ollama_http": official_ollama_http,
        "authoritative_evidence": authoritative_evidence,
        "judge_id": spec.judge_id,
        "model_family": spec.model_family,
        "ollama_model": spec.ollama_model,
        "sample_id": row.sample_id,
        "verdict": verdict,
        "prompt_sha256": _sha256_bytes(PROMPT.encode("utf-8")),
        "runner_script_sha256": runner_script_sha256,
        "model_manifest_sha256": spec.model_manifest_sha256,
        "model_weight_layer_sha256": spec.model_weight_layer_sha256,
        "model_config_sha256": spec.model_config_sha256,
        "server_model_digest": server_binding.model_digest,
        "server_model_families": list(server_binding.model_families),
        "server_capabilities": list(server_binding.capabilities),
        "server_digest_contract": SERVER_DIGEST_CONTRACT,
        "server_tags_response_sha256": server_binding.tags_response_sha256,
        "server_show_response_sha256": server_binding.show_response_sha256,
        "postchat_tags_response": json.loads(postchat_tags_bytes),
        "postchat_tags_response_sha256": postchat_tags.response_sha256,
        "postchat_server_model_digest": postchat_tags.model_digest,
        "postchat_server_model_families": list(postchat_tags.model_families),
        "canonical_raw_response": stored_raw_response,
        "canonical_raw_response_sha256": canonical_raw_response_sha256,
        "source_sha256": row.source_sha256,
        "crop_sha256": row.crop_sha256,
    }
    return {
        **vote_payload,
        "vote_binding_sha256": _sha256_bytes(_canonical_json_bytes(vote_payload)),
    }


def _validate_vote_coverage(
    rows: Sequence[BackgroundRow],
    specs: Sequence[JudgeSpec],
    votes: Sequence[Mapping[str, Any]],
) -> None:
    for vote in votes:
        postchat_tags = vote.get("postchat_tags_response")
        if not isinstance(postchat_tags, Mapping):
            raise ValueError("vote postchat_tags_response must be an object")
        declared_postchat_tags_sha = _declared_sha256(
            vote.get("postchat_tags_response_sha256"),
            field="vote.postchat_tags_response_sha256",
        )
        if declared_postchat_tags_sha != _sha256_bytes(
            _canonical_json_bytes(dict(postchat_tags))
        ):
            raise ValueError("post-chat tags response SHA-256 mismatch")
        raw_response = vote.get("canonical_raw_response")
        if not isinstance(raw_response, Mapping):
            raise ValueError("vote canonical_raw_response must be an object")
        declared_raw_response_sha = _declared_sha256(
            vote.get("canonical_raw_response_sha256"),
            field="vote.canonical_raw_response_sha256",
        )
        actual_raw_response_sha = _sha256_bytes(
            _canonical_json_bytes(dict(raw_response))
        )
        if declared_raw_response_sha != actual_raw_response_sha:
            raise ValueError("canonical raw response SHA-256 mismatch")
        declared_binding = _declared_sha256(
            vote.get("vote_binding_sha256"), field="vote.vote_binding_sha256"
        )
        payload = {
            key: value for key, value in vote.items() if key != "vote_binding_sha256"
        }
        actual_binding = _sha256_bytes(_canonical_json_bytes(payload))
        if declared_binding != actual_binding:
            raise ValueError("vote binding SHA-256 mismatch")
    expected = {
        (row.sample_id, spec.judge_id) for row in rows for spec in specs
    }
    counts = Counter(
        (str(vote.get("sample_id", "")), str(vote.get("judge_id", "")))
        for vote in votes
    )
    missing = sorted(expected - set(counts))
    unexpected = sorted(set(counts) - expected)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if missing or unexpected or duplicates or len(votes) != len(expected):
        raise ValueError(
            "judge vote coverage must contain exactly one vote per judge per row; "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"duplicate={len(duplicates)}"
        )


def _exclusive_publish_pair(
    *,
    output_jsonl: Path,
    jsonl_bytes: bytes,
    output_report: Path,
    report_bytes: bytes,
) -> None:
    targets = (output_jsonl.resolve(), output_report.resolve())
    if targets[0] == targets[1]:
        raise ValueError("output JSONL and report paths must be different")
    existing = [path for path in targets if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")

    staged: list[Path] = []
    published: list[Path] = []
    try:
        for target, content in zip(targets, (jsonl_bytes, report_bytes)):
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw_temp = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temp = Path(raw_temp)
            staged.append(temp)
            with os.fdopen(descriptor, "wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
        for temp, target in zip(staged, targets):
            os.link(temp, target)
            published.append(target)
    except BaseException:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        raise
    finally:
        for temp in staged:
            temp.unlink(missing_ok=True)


def run_independent_visual_judges(
    *,
    input_manifest: Path,
    judge_spec_paths: Sequence[Path],
    output_jsonl: Path,
    output_report: Path,
    timeout: float = 120.0,
    teacher_model_families: Sequence[str] = DEFAULT_TEACHER_FAMILIES,
    api_client: ApiClient | None = None,
) -> dict[str, Any]:
    """Run every independent judge exactly once for every strict input row."""

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    if output_jsonl.resolve() == output_report.resolve():
        raise ValueError("output JSONL and report paths must be different")
    for target in (output_jsonl, output_report):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output: {target}")

    specs = load_judge_specs(
        judge_spec_paths, teacher_model_families=teacher_model_families
    )
    rows, input_manifest_sha256 = load_background_rows(input_manifest)
    runner_script_path = Path(__file__).resolve()
    runner_script_sha256 = _sha256_file(runner_script_path)
    official_ollama_http = api_client is None
    authoritative_evidence = official_ollama_http
    client = api_client or _ollama_api_request
    blocked_families = _blocked_families(teacher_model_families)
    server_bindings = {
        spec.judge_id: _preflight_ollama_model(
            spec,
            client=client,
            timeout=timeout,
            blocked_families=blocked_families,
        )
        for spec in specs
    }
    evidence_pair_seed = {
        "evidence_schema": EVIDENCE_SCHEMA,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_pair_contract": EVIDENCE_PAIR_CONTRACT,
        "input_manifest_sha256": input_manifest_sha256,
        "prompt_sha256": _sha256_bytes(PROMPT.encode("utf-8")),
        "runner_script_sha256": runner_script_sha256,
        "official_ollama_http": official_ollama_http,
        "authoritative_evidence": authoritative_evidence,
        "judges": [
            {
                "judge_id": spec.judge_id,
                "judge_spec_sha256": spec.spec_sha256,
                "model_manifest_sha256": spec.model_manifest_sha256,
                "model_weight_layer_sha256": spec.model_weight_layer_sha256,
                "model_config_sha256": spec.model_config_sha256,
                "preflight_tags_response_sha256": server_bindings[
                    spec.judge_id
                ].tags_response_sha256,
                "preflight_show_response_sha256": server_bindings[
                    spec.judge_id
                ].show_response_sha256,
            }
            for spec in specs
        ],
    }
    evidence_pair_id = _sha256_bytes(
        _canonical_json_bytes(evidence_pair_seed)
    )
    votes: list[dict[str, Any]] = []
    for row in rows:
        source_bytes = _read_verified_image(
            row.source_path,
            expected_sha256=row.source_sha256,
            field=f"sample {row.sample_id!r} source",
        )
        crop_bytes = _read_verified_image(
            row.crop_path,
            expected_sha256=row.crop_sha256,
            field=f"sample {row.sample_id!r} crop",
        )
        for spec in specs:
            payload = build_ollama_payload(
                spec, source_bytes=source_bytes, crop_bytes=crop_bytes
            )
            raw_response = client(
                spec, "POST", "/api/chat", payload, timeout
            )
            response, response_sha256 = _canonical_response_snapshot(
                raw_response, judge_id=spec.judge_id, endpoint="/api/chat"
            )
            postchat_tags = _validate_tags_evidence(
                spec,
                raw_response=client(spec, "GET", "/api/tags", None, timeout),
                blocked_families=blocked_families,
            )
            _require_same_tag_identity(
                server_bindings[spec.judge_id],
                postchat_tags,
                judge_id=spec.judge_id,
            )
            _reject_forbidden_chat_response(
                response,
                forbidden_echo_values=(
                    PROMPT,
                    base64.b64encode(source_bytes).decode("ascii"),
                    base64.b64encode(crop_bytes).decode("ascii"),
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            verdict = parse_verdict(response, spec=spec)
            votes.append(
                _vote(
                    spec,
                    server_bindings[spec.judge_id],
                    row,
                    verdict,
                    postchat_tags=postchat_tags,
                    canonical_raw_response=response,
                    canonical_raw_response_sha256=response_sha256,
                    runner_script_sha256=runner_script_sha256,
                    evidence_pair_id=evidence_pair_id,
                    official_ollama_http=official_ollama_http,
                    authoritative_evidence=authoritative_evidence,
                )
            )

    postflight_bindings = {
        spec.judge_id: _preflight_ollama_model(
            spec,
            client=client,
            timeout=timeout,
            blocked_families=blocked_families,
        )
        for spec in specs
    }
    for spec in specs:
        _require_same_server_identity(
            server_bindings[spec.judge_id],
            postflight_bindings[spec.judge_id],
            judge_id=spec.judge_id,
            stage="postflight",
        )
    postflight_identity_set = [
        {
            "judge_id": spec.judge_id,
            "postflight_tags_response_sha256": postflight_bindings[
                spec.judge_id
            ].tags_response_sha256,
            "postflight_show_response_sha256": postflight_bindings[
                spec.judge_id
            ].show_response_sha256,
        }
        for spec in specs
    ]
    postflight_identity_set_sha256 = _sha256_bytes(
        _canonical_json_bytes(postflight_identity_set)
    )
    evidence_pair_seed["postflight_identity_set"] = postflight_identity_set
    evidence_pair_id = _sha256_bytes(_canonical_json_bytes(evidence_pair_seed))
    rebound_votes: list[dict[str, Any]] = []
    for vote in votes:
        payload = {
            key: value
            for key, value in vote.items()
            if key != "vote_binding_sha256"
        }
        payload["evidence_pair_id"] = evidence_pair_id
        payload["postflight_identity_set_sha256"] = (
            postflight_identity_set_sha256
        )
        rebound_votes.append(
            {
                **payload,
                "vote_binding_sha256": _sha256_bytes(
                    _canonical_json_bytes(payload)
                ),
            }
        )
    votes = rebound_votes

    if _sha256_file(runner_script_path) != runner_script_sha256:
        raise RuntimeError("runner script changed during judge execution")
    _validate_vote_coverage(rows, specs, votes)
    jsonl_bytes = b"".join(_canonical_json_bytes(vote) for vote in votes)
    evidence_jsonl_sha256 = _sha256_bytes(jsonl_bytes)
    evidence_jsonl_line_count = jsonl_bytes.count(b"\n")
    if evidence_jsonl_line_count != len(votes):
        raise RuntimeError("evidence JSONL line count does not match vote count")
    verdict_counts = {
        spec.judge_id: {
            verdict: sum(
                vote["judge_id"] == spec.judge_id and vote["verdict"] == verdict
                for vote in votes
            )
            for verdict in sorted(ALLOWED_VERDICTS)
        }
        for spec in specs
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "report_schema": REPORT_SCHEMA,
        "evidence_schema": EVIDENCE_SCHEMA,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
        "evidence_pair_contract": EVIDENCE_PAIR_CONTRACT,
        "evidence_pair_id": evidence_pair_id,
        "evidence_pair_seed": evidence_pair_seed,
        "postflight_identity_set": postflight_identity_set,
        "postflight_identity_set_sha256": postflight_identity_set_sha256,
        "official_ollama_http": official_ollama_http,
        "authoritative_evidence": authoritative_evidence,
        "artifact_pair": {
            "complete": True,
            "required_members": ["evidence_jsonl", "report_json"],
            "report_pins_exact_evidence_jsonl": True,
        },
        "artifact_role": "diagnostic_veto_only_not_promotion_authority",
        "authority": {
            "promotion_authority": False,
            "ground_truth_authority": False,
            "may_relabel_truth": False,
            "may_tune_thresholds": False,
            "allowed_actions": ["diagnostic", "veto", "request_more_evidence"],
        },
        "input_manifest_sha256": input_manifest_sha256,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": _sha256_bytes(PROMPT.encode("utf-8")),
        "runner_script_sha256": runner_script_sha256,
        "server_digest_contract": SERVER_DIGEST_CONTRACT,
        "row_count": len(rows),
        "judge_count": len(specs),
        "vote_count": len(votes),
        "expected_vote_count": len(rows) * len(specs),
        "coverage": {"every_judge_exactly_one_vote_per_row": True},
        "candidate_metadata_exposed_to_prompt": False,
        "raw_response_content_stored": True,
        "request_content_stored": False,
        "image_content_stored": False,
        "evidence_jsonl_sha256": evidence_jsonl_sha256,
        "evidence_jsonl_line_count": evidence_jsonl_line_count,
        "canonical_raw_response_sha256_by_vote": [
            {
                "sample_id": vote["sample_id"],
                "judge_id": vote["judge_id"],
                "canonical_raw_response_sha256": vote[
                    "canonical_raw_response_sha256"
                ],
            }
            for vote in votes
        ],
        "judges": [
            {
                "judge_id": spec.judge_id,
                "model_family": spec.model_family,
                "ollama_model": spec.ollama_model,
                "model_manifest_sha256": spec.model_manifest_sha256,
                "model_weight_layer_sha256": spec.model_weight_layer_sha256,
                "model_config_sha256": spec.model_config_sha256,
                "model_config_families": list(spec.model_config_families),
                "judge_spec_sha256": spec.spec_sha256,
                "server_model_digest": server_bindings[
                    spec.judge_id
                ].model_digest,
                "server_model_families": list(
                    server_bindings[spec.judge_id].model_families
                ),
                "server_capabilities": list(
                    server_bindings[spec.judge_id].capabilities
                ),
                "server_tags_response_sha256": server_bindings[
                    spec.judge_id
                ].tags_response_sha256,
                "server_show_response_sha256": server_bindings[
                    spec.judge_id
                ].show_response_sha256,
                "preflight_tags_response": server_bindings[
                    spec.judge_id
                ].tags_response,
                "preflight_tags_response_sha256": server_bindings[
                    spec.judge_id
                ].tags_response_sha256,
                "preflight_show_response": server_bindings[
                    spec.judge_id
                ].show_response,
                "preflight_show_response_sha256": server_bindings[
                    spec.judge_id
                ].show_response_sha256,
                "postflight_tags_response": postflight_bindings[
                    spec.judge_id
                ].tags_response,
                "postflight_tags_response_sha256": postflight_bindings[
                    spec.judge_id
                ].tags_response_sha256,
                "postflight_show_response": postflight_bindings[
                    spec.judge_id
                ].show_response,
                "postflight_show_response_sha256": postflight_bindings[
                    spec.judge_id
                ].show_response_sha256,
                "postflight_identity_matches_preflight": True,
                "prompt_sha256": _sha256_bytes(PROMPT.encode("utf-8")),
                "runner_script_sha256": runner_script_sha256,
            }
            for spec in specs
        ],
        "verdict_counts_by_judge": verdict_counts,
        "results_jsonl_sha256": evidence_jsonl_sha256,
        "generated_by": "scripts/run_independent_visual_judges.py",
    }
    report_bytes = _canonical_json_bytes(report)
    if _sha256_file(runner_script_path) != runner_script_sha256:
        raise RuntimeError("runner script changed before artifact publication")
    _exclusive_publish_pair(
        output_jsonl=output_jsonl,
        jsonl_bytes=jsonl_bytes,
        output_report=output_report,
        report_bytes=report_bytes,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument(
        "--judge-spec",
        action="append",
        required=True,
        type=Path,
        help="Repeat for at least two independent judge JSON specs.",
    )
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--teacher-model-family",
        action="append",
        help="Project teacher family to reject; defaults to qwen.",
    )
    args = parser.parse_args(argv)
    report = run_independent_visual_judges(
        input_manifest=args.input_manifest,
        judge_spec_paths=args.judge_spec,
        output_jsonl=args.output_jsonl,
        output_report=args.output_report,
        timeout=args.timeout,
        teacher_model_families=(
            args.teacher_model_family
            if args.teacher_model_family is not None
            else DEFAULT_TEACHER_FAMILIES
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
