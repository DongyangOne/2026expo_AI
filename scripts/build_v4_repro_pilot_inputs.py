"""Build immutable, diagnostic-only inputs for the v4 batch=1 replay pilot.

The builder resolves the source YOLO YAML with the same helpers used by the
proposal generator, admits only explicit empty labels or exactly one valid
0..8 label, quarantines byte-identical sources that cross train/validation,
and applies a deterministic per-split/per-stratum quota.  When current
explicit-empty sources cannot fill the background input quota, a source that
was historically emitted as background may be selected as a *probe* while its
current non-empty YOLO ground truth remains unchanged.  Historical v4
artifacts only select probes or prioritize bounded drift examples; they never
provide labels, replay truth, or promotion authority.

The output directory is exclusive.  ``input_ready.json`` is published last;
any earlier failure publishes ``failed.txt`` and can never leave a ready
marker behind.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

try:
    from scripts import prepare_proposal_verifier_dataset as proposal_dataset
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import prepare_proposal_verifier_dataset as proposal_dataset  # type: ignore[no-redef]


CLASS_NAMES = proposal_dataset.CLASS_NAMES
OUTPUT_STRATA = (*CLASS_NAMES, "background")
SELECTION_CONTRACT = (
    "v4_repro_pilot_inputs.gt_stratified_historical_observation_priority_blake2b.v3"
)
ARTIFACT_ROLE = (
    "v4_batch1_reproducibility_pilot_inputs_diagnostic_only_"
    "not_training_blind_or_deployment_authority"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class ScannedSource:
    path: Path
    split: str
    source_sha256: str
    label_path: Path | None
    label_sha256: str | None
    stratum: str | None
    gt_class_id: int | None
    gt_xywhn: tuple[float, float, float, float] | None
    reasons: set[str] = field(default_factory=set)
    historical_categories: tuple[str, ...] = ()
    anchor: bool = False

    @property
    def semantic_label(self) -> tuple[object, ...] | None:
        if self.stratum is None:
            return None
        if self.stratum == "background":
            return ("background",)
        return (self.gt_class_id, *(self.gt_xywhn or ()))


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _stable_bytes(path: Path, *, description: str) -> bytes:
    resolved = path.resolve()
    before = resolved.stat()
    content = resolved.read_bytes()
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{description} changed while being read: {resolved}")
    return content


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _stable_sha256(path: Path, *, description: str) -> str:
    return _sha256_bytes(_stable_bytes(path, description=description))


def _publish_exclusive(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output artifact: {path}")
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not published:
                raise


def _publish_failure(output_dir: Path, error: BaseException) -> None:
    if not output_dir.is_dir():
        return
    failed = output_dir / "failed.txt"
    # A ready marker may have been forged before this process started, or a
    # failure may race immediately after terminal publication.  Consumers
    # require ready AND absence of failed, so never let ready suppress failure.
    if failed.exists() or failed.is_symlink():
        return
    message = f"{type(error).__name__}: {error}\n".encode("utf-8", errors="replace")
    try:
        _publish_exclusive(failed, message)
    except OSError:
        pass


def _absolute_posix(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_absolute():
        raise ValueError(f"path did not resolve to an absolute path: {path}")
    return resolved.as_posix()


def _source_score(*, seed: int, split: str, stratum: str, source_sha256: str) -> str:
    value = f"{seed}|{split}|{stratum}|{source_sha256}"
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).hexdigest()


def _ground_truth_reason(reason: str) -> str:
    if reason == "not_single_object":
        return "multi_object_label"
    return f"malformed_label/{reason}"


def _scan_sources(
    split_images: Mapping[str, Iterable[Path]],
) -> list[ScannedSource]:
    resolved_splits = {
        split: list(split_images.get(split, ()))
        for split in ("training", "validation")
    }
    total = sum(len(paths) for paths in resolved_splits.values())
    processed = 0
    started = time.monotonic()
    scanned: list[ScannedSource] = []
    for split in ("training", "validation"):
        for raw_path in resolved_splits[split]:
            path = raw_path.resolve()
            source_content = _stable_bytes(path, description="source image")
            source_sha256 = _sha256_bytes(source_content)
            label_path: Path | None = None
            label_sha256: str | None = None
            stratum: str | None = None
            gt_class_id: int | None = None
            gt_xywhn: tuple[float, float, float, float] | None = None
            reasons: set[str] = set()
            if proposal_dataset._read_image(path) is None:
                reasons.add("unreadable_image")
            try:
                label_path = proposal_dataset._label_path(path).resolve(strict=False)
            except ValueError:
                reasons.add("unresolved_label_path")
            if label_path is not None and not label_path.is_file():
                reasons.add("missing_label_file")
            elif label_path is not None:
                try:
                    label_content = _stable_bytes(label_path, description="YOLO label")
                    label_sha256 = _sha256_bytes(label_content)
                    label_text = label_content.decode("utf-8")
                except UnicodeError:
                    reasons.add("unreadable_label")
                else:
                    ground_truth, reason = proposal_dataset.parse_yolo_label_text(
                        label_text
                    )
                    if reason is not None:
                        reasons.add(_ground_truth_reason(reason))
                    elif ground_truth is None:
                        stratum = "background"
                    else:
                        gt_class_id = ground_truth.class_id
                        gt_xywhn = ground_truth.xywhn
                        stratum = CLASS_NAMES[ground_truth.class_id]
            scanned.append(
                ScannedSource(
                    path=path,
                    split=split,
                    source_sha256=source_sha256,
                    label_path=label_path,
                    label_sha256=label_sha256,
                    stratum=stratum,
                    gt_class_id=gt_class_id,
                    gt_xywhn=gt_xywhn,
                    reasons=reasons,
                )
            )
            processed += 1
            if processed % 1_000 == 0 or processed == total:
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = processed / elapsed
                remaining = max(total - processed, 0)
                print(
                    json.dumps(
                        {
                            "event": "v4_pilot_source_scan_progress",
                            "processed": processed,
                            "total": total,
                            "split": split,
                            "sources_per_second": round(rate, 3),
                            "eta_seconds": round(remaining / rate, 1),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    return scanned


def _quarantine_duplicates(scanned: list[ScannedSource]) -> None:
    by_source: dict[str, list[ScannedSource]] = defaultdict(list)
    for record in scanned:
        by_source[record.source_sha256].append(record)

    for records in by_source.values():
        splits = {record.split for record in records}
        if len(splits) > 1:
            for record in records:
                record.reasons.add("duplicate_source_content_cross_split")
            continue

        eligible = [record for record in records if not record.reasons]
        if len(eligible) <= 1:
            continue
        semantics = {record.semantic_label for record in eligible}
        if len(semantics) > 1:
            for record in eligible:
                record.reasons.add("duplicate_source_content_conflicting_ground_truth")
            continue
        canonical = min(eligible, key=lambda record: _absolute_posix(record.path))
        for record in eligible:
            if record is not canonical:
                record.reasons.add("duplicate_source_content_same_split")


def _read_historical_manifest(
    path: Path | None,
) -> tuple[dict[str, tuple[tuple[str, str], ...]], dict[str, object] | None]:
    if path is None:
        return {}, None
    content = _stable_bytes(path, description="historical manifest")
    try:
        reader = csv.DictReader(content.decode("utf-8-sig").splitlines())
        rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError(f"invalid historical manifest: {path}") from error
    required = {"source_id", "split", "category"}
    if not required.issubset(reader.fieldnames or ()):
        raise ValueError("historical manifest must contain source_id, split and category")
    indexed: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for line, row in enumerate(rows, start=2):
        source_id = str(row.get("source_id", "")).strip().casefold()
        split = str(row.get("split", "")).strip()
        category = str(row.get("category", "")).strip().casefold()
        if not SHA256_RE.fullmatch(source_id):
            raise ValueError(f"historical manifest row {line} has invalid source_id")
        if split not in {"training", "validation"}:
            raise ValueError(f"historical manifest row {line} has invalid split")
        if category not in OUTPUT_STRATA:
            raise ValueError(f"historical manifest row {line} has invalid category")
        indexed[source_id].add((split, category))
    normalized = {
        source_id: tuple(sorted(values)) for source_id, values in sorted(indexed.items())
    }
    return normalized, {
        "path": _absolute_posix(path),
        "sha256": _sha256_bytes(content),
        "rows": len(rows),
    }


def _source_ids_from_examples(value: object, *, location: str) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list):
        raise ValueError(f"drift report {location} must contain a list")
    found: set[str] = set()
    for index, example in enumerate(value):
        if not isinstance(example, Mapping):
            raise ValueError(f"drift report {location}[{index}] must contain an object")
        raw = example.get("source_id")
        if raw is None:
            continue
        normalized = str(raw).strip().casefold()
        if not SHA256_RE.fullmatch(normalized):
            raise ValueError(
                f"drift report {location}[{index}] has invalid source_id"
            )
        found.add(normalized)
    return found


def _collect_allowlisted_report_source_ids(value: Mapping[str, object]) -> set[str]:
    """Read only the bounded replay-example paths approved for pilot anchors."""

    replay = value.get("replay")
    if not isinstance(replay, Mapping):
        raise ValueError("historical drift report must contain a replay object")
    found: set[str] = set()

    hard = replay.get("hard_semantic_mismatch_examples")
    if hard is not None:
        if not isinstance(hard, Mapping):
            raise ValueError(
                "drift report replay.hard_semantic_mismatch_examples must be an object"
            )
        for name, examples in hard.items():
            found.update(
                _source_ids_from_examples(
                    examples,
                    location=f"replay.hard_semantic_mismatch_examples.{name}",
                )
            )

    for section_name in (
        "confidence_abs_drift",
        "bbox_max_abs_drift",
        "declared_vs_replayed_crop_bounds",
    ):
        section = replay.get(section_name)
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise ValueError(f"drift report replay.{section_name} must be an object")
        found.update(
            _source_ids_from_examples(
                section.get("max_examples"),
                location=f"replay.{section_name}.max_examples",
            )
        )

    thresholds = replay.get("fixed_threshold_diagnostics")
    if thresholds is not None:
        if not isinstance(thresholds, Mapping):
            raise ValueError(
                "drift report replay.fixed_threshold_diagnostics must be an object"
            )
        for name, examples in thresholds.items():
            if str(name).endswith("_nearest_examples"):
                found.update(
                    _source_ids_from_examples(
                        examples,
                        location=f"replay.fixed_threshold_diagnostics.{name}",
                    )
                )
    return found


def _read_drift_report(
    path: Path | None,
    historical: Mapping[str, tuple[tuple[str, str], ...]],
) -> tuple[set[str], dict[str, object] | None]:
    if path is None:
        return set(), None
    if not historical:
        raise ValueError("--drift-report requires --old-manifest")
    content = _stable_bytes(path, description="historical drift report")
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid historical drift report: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("historical drift report must contain a JSON object")
    source_ids = _collect_allowlisted_report_source_ids(value)
    missing = sorted(source_ids - set(historical))
    if missing:
        raise ValueError(
            "drift report source_id is absent from historical manifest: "
            + ", ".join(missing[:3])
        )
    return source_ids, {
        "path": _absolute_posix(path),
        "sha256": _sha256_bytes(content),
        "anchor_source_ids": len(source_ids),
    }


def _apply_historical_selection_metadata(
    records: Iterable[ScannedSource],
    *,
    historical: Mapping[str, tuple[tuple[str, str], ...]],
    anchors: set[str],
) -> None:
    for record in records:
        entries = historical.get(record.source_sha256, ())
        record.historical_categories = tuple(
            sorted(category for split, category in entries if split == record.split)
        )
        record.anchor = bool(record.historical_categories) and (
            record.source_sha256 in anchors
        )


def _selection_rows(
    scanned: list[ScannedSource],
    *,
    seed: int,
    training_quota: int,
    validation_quota: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    eligible: dict[tuple[str, str], list[ScannedSource]] = defaultdict(list)
    for record in scanned:
        if not record.reasons:
            if record.stratum not in OUTPUT_STRATA:
                raise RuntimeError("eligible source is missing a valid stratum")
            eligible[(record.split, record.stratum)].append(record)
            if (
                record.stratum != "background"
                and "background" in record.historical_categories
            ):
                # Historical background is a selection hint only.  The source
                # keeps its current non-empty YOLO label, and the frozen
                # batch=1 replay decides whether the new proposal is actually
                # background.
                eligible[(record.split, "background")].append(record)

    selected_rows: list[dict[str, object]] = []
    eligible_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    shortages: dict[str, int] = {}
    eligible_material_historical_observed_counts: dict[str, int] = {}
    for split in ("training", "validation"):
        quota = training_quota if split == "training" else validation_quota
        material_available = {
            stratum: len(eligible.get((split, stratum), []))
            for stratum in CLASS_NAMES
        }
        reserved_records: set[int] = set()
        # Reserve background probes first so a source can never satisfy both
        # the background input quota and its current material quota.
        for stratum in ("background", *CLASS_NAMES):
            key = (split, stratum)
            candidates = [
                record
                for record in eligible.get(key, [])
                if id(record) not in reserved_records
            ]

            def blake_key(record: ScannedSource) -> tuple[str, str, str]:
                return (
                    _source_score(
                        seed=seed,
                        split=split,
                        stratum=stratum,
                        source_sha256=record.source_sha256,
                    ),
                    record.source_sha256,
                    _absolute_posix(record.path),
                )

            candidates.sort(key=blake_key)
            label = f"{split}/{stratum}"
            eligible_counts[label] = len(candidates)
            anchor_priority_cap = max(1, quota // 5)
            explicit_background: list[ScannedSource] = []
            if stratum == "background":
                explicit_background = [
                    record for record in candidates if record.stratum == "background"
                ][:quota]
            explicit_ids = {id(record) for record in explicit_background}
            remaining_quota = max(0, quota - len(explicit_background))
            if stratum == "background":
                # A historical-background probe still has current material GT.
                # Reserve only the surplus above that material's own quota so a
                # greedy background choice cannot make a feasible material
                # allocation fail later.
                probe_capacity = {
                    material: max(0, material_available[material] - quota)
                    for material in CLASS_NAMES
                }
                probe_consumed: Counter[str] = Counter()

                def take_safe_probes(
                    pool: Iterable[ScannedSource], *, limit: int
                ) -> list[ScannedSource]:
                    chosen_probes: list[ScannedSource] = []
                    for record in pool:
                        if len(chosen_probes) >= limit:
                            break
                        material = record.stratum
                        if material not in probe_capacity:
                            continue
                        if probe_consumed[material] >= probe_capacity[material]:
                            continue
                        chosen_probes.append(record)
                        probe_consumed[material] += 1
                    return chosen_probes

                priority = take_safe_probes(
                    sorted(
                        (
                            record
                            for record in candidates
                            if record.anchor and id(record) not in explicit_ids
                        ),
                        key=blake_key,
                    ),
                    limit=min(anchor_priority_cap, remaining_quota),
                )
                priority_ids = {id(record) for record in priority}
                remainder = take_safe_probes(
                    (
                        record
                        for record in candidates
                        if id(record) not in explicit_ids
                        and id(record) not in priority_ids
                    ),
                    limit=max(0, remaining_quota - len(priority)),
                )
            else:
                eligible_material_historical_observed_counts[label] = sum(
                    1 for record in candidates if record.historical_categories
                )
                priority = sorted(
                    (
                        record
                        for record in candidates
                        if record.anchor and id(record) not in explicit_ids
                    ),
                    key=blake_key,
                )[: min(anchor_priority_cap, remaining_quota)]
                priority_ids = {id(record) for record in priority}
                remaining_candidates = [
                    record
                    for record in candidates
                    if id(record) not in explicit_ids
                    and id(record) not in priority_ids
                ]
                # Historical observations from the same split are a selection
                # hint only.  Prefer detector-observed sources after the bounded
                # drift-anchor slice, while preserving current YOLO GT and the
                # original BLAKE2 order inside each tier.  Previously unseen
                # sources remain a deterministic fallback.
                historically_observed = [
                    record
                    for record in remaining_candidates
                    if record.historical_categories
                ]
                unseen = [
                    record
                    for record in remaining_candidates
                    if not record.historical_categories
                ]
                remainder = [*historically_observed, *unseen]
            chosen = [
                *explicit_background,
                *priority,
                *remainder[: max(0, remaining_quota - len(priority))],
            ]
            selected_counts[label] = len(chosen)
            shortages[label] = max(0, quota - len(chosen))
            reserved_records.update(id(record) for record in chosen)
            for rank, record in enumerate(chosen, start=1):
                if record.label_path is None or record.label_sha256 is None:
                    raise RuntimeError("selected source lacks an explicit label binding")
                explicit_empty = record.stratum == "background"
                historical_background_probe = (
                    stratum == "background" and not explicit_empty
                )
                if historical_background_probe and (
                    "background" not in record.historical_categories
                    or record.gt_class_id is None
                    or record.gt_xywhn is None
                ):
                    raise RuntimeError(
                        "historical background probe lacks current GT or historical membership"
                    )
                if explicit_empty:
                    selection_reason = "current_explicit_empty_label"
                elif id(record) in priority_ids:
                    selection_reason = "drift_anchor_priority"
                elif historical_background_probe:
                    selection_reason = "historical_background_probe_blake2"
                elif stratum != "background" and record.historical_categories:
                    selection_reason = "historical_observation_priority_blake2"
                else:
                    selection_reason = "deterministic_blake2"
                selected_rows.append(
                    {
                        "split": split,
                        "stratum": stratum,
                        "selection_stratum": stratum,
                        "current_gt_stratum": record.stratum,
                        "selection_cohort": (
                            "historical_background_probe"
                            if historical_background_probe
                            else "current_yolo_ground_truth"
                        ),
                        "selection_rank_within_stratum": rank,
                        "selection_score_blake2b128": _source_score(
                            seed=seed,
                            split=split,
                            stratum=stratum,
                            source_sha256=record.source_sha256,
                        ),
                        "drift_anchor": record.anchor,
                        "selection_reason": selection_reason,
                        "path": _absolute_posix(record.path),
                        "source_sha256": record.source_sha256,
                        "label_path": _absolute_posix(record.label_path),
                        "label_sha256": record.label_sha256,
                        "explicit_empty_label": explicit_empty,
                        "historical_background_probe_selection_only": (
                            historical_background_probe
                        ),
                        "gt_class_id": record.gt_class_id,
                        "gt_xywhn": list(record.gt_xywhn) if record.gt_xywhn else None,
                        "historical_categories_selection_only": list(
                            record.historical_categories
                        ),
                    }
                )
    selected_rows.sort(
        key=lambda row: (
            str(row["split"]),
            OUTPUT_STRATA.index(str(row["stratum"])),
            int(row["selection_rank_within_stratum"]),
            str(row["path"]),
        )
    )
    return (
        selected_rows,
        eligible_counts,
        selected_counts,
        shortages,
        eligible_material_historical_observed_counts,
    )


def _universe_binding(scanned: Iterable[ScannedSource]) -> tuple[str, int]:
    rows = []
    for record in scanned:
        rows.append(
            {
                "path": _absolute_posix(record.path),
                "split": record.split,
                "source_sha256": record.source_sha256,
                "label_path": (
                    _absolute_posix(record.label_path) if record.label_path else None
                ),
                "label_sha256": record.label_sha256,
                "stratum": record.stratum,
                "gt_class_id": record.gt_class_id,
                "gt_xywhn": list(record.gt_xywhn) if record.gt_xywhn else None,
                "reasons": sorted(record.reasons),
            }
        )
    rows.sort(key=lambda row: (str(row["split"]), str(row["path"])))
    return _sha256_bytes(_json_bytes(rows)), len(rows)


def _rejection_summary(scanned: Iterable[ScannedSource]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    rejected_sources = 0
    for record in sorted(scanned, key=lambda item: (item.split, _absolute_posix(item.path))):
        if not record.reasons:
            continue
        rejected_sources += 1
        counts.update(record.reasons)
        if len(examples) < 100:
            examples.append(
                {
                    "split": record.split,
                    "path": _absolute_posix(record.path),
                    "source_sha256": record.source_sha256,
                    "label_path": (
                        _absolute_posix(record.label_path) if record.label_path else None
                    ),
                    "label_sha256": record.label_sha256,
                    "reasons": sorted(record.reasons),
                }
            )
    return {
        "rejected_sources": rejected_sources,
        "counts": dict(sorted(counts.items())),
        "examples_bounded_to_100": examples,
    }


def _yaml_bytes(
    *, dataset_dir: Path, train_list: Path, validation_list: Path
) -> bytes:
    lines = [
        f"path: {json.dumps(_absolute_posix(dataset_dir), ensure_ascii=False)}",
        f"train: {json.dumps(_absolute_posix(train_list), ensure_ascii=False)}",
        f"val: {json.dumps(_absolute_posix(validation_list), ensure_ascii=False)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _list_bytes(selected: Iterable[Mapping[str, object]], split: str) -> bytes:
    paths = sorted(str(row["path"]) for row in selected if row["split"] == split)
    if not paths:
        raise RuntimeError(f"pilot selection is empty for split: {split}")
    if any(not Path(path).is_absolute() for path in paths):
        raise ValueError(f"pilot {split} list contains a non-absolute path")
    return ("\n".join(paths) + "\n").encode("utf-8")


def _verify_selected_bindings(selected: Iterable[Mapping[str, object]]) -> None:
    for row in selected:
        source = Path(str(row["path"]))
        label = Path(str(row["label_path"]))
        if _stable_sha256(source, description="selected source final rehash") != row[
            "source_sha256"
        ]:
            raise RuntimeError(f"selected source changed during input build: {source}")
        if _stable_sha256(label, description="selected label final rehash") != row[
            "label_sha256"
        ]:
            raise RuntimeError(f"selected label changed during input build: {label}")


def _verify_selected_semantics(selected: Iterable[Mapping[str, object]]) -> None:
    paths: set[str] = set()
    source_hashes: set[str] = set()
    for row in selected:
        split = row.get("split")
        selection = row.get("selection_stratum")
        current = row.get("current_gt_stratum")
        cohort = row.get("selection_cohort")
        path = str(row.get("path", ""))
        source_sha = str(row.get("source_sha256", ""))
        explicit = row.get("explicit_empty_label")
        probe = row.get("historical_background_probe_selection_only")
        categories = row.get("historical_categories_selection_only")
        if split not in {"training", "validation"}:
            raise RuntimeError("selected source has invalid split")
        if selection not in OUTPUT_STRATA or row.get("stratum") != selection:
            raise RuntimeError("selected source has invalid selection stratum")
        if current not in OUTPUT_STRATA:
            raise RuntimeError("selected source has invalid current GT stratum")
        if path in paths or source_sha in source_hashes:
            raise RuntimeError("selected source path or SHA is duplicated")
        paths.add(path)
        source_hashes.add(source_sha)
        if not isinstance(categories, list):
            raise RuntimeError("selected source historical categories are invalid")
        if probe is True:
            if (
                selection != "background"
                or current == "background"
                or explicit is not False
                or cohort != "historical_background_probe"
                or "background" not in categories
                or row.get("gt_class_id") is None
                or row.get("gt_xywhn") is None
            ):
                raise RuntimeError("historical background probe semantics are invalid")
        else:
            if (
                probe is not False
                or cohort != "current_yolo_ground_truth"
                or selection != current
                or explicit is not (current == "background")
            ):
                raise RuntimeError("current YOLO ground-truth selection semantics are invalid")
            reason = row.get("selection_reason")
            if current == "background":
                if reason != "current_explicit_empty_label":
                    raise RuntimeError("explicit background selection reason is invalid")
            elif reason == "drift_anchor_priority":
                if row.get("drift_anchor") is not True:
                    raise RuntimeError("drift anchor priority lacks an anchor")
            elif categories:
                if reason != "historical_observation_priority_blake2":
                    raise RuntimeError("historical observation selection reason is invalid")
            elif reason != "deterministic_blake2":
                raise RuntimeError("unseen material selection reason is invalid")


def build_pilot_inputs(
    *,
    data_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    seed: int = 20260901,
    training_quota: int = 250,
    validation_quota: int = 100,
    old_manifest: Path | None = None,
    drift_report: Path | None = None,
) -> dict[str, object]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if training_quota < 1 or validation_quota < 1:
        raise ValueError("training and validation quotas must be positive")
    if drift_report is not None and old_manifest is None:
        raise ValueError("drift_report requires old_manifest")

    data_path = data_path.resolve()
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve(strict=False)
    if not data_path.is_file():
        raise FileNotFoundError(f"missing YOLO data YAML: {data_path}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"missing dataset directory: {dataset_dir}")
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        split_images = proposal_dataset.resolve_split_images(data_path, dataset_dir)
        scanned = _scan_sources(split_images)
        if not scanned:
            raise RuntimeError("YOLO data YAML resolved no source images")
        _quarantine_duplicates(scanned)

        historical, historical_info = _read_historical_manifest(old_manifest)
        anchors, drift_info = _read_drift_report(drift_report, historical)
        _apply_historical_selection_metadata(
            scanned, historical=historical, anchors=anchors
        )
        (
            selected,
            eligible_counts,
            selected_counts,
            shortages,
            eligible_material_historical_observed_counts,
        ) = _selection_rows(
            scanned,
            seed=seed,
            training_quota=training_quota,
            validation_quota=validation_quota,
        )
        if any(shortages.values()):
            detail = ", ".join(
                f"{name}={amount}"
                for name, amount in shortages.items()
                if amount
            )
            raise RuntimeError(f"balanced pilot quota shortage: {detail}")
        _verify_selected_semantics(selected)
        _verify_selected_bindings(selected)
        universe_sha256, universe_sources = _universe_binding(scanned)
        selected_current_gt_counts = Counter(
            f"{row['split']}/{row['current_gt_stratum']}" for row in selected
        )
        selected_cohort_counts = Counter(
            f"{row['split']}/{row['selection_cohort']}" for row in selected
        )
        background_quota_composition = {}
        eligible_background_probe_counts = {}
        eligible_explicit_empty_counts = {}
        for split in ("training", "validation"):
            background_rows = [
                row
                for row in selected
                if row["split"] == split and row["selection_stratum"] == "background"
            ]
            explicit = sum(bool(row["explicit_empty_label"]) for row in background_rows)
            probes = sum(
                bool(row["historical_background_probe_selection_only"])
                for row in background_rows
            )
            background_quota_composition[split] = {
                "current_explicit_empty_label": explicit,
                "historical_background_probe": probes,
                "total": len(background_rows),
            }
            eligible_explicit_empty_counts[split] = sum(
                1
                for record in scanned
                if not record.reasons
                and record.split == split
                and record.stratum == "background"
            )
            eligible_background_probe_counts[split] = sum(
                1
                for record in scanned
                if not record.reasons
                and record.split == split
                and record.stratum != "background"
                and "background" in record.historical_categories
            )

        train_path = output_dir / "train_pilot.txt"
        validation_path = output_dir / "validation_pilot.txt"
        yaml_path = output_dir / "pilot_dataset.yaml"
        inventory_path = output_dir / "selection_inventory.json"
        marker_path = output_dir / "inputs.sha256"
        ready_path = output_dir / "input_ready.json"

        train_content = _list_bytes(selected, "training")
        validation_content = _list_bytes(selected, "validation")
        yaml_content = _yaml_bytes(
            dataset_dir=dataset_dir,
            train_list=train_path,
            validation_list=validation_path,
        )
        selector_path = Path(__file__).resolve()
        generator_path = Path(proposal_dataset.__file__).resolve()
        data_sha256 = _stable_sha256(data_path, description="YOLO data YAML")
        inventory = {
            "schema_version": 1,
            "artifact_role": ARTIFACT_ROLE,
            "selection_contract": SELECTION_CONTRACT,
            "status": "selection_complete_not_replay_validated",
            "seed": seed,
            "quota_per_stratum": {
                "training": training_quota,
                "validation": validation_quota,
            },
            "classes": list(CLASS_NAMES),
            "strata": list(OUTPUT_STRATA),
            "source_contract": {
                "explicit_label_file_required": True,
                "background_prefers_current_explicit_empty_label": True,
                "historical_background_probe_requires_current_single_object_label": True,
                "historical_background_category_is_selection_only": True,
                "historical_background_category_is_not_ground_truth": True,
                "historical_observation_priority_is_selection_only": True,
                "current_batch1_replay_decides_emitted_category": True,
                "material_requires_exactly_one_valid_yolo_label": True,
                "multi_object_excluded": True,
                "cross_split_content_duplicates_quarantined": True,
                "same_split_conflicting_ground_truth_quarantined": True,
            },
            "bindings": {
                "data_path": _absolute_posix(data_path),
                "data_sha256": data_sha256,
                "dataset_dir": _absolute_posix(dataset_dir),
                "selector_path": _absolute_posix(selector_path),
                "selector_sha256": _stable_sha256(
                    selector_path, description="pilot selector"
                ),
                "proposal_generator_path": _absolute_posix(generator_path),
                "proposal_generator_sha256": _stable_sha256(
                    generator_path, description="proposal generator"
                ),
                "resolved_universe_sha256": universe_sha256,
                "resolved_universe_sources": universe_sources,
            },
            "historical_selection_evidence": {
                "used_for_selection_only": bool(historical_info),
                "ground_truth_authority": False,
                "replay_validation_authority": False,
                "background_category_authority": False,
                "old_manifest": historical_info,
                "drift_report": drift_info,
                "anchors_matched_to_eligible_sources": sum(
                    1 for record in scanned if not record.reasons and record.anchor
                ),
                "anchors_selected": sum(
                    1 for row in selected if bool(row["drift_anchor"])
                ),
                "anchors_priority_selected": sum(
                    1
                    for row in selected
                    if row["selection_reason"] == "drift_anchor_priority"
                ),
                "historical_observation_priority_selected": sum(
                    1
                    for row in selected
                    if row["selection_reason"]
                    == "historical_observation_priority_blake2"
                ),
                "eligible_material_historical_observed_counts": dict(
                    sorted(eligible_material_historical_observed_counts.items())
                ),
                "eligible_current_explicit_empty_counts": dict(
                    sorted(eligible_explicit_empty_counts.items())
                ),
                "eligible_historical_background_probe_counts": dict(
                    sorted(eligible_background_probe_counts.items())
                ),
            },
            "rejections": _rejection_summary(scanned),
            "eligible_counts": eligible_counts,
            "selected_counts": selected_counts,
            "selected_current_gt_counts": dict(
                sorted(selected_current_gt_counts.items())
            ),
            "selected_cohort_counts": dict(sorted(selected_cohort_counts.items())),
            "background_quota_composition": background_quota_composition,
            "quota_shortages": shortages,
            "full_quota_met": not any(shortages.values()),
            "selected_sources": selected,
            "authority": {
                "raw_generation_authorized": False,
                "validator_authority": False,
                "training_authorized": False,
                "blind_test_authorized": False,
                "production_deployment_authorized": False,
            },
        }
        inventory_content = _json_bytes(inventory)

        artifact_contents = {
            "train_pilot.txt": train_content,
            "validation_pilot.txt": validation_content,
            "pilot_dataset.yaml": yaml_content,
            "selection_inventory.json": inventory_content,
        }
        for name, content in artifact_contents.items():
            _publish_exclusive(output_dir / name, content)

        marker_lines = [
            f"{_sha256_bytes(artifact_contents[name])}  {name}"
            for name in sorted(artifact_contents)
        ]
        marker_content = ("\n".join(marker_lines) + "\n").encode("ascii")
        _publish_exclusive(marker_path, marker_content)

        # Bind the just-published bytes before the terminal ready marker.
        for name, expected in artifact_contents.items():
            actual = _stable_bytes(output_dir / name, description="published pilot input")
            if actual != expected:
                raise RuntimeError(f"published pilot input changed: {name}")
        if _stable_bytes(marker_path, description="pilot input marker") != marker_content:
            raise RuntimeError("published pilot input marker changed")
        _verify_selected_bindings(selected)
        if _stable_sha256(data_path, description="YOLO data YAML final rehash") != data_sha256:
            raise RuntimeError("YOLO data YAML changed during input build")
        if old_manifest is not None and historical_info is not None:
            if _stable_sha256(
                old_manifest, description="historical manifest final rehash"
            ) != historical_info["sha256"]:
                raise RuntimeError("historical manifest changed during input build")
        if drift_report is not None and drift_info is not None:
            if _stable_sha256(
                drift_report, description="historical drift report final rehash"
            ) != drift_info["sha256"]:
                raise RuntimeError("historical drift report changed during input build")

        ready = {
            "schema_version": 1,
            "artifact_role": ARTIFACT_ROLE,
            "status": "pilot_inputs_ready",
            "selection_contract": SELECTION_CONTRACT,
            "seed": seed,
            "selected_sources": len(selected),
            "selected_counts": selected_counts,
            "selected_current_gt_counts": dict(
                sorted(selected_current_gt_counts.items())
            ),
            "selected_cohort_counts": dict(sorted(selected_cohort_counts.items())),
            "background_quota_composition": background_quota_composition,
            "full_quota_met": not any(shortages.values()),
            "bindings": {
                "inputs_marker_sha256": _sha256_bytes(marker_content),
                "artifacts": {
                    name: _sha256_bytes(content)
                    for name, content in sorted(artifact_contents.items())
                },
                "resolved_universe_sha256": universe_sha256,
            },
            "historical_selection_only": bool(historical_info),
            "validator_authority": False,
            "training_authorized": False,
            "blind_test_authorized": False,
            "production_deployment_authorized": False,
        }
        # Terminal publication: no validation, writes, or reads may follow this call.
        _publish_exclusive(ready_path, _json_bytes(ready))
        return ready
    except BaseException as error:
        _publish_failure(output_dir, error)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--train-quota-per-stratum", type=int, default=250)
    parser.add_argument("--validation-quota-per-stratum", type=int, default=100)
    parser.add_argument("--old-manifest", type=Path)
    parser.add_argument("--drift-report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_pilot_inputs(
        data_path=args.data,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        training_quota=args.train_quota_per_stratum,
        validation_quota=args.validation_quota_per_stratum,
        old_manifest=args.old_manifest,
        drift_report=args.drift_report,
    )


if __name__ == "__main__":
    main()
