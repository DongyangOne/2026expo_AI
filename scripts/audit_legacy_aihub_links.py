"""Probe legacy converter source links by exact regenerated JPEG bytes, not indices as truth."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

CONVERTER_SHA256 = "52dee1d692f979c352f32510a74a84ea6e8892c08a47e1d29b8922b31ff95c22"
REMAINDER_SHA256 = "46cd95812d9e3fea0938d57a7e3ba48479dc897ca144a6e536c7e3cab8ae0c01"
AUTHORITY = dict(training_authorized=False, blind_test_authorized=False,
                 deployment_authorized=False, complete_original_lineage=False,
                 original_alias_uniqueness_proven=False)
SHA = re.compile(r"[0-9a-f]{64}")


def require(value, message):
    if not value:
        raise ValueError(message)


def _path(path):
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe input/output path")
    return path


def read_stable(path, maximum=128 * 1024**2):
    path = _path(path)
    before = path.stat()
    require(path.is_file() and before.st_size <= maximum, "file size/type guard")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        data = handle.read(maximum + 1)
        consumed = os.fstat(handle.fileno())
    after = _path(path).stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    require(len(data) == before.st_size and len({identity(s) for s in (before, opened, consumed, after)}) == 1
            and before.st_ctime_ns == after.st_ctime_ns and opened.st_ctime_ns == consumed.st_ctime_ns, "input changed during read")
    return data


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _remember(path, bindings, expected=None, maximum=128 * 1024**2):
    path = _path(path)
    data = read_stable(path, maximum)
    digest = _sha(data)
    require(expected is None or digest == expected, "input SHA mismatch")
    require(path not in bindings or bindings[path] == digest, "input changed between reads")
    bindings[path] = digest
    return data


def _rehash(bindings):
    for path, expected in bindings.items():
        require(_sha(read_stable(path)) == expected, "input changed after consumption")


def _decode(value):
    require(type(value) is str, "missing encoded path")
    raw = base64.b64decode(value, altchars=b"-_", validate=True)
    path = Path(os.fsdecode(raw))
    require(path.is_absolute() and ".." not in path.parts and b"\x00" not in raw, "invalid encoded path")
    return path, raw


def _encode(path):
    return base64.urlsafe_b64encode(os.fsencode(path)).decode("ascii")


def iter_pairs(split_dir, cap, *, remainder=False, max_index=None):
    """Historical order, including unsorted source directories; stop before later candidate stats."""
    src_map = defaultdict(list)
    src_base, labels = Path(split_dir) / "01.원천데이터", Path(split_dir) / "02.라벨링데이터"
    if src_base.exists():
        for directory in src_base.iterdir():
            if directory.is_dir():
                src_map[re.sub(r"_\d+$", "", directory.name)].append(directory)
    if not labels.exists():
        return
    index = 0
    for directory in sorted(labels.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(("TL_", "VL_")):
            continue
        prefix = ("TS_" if directory.name.startswith("TL_") else "VS_") + directory.name[3:]
        items = sorted(str(p) for p in directory.rglob("*.json"))
        if cap and len(items) > cap:
            chosen = {int(i * (len(items) / cap)) for i in range(cap)}
            items = [value for i, value in enumerate(items) if (i not in chosen if remainder else i in chosen)]
        elif remainder:
            items = []
        for label in items:
            hit = None
            for source_dir in src_map[prefix]:
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                    candidate = source_dir / (Path(label).stem + ext)
                    if candidate.exists():
                        hit = candidate
                        break
                if hit is not None:
                    break
            if hit is not None:
                yield hit, Path(label)
                if max_index is not None and index >= max_index:
                    return
                index += 1


def _legacy_kind(path):
    if path.parent.name != "images" or path.parent.parent.parent.name != "yolo_dataset_9class_v2":
        return None
    match = re.fullmatch(r"(train_r|train_|val_)([0-9]{7})\.jpg", path.name)
    if not match:
        return None
    kind = match[1].rstrip("_")
    require(path.parent.parent.name == ("val" if kind == "val" else "train"), "legacy split/path mismatch")
    return kind, int(match[2])


def _label_reference(namespace, payload, width, height):
    lines = []
    for ann in payload.get("ANNOTATION_INFO", []):
        box = namespace["points_to_bbox"](ann.get("POINTS", []))
        if box is None:
            continue
        cls = namespace["find_category_id"](ann.get("CLASS", ""), ann.get("DETAILS", ""))
        cx, cy, w, h = namespace["to_yolo"](box, width, height)
        if w > 0 and h > 0:
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return "".join(lines).encode("utf-8")


def _publish(path, payload):
    content = (json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()
    with _path(path).open("xb") as handle:
        handle.write(content)
    return content


def audit_links(*, protected_report, protected_report_sha256, v3_manifest, v3_manifest_sha256,
                converter, converter_sha256, remainder, remainder_sha256, dataset_dir, output, max_per_kind=3):
    require(type(max_per_kind) is int and max_per_kind >= 0, "invalid per-kind limit")
    inputs = [(protected_report, protected_report_sha256), (v3_manifest, v3_manifest_sha256),
              (converter, converter_sha256), (remainder, remainder_sha256)]
    inputs = [(_path(p), sha) for p, sha in inputs]
    require(all(type(sha) is str and SHA.fullmatch(sha) for _, sha in inputs), "invalid metadata pin")
    require(converter_sha256 == CONVERTER_SHA256 and remainder_sha256 == REMAINDER_SHA256, "unsupported historical converter code")
    dataset_dir, output = _path(dataset_dir), _path(output)
    require(dataset_dir.is_dir() and not output.exists(), "dataset missing or output already exists")
    require(not any(output.is_relative_to(p) for p in [dataset_dir, inputs[0][0].parent, inputs[1][0].parent]), "output overlaps input tree")
    bindings, results = {}, []
    protected_blob = _remember(inputs[0][0], bindings, inputs[0][1])
    snapshot = json.loads(protected_blob)
    require(type(snapshot) is dict and type(snapshot.get("records")) is list, "invalid protected snapshot")
    for row in snapshot["records"]:
        path, _ = _decode(row["source_path_b64"])
        if _legacy_kind(path) is not None:
            require(not output.is_relative_to(path.parent.parent.parent), "output overlaps legacy dataset")
    output.mkdir(parents=True, exist_ok=False)
    try:
        blobs = [protected_blob, *(_remember(p, bindings, sha) for p, sha in inputs[1:])]
        _remember(Path(__file__), bindings)
        require(snapshot.get("schema") == "protected_image_fingerprint_snapshot.v1"
                and snapshot.get("status") == "snapshot_complete" and snapshot.get("missing_sources") == 0
                and snapshot.get("training_authorized") is False and snapshot.get("deployment_authorized") is False
                and snapshot.get("expected_sources") == snapshot.get("verified_sources") == len(snapshot["records"]), "invalid protected snapshot")
        require(not (inputs[0][0].parent / "failed.json").exists(), "protected snapshot failed")
        pool, seen_ids = defaultdict(list), set()
        for row in csv.DictReader(io.StringIO(blobs[1].decode("utf-8-sig"))):
            sid = row.get("source_id", "")
            require(re.fullmatch(r"[0-9a-f]{20}", sid) and sid not in seen_ids, "invalid/duplicate v3 source ID")
            require(row.get("split") in ("training", "validation"), "invalid v3 split")
            seen_ids.add(sid)
            pool[_decode(row["source_path_b64"])[1]].append(sid)
        groups, seen_shas = defaultdict(dict), set()
        for row in snapshot["records"]:
            sha = row.get("source_sha256")
            require(type(sha) is str and SHA.fullmatch(sha) and sha not in seen_shas, "invalid/duplicate protected SHA")
            seen_shas.add(sha)
            path, _ = _decode(row["source_path_b64"])
            identity = _legacy_kind(path)
            if identity is not None:
                kind, index = identity
                require(index not in groups[kind], "duplicate legacy index")
                require(not output.is_relative_to(path.parent.parent.parent), "output overlaps legacy dataset")
                groups[kind][index] = (path, sha)
        require(bool(groups), "no protected legacy sources")
        namespaces = []
        for path, code in zip((inputs[2][0], inputs[3][0]), blobs[2:]):
            namespace = {"__name__": "legacy_reference", "__file__": str(path)}
            exec(compile(code, str(path), "exec"), namespace)
            namespaces.append(namespace)
        cv2.setNumThreads(1)
        for kind in ("train", "train_r", "val"):
            indices = sorted(groups[kind])[:max_per_kind or None]
            if not indices:
                continue
            chosen = set(indices)
            split = "Validation" if kind == "val" else "Training"
            pairs = iter_pairs(dataset_dir / "01-1.정식개방데이터" / split,
                               2000 if kind == "val" else 15000, remainder=kind == "train_r", max_index=indices[-1])
            found = {i: pair for i, pair in enumerate(pairs) if i in chosen}
            for index in indices:
                legacy, expected = groups[kind][index]
                legacy_label = legacy.parent.parent / "labels" / (legacy.stem + ".txt")
                _remember(legacy, bindings, expected)
                sidecar = _remember(legacy_label, bindings)
                result = dict(kind=kind, index=index, legacy_sha256=expected, legacy_path_b64=_encode(legacy),
                              legacy_label_sha256=_sha(sidecar), status="unresolved", reason="candidate_index_unavailable")
                if index in found:
                    source, label = map(_path, found[index])
                    require(source.is_relative_to(dataset_dir) and label.is_relative_to(dataset_dir), "candidate outside dataset")
                    image_data, label_data = _remember(source, bindings), _remember(label, bindings, maximum=1024**2)
                    pixels = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                    require(pixels is not None, "candidate image cannot be decoded")
                    h, w = pixels.shape[:2]
                    scale = 640 / float(max(h, w))
                    resized = cv2.resize(pixels, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else pixels
                    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    require(ok, "JPEG reproduction failed")
                    ids = sorted(pool.get(os.fsencode(source), []))
                    reference = _label_reference(namespaces[kind == "train_r"], json.loads(label_data), w, h)
                    matched = _sha(encoded.tobytes()) == expected
                    result.update(status="verified_source_link" if matched else "unresolved", reason="exact_legacy_jpeg_bytes" if matched else "legacy_jpeg_bytes_differ",
                                  source_path_b64=_encode(source), source_sha256=_sha(image_data), annotation_path_b64=_encode(label), annotation_sha256=_sha(label_data),
                                  regenerated_legacy_sha256=_sha(encoded.tobytes()), v3_source_ids=ids,
                                  membership="in_v3" if ids else "outside_v3", legacy_label_reproduction_matches=reference == sidecar)
                results.append(result)
                print(json.dumps({"processed": len(results), "kind": kind, "status": result["status"]}), flush=True)
        report = dict(schema="legacy_aihub_source_link_probe.v1", status="probe_complete", records=results,
                      protected_legacy_counts={k: len(v) for k, v in groups.items()}, max_per_kind=max_per_kind,
                      partial_selection=len(results) < sum(map(len, groups.values())), candidate_index_is_search_only=True,
                      status_counts=dict(Counter(r["status"] for r in results)), cv2_version=cv2.__version__,
                      cv2_build_sha256=_sha(cv2.getBuildInformation().encode()),
                      metadata_and_consumed_inputs=[dict(path_b64=_encode(p), sha256=s) for p, s in bindings.items()], **AUTHORITY)
        _rehash(bindings)
        content = _publish(output / "report.json", report)
        _rehash(bindings)
        require(read_stable(output / "report.json") == content, "report changed during publication")
        return report
    except BaseException as exc:
        _publish(output / "failed.json", dict(status="failed", exception_type=type(exc).__name__, completed_records=len(results), **AUTHORITY))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protected-report", "v3-manifest", "converter", "remainder"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-kind", type=int, default=3)
    audit_links(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
