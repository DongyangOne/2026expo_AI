"""Build a source-only fingerprint input from verified full legacy links.

This writes no images or training labels. The existing fingerprint auditor must
still consume the actual recovered originals before the cohort uses their pHash.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

try:
    from scripts import plan_aihub_original_cohort as planner
    from scripts import audit_protected_image_fingerprints as fingerprint
except ModuleNotFoundError:
    import plan_aihub_original_cohort as planner
    import audit_protected_image_fingerprints as fingerprint


def build_inventory(*, link_report: Path, link_sha256: str, protected_report: Path,
                    protected_sha256: str, output: Path) -> dict:
    link_report, protected_report = map(fingerprint._path, (link_report, protected_report))
    output = fingerprint._path(output, exists=False)
    if output.exists() or any(output.is_relative_to(path.parent) for path in (link_report, protected_report)):
        raise ValueError("fresh output outside input report trees required")
    protected = planner.load_pinned(protected_report, protected_sha256)
    checked = planner.validate_legacy_link_report(
        link_report=link_report, link_sha256=link_sha256, protected_report=protected_report,
        protected_sha256=protected_sha256, protected=protected["records"],
    )
    legacy_roles = {row["source_sha256"]: row["roles"] for row in protected["records"]}
    groups = defaultdict(lambda: {"paths": set(), "roles": set()})
    for row in checked["verified_records"]:
        sha = row["source_sha256"]
        raw_path = base64.b64decode(row["source_path_b64"], altchars=b"-_", validate=True)
        path = fingerprint._path(Path(os.fsdecode(raw_path)))
        if output.is_relative_to(path.parent):
            raise ValueError("output overlaps recovered original source directory")
        roles = legacy_roles[row["legacy_sha256"]]
        if (type(roles) is not list or not roles
                or any(type(role) is not str or role not in fingerprint.ROLES for role in roles)
                or len(roles) != len(set(roles))):
            raise ValueError("recovered source must inherit actual protected roles")
        groups[sha]["paths"].add(os.fsencode(path))
        groups[sha]["roles"].update(roles)
    if not groups:
        raise ValueError("no verified recovered originals to fingerprint")
    records = [{"sha256": sha, "path": os.fsdecode(min(value["paths"])),
                "roles": sorted(value["roles"])} for sha, value in sorted(groups.items())]
    bindings = list(checked["bindings"])
    for module in (Path(__file__), Path(planner.__file__), Path(planner.original.__file__), Path(fingerprint.__file__)):
        module = fingerprint._path(module.absolute())
        bindings.append({"path": str(module), "sha256": planner.original.digest_file(module)})
    if len({row["path"] for row in bindings}) != len(bindings):
        raise ValueError("duplicate inventory metadata binding")
    data = {"records": records, "metadata_bindings": bindings}
    payload = (json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    summary = {"status": "fingerprint_inventory_prepared", "unique_original_sources": len(records),
               "verified_legacy_references": len(checked["verified_records"]),
               "expected_legacy_references": checked["expected_legacy_references"],
               "unresolved": checked["unresolved"], "inventory_sha256": hashlib.sha256(payload).hexdigest(),
               "original_pixels_fingerprinted": False, "training_authorized": False,
               "blind_test_authorized": False, "deployment_authorized": False}
    summary_payload = (json.dumps(summary, ensure_ascii=True, sort_keys=True) + "\n").encode()
    reports = (link_report, protected_report)
    planner.recheck_legacy_bindings(bindings, reports)
    output.mkdir(parents=True, exist_ok=False)
    identity = (output.stat().st_dev, output.stat().st_ino)
    try:
        with (output / "inventory.json").open("xb") as handle:
            handle.write(payload)
        with (output / "summary.json").open("xb") as handle:
            handle.write(summary_payload)
        planner.recheck_legacy_bindings(bindings, reports)
        if (output / "inventory.json").read_bytes() != payload:
            raise ValueError("inventory changed during publication")
        if (output / "summary.json").read_bytes() != summary_payload:
            raise ValueError("summary changed during publication")
    except BaseException:
        fingerprint._path(output)
        if (output.stat().st_dev, output.stat().st_ino) == identity:
            inventory_path = output / "inventory.json"
            if inventory_path.is_file() and not inventory_path.is_symlink() and inventory_path.read_bytes() == payload:
                inventory_path.unlink()  # Remove only this run's exact failed publication.
            with (output / "failed.json").open("x", encoding="utf-8") as handle:
                json.dump({"status": "failed", "training_authorized": False, "deployment_authorized": False}, handle)
        raise
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link-report", type=Path, required=True)
    parser.add_argument("--link-sha256", required=True)
    parser.add_argument("--protected-report", type=Path, required=True)
    parser.add_argument("--protected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build_inventory(**vars(parser.parse_args())), sort_keys=True), flush=True)
