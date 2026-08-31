import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from scripts import evaluate_v4_candidate_judge as candidate_judge
from scripts.evaluate_v4_candidate_judge import (
    GateThresholds,
    READY_MARKER_NAME,
    REPORT_NAME,
    _lineage_digest,
    evaluate_v4_candidate,
    main,
)
from scripts.replay_v4_candidate_metrics import replay_validation


MATERIAL_CLASSES = [
    "can",
    "pet",
    "paper",
    "plastic",
    "styrofoam",
    "vinyl",
    "glass",
    "battery",
    "fluorescent",
]
STRICT_FIELDS = [
    "filepath",
    "split",
    "source_id",
    "material",
    "category",
    "dent",
    "label",
    "foreign_material",
    "source_object_count",
    "crop_object_count",
    "sample_id",
    "source_sha256",
    "image_sha256",
    "object_group",
    "capture_session",
    "role",
    "fold",
    "origin",
]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _Info:
    def __init__(self, name: str, shape: list[object]):
        self.name = name
        self.shape = shape
        self.type = "tensor(float)"


class _Session:
    def __init__(self, predictions: list[list[int]]):
        self.predictions = predictions
        self.offset = 0

    def get_inputs(self):
        return [_Info("img", ["batch", 3, 1, 1])]

    def get_outputs(self):
        return [
            _Info("objectness", ["batch", 2]),
            _Info("material", ["batch", 9]),
        ]

    def run(self, output_names, inputs):
        assert output_names == ["objectness", "material"]
        batch = next(iter(inputs.values())).shape[0]
        selected = self.predictions[self.offset : self.offset + batch]
        self.offset += batch
        objectness = np.full((batch, 2), -2.0, dtype=np.float32)
        material = np.full((batch, 9), -3.0, dtype=np.float32)
        for index, (objectness_id, material_id) in enumerate(selected):
            objectness[index, objectness_id] = 2.0
            material[index, material_id] = 3.0
        return [objectness, material]


def _inference_spec() -> dict[str, object]:
    return {
        "format_version": 1,
        "artifact_role": "offline_candidate_spec_not_production_authorization",
        "detector_classes": MATERIAL_CLASSES,
        "crop": {
            "source": "selected_detector_bbox",
            "clip_to_source": True,
            "resize": "aspect_preserving_letterbox",
            "preprocessing_contract_version": "offline_verifier_crop.v1",
            "bbox_rounding": {"min_edges": "floor", "max_edges": "ceil"},
            "resize_rounding": "nearest_ties_to_even",
            "resize_interpolation": {
                "downscale": "INTER_AREA",
                "upscale": "INTER_LINEAR",
                "equal": "identity",
            },
            "letterbox_alignment": {
                "horizontal": "center_floor",
                "vertical": "center_floor",
            },
            "color_conversion": "BGR_TO_RGB",
            "padding_ratio": 0.0,
            "letterbox_size": 1,
            "letterbox_fill": 114,
            "normalization": {
                "layout": "NCHW",
                "dtype": "float32",
                "input_scale": 255.0,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "verifier_contract": {
            "objectness": "binary_material_vs_background",
            "material": "positive_only_9_class_softmax",
            "blind_gate_threshold_overrides": False,
        },
        "safety": {"production_model_replacement": False},
    }


def _materialize_images(root: Path, rows: list[dict[str, object]]) -> None:
    image_directory = root / "images"
    image_directory.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        image = image_directory / f"{row['sample_id']}.ppm"
        material = int(row["material"])
        red = 250 if material == 9 else 10 + material * 20
        pixel = bytes((red, index % 256, (index // 256) % 256))
        image.write_bytes(b"P6\n1 1\n255\n" + pixel)
        row["filepath"] = image.relative_to(root).as_posix()
        row["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()


def _write_tiny_onnx(path: Path, *, mode: str) -> None:
    red_targets = np.asarray(
        [((10 + material * 20) / 255.0 - 0.485) / 0.229 for material in range(9)],
        dtype=np.float32,
    )
    threshold = np.float32(((210 / 255.0) - 0.485) / 0.229)
    objectness_weight = np.zeros((3, 2), dtype=np.float32)
    objectness_bias = np.asarray([-threshold, threshold], dtype=np.float32)
    objectness_weight[0] = [1.0, -1.0]
    if mode == "all_background":
        objectness_weight.fill(0.0)
        objectness_bias[:] = [1.0, -1.0]
    elif mode == "all_material":
        objectness_weight.fill(0.0)
        objectness_bias[:] = [-1.0, 1.0]
    material_weight = np.zeros((3, 9), dtype=np.float32)
    material_weight[0] = 2.0 * red_targets
    material_bias = -(red_targets ** 2)
    if mode == "material_zero_wrong":
        material_weight[:, 0] = 0.0
        material_bias[0] = -100.0
    input_info = helper.make_tensor_value_info(
        "img", TensorProto.FLOAT, ["batch", 3, 1, 1]
    )
    objectness_info = helper.make_tensor_value_info(
        "objectness", TensorProto.FLOAT, ["batch", 2]
    )
    material_info = helper.make_tensor_value_info(
        "material", TensorProto.FLOAT, ["batch", 9]
    )
    nodes = [
        helper.make_node("Flatten", ["img"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "objectness_weight"], ["objectness_mm"]),
        helper.make_node("Add", ["objectness_mm", "objectness_bias"], ["objectness"]),
        helper.make_node("MatMul", ["flat", "material_weight"], ["material_mm"]),
        helper.make_node("Add", ["material_mm", "material_bias"], ["material"]),
    ]
    graph = helper.make_graph(
        nodes,
        f"tiny-{path.name}-{mode}",
        [input_info],
        [objectness_info, material_info],
        [
            numpy_helper.from_array(objectness_weight, "objectness_weight"),
            numpy_helper.from_array(objectness_bias, "objectness_bias"),
            numpy_helper.from_array(material_weight, "material_weight"),
            numpy_helper.from_array(material_bias, "material_bias"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name=f"judge-test-{path.name}",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


def _row(role: str, material: int, index: int) -> dict[str, object]:
    token = f"{role}-{material}-{index}"
    return {
        "filepath": f"images/{token}.jpg",
        "split": "validation" if role == "model_validation" else "training",
        "source_id": f"source-{token}",
        "material": material,
        "category": "background" if material == 9 else MATERIAL_CLASSES[material],
        "dent": -1,
        "label": -1,
        "foreign_material": -1,
        "source_object_count": 0 if material == 9 else 1,
        "crop_object_count": 0 if material == 9 else 1,
        "sample_id": f"sample-{token}",
        "source_sha256": _sha(f"source-{token}"),
        "image_sha256": _sha(f"image-{token}"),
        "object_group": f"object-{token}",
        "capture_session": f"session-{token}",
        "role": role,
        "fold": f"fold-{role}",
        "origin": "judge-test",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=STRICT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _normalized_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return [{key: str(value) for key, value in row.items()} for row in rows]


def _metrics(
    *,
    background_support: int,
    background_recall: float = 0.95,
    material_objectness_recall: float = 0.98,
    material_recalls: list[float] | None = None,
) -> dict[str, object]:
    recalls = material_recalls or [0.90] * 9
    objectness_recalls = [background_recall, material_objectness_recall]
    return {
        "validation": {
            "objectness": {
                "count": background_support + 9,
                "support": [background_support, 9],
                "per_class_recall": objectness_recalls,
                "balanced_accuracy": sum(objectness_recalls) / 2,
            },
            "material": {
                "count": 9,
                "support": [1] * 9,
                "per_class_recall": recalls,
                "balanced_accuracy": sum(recalls) / 9,
            },
        }
    }


def _confusion_metrics(confusion: list[list[int]]) -> dict[str, object]:
    support = [sum(row) for row in confusion]
    recalls = [confusion[index][index] / support[index] for index in range(len(support))]
    count = sum(support)
    return {
        "count": count,
        "support": support,
        "per_class_recall": recalls,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "accuracy": sum(confusion[index][index] for index in range(len(support))) / count,
        "confusion": confusion,
    }


def _replay_rows_and_metrics(
    rows: list[dict[str, object]], *, mode: str = "correct"
) -> tuple[list[dict[str, object]], dict[str, object]]:
    replay_rows: list[dict[str, object]] = []
    objectness_confusion = [[0, 0], [0, 0]]
    material_confusion = [[0 for _ in range(9)] for _ in range(9)]
    for row in sorted(
        (item for item in rows if item["role"] == "model_validation"),
        key=lambda item: str(item["sample_id"]),
    ):
        truth_objectness = 0 if row["material"] == 9 else 1
        truth_material = None if truth_objectness == 0 else int(row["material"])
        if mode == "all_background":
            predicted_objectness = 0
        elif mode == "all_material":
            predicted_objectness = 1
        else:
            predicted_objectness = truth_objectness
        predicted_material = 0 if truth_material is None else truth_material
        if mode == "material_zero_wrong" and truth_material == 0:
            predicted_material = 1
        objectness_logits = [-2.0, -2.0]
        objectness_logits[predicted_objectness] = 2.0
        material_logits = [-3.0] * 9
        material_logits[predicted_material] = 3.0
        objectness_confusion[truth_objectness][predicted_objectness] += 1
        if truth_material is not None:
            material_confusion[truth_material][predicted_material] += 1
        replay_rows.append(
            {
                "sample_id": row["sample_id"],
                "source_sha256": row["source_sha256"],
                "image_sha256": row["image_sha256"],
                "object_group": row["object_group"],
                "capture_session": row["capture_session"],
                "fold": row["fold"],
                "role": "model_validation",
                "truth_objectness": truth_objectness,
                "truth_material": truth_material,
                "input_tensor_sha256": _sha(f"tensor-{row['sample_id']}"),
                "objectness_logits": objectness_logits,
                "material_logits": material_logits,
                "predicted_objectness": predicted_objectness,
                "predicted_material_head": predicted_material,
                "cascaded_material": (
                    predicted_material if predicted_objectness == 1 else None
                ),
            }
        )
    return replay_rows, {
        "objectness": _confusion_metrics(objectness_confusion),
        "material": _confusion_metrics(material_confusion),
    }


def _write_replay_bundle(
    *,
    rows: list[dict[str, object]],
    manifest: Path,
    metadata: Path,
    model: Path,
    spec: Path,
    predictions: Path,
    attestation: Path,
    mode: str = "correct",
) -> None:
    replay_rows, replay_metrics = _replay_rows_and_metrics(rows, mode=mode)
    claimed_metrics = {
        "validation": {
            name: {
                key: value
                for key, value in head.items()
                if key in {"count", "support", "per_class_recall", "balanced_accuracy"}
            }
            for name, head in replay_metrics.items()
        }
    }
    _write_tiny_onnx(model, mode=mode)
    _write_metadata(
        metadata,
        manifest,
        rows,
        metrics=claimed_metrics,
        onnx_name=model.name,
    )
    predictions.unlink(missing_ok=True)
    attestation.unlink(missing_ok=True)
    replay_validation(
        manifest_paths=[manifest],
        verifier_onnx=model,
        verifier_metadata=metadata,
        inference_spec=spec,
        output_jsonl=predictions,
        output_attestation=attestation,
    )


def _write_metadata(
    path: Path,
    manifest: Path,
    rows: list[dict[str, object]],
    *,
    metrics: dict[str, object] | None = None,
    onnx_name: str = "candidate.onnx",
) -> None:
    role_counts: dict[str, int] = {}
    for row in rows:
        role = str(row["role"])
        role_counts[role] = role_counts.get(role, 0) + 1
    metadata = {
        "format_version": 3,
        "architecture": "multitask_crop_verifier",
        "candidate_only": True,
        "production_runtime_modified": False,
        "onnx": onnx_name,
        "model_config": {"input_size": 1},
        "preprocessing": {
            "color_space": "RGB",
            "resize": [1, 1],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "objectness_classes": ["background", "material"],
        "material_classes": MATERIAL_CLASSES,
        "output_contract": {
            "version": "multitask_verifier.v3",
            "output_order": ["objectness", "material"],
            "material_background_class_id": None,
            "outputs": [
                {
                    "name": "objectness",
                    "class_names": ["background", "material"],
                    "shape": ["batch", 2],
                },
                {
                    "name": "material",
                    "class_names": MATERIAL_CLASSES,
                    "shape": ["batch", 9],
                },
            ],
        },
        "best_metrics": metrics
        or _metrics(
            background_support=sum(
                row["role"] == "model_validation" and row["material"] == 9
                for row in rows
            )
        ),
        "manifest_summary": {
            "strict": True,
            "rows": len(rows),
            "lineage_sha256": _lineage_digest(_normalized_rows(rows)),
            "input_manifests": [
                {
                    "path": str(manifest.resolve()),
                    "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                }
            ],
            "role_counts": dict(sorted(role_counts.items())),
        },
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _visual_judges() -> list[dict[str, object]]:
    judges: list[dict[str, object]] = []
    for suffix, family in (("a", "llava"), ("b", "minicpmv")):
        manifest_sha = _sha(f"model-manifest-{suffix}")
        tag = f"{family}:test"
        tags_response = {
            "models": [
                {
                    "name": tag,
                    "model": tag,
                    "digest": manifest_sha,
                    "details": {"family": family, "families": [family]},
                }
            ]
        }
        show_response = {
            "details": {"family": family, "families": [family]},
            "capabilities": ["completion", "vision"],
        }
        judges.append(
            {
                "judge_id": f"judge-{suffix}",
                "model_family": family,
                "ollama_model": tag,
                "model_manifest_sha256": manifest_sha,
                "model_weight_layer_sha256": _sha(f"model-weight-{suffix}"),
                "model_config_sha256": _sha(f"model-config-{suffix}"),
                "model_config_families": [family],
                "judge_spec_sha256": _sha(f"judge-spec-{suffix}"),
                "server_model_digest": manifest_sha,
                "server_model_families": [family],
                "server_capabilities": ["completion", "vision"],
                "server_tags_response_sha256": hashlib.sha256(
                    _canonical(tags_response)
                ).hexdigest(),
                "server_show_response_sha256": hashlib.sha256(
                    _canonical(show_response)
                ).hexdigest(),
                "preflight_tags_response": tags_response,
                "preflight_show_response": show_response,
                "postflight_tags_response": tags_response,
                "postflight_show_response": show_response,
            }
        )
    return judges


def _write_visual_bundle(
    report_path: Path,
    evidence_path: Path,
    rows: list[dict[str, object]],
    *,
    veto_sample: str | None = None,
    forbidden_raw_field: str | None = None,
    official_ollama_http: bool = True,
    postflight_digest_override: str | None = None,
) -> None:
    prompt_sha = _sha("fixed-prompt")
    runner_sha = _sha("fixed-runner")
    judges = _visual_judges()
    if postflight_digest_override is not None:
        postflight = json.loads(
            json.dumps(judges[0]["postflight_tags_response"])
        )
        postflight["models"][0]["digest"] = postflight_digest_override
        judges[0]["postflight_tags_response"] = postflight
    for judge in judges:
        judge["preflight_tags_response_sha256"] = hashlib.sha256(
            _canonical(judge["preflight_tags_response"])
        ).hexdigest()
        judge["preflight_show_response_sha256"] = hashlib.sha256(
            _canonical(judge["preflight_show_response"])
        ).hexdigest()
        judge["postflight_tags_response_sha256"] = hashlib.sha256(
            _canonical(judge["postflight_tags_response"])
        ).hexdigest()
        judge["postflight_show_response_sha256"] = hashlib.sha256(
            _canonical(judge["postflight_show_response"])
        ).hexdigest()
    postflight_identity_set = [
        {
            "judge_id": judge["judge_id"],
            "postflight_tags_response_sha256": judge[
                "postflight_tags_response_sha256"
            ],
            "postflight_show_response_sha256": judge[
                "postflight_show_response_sha256"
            ],
        }
        for judge in judges
    ]
    postflight_identity_sha = hashlib.sha256(
        _canonical(postflight_identity_set)
    ).hexdigest()
    pair_seed = {
        "evidence_schema": "independent_visual_judge_evidence.v1",
        "evidence_schema_version": 1,
        "evidence_pair_contract": (
            "votes_share_pair_id_and_report_pins_exact_jsonl_sha256.v1"
        ),
        "input_manifest_sha256": _sha("visual-input-manifest"),
        "prompt_sha256": prompt_sha,
        "runner_script_sha256": runner_sha,
        "official_ollama_http": official_ollama_http,
        "authoritative_evidence": official_ollama_http,
        "judges": [
            {
                "judge_id": judge["judge_id"],
                "judge_spec_sha256": judge["judge_spec_sha256"],
                "model_manifest_sha256": judge["model_manifest_sha256"],
                "model_weight_layer_sha256": judge[
                    "model_weight_layer_sha256"
                ],
                "model_config_sha256": judge["model_config_sha256"],
                "preflight_tags_response_sha256": judge[
                    "preflight_tags_response_sha256"
                ],
                "preflight_show_response_sha256": judge[
                    "preflight_show_response_sha256"
                ],
            }
            for judge in judges
        ],
        "postflight_identity_set": postflight_identity_set,
    }
    evidence_pair_id = hashlib.sha256(_canonical(pair_seed)).hexdigest()
    backgrounds = [
        row
        for row in rows
        if row["role"] == "model_validation" and row["material"] == 9
    ]
    votes: list[dict[str, object]] = []
    verdict_counts = {
        str(judge["judge_id"]): {
            "ambiguous": 0,
            "background": 0,
            "material": 0,
        }
        for judge in judges
    }
    for row in backgrounds:
        for judge in judges:
            verdict = (
                "ambiguous"
                if veto_sample == row["sample_id"]
                and judge["judge_id"] == "judge-b"
                else "background"
            )
            raw_response: dict[str, object] = {
                "done": True,
                "message": {
                    "content": json.dumps(
                        {"verdict": verdict}, separators=(",", ":")
                    )
                },
                "model": judge["ollama_model"],
            }
            if forbidden_raw_field and not votes:
                raw_response[forbidden_raw_field] = "background"
            raw_sha = hashlib.sha256(_canonical(raw_response)).hexdigest()
            vote_payload = {
                "schema_version": 1,
                "vote_schema": "independent_visual_judge_vote.v1",
                "evidence_schema": "independent_visual_judge_evidence.v1",
                "evidence_schema_version": 1,
                "canonical_json_contract": (
                    "utf8_sorted_keys_compact_separators_trailing_newline.v1"
                ),
                "evidence_pair_contract": (
                    "votes_share_pair_id_and_report_pins_exact_jsonl_sha256.v1"
                ),
                "evidence_pair_id": evidence_pair_id,
                "official_ollama_http": official_ollama_http,
                "authoritative_evidence": official_ollama_http,
                "postflight_identity_set_sha256": postflight_identity_sha,
                "judge_id": judge["judge_id"],
                "model_family": judge["model_family"],
                "ollama_model": judge["ollama_model"],
                "sample_id": row["sample_id"],
                "verdict": verdict,
                "prompt_sha256": prompt_sha,
                "runner_script_sha256": runner_sha,
                "model_manifest_sha256": judge["model_manifest_sha256"],
                "model_weight_layer_sha256": judge[
                    "model_weight_layer_sha256"
                ],
                "model_config_sha256": judge["model_config_sha256"],
                "server_model_digest": judge["server_model_digest"],
                "server_model_families": judge["server_model_families"],
                "server_capabilities": judge["server_capabilities"],
                "server_digest_contract": (
                    "ollama_api_tags_digest_equals_sha256_of_local_oci_tag_manifest_bytes.v1"
                ),
                "server_tags_response_sha256": judge[
                    "server_tags_response_sha256"
                ],
                "server_show_response_sha256": judge[
                    "server_show_response_sha256"
                ],
                "postchat_tags_response": judge["preflight_tags_response"],
                "postchat_tags_response_sha256": judge[
                    "server_tags_response_sha256"
                ],
                "postchat_server_model_digest": judge["server_model_digest"],
                "postchat_server_model_families": judge[
                    "server_model_families"
                ],
                "canonical_raw_response": raw_response,
                "canonical_raw_response_sha256": raw_sha,
                "source_sha256": row["source_sha256"],
                "crop_sha256": row["image_sha256"],
            }
            vote = {
                **vote_payload,
                "vote_binding_sha256": hashlib.sha256(
                    _canonical(vote_payload)
                ).hexdigest(),
            }
            votes.append(vote)
            verdict_counts[str(judge["judge_id"])][verdict] += 1
    evidence_bytes = b"".join(_canonical(vote) for vote in votes)
    evidence_path.write_bytes(evidence_bytes)
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
    report = {
        "schema_version": 1,
        "report_schema": "independent_visual_judge_report.v1",
        "evidence_schema": "independent_visual_judge_evidence.v1",
        "evidence_schema_version": 1,
        "canonical_json_contract": (
            "utf8_sorted_keys_compact_separators_trailing_newline.v1"
        ),
        "evidence_pair_contract": (
            "votes_share_pair_id_and_report_pins_exact_jsonl_sha256.v1"
        ),
        "evidence_pair_id": evidence_pair_id,
        "evidence_pair_seed": pair_seed,
        "postflight_identity_set": postflight_identity_set,
        "postflight_identity_set_sha256": postflight_identity_sha,
        "official_ollama_http": official_ollama_http,
        "authoritative_evidence": official_ollama_http,
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
        "input_manifest_sha256": _sha("visual-input-manifest"),
        "prompt_version": "independent_background_material_judge.v1",
        "prompt_sha256": prompt_sha,
        "runner_script_sha256": runner_sha,
        "server_digest_contract": (
            "ollama_api_tags_digest_equals_sha256_of_local_oci_tag_manifest_bytes.v1"
        ),
        "row_count": len(backgrounds),
        "judge_count": len(judges),
        "vote_count": len(votes),
        "expected_vote_count": len(backgrounds) * len(judges),
        "coverage": {"every_judge_exactly_one_vote_per_row": True},
        "candidate_metadata_exposed_to_prompt": False,
        "raw_response_content_stored": True,
        "request_content_stored": False,
        "image_content_stored": False,
        "evidence_jsonl_sha256": evidence_sha,
        "evidence_jsonl_line_count": len(votes),
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
                **judge,
                "preflight_tags_response_sha256": judge[
                    "preflight_tags_response_sha256"
                ],
                "preflight_show_response_sha256": judge[
                    "preflight_show_response_sha256"
                ],
                "postflight_tags_response_sha256": judge[
                    "postflight_tags_response_sha256"
                ],
                "postflight_show_response_sha256": judge[
                    "postflight_show_response_sha256"
                ],
                "postflight_identity_matches_preflight": True,
                "prompt_sha256": prompt_sha,
                "runner_script_sha256": runner_sha,
            }
            for judge in judges
        ],
        "verdict_counts_by_judge": verdict_counts,
        "results_jsonl_sha256": evidence_sha,
        "generated_by": "scripts/run_independent_visual_judges.py",
    }
    report_path.write_bytes(_canonical(report))


def _write_trusted_policy(
    path: Path,
    *,
    rows: list[dict[str, object]],
    manifest: Path,
    baseline_model: Path,
    baseline_metadata: Path,
) -> None:
    judges = [
        {
            key: judge[key]
            for key in (
                "judge_id",
                "model_family",
                "ollama_model",
                "model_manifest_sha256",
                "model_weight_layer_sha256",
                "model_config_sha256",
                "server_model_digest",
                "server_model_families",
            )
        }
        for judge in _visual_judges()
    ]
    policy = {
        "schema_version": 1,
        "policy_schema": "v4_candidate_judge_trusted_policy.v1",
        "artifact_role": "trusted_frozen_v4_candidate_judge_policy",
        "frozen": True,
        "approved_baseline": {
            "model_sha256": hashlib.sha256(baseline_model.read_bytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(
                baseline_metadata.read_bytes()
            ).hexdigest(),
        },
        "strict_validation": {
            "manifest_sha256": [hashlib.sha256(manifest.read_bytes()).hexdigest()],
            "lineage_sha256": _lineage_digest(_normalized_rows(rows)),
        },
        "visual_judges": {
            "input_manifest_sha256": _sha("visual-input-manifest"),
            "prompt_sha256": _sha("fixed-prompt"),
            "runner_script_sha256": _sha("fixed-runner"),
            "judges": judges,
        },
    }
    path.write_bytes(_canonical(policy))


def _pin_test_trusted_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    repository_root: Path,
    policy_path: Path,
) -> None:
    monkeypatch.setattr(candidate_judge, "REPO_ROOT", repository_root)
    monkeypatch.setattr(
        candidate_judge,
        "TRUSTED_POLICY_RELATIVE_PATH",
        policy_path.relative_to(repository_root),
    )
    monkeypatch.setattr(
        candidate_judge,
        "APPROVED_TRUSTED_POLICY_SHA256",
        hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    )


@pytest.fixture
def candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    rows = [_row("train", material, 0) for material in range(10)]
    rows.extend(_row("model_validation", material, 0) for material in range(9))
    rows.extend(_row("model_validation", 9, index) for index in range(200))
    _materialize_images(tmp_path, rows)
    manifest = tmp_path / "strict.csv"
    metadata = tmp_path / "multitask_verifier_metadata.json"
    baseline_metadata = tmp_path / "baseline_multitask_verifier_metadata.json"
    candidate_onnx = tmp_path / "candidate.onnx"
    baseline_onnx = tmp_path / "baseline.onnx"
    spec = tmp_path / "inference-spec.json"
    replay_predictions = tmp_path / "candidate-replay.jsonl"
    replay_attestation = tmp_path / "candidate-replay-attestation.json"
    baseline_replay_predictions = tmp_path / "baseline-replay.jsonl"
    baseline_replay_attestation = tmp_path / "baseline-replay-attestation.json"
    trusted_policy = tmp_path / "trusted-policy.json"
    visual_report = tmp_path / "visual-judge-report.json"
    visual_evidence = tmp_path / "visual-judge-evidence.jsonl"
    _write_csv(manifest, rows)
    spec.write_text(json.dumps(_inference_spec()), encoding="utf-8")
    _write_replay_bundle(
        rows=rows,
        manifest=manifest,
        metadata=metadata,
        model=candidate_onnx,
        spec=spec,
        predictions=replay_predictions,
        attestation=replay_attestation,
    )
    _write_replay_bundle(
        rows=rows,
        manifest=manifest,
        metadata=baseline_metadata,
        model=baseline_onnx,
        spec=spec,
        predictions=baseline_replay_predictions,
        attestation=baseline_replay_attestation,
    )
    _write_visual_bundle(visual_report, visual_evidence, rows)
    _write_trusted_policy(
        trusted_policy,
        rows=rows,
        manifest=manifest,
        baseline_model=baseline_onnx,
        baseline_metadata=baseline_metadata,
    )
    _pin_test_trusted_policy(
        monkeypatch,
        repository_root=tmp_path,
        policy_path=trusted_policy,
    )
    return {
        "rows": rows,
        "manifest": manifest,
        "metadata": metadata,
        "baseline_metadata": baseline_metadata,
        "candidate_onnx": candidate_onnx,
        "baseline_onnx": baseline_onnx,
        "spec": spec,
        "replay_predictions": replay_predictions,
        "replay_attestation": replay_attestation,
        "baseline_replay_predictions": baseline_replay_predictions,
        "baseline_replay_attestation": baseline_replay_attestation,
        "trusted_policy": trusted_policy,
        "trust_root_repository": tmp_path,
        "visual_report": visual_report,
        "visual_evidence": visual_evidence,
        "thresholds": GateThresholds(),
    }


def _evaluate(candidate: dict[str, object], output_dir: Path, **overrides):
    arguments = {
        "metadata_path": candidate["metadata"],
        "manifest_paths": [candidate["manifest"]],
        "candidate_onnx_path": candidate["candidate_onnx"],
        "inference_spec_path": candidate["spec"],
        "replay_predictions_path": candidate["replay_predictions"],
        "replay_attestation_path": candidate["replay_attestation"],
        "baseline_metadata_path": candidate["baseline_metadata"],
        "baseline_onnx_path": candidate["baseline_onnx"],
        "baseline_replay_predictions_path": candidate["baseline_replay_predictions"],
        "baseline_replay_attestation_path": candidate["baseline_replay_attestation"],
        "trusted_policy_path": candidate["trusted_policy"],
        "visual_judge_report_path": candidate["visual_report"],
        "visual_judge_evidence_path": candidate["visual_evidence"],
        "output_dir": output_dir,
        "thresholds": candidate["thresholds"],
    }
    arguments.update(overrides)
    return evaluate_v4_candidate(**arguments)


def _cli_args(candidate: dict[str, object], output_dir: Path) -> list[str]:
    return [
        "--metadata",
        str(candidate["metadata"]),
        "--manifest",
        str(candidate["manifest"]),
        "--candidate-onnx",
        str(candidate["candidate_onnx"]),
        "--inference-spec",
        str(candidate["spec"]),
        "--replay-predictions",
        str(candidate["replay_predictions"]),
        "--replay-attestation",
        str(candidate["replay_attestation"]),
        "--baseline-metadata",
        str(candidate["baseline_metadata"]),
        "--baseline-onnx",
        str(candidate["baseline_onnx"]),
        "--baseline-replay-predictions",
        str(candidate["baseline_replay_predictions"]),
        "--baseline-replay-attestation",
        str(candidate["baseline_replay_attestation"]),
        "--trusted-policy",
        str(candidate["trusted_policy"]),
        "--visual-judge-report",
        str(candidate["visual_report"]),
        "--visual-judge-evidence",
        str(candidate["visual_evidence"]),
        "--output-dir",
        str(output_dir),
    ]


def test_pass_is_hash_bound_immutable_and_never_authorizes_production(
    candidate, tmp_path
):
    output_dir = tmp_path / "judge-output"

    report = _evaluate(candidate, output_dir)

    assert report["status"] == "passed"
    assert report["candidate_ready"] is True
    assert report["trust_root_method"] == "git_bundled_code_sha256_pin"
    assert report["trust_root_verified"] is True
    assert report["production_deployment_authorized"] is False
    assert report["gates"]["visual_judges"]["judge_count"] == 2
    assert report["gates"]["visual_judges"]["truth_relabels"] == 0
    assert report["gates"]["candidate_replay"]["runtime_replay"][
        "actual_onnx_inference"
    ] is True
    marker = json.loads((output_dir / READY_MARKER_NAME).read_text())
    assert marker["trusted_policy_sha256"] == hashlib.sha256(
        candidate["trusted_policy"].read_bytes()
    ).hexdigest()
    assert marker["trust_root_method"] == "git_bundled_code_sha256_pin"
    assert marker["trust_root_verified"] is True
    assert marker["report_sha256"] == hashlib.sha256(
        (output_dir / REPORT_NAME).read_bytes()
    ).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _evaluate(candidate, output_dir)


def test_configured_cli_subprocess_passes_with_real_tiny_onnx(candidate, tmp_path):
    output_dir = tmp_path / "subprocess-output"
    policy_sha256 = hashlib.sha256(candidate["trusted_policy"].read_bytes()).hexdigest()
    bootstrap = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from scripts import evaluate_v4_candidate_judge as gate",
            f"gate.REPO_ROOT = Path({str(candidate['trust_root_repository'])!r})",
            (
                "gate.TRUSTED_POLICY_RELATIVE_PATH = "
                f"Path({candidate['trusted_policy'].name!r})"
            ),
            f"gate.APPROVED_TRUSTED_POLICY_SHA256 = {policy_sha256!r}",
            "raise SystemExit(gate.main(sys.argv[1:]))",
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            bootstrap,
            *_cli_args(candidate, output_dir),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / READY_MARKER_NAME).is_file()


def test_stock_cli_unconfigured_pin_exits_two_without_artifacts(candidate, tmp_path):
    output_dir = tmp_path / "stock-unconfigured-output"
    args = _cli_args(candidate, output_dir)
    policy_index = args.index("--trusted-policy") + 1
    args[policy_index] = str(
        Path(__file__).parents[1]
        / "configs/v4_candidate_judge_trusted_policy.json"
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_v4_candidate_judge.py",
            *args,
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "UNCONFIGURED" in result.stderr
    assert not (output_dir / REPORT_NAME).exists()
    assert not (output_dir / READY_MARKER_NAME).exists()


def test_unconfigured_code_pin_fails_closed_without_artifacts(
    candidate, tmp_path, monkeypatch
):
    output_dir = tmp_path / "unconfigured-trust-root-output"
    monkeypatch.setattr(
        candidate_judge,
        "APPROVED_TRUSTED_POLICY_SHA256",
        candidate_judge.UNCONFIGURED_TRUST_ROOT,
    )

    with pytest.raises(ValueError, match="UNCONFIGURED"):
        _evaluate(candidate, output_dir)

    assert not (output_dir / REPORT_NAME).exists()
    assert not (output_dir / READY_MARKER_NAME).exists()


def test_caller_cannot_select_another_trusted_policy_path(candidate, tmp_path):
    output_dir = tmp_path / "wrong-policy-path-output"
    caller_policy = tmp_path / "caller-policy.json"
    caller_policy.write_bytes(candidate["trusted_policy"].read_bytes())

    with pytest.raises(ValueError, match="repository-pinned trust root"):
        _evaluate(
            candidate,
            output_dir,
            trusted_policy_path=caller_policy,
        )

    assert not (output_dir / REPORT_NAME).exists()
    assert not (output_dir / READY_MARKER_NAME).exists()


def test_wrong_code_pinned_policy_hash_fails_closed(
    candidate, tmp_path, monkeypatch
):
    output_dir = tmp_path / "wrong-policy-hash-output"
    monkeypatch.setattr(
        candidate_judge,
        "APPROVED_TRUSTED_POLICY_SHA256",
        _sha("different-reviewed-policy"),
    )

    with pytest.raises(ValueError, match="SHA-256 differs"):
        _evaluate(candidate, output_dir)

    assert not (output_dir / REPORT_NAME).exists()
    assert not (output_dir / READY_MARKER_NAME).exists()


def test_caller_crafted_policy_at_fixed_path_cannot_self_root(candidate, tmp_path):
    output_dir = tmp_path / "caller-crafted-policy-output"
    crafted = json.loads(candidate["trusted_policy"].read_text())
    crafted["approved_baseline"]["model_sha256"] = _sha("caller-baseline")
    candidate["trusted_policy"].write_bytes(_canonical(crafted))

    with pytest.raises(ValueError, match="SHA-256 differs"):
        _evaluate(candidate, output_dir)

    assert not (output_dir / REPORT_NAME).exists()
    assert not (output_dir / READY_MARKER_NAME).exists()


def test_invalid_onnx_cannot_pass_with_stale_replay(candidate, tmp_path):
    candidate["candidate_onnx"].write_bytes(b"not-an-onnx-model")
    output_dir = tmp_path / "invalid-onnx-output"

    report = _evaluate(candidate, output_dir)

    assert report["candidate_ready"] is False
    assert any("runtime ONNX replay failed" in p for p in report["problems"])
    assert not (output_dir / READY_MARKER_NAME).exists()


def test_missing_visual_evidence_fails_closed(candidate, tmp_path):
    output_dir = tmp_path / "missing-output"
    report = _evaluate(
        candidate,
        output_dir,
        visual_judge_evidence_path=tmp_path / "missing.jsonl",
    )

    assert report["status"] == "rejected"
    assert any("visual judge evidence" in p for p in report["problems"])
    assert not (output_dir / READY_MARKER_NAME).exists()


def test_ambiguous_vote_is_diagnostic_veto_only(candidate, tmp_path):
    veto_sample = next(
        row["sample_id"]
        for row in candidate["rows"]
        if row["role"] == "model_validation" and row["material"] == 9
    )
    _write_visual_bundle(
        candidate["visual_report"],
        candidate["visual_evidence"],
        candidate["rows"],
        veto_sample=str(veto_sample),
    )

    report = _evaluate(candidate, tmp_path / "veto-output")

    assert report["candidate_ready"] is False
    assert report["gates"]["visual_judges"]["veto_count"] == 1
    assert report["gates"]["visual_judges"]["truth_relabels"] == 0


def test_all_background_collapse_is_explicitly_rejected(candidate, tmp_path):
    _write_replay_bundle(
        rows=candidate["rows"],
        manifest=candidate["manifest"],
        metadata=candidate["metadata"],
        model=candidate["candidate_onnx"],
        spec=candidate["spec"],
        predictions=candidate["replay_predictions"],
        attestation=candidate["replay_attestation"],
        mode="all_background",
    )

    report = _evaluate(candidate, tmp_path / "collapse-output")

    assert report["gates"]["candidate_metrics"]["collapse"] == "all_background"
    assert any("all-background" in p for p in report["problems"])


def test_baseline_recall_drop_is_rejected(candidate, tmp_path):
    _write_replay_bundle(
        rows=candidate["rows"],
        manifest=candidate["manifest"],
        metadata=candidate["metadata"],
        model=candidate["candidate_onnx"],
        spec=candidate["spec"],
        predictions=candidate["replay_predictions"],
        attestation=candidate["replay_attestation"],
        mode="material_zero_wrong",
    )

    report = _evaluate(candidate, tmp_path / "baseline-output")

    regressions = report["gates"]["candidate_metrics"][
        "baseline_non_regression"
    ]["regressions"]
    assert regressions[0]["metric"] == "material/can"


def test_metadata_metrics_are_recomputed_not_trusted(candidate, tmp_path):
    metadata = json.loads(candidate["metadata"].read_text())
    metadata["best_metrics"] = _metrics(background_support=201)
    candidate["metadata"].write_text(json.dumps(metadata), encoding="utf-8")

    report = _evaluate(candidate, tmp_path / "support-output")

    assert any("metadata best_metrics differ" in p for p in report["problems"])


def test_v4_hard_negative_source_one_crop_zero_is_valid(
    candidate, tmp_path, monkeypatch
):
    hard_negative = next(
        row
        for row in candidate["rows"]
        if row["role"] == "model_validation" and row["material"] == 9
    )
    hard_negative["source_object_count"] = 1
    hard_negative["crop_object_count"] = 0
    _write_csv(candidate["manifest"], candidate["rows"])
    for prefix in ("", "baseline_"):
        _write_replay_bundle(
            rows=candidate["rows"],
            manifest=candidate["manifest"],
            metadata=candidate[f"{prefix}metadata"],
            model=candidate[f"{prefix}onnx"] if prefix else candidate["candidate_onnx"],
            spec=candidate["spec"],
            predictions=candidate[f"{prefix}replay_predictions"],
            attestation=candidate[f"{prefix}replay_attestation"],
        )
    _write_trusted_policy(
        candidate["trusted_policy"],
        rows=candidate["rows"],
        manifest=candidate["manifest"],
        baseline_model=candidate["baseline_onnx"],
        baseline_metadata=candidate["baseline_metadata"],
    )
    _pin_test_trusted_policy(
        monkeypatch,
        repository_root=candidate["trust_root_repository"],
        policy_path=candidate["trusted_policy"],
    )

    report = _evaluate(candidate, tmp_path / "hard-negative-output")

    assert report["status"] == "passed"


def test_trusted_policy_pins_baseline_identity(candidate, tmp_path, monkeypatch):
    policy = json.loads(candidate["trusted_policy"].read_text())
    policy["approved_baseline"]["model_sha256"] = _sha("unapproved-baseline")
    candidate["trusted_policy"].write_bytes(_canonical(policy))
    _pin_test_trusted_policy(
        monkeypatch,
        repository_root=candidate["trust_root_repository"],
        policy_path=candidate["trusted_policy"],
    )

    report = _evaluate(candidate, tmp_path / "policy-output")

    assert report["candidate_ready"] is False
    assert any("baseline model SHA-256 differs" in p for p in report["problems"])


def test_trusted_policy_rejects_shared_judge_weight_layer(
    candidate, tmp_path, monkeypatch
):
    policy = json.loads(candidate["trusted_policy"].read_text())
    policy["visual_judges"]["judges"][1]["model_weight_layer_sha256"] = policy[
        "visual_judges"
    ]["judges"][0]["model_weight_layer_sha256"]
    candidate["trusted_policy"].write_bytes(_canonical(policy))
    _pin_test_trusted_policy(
        monkeypatch,
        repository_root=candidate["trust_root_repository"],
        policy_path=candidate["trusted_policy"],
    )

    report = _evaluate(candidate, tmp_path / "shared-weight-output")

    assert report["candidate_ready"] is False
    assert any("weight_layer_sha256 values must be distinct" in p for p in report["problems"])


def test_thresholds_can_only_tighten_frozen_defaults(candidate, tmp_path):
    output_dir = tmp_path / "weakened-threshold-output"
    with pytest.raises(ValueError, match="cannot weaken the frozen default"):
        _evaluate(
            candidate,
            output_dir,
            thresholds=GateThresholds(min_background_recall=0.0),
        )
    assert not output_dir.exists()


def test_report_names_cannot_escape_output_directory(candidate, tmp_path):
    output_dir = tmp_path / "contained-output"
    with pytest.raises(ValueError, match="must be a basename"):
        _evaluate(candidate, output_dir, report_name="../escaped.json")
    assert not output_dir.exists()


def test_candidate_replay_logits_are_recomputed_not_trusted(candidate, tmp_path):
    rows = [
        json.loads(line)
        for line in candidate["replay_predictions"].read_text().splitlines()
    ]
    material_row = next(row for row in rows if row["truth_material"] == 0)
    material_row["material_logits"] = [-3.0] * 9
    material_row["material_logits"][1] = 3.0
    material_row["predicted_material_head"] = 1
    material_row["cascaded_material"] = 1
    prediction_bytes = b"".join(_canonical(row) for row in rows)
    candidate["replay_predictions"].write_bytes(prediction_bytes)
    attestation = json.loads(candidate["replay_attestation"].read_text())
    attestation["predictions_sha256"] = hashlib.sha256(prediction_bytes).hexdigest()
    candidate["replay_attestation"].write_text(json.dumps(attestation))

    report = _evaluate(candidate, tmp_path / "recomputed-output")

    assert report["candidate_ready"] is False
    assert any("independent recomputation" in p for p in report["problems"])


def test_gate_reads_original_replay_once_then_uses_immutable_snapshot(
    candidate, tmp_path, monkeypatch
):
    target = candidate["replay_predictions"].resolve()
    original_read_bytes = Path.read_bytes
    calls = 0

    def adversarial_read(path: Path) -> bytes:
        nonlocal calls
        if path.resolve() == target:
            calls += 1
            if calls > 1:
                return b'{"fabricated":"high-metrics"}\n'
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", adversarial_read)
    report = _evaluate(candidate, tmp_path / "single-read-output")

    assert report["status"] == "passed"
    assert calls == 1


def test_original_input_mutation_after_snapshot_rejects_ready_marker(
    candidate, tmp_path, monkeypatch
):
    target = candidate["replay_predictions"].resolve()
    original_read_bytes = Path.read_bytes
    mutated = False

    def mutate_after_snapshot(path: Path) -> bytes:
        nonlocal mutated
        raw = original_read_bytes(path)
        if path.resolve() == target and not mutated:
            target.write_bytes(b'{"fabricated":"changed-after-snapshot"}\n')
            mutated = True
        return raw

    monkeypatch.setattr(Path, "read_bytes", mutate_after_snapshot)
    output_dir = tmp_path / "changed-after-snapshot-output"
    report = _evaluate(candidate, output_dir)

    assert report["candidate_ready"] is False
    assert any("input artifact changed" in p for p in report["problems"])
    assert not (output_dir / READY_MARKER_NAME).exists()


def test_baseline_replay_is_mandatory(candidate, tmp_path):
    output_dir = tmp_path / "missing-baseline-output"
    report = _evaluate(
        candidate,
        output_dir,
        baseline_replay_predictions_path=tmp_path / "missing.jsonl",
    )
    assert report["candidate_ready"] is False
    assert any("baseline replay" in p for p in report["problems"])
    assert not (output_dir / READY_MARKER_NAME).exists()


def test_candidate_cannot_be_reused_as_its_own_baseline(candidate, tmp_path):
    report = _evaluate(
        candidate,
        tmp_path / "same-model-output",
        baseline_onnx_path=candidate["candidate_onnx"],
    )
    assert report["candidate_ready"] is False
    assert any("baseline model must be distinct" in p for p in report["problems"])


def test_policy_pins_visual_prompt_and_runner(candidate, tmp_path, monkeypatch):
    policy = json.loads(candidate["trusted_policy"].read_text())
    policy["visual_judges"]["runner_script_sha256"] = _sha("other-runner")
    candidate["trusted_policy"].write_bytes(_canonical(policy))
    _pin_test_trusted_policy(
        monkeypatch,
        repository_root=candidate["trust_root_repository"],
        policy_path=candidate["trusted_policy"],
    )

    report = _evaluate(candidate, tmp_path / "runner-policy-output")

    assert report["candidate_ready"] is False
    assert any("invalid runner_script_sha256" in p for p in report["problems"])


def test_rehashed_raw_response_with_forbidden_truth_is_rejected(candidate, tmp_path):
    _write_visual_bundle(
        candidate["visual_report"],
        candidate["visual_evidence"],
        candidate["rows"],
        forbidden_raw_field="ground_truth",
    )

    report = _evaluate(candidate, tmp_path / "forbidden-raw-output")

    assert report["candidate_ready"] is False
    assert any("raw response exposes forbidden fields" in p for p in report["problems"])


def test_custom_transport_visual_evidence_is_never_authoritative(candidate, tmp_path):
    _write_visual_bundle(
        candidate["visual_report"],
        candidate["visual_evidence"],
        candidate["rows"],
        official_ollama_http=False,
    )

    report = _evaluate(candidate, tmp_path / "custom-transport-output")

    assert report["candidate_ready"] is False
    assert any("official_ollama_http" in p for p in report["problems"])


def test_rehashed_postflight_identity_swap_is_rejected(candidate, tmp_path):
    _write_visual_bundle(
        candidate["visual_report"],
        candidate["visual_evidence"],
        candidate["rows"],
        postflight_digest_override=_sha("swapped-served-model"),
    )

    report = _evaluate(candidate, tmp_path / "postflight-swap-output")

    assert report["candidate_ready"] is False
    assert any("postflight /api/tags" in p for p in report["problems"])


def test_custom_replay_factory_can_never_create_ready_marker(candidate, tmp_path):
    stored = {}
    for model_key, prediction_key in (
        ("candidate_onnx", "replay_predictions"),
        ("baseline_onnx", "baseline_replay_predictions"),
    ):
        stored[str(candidate[model_key].resolve())] = [
            [row["predicted_objectness"], row["predicted_material_head"]]
            for row in (
                json.loads(line)
                for line in candidate[prediction_key].read_text().splitlines()
            )
        ]

    def factory(path: Path):
        return _Session(stored[str(path.resolve())])

    output_dir = tmp_path / "custom-session-output"
    report = _evaluate(candidate, output_dir, replay_session_factory=factory)

    assert report["candidate_ready"] is False
    assert any("test-only" in p for p in report["problems"])
    assert not (output_dir / READY_MARKER_NAME).exists()


def test_cli_returns_one_when_visual_evidence_is_missing(candidate, tmp_path):
    output_dir = tmp_path / "cli-rejected-output"
    args = _cli_args(candidate, output_dir)
    missing_index = args.index("--visual-judge-evidence") + 1
    args[missing_index] = str(tmp_path / "missing-visual.jsonl")

    assert main(args) == 1
    assert not (output_dir / READY_MARKER_NAME).exists()
