"""Single trusted definition of the operational VLM teacher contract."""

from __future__ import annotations

import hashlib
import json
import copy
from collections.abc import Mapping
from pathlib import Path


MATERIALS = [
    "can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass",
    "battery", "fluorescent", "negative", "exclude",
]
TEACHER_LABEL_SCHEMA_VERSION = "operational_teacher_label.v3"
TEACHER_CONTRACT_SCHEMA_VERSION = "operational_teacher_contract.v3"
QUALITY_REASONS = [
    "usable",
    "severe_frame_crop",
    "person_occlusion_or_dominance",
    "clutter_or_multiple_objects",
    "boundary_unreadable",
]
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "material": {"type": "string", "enum": MATERIALS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "single_object": {"type": "boolean"},
        "foreign_material": {"type": "boolean"},
        "training_usable": {"type": "boolean"},
        "quality_reason": {"type": "string", "enum": QUALITY_REASONS},
    },
    "required": [
        "material", "confidence", "single_object", "foreign_material",
        "training_usable", "quality_reason",
    ],
    "additionalProperties": False,
}

TEACHER_LABEL_BASE_FIELDS = frozenset({
    "schema_version", "sha256", "image_ref", "input_image_sha256",
    "teacher_contract", "teacher_contract_sha256", "model", "model_digest",
    "passes", "errors", "consensus", "consensus_decision",
    "minimum_confidence",
})
KNOWN_AUDIT_SPLITS = frozenset({"train", "validation", "protected_validation"})


def load_known_audit(path: Path) -> dict[str, object]:
    """Load a strict, case-insensitive SHA-keyed audit without ambiguity."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("known_audit must be a SHA-keyed JSON object")
    normalized: dict[str, object] = {}
    for raw_key, row in value.items():
        key = valid_sha256(raw_key)
        if key is None:
            raise ValueError("known_audit contains an invalid SHA-256 key")
        if key in normalized:
            raise ValueError("known_audit contains a case-insensitive duplicate SHA")
        if not isinstance(row, Mapping) or isinstance(row, bool) or not row:
            raise ValueError("known_audit values must be non-empty mapping objects")
        split = row.get("split")
        if split not in KNOWN_AUDIT_SPLITS:
            raise ValueError("known_audit split is invalid")
        normalized[key] = dict(row)
    return normalized

BASE_PROMPTS = (
    """사진 중앙 투입구에 들어온 쓰레기의 주재질 하나를 분류하세요. 가능한 값은
can(금속 캔), pet(PET 음료병), paper(종이/종이상자), plastic(PET 아닌 플라스틱),
styrofoam(스티로폼), vinyl(비닐/봉투), glass, battery, fluorescent,
negative(쓰레기 없음), exclude(판별 불가/여러 쓰레기 혼합)입니다. 빨대처럼 같은
재질의 작은 부속품은 허용하지만 종이컵 띠지처럼 다른 재질이 붙었으면
foreign_material=true로 표시하세요. 모델의 기존 예측은 참고하지 말고 사진만 보세요.""",
    """재활용 키오스크 카메라 사진을 독립적으로 재검토하세요. 화면의 주된 물체가
정확히 하나인지, 그 물체의 주재질이 9종 중 무엇인지 판단하세요. 물체가 없으면
negative, 여러 물체/너무 불명확하면 exclude를 사용하세요. 주재질과 다른 부착물이나
혼합물이 보이면 foreign_material=true입니다. 자신 없으면 confidence를 낮추세요.""",
)
QUALITY_INSTRUCTION = """추가로 이 사진 자체가 학습에 쓸 수 있는지 training_usable로
판단하세요. 다음 네 경우에만 false로 하고 quality_reason을 각각 지정하세요:
(1) 주 대상의 재질/형태를 판단할 핵심 부분이 프레임 밖으로 심하게 잘린
severe_frame_crop, (2) 사람 손/팔이 대상 경계를 가리거나 화면을 지배하는
person_occlusion_or_dominance, (3) 대상이 너무 작을 만큼 불필요한 배경이 과도하거나
여러 쓰레기·배경 물체가 겹쳐 주 대상 하나의 경계를 정할 수 없는
clutter_or_multiple_objects, (4) 반사·가림·구도 때문에 대상
경계를 판독할 수 없는 boundary_unreadable. 그 외에는 true와 usable을 사용하세요.
정상적인 키오스크 배경, 대상 경계를 가리지 않는 가벼운 손끝 접촉, 작은 동일 재질
부속품만 있다는 이유로 false 처리하지 마세요. 손/팔 자체는 쓰레기 개수에 포함하지
말고, 가림·지배 정도만 품질로 판단하세요. 반대로 손/팔과 불필요 배경이 함께 화면
대부분을 차지해 대상 crop 경계를 신뢰할 수 없으면 false입니다. negative도 깨끗한 빈
투입구 장면이면 training_usable=true입니다. confidence는 재질과 이 품질 판정을
포함한 전체 JSON 판정의 확신도입니다. 캔·병·종이 등 대상 자체가 구겨지거나
찌그러진 것은 중요한 hard case이며 촬영 실패가 아니므로, 대상 경계와 재질을 판독할
수 있다면 training_usable=true로 유지하세요."""
PROMPTS = tuple(f"{prompt}\n\n{QUALITY_INSTRUCTION}" for prompt in BASE_PROMPTS)
ADJUDICATION_PROMPT = """같은 재활용 키오스크 사진을 최종 독립 판정하세요. 사진만 보고
주재질 하나, 단일 물체 여부, 다른 재질 이물질 부착 여부를 보수적으로 결정하세요.
이전 모델 답은 제공되지 않으며, 불명확하거나 여러 물체면 exclude를 사용하세요.

""" + QUALITY_INSTRUCTION
JSON_ONLY_SUFFIX = " 설명이나 추론 없이 JSON 스키마 필드만 반환하세요."
REQUEST_OPTIONS = {"temperature": 0, "seed": 20260819, "num_predict": 1024}
REQUEST_CONTRACT = {
    "endpoint": "/api/chat",
    "stream": False,
    "think": False,
    "keep_alive": "30m",
    "options": REQUEST_OPTIONS,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def valid_sha256(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64:
        return None
    try:
        int(normalized, 16)
    except ValueError:
        return None
    return normalized


def render_prompt(prompt: str) -> str:
    return "/no_think\n" + prompt + JSON_ONLY_SUFFIX


def build_teacher_contract(model_identifier: str, model_digest: str) -> tuple[dict, str]:
    digest = valid_sha256(model_digest)
    if digest is None:
        raise ValueError("model_digest must be exactly 64 hexadecimal characters")
    if not isinstance(model_identifier, str) or not model_identifier.strip():
        raise ValueError("model_identifier must not be empty")
    contract = {
        "schema_version": TEACHER_CONTRACT_SCHEMA_VERSION,
        "teacher_label_schema_version": TEACHER_LABEL_SCHEMA_VERSION,
        "model_identifier": model_identifier,
        "model_digest": digest,
        "decision_schema": copy.deepcopy(DECISION_SCHEMA),
        "prompts": {
            "initial": list(PROMPTS),
            "adjudication": ADJUDICATION_PROMPT,
        },
        "rendered_prompts": {
            "initial": [render_prompt(prompt) for prompt in PROMPTS],
            "adjudication": render_prompt(ADJUDICATION_PROMPT),
        },
        "request": copy.deepcopy(REQUEST_CONTRACT),
    }
    return contract, sha256_bytes(canonical_json(contract).encode("utf-8"))
