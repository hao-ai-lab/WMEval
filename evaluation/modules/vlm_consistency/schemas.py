
from __future__ import annotations

import copy
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

__all__ = [
    "RUBRIC_AGENTS",
    "VALID_GROUNDINGS",
    "VALID_WEIGHTS",
    "TopLevelExplanation",
    "RubricItem",
    "RubricMetadata",
    "Rubric",
    "AgentResult",
    "JudgeResult",
    "ItemScore",
    "VLMItemScore",
    "VLMJudgeOutput",
    "VLMCandidateItem",
    "VLMCandidateOutput",
    "export_candidate_schema",
    "export_judge_schema",
]


class TopLevelExplanation(BaseModel):
    summary: str = Field(
        ...,
        description="Overall summary of alignment quality",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific issues found across all criteria",
    )


_OPENAI_STRIP = {
    "examples", "default", "title",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "minItems", "maxItems",
    "minProperties", "maxProperties",
}

_ANTHROPIC_STRIP = _OPENAI_STRIP

_GEMINI_STRIP = {"examples", "default", "title", "$defs", "$ref"}


def _transform_for_openai(schema: Dict[str, Any]) -> Dict[str, Any]:
    schema = copy.deepcopy(schema)
    _openai_fixup(schema)
    return schema


def _openai_fixup(node: Any) -> None:
    if not isinstance(node, dict):
        return

    for key in list(node.keys()):
        if key in _OPENAI_STRIP:
            del node[key]

    if node.get("type") == "object" and "properties" in node:
        node["additionalProperties"] = False
        node["required"] = list(node["properties"].keys())

    for v in node.values():
        if isinstance(v, dict):
            _openai_fixup(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _openai_fixup(item)


def _transform_for_anthropic(schema: Dict[str, Any]) -> Dict[str, Any]:
    schema = copy.deepcopy(schema)
    _strip_keys_recursive(schema, _ANTHROPIC_STRIP)
    return schema


def _transform_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})
    _inline_refs(schema, defs)
    _strip_keys_recursive(schema, _GEMINI_STRIP)
    return schema


def _strip_keys_recursive(node: Any, keys: set) -> None:
    if not isinstance(node, dict):
        return
    for key in list(node.keys()):
        if key in keys:
            del node[key]
    for v in node.values():
        if isinstance(v, dict):
            _strip_keys_recursive(v, keys)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _strip_keys_recursive(item, keys)


def _inline_refs(node: Any, defs: Dict[str, Any]) -> None:
    if not isinstance(node, dict):
        return
    for k, v in list(node.items()):
        if isinstance(v, dict):
            if "$ref" in v:
                ref_name = v["$ref"].split("/")[-1]
                if ref_name in defs:
                    resolved = copy.deepcopy(defs[ref_name])
                    _inline_refs(resolved, defs)
                    node[k] = resolved
            else:
                _inline_refs(v, defs)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    if "$ref" in item:
                        ref_name = item["$ref"].split("/")[-1]
                        if ref_name in defs:
                            resolved = copy.deepcopy(defs[ref_name])
                            _inline_refs(resolved, defs)
                            v[i] = resolved
                    else:
                        _inline_refs(item, defs)


RUBRIC_AGENTS: Tuple[str, ...] = ("environment", "object")

VALID_GROUNDINGS = ("prompt_grounded", "reference_grounded")
VALID_WEIGHTS = (1.0, 1.5, 2.0)


class RubricItem(BaseModel):
    rubric_id: str = Field(..., description="Unique identifier, e.g. env_01 / obj_01")
    agent: Literal["environment", "object"] = Field(
        ...,
        description="environment = scene-level attributes; object = salient entities and their attributes",
    )
    weight: float = Field(
        default=1.0,
        description="1.0 = standard, 1.5 = important, 2.0 = scene-defining anchor",
    )
    rubric_text: str = Field(
        ...,
        description="A single clear sentence describing what to evaluate",
    )


class RubricMetadata(BaseModel):
    generator_provider: str
    generator_model: str
    generated_at: str = Field(..., description="ISO-8601 timestamp")
    num_candidates_generated: int = Field(default=0)


class Rubric(BaseModel):
    scene_name: str
    prompt_text: str = Field(default="")
    rubric_items: List[RubricItem] = Field(
        ...,
        min_length=0,
        description="Evaluation rubrics from both agents (environment + object)",
    )
    metadata: RubricMetadata

    @property
    def environment_items(self) -> List[RubricItem]:
        return [i for i in self.rubric_items if i.agent == "environment"]

    @property
    def object_items(self) -> List[RubricItem]:
        return [i for i in self.rubric_items if i.agent == "object"]


class ItemScore(BaseModel):
    rubric_id: str
    reasoning: str
    frame_evidence: List[str]
    judged_score: int = Field(..., ge=0, le=3)


class AgentResult(BaseModel):
    agent: Literal["environment", "object"]
    item_scores: List[ItemScore]
    score: Optional[float] = Field(
        ..., ge=0.0, le=1.0,
        description="Weighted average judged_score [0,1], or null if no rubrics for this agent",
    )
    explanations: TopLevelExplanation


class JudgeCallRecord(BaseModel):
    """One hosted-judge request and the response it got back.

    ``request`` holds what identifies the call without the image payloads
    (system-prompt hash, user text, schema hash, per-image label + sha256);
    ``response`` holds what the provider returned (resolved model snapshot,
    response id, usage, attempts, raw output text).
    """

    agent: Literal["environment", "object"]
    request: Dict[str, Any]
    response: Dict[str, Any]


class JudgeResult(BaseModel):
    scene_name: str
    provider: str
    model: str
    rubric_metadata: RubricMetadata
    alpha: float = Field(
        ...,
        description="Weight for environment component in overall score (0-1)",
    )
    num_frames: int = Field(..., description="Total candidate frames evaluated")
    environment_result: AgentResult
    object_result: AgentResult
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="alpha * environment_score + (1-alpha) * object_score",
    )
    judge_calls: List[JudgeCallRecord] = Field(
        default_factory=list,
        description="Every judge request/response behind this result",
    )


class VLMItemScore(BaseModel):
    rubric_id: str
    reasoning: str = Field(
        ...,
        description="Step-by-step analysis of the frames against the rubric. "
        "Complete this BEFORE assigning judged_score.",
    )
    frame_evidence: List[str] = Field(
        ...,
        description="Specific frame labels cited as evidence, e.g. ['frame_t=0.5s', 'frame_t=2.0s'].",
    )
    judged_score: int = Field(
        ..., ge=0, le=3,
        description="Assign ONLY after completing reasoning and frame_evidence.",
    )


class VLMJudgeOutput(BaseModel):
    item_scores: List[VLMItemScore]
    explanations: TopLevelExplanation


def export_judge_schema(
    provider: Literal["openai", "anthropic", "gemini"],
) -> Dict[str, Any]:
    raw = VLMJudgeOutput.model_json_schema()
    if provider == "openai":
        return _transform_for_openai(raw)
    elif provider == "anthropic":
        return _transform_for_anthropic(raw)
    return _transform_for_gemini(raw)


class VLMCandidateItem(BaseModel):
    rubric_id: str
    agent: str = Field(
        ...,
        description="One of: environment, object",
    )
    weight: float = Field(default=1.0)
    rubric_text: str = Field(default="", description="Declarative criterion sentence")
    question: str = Field(default="", description="Yes/no question form of the criterion")
    rationale: str = Field(default="", description="Why this criterion matters for the evaluated dimension")


class VLMCandidateOutput(BaseModel):
    rubric_items: List[VLMCandidateItem]


def export_candidate_schema(
    provider: Literal["openai", "anthropic", "gemini"],
) -> Dict[str, Any]:
    raw = VLMCandidateOutput.model_json_schema()
    if provider == "openai":
        return _transform_for_openai(raw)
    elif provider == "anthropic":
        return _transform_for_anthropic(raw)
    return _transform_for_gemini(raw)
