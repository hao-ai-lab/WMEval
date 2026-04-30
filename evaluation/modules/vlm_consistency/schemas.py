"""Pydantic models and JSON schema utilities for VLM consistency evaluation.

Design:
- Two rubric agents split by semantic dimension: environment | object.
  Each agent independently covers prompt coverage, atomic feature fidelity,
  and late-frame drift detection within its semantic scope.
- Single judge call on all timestamped frames.
- Scoring: EnvironmentScore + ObjectScore -> alpha-weighted overall.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Literal, Tuple

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


# ---------------------------------------------------------------------------
# Top-level explanation block (shared by VLM judge output and AgentResult)
# ---------------------------------------------------------------------------

class TopLevelExplanation(BaseModel):
    summary: str = Field(
        ...,
        description="Overall summary of alignment quality",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific issues found across all criteria",
    )


# ---------------------------------------------------------------------------
# Provider-specific JSON schema transforms
# ---------------------------------------------------------------------------

_OPENAI_STRIP = {
    "examples", "default", "title",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "minItems", "maxItems",
    "minProperties", "maxProperties",
}

_ANTHROPIC_STRIP = _OPENAI_STRIP

_GEMINI_STRIP = {"examples", "default", "title", "$defs", "$ref"}


def _transform_for_openai(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Make a JSON schema OpenAI-strict-mode compatible.

    OpenAI requires:
    - additionalProperties: false on every object
    - ALL properties listed in required
    - No min/max/minItems/maxItems/minLength/maxLength
    """
    schema = copy.deepcopy(schema)
    _openai_fixup(schema)
    return schema


def _openai_fixup(node: Any) -> None:
    """Recursively fix a schema node for OpenAI strict mode."""
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
    """Make a JSON schema Anthropic-compatible."""
    schema = copy.deepcopy(schema)
    _strip_keys_recursive(schema, _ANTHROPIC_STRIP)
    return schema


def _transform_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Make a JSON schema Gemini-compatible.

    Gemini doesn't support $defs/$ref, so we inline definitions.
    """
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
    """Recursively replace $ref with the inlined definition."""
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


# ---------------------------------------------------------------------------
# Taxonomy constants
# ---------------------------------------------------------------------------

RUBRIC_AGENTS: Tuple[str, ...] = ("environment", "object")

VALID_GROUNDINGS = ("prompt_grounded", "reference_grounded")
VALID_WEIGHTS = (1.0, 1.5, 2.0)


# ---------------------------------------------------------------------------
# Rubric item
# ---------------------------------------------------------------------------

class RubricItem(BaseModel):
    rubric_id: str = Field(..., description="Unique identifier, e.g. env_01 / obj_01")
    agent: Literal["environment", "object"] = Field(
        ...,
        description="environment = scene-level attributes; object = salient entities and their attributes",
    )
    grounding: Literal["prompt_grounded", "reference_grounded"] = Field(
        ...,
        description="Whether this rubric is grounded in the prompt text or GT reference frame",
    )
    weight: float = Field(
        default=1.0,
        description="1.0 = standard, 1.5 = important, 2.0 = scene-defining anchor",
    )
    rubric_text: str = Field(
        ...,
        description="A single clear sentence describing what to evaluate",
    )


# ---------------------------------------------------------------------------
# Rubric metadata & container
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Judge result schemas — dual-agent
# ---------------------------------------------------------------------------

class ItemScore(BaseModel):
    """Per-item judge result with chain-of-thought fields."""
    rubric_id: str
    reasoning: str
    frame_evidence: List[str]
    judged_score: int = Field(..., ge=0, le=3)


class AgentResult(BaseModel):
    """Score and explanations for one rubric agent."""
    agent: Literal["environment", "object"]
    item_scores: List[ItemScore]
    score: float = Field(..., ge=0.0, le=1.0, description="Weighted average judged_score [0,1]")
    explanations: TopLevelExplanation


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


# ---------------------------------------------------------------------------
# VLM judge output schemas (no checklist, no derived_score)
# ---------------------------------------------------------------------------

class VLMItemScore(BaseModel):
    """VLM output per rubric item — chain-of-thought order enforced by field sequence."""
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
    """VLM judge output schema."""
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


# ---------------------------------------------------------------------------
# VLM-facing candidate generation schema
# ---------------------------------------------------------------------------

class VLMCandidateItem(BaseModel):
    """One candidate rubric produced by an agent LLM call."""
    rubric_id: str
    agent: str = Field(
        ...,
        description="One of: environment, object",
    )
    grounding: str = Field(
        ...,
        description="One of: prompt_grounded, reference_grounded",
    )
    weight: float = Field(default=1.0)
    rubric_text: str = Field(default="", description="Declarative criterion sentence")
    question: str = Field(default="", description="Yes/no question form of the criterion")
    rationale: str = Field(default="", description="Why this criterion matters for the evaluated dimension")


class VLMCandidateOutput(BaseModel):
    """Full output of a single agent generation call."""
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
