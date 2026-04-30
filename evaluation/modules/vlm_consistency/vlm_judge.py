"""Unified VLM judge: dual-agent (environment / object), single call on all
timestamped frames.

All candidate frames are passed together, labeled with real timestamps
(frame_t=X.Xs). A single system prompt instructs the VLM to evaluate every
rubric, jointly considering prompt fidelity in early frames and drift in
late frames within each agent's semantic scope.

Scoring hierarchy: item → agent (environment / object) → overall.
  EnvironmentScore = weighted_avg(environment rubric judged_scores)
  ObjectScore      = weighted_avg(object rubric judged_scores)
  OverallScore     = alpha * EnvironmentScore + (1-alpha) * ObjectScore
If one agent has 0 rubrics, its component is excluded from the overall score.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Tuple

from .schemas import (
    AgentResult,
    ItemScore,
    JudgeResult,
    RubricItem,
    Rubric,
    TopLevelExplanation,
    VLMJudgeOutput,
    export_judge_schema,
)

# ---------------------------------------------------------------------------
# Score definitions
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
Task: Evaluate scene alignment between a text prompt and a generated video.

Input:
- Text prompt: target scene description.
- Candidate frames: uniformly sampled from the video, labeled frame_t=X.Xs.
- Reference frames (optional): labeled gt_frame_t=X.Xs.

Rubric structure (each rubric is a binary yes/no question):
- Environment rubrics evaluate scene-level attributes (structure, layout, lighting, sky, terrain, style).
- Object rubrics evaluate salient entities and their attributes (color, material, texture, shape).
Within each agent, a rubric may target either (a) prompt-fidelity in early frames or (b) drift in late
frames. For drift questions, use OR logic: pass if late frames satisfy EITHER cross-frame consistency
with early frames OR alignment with the prompt; fail only when both are violated.

Per-item evaluation order: reasoning → frame_evidence → judged_score.

Scoring (0–3):
- 3: criterion fully met; no meaningful deviation.
- 2: criterion met in the vast majority of frames; only minor deviations permitted (e.g., lighting
     flicker, transient occlusion). Shape changes, wrong colors, or partial absence must not score 2.
- 1: criterion partially met; satisfied in some frames but notably absent or wrong in others.
- 0: not met — element absent, incorrect, or contradicted. Also 0 if most frames are corrupted.

Notes:
- Ignore natural temporal dynamics unrelated to the rubric criterion.
- Do not penalize natural motion or camera rotation; penalize only when late frames violate both
  early-frame consistency and the prompt description simultaneously.

After all items, provide a concise top-level summary and notable issues.
Return valid JSON matching the provided schema.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_provider_module(provider: str):
    if provider == "openai":
        from .providers import openai as mod
    elif provider == "anthropic":
        from .providers import anthropic as mod
    elif provider == "gemini":
        from .providers import gemini as mod
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return mod


def _resolve_api_key(provider: str) -> str:
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    key = os.environ.get(env_map[provider], "")
    if not key:
        raise RuntimeError(f"{env_map[provider]} environment variable is not set")
    return key


def _format_items_for_prompt(items: List[RubricItem]) -> str:
    """Format rubric items as a text block, organized by agent."""
    env_items = [i for i in items if i.agent == "environment"]
    obj_items = [i for i in items if i.agent == "object"]

    lines = ["## Rubric Items\n"]

    if env_items:
        lines.append("### Environment Rubrics (scene-level attributes)\n")
        for item in env_items:
            lines.append(f"#### {item.rubric_id}")
            lines.append(f"Rubric: {item.rubric_text}")
            lines.append(f"Grounding: {item.grounding}")
            lines.append(f"Weight: {item.weight}")
            lines.append("")

    if obj_items:
        lines.append("### Object Rubrics (salient entities and attributes)\n")
        for item in obj_items:
            lines.append(f"#### {item.rubric_id}")
            lines.append(f"Rubric: {item.rubric_text}")
            lines.append(f"Grounding: {item.grounding}")
            lines.append(f"Weight: {item.weight}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_agent(
    item_scores: List[ItemScore],
    rubric_items: List[RubricItem],
) -> float:
    """Compute weighted normalized score for one agent's rubrics. Returns [0, 1]."""
    weight_map = {ri.rubric_id: ri.weight for ri in rubric_items}
    total_weighted = 0.0
    total_weight = 0.0
    for item in item_scores:
        w = weight_map.get(item.rubric_id, 1.0)
        total_weighted += w * item.judged_score
        total_weight += w
    if total_weight == 0:
        return 0.0
    return total_weighted / (3.0 * total_weight)


# ---------------------------------------------------------------------------
# VLM output parsing
# ---------------------------------------------------------------------------

def _parse_vlm_output(raw: Dict[str, Any]) -> Tuple[List[ItemScore], TopLevelExplanation]:
    vlm_output = VLMJudgeOutput.model_validate(raw)
    item_scores = [
        ItemScore(
            rubric_id=vs.rubric_id,
            reasoning=vs.reasoning,
            frame_evidence=vs.frame_evidence,
            judged_score=vs.judged_score,
        )
        for vs in vlm_output.item_scores
    ]
    explanations = TopLevelExplanation(
        summary=vlm_output.explanations.summary,
        issues=vlm_output.explanations.issues,
    )
    return item_scores, explanations


# ---------------------------------------------------------------------------
# Build AgentResult objects from VLM output
# ---------------------------------------------------------------------------

def _build_agent_results(
    all_item_scores: List[ItemScore],
    rubric_items: List[RubricItem],
    explanations: TopLevelExplanation,
) -> Tuple[AgentResult, AgentResult]:
    """Split item scores by agent and compute per-agent weighted scores."""
    item_map: Dict[str, RubricItem] = {ri.rubric_id: ri for ri in rubric_items}

    env_scores: List[ItemScore] = []
    obj_scores: List[ItemScore] = []
    env_rubrics: List[RubricItem] = []
    obj_rubrics: List[RubricItem] = []

    for score in all_item_scores:
        ri = item_map.get(score.rubric_id)
        if ri is None:
            continue
        if ri.agent == "environment":
            env_scores.append(score)
            env_rubrics.append(ri)
        else:
            obj_scores.append(score)
            obj_rubrics.append(ri)

    env_result = AgentResult(
        agent="environment",
        item_scores=env_scores,
        score=round(score_agent(env_scores, env_rubrics), 4),
        explanations=explanations,
    )
    obj_result = AgentResult(
        agent="object",
        item_scores=obj_scores,
        score=round(score_agent(obj_scores, obj_rubrics), 4),
        explanations=explanations,
    )
    return env_result, obj_result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge_scene(
    rubric: Rubric,
    all_frames: List[Tuple[str, str]],
    provider: Literal["openai", "anthropic", "gemini"] = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    alpha: float = 0.6,
    gt_frames: Optional[List[Tuple[str, str]]] = None,
) -> JudgeResult:
    """Run unified VLM scene alignment judgment.

    Parameters
    ----------
    rubric : Rubric
        Rubric loaded from cache; contains environment + object items.
    all_frames : list of (label, base64)
        All candidate frames, labeled with timestamps (frame_t=X.Xs).
    gt_frames : list of (label, base64), optional
        GT reference frames labeled gt_frame_t=X.Xs.
    alpha : float
        Weight for environment component in overall score (default 0.6).
    """
    mod = _get_provider_module(provider)
    resolved_key = api_key or _resolve_api_key(provider)
    resolved_model = model or mod.DEFAULT_MODEL

    rubric_items = rubric.rubric_items
    if not rubric_items:
        raise ValueError("Rubric has no items")

    # Combine candidate + GT reference frames
    images_b64 = list(all_frames)
    if gt_frames:
        images_b64.extend(gt_frames)

    schema = export_judge_schema(provider)
    rubric_text = _format_items_for_prompt(rubric_items)
    frame_labels = [label for label, _ in all_frames]

    user_text = (
        "## Evaluation Task\n"
        "Evaluate the candidate video frames against each rubric item below.\n\n"
        f"Candidate frames provided: {', '.join(frame_labels)}\n"
        + (f"GT reference frames: {', '.join(l for l, _ in gt_frames)}\n" if gt_frames else "")
        + f"\nPrompt:\n{rubric.prompt_text}\n\n"
        f"{rubric_text}\n"
        "Evaluate each rubric item and return structured JSON."
    )

    raw = mod.call_vlm(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_text=user_text,
        images_b64=images_b64,
        schema=schema,
        api_key=resolved_key,
        model=resolved_model,
    )

    all_item_scores, explanations = _parse_vlm_output(raw)
    env_result, obj_result = _build_agent_results(all_item_scores, rubric_items, explanations)

    # Overall score: weighted combination, excluding empty agents
    has_env = len(env_result.item_scores) > 0
    has_obj = len(obj_result.item_scores) > 0

    if has_env and has_obj:
        overall = round(alpha * env_result.score + (1 - alpha) * obj_result.score, 4)
    elif has_env:
        overall = env_result.score
    elif has_obj:
        overall = obj_result.score
    else:
        overall = 0.0

    # Count only candidate frames (exclude gt_ labeled frames)
    n_candidate = sum(1 for l, _ in all_frames if not l.startswith("gt_"))

    return JudgeResult(
        scene_name=rubric.scene_name,
        provider=provider,
        model=resolved_model,
        rubric_metadata=rubric.metadata,
        alpha=alpha,
        num_frames=n_candidate,
        environment_result=env_result,
        object_result=obj_result,
        overall_score=overall,
    )
