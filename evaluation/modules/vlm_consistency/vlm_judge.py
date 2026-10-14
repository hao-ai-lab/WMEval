
from __future__ import annotations

import base64
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Tuple

from .schemas import (
    AgentResult,
    ItemScore,
    JudgeCallRecord,
    JudgeResult,
    RubricItem,
    Rubric,
    TopLevelExplanation,
    VLMJudgeOutput,
    export_judge_schema,
)


ENV_JUDGE_SYSTEM_PROMPT = """\
### Role
You are an expert video evaluator for prompt-scene consistency. 
Your task is to judge whether the environmental attributes in a generated video are faithful to the text prompt.

### Evaluation Scope
Evaluate only environment-level attributes, including ground, sky, terrain, lighting, background structure, global style, and color palette.

Do not evaluate salient entities such as characters, vehicles, animals, controllable agents, or primary foreground objects. These are evaluated by a separate object judge. If a rubric item mixes environment and object properties, judge only the environmental component.

### Inputs
- **Text prompt**: The primary source of truth. All judgments must be grounded in attributes explicitly stated or directly implied by the prompt.
- **Candidate frames**: Uniformly sampled frames from the generated video, labeled as frame_t=X.Xs.
- **Reference image**: A frame labeled ref_frame, providing visual context for the intended scene appearance. Use it only to clarify ambiguous prompt wording or visible environmental attributes.
- **Rubric items**: Each item targets one environment-level attribute and should be judged independently.

### Evaluation Protocol
For each rubric item, identify the targeted environmental attribute and interpret it in the context of the full text prompt. If the prompt explicitly describes an environmental change, state transition, or exception, account for it when scoring the relevant attribute. Otherwise, assume the environment is stable: camera motion may change what is visible, but should not change the underlying environmental properties.

Base each judgment on direct visual evidence from the sampled frames. For each rubric item, first inspect the relevant frames, then write step-by-step reasoning that cites specific frame labels and explains how the targeted environmental attribute matches or deviates from the prompt; only then assign the final score. Be conservative: if an attribute cannot be clearly verified, assign a lower score.

### Scoring Rubric
Use an integer score from 0 to 3 for each rubric item.

- **3**: The attribute clearly matches the prompt across all relevant frames, with no meaningful ambiguity, omission, or inconsistency.
- **2**: The attribute matches the prompt in the vast majority of relevant frames, with only minor visual ambiguity, brief occlusion, or isolated low-severity artifacts. Do not assign 2 for clear semantic, material, or structural changes that contradict the prompt.
- **1**: The attribute matches the prompt in some frames but is wrong, missing, unstable, or ambiguous in a non-trivial portion of the video.
- **0**: The attribute contradicts the prompt, is absent or unverifiable in most relevant frames, changes substantially without prompt support, or the video is too corrupted to evaluate.

When uncertain between two adjacent scores, choose the lower score.

### Output Requirements
Return valid JSON only.

Additionally, provide a top-level explanations object with:
- summary: a concise overall summary of environment-consistency quality.
- issues: a list of specific environment-consistency issues found across rubrics.

### Output Schema
{
  "item_scores": [
    {
      "rubric_id": "<rubric_id>",
      "reasoning": "<your evidence-based reasoning with cited frame labels>",
      "frame_evidence": ["<frame_label_1>", "<frame_label_2>"],
      "judged_score": <integer_0_to_3>
    }
  ],
  "explanations": {
    "summary": "<concise summary of overall environment-consistency quality>",
    "issues": ["<specific issue 1>", "<specific issue 2>"]
  }
}
"""

OBJ_JUDGE_SYSTEM_PROMPT = """\
### Role
You are an expert video evaluator for prompt-object consistency.
Your task is to judge whether the salient entities in a generated video are faithful to the text prompt and visually consistent when visible.

### Evaluation Scope
Evaluate only salient entities mentioned in the prompt, including characters, vehicles, animals, props, controllable agents, and primary foreground objects, as well as their explicitly stated or directly implied attributes.

Do not evaluate environment-level attributes such as ground, sky, terrain, lighting, background structure, global style, or color palette. These are evaluated by a separate environment judge. If a rubric item mixes object and environment properties, judge only the object-level component.

### Inputs
- **Text prompt**: The primary source of truth. All judgments must be grounded in entities and attributes explicitly stated or directly implied by the prompt.
- **Candidate frames**: Uniformly sampled frames from the generated video, labeled as frame_t=X.Xs.
- **Reference image**: A frame labeled ref_frame, providing visual context for the intended scene appearance. Use it only to clarify ambiguous prompt wording or visible object attributes.
- **Rubric items**: Each item targets one salient entity or one object-level attribute and should be judged independently.

### Evaluation Protocol
For each rubric item, identify the targeted entity and the attribute being evaluated, such as color, material, shape, texture, count, or identity. Interpret the item in the context of the full text prompt.

Scan the sampled frames and determine which frames are relevant for judging the item. For object-level attributes, only frames where the targeted entity is visible should contribute to scoring.

Base each judgment on direct visual evidence from the sampled frames. For each rubric item, first inspect the relevant visible frames, then write step-by-step reasoning that cites specific frame labels and explains how the targeted entity or attribute matches or deviates from the prompt; only then assign the final score. Be conservative: if an entity or attribute cannot be clearly verified, assign a lower score.

### Scoring Rubric
Use an integer score from 0 to 3 for each rubric item.

- **3**: The targeted entity is visible and its prompt-specified attributes clearly match the prompt in all visible, judgeable frames. No meaningful ambiguity, omission, or inconsistency is observed.
- **2**: The targeted entity is visible and its attributes match the prompt in the vast majority of visible frames, with only minor visual ambiguity, brief occlusion, or isolated low-severity artifacts. Do not assign 2 for clear identity errors, wrong colors, wrong shapes, missing required parts, or persistent attribute errors.
- **1**: The targeted entity is visible in some frames, but its identity or prompt-specified attributes are wrong, missing, unstable, or ambiguous in a non-trivial portion of visible, judgeable frames.
- **0**: The targeted entity is absent throughout the sampled frames, contradicts the prompt, has the wrong identity or attributes in most visible frames, or the video is too corrupted to evaluate.

When uncertain between two adjacent scores, choose the lower score.

### Output Requirements
Return valid JSON only.

Additionally, provide a top-level explanations object with:
- summary: a concise overall summary of object-consistency quality.
- issues: a list of specific object-consistency issues found across rubrics.

### Output Schema
{
  "item_scores": [
    {
      "rubric_id": "<rubric_id>",
      "reasoning": "<your evidence-based reasoning with cited frame labels>",
      "frame_evidence": ["<frame_label_1>", "<frame_label_2>"],
      "judged_score": <integer_0_to_3>
    }
  ],
  "explanations": {
    "summary": "<concise summary of overall object-consistency quality>",
    "issues": ["<specific issue 1>", "<specific issue 2>"]
  }
}
"""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Identifies the judge instructions a score was produced under; a frozen
# reference score is only comparable while these are unchanged.
SYSTEM_PROMPT_SHA256 = {
    "environment": _sha256_text(ENV_JUDGE_SYSTEM_PROMPT),
    "object": _sha256_text(OBJ_JUDGE_SYSTEM_PROMPT),
}


def _request_record(
    *,
    system_prompt: str,
    user_text: str,
    schema: Dict[str, Any],
    images_b64: List[Tuple[str, str]],
    model: str,
) -> Dict[str, Any]:
    """What was sent, minus the image payloads (kept as per-image hashes)."""
    return {
        "model": model,
        "system_prompt_sha256": _sha256_text(system_prompt),
        "schema_sha256": _sha256_text(json.dumps(schema, sort_keys=True)),
        "user_text": user_text,
        "images": [
            {"label": label, "sha256": hashlib.sha256(base64.b64decode(b64)).hexdigest()}
            for label, b64 in images_b64
        ],
    }


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


def _format_items_block(items: List[RubricItem], group_label: str) -> str:
    lines = [f"## {group_label} Rubric Items\n"]
    for item in items:
        lines.append(f"### {item.rubric_id}")
        lines.append(f"Rubric: {item.rubric_text}")
        lines.append("")
    return "\n".join(lines)


def score_agent(
    item_scores: List[ItemScore],
    rubric_items: List[RubricItem],
) -> float:
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


def _run_agent_call(
    *,
    agent_label: str,
    rubric_items: List[RubricItem],
    prompt_text: str,
    images_b64: List[Tuple[str, str]],
    candidate_labels: List[str],
    system_prompt: str,
    schema: Dict[str, Any],
    mod: Any,
    api_key: str,
    model: str,
) -> Tuple[List[ItemScore], TopLevelExplanation, Optional[JudgeCallRecord]]:
    if not rubric_items:
        return [], TopLevelExplanation(
            summary=f"(no {agent_label.lower()} rubrics to evaluate)",
            issues=[],
        ), None

    rubric_block = _format_items_block(rubric_items, agent_label)
    ref_labels = [l for l in candidate_labels if l == "ref_frame"]
    cand_labels_only = [l for l in candidate_labels if l != "ref_frame"]
    user_text = (
        "## Evaluation Task\n"
        "Evaluate the candidate video frames against each rubric item below.\n\n"
        + (f"Reference image: {', '.join(ref_labels)}\n" if ref_labels else "")
        + f"Candidate frames: {', '.join(cand_labels_only)}\n"
        + f"\nPrompt:\n{prompt_text}\n\n"
        f"{rubric_block}\n"
        "Evaluate each rubric item and return structured JSON."
    )

    response: Dict[str, Any] = {}
    raw = mod.call_vlm(
        system_prompt=system_prompt,
        user_text=user_text,
        images_b64=images_b64,
        schema=schema,
        api_key=api_key,
        model=model,
        record=response,
    )
    item_scores, explanations = _parse_vlm_output(raw)
    call = JudgeCallRecord(
        agent=agent_label.lower(),
        request=_request_record(
            system_prompt=system_prompt,
            user_text=user_text,
            schema=schema,
            images_b64=images_b64,
            model=model,
        ),
        response=response,
    )
    return item_scores, explanations, call


def judge_scene(
    rubric: Rubric,
    all_frames: List[Tuple[str, str]],
    provider: Literal["openai", "anthropic", "gemini"] = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    alpha: float = 0.6,
) -> JudgeResult:
    mod = _get_provider_module(provider)
    resolved_key = api_key or _resolve_api_key(provider)
    resolved_model = model or mod.DEFAULT_MODEL

    rubric_items = rubric.rubric_items
    if not rubric_items:
        raise ValueError("Rubric has no items")

    env_items = [i for i in rubric_items if i.agent == "environment"]
    obj_items = [i for i in rubric_items if i.agent == "object"]

    images_b64 = list(all_frames)
    candidate_labels = [label for label, _ in all_frames]

    schema = export_judge_schema(provider)

    common = dict(
        prompt_text=rubric.prompt_text,
        images_b64=images_b64,
        candidate_labels=candidate_labels,
        schema=schema,
        mod=mod,
        api_key=resolved_key,
        model=resolved_model,
    )

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_env = ex.submit(
            _run_agent_call,
            agent_label="Environment",
            rubric_items=env_items,
            system_prompt=ENV_JUDGE_SYSTEM_PROMPT,
            **common,
        )
        fut_obj = ex.submit(
            _run_agent_call,
            agent_label="Object",
            rubric_items=obj_items,
            system_prompt=OBJ_JUDGE_SYSTEM_PROMPT,
            **common,
        )
        env_scores, env_expl, env_call = fut_env.result()
        obj_scores, obj_expl, obj_call = fut_obj.result()

    env_result = AgentResult(
        agent="environment",
        item_scores=env_scores,
        score=round(score_agent(env_scores, env_items), 4) if env_items else None,
        explanations=env_expl,
    )
    obj_result = AgentResult(
        agent="object",
        item_scores=obj_scores,
        score=round(score_agent(obj_scores, obj_items), 4) if obj_items else None,
        explanations=obj_expl,
    )

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

    n_candidate = sum(1 for l, _ in all_frames if l != "ref_frame")

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
        judge_calls=[call for call in (env_call, obj_call) if call is not None],
    )
