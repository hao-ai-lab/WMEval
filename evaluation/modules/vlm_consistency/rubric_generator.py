"""Rubric generation: dual-agent pipeline split by semantic dimension.

Pipeline overview:
  Step 1a  _generate_agent_rubrics("environment") — LLM generates 0-4 environment rubrics
  Step 1b  _generate_agent_rubrics("object")      — LLM generates 0-4 object rubrics
  Step 2   select_final_rubrics                    — cap to max 5 per agent
  Step 3   inject mandatory holistic coverage rubrics (one per agent)

Each agent's rubrics independently cover three sub-criteria within their semantic
scope: prompt coverage, atomic feature fidelity, and late-frame drift.

Public entry point: generate_rubric(...)
"""

from __future__ import annotations

import base64
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from .schemas import (
    VALID_GROUNDINGS,
    VALID_WEIGHTS,
    RubricItem,
    RubricMetadata,
    Rubric,
    VLMCandidateOutput,
    export_candidate_schema,
)

# ═══════════════════════════════════════════════════════════════════════════
# Agent 1: Environment — scene-level attributes
# ═══════════════════════════════════════════════════════════════════════════

ENVIRONMENT_AGENT_PROMPT = """\
Task: Generate ENVIRONMENT rubrics for judging how well a generated video renders
the scene-level (non-object) content described in the prompt. Each rubric is a
binary (yes/no) question a VLM judge can answer from timestamped frames.

Each rubric covers ONE of the following sub-criteria within the environment scope:
  (a) Atomic feature fidelity — a specific environment attribute (structure, layout,
      lighting, style, color, texture) matches the prompt in early frames.
  (b) Late-frame drift — a specific environment attribute remains coherent in late
      frames. Answer "yes" if late frames satisfy EITHER cross-frame consistency
      with early frames OR alignment with the prompt; "no" only when both fail.
      Do not penalize natural camera rotation or shift.

Assign rubric_id with prefix "env_" (env_01, env_02, …). Generate 0–4 rubrics;
prefer fewer, high-confidence criteria over exhaustive lists.

Each rubric must satisfy all of the following:
- Be atomic: evaluate a single environment attribute per rubric.
- Be scene-level (structure, layout, lighting, sky, terrain, style) — NOT a salient
  entity or character (reserved for the Object agent).
- Specify grounding: "prompt_grounded" if supported by explicit prompt text;
  "reference_grounded" only when a GT reference frame is provided and the target
  attribute is clearly visible in that frame.

Invalid and must not appear:
- Rubrics requiring exact spatial coordinates, pairwise distances, or exact counts
  (unless the prompt states a number).
- Rubrics dependent on occlusion state or camera framing.
- Rubrics assessing action quality, physics simulation, or task completion.
- Rubrics about salient objects or characters (handled by the Object agent).
- The holistic question "do all environmental elements in the prompt appear?" is
  injected automatically — do NOT generate it here.

Output format example:
{
  "rubric_items": [
    {
      "rubric_id": "env_01",
      "agent": "environment",
      "grounding": "prompt_grounded",
      "question": "Is the ground depicted as a black-and-white chessboard pattern?",
      "rationale": "The chessboard ground is a scene-defining environment feature."
    }
  ]
}

Return valid JSON matching the provided schema.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Agent 2: Object — salient entities and their attributes
# ═══════════════════════════════════════════════════════════════════════════

OBJECT_AGENT_PROMPT = """\
Task: Generate OBJECT rubrics for judging how well a generated video renders the
salient entities (characters, vehicles, props, primary objects) described in the
prompt. Each rubric is a binary (yes/no) question a VLM judge can answer from
timestamped frames.

Each rubric covers ONE of the following sub-criteria within the object scope:
  (a) Atomic feature fidelity — a specific object attribute (color, material,
      texture, shape) matches the prompt in early frames.
  (b) Late-frame drift — a specific object remains coherent in late frames.
      Answer "yes" if late frames satisfy EITHER cross-frame consistency with
      early frames OR alignment with the prompt; "no" only when both fail.
      Do not penalize natural motion or expected interaction.

Assign rubric_id with prefix "obj_" (obj_01, obj_02, …). Generate 0–4 rubrics;
prefer fewer, high-confidence criteria over exhaustive lists.

Each rubric must satisfy all of the following:
- Be atomic: evaluate a single object or single attribute per rubric.
- Target a salient entity or character (NOT scene-level structure or lighting,
  which are reserved for the Environment agent).
- Specify grounding: "prompt_grounded" if supported by explicit prompt text;
  "reference_grounded" only when a GT reference frame is provided and the target
  entity is clearly visible in that frame.

Invalid and must not appear:
- Rubrics requiring exact spatial coordinates, pairwise distances, or exact counts
  (unless the prompt states a number).
- Rubrics dependent on occlusion state or camera framing.
- Rubrics assessing action quality, physics simulation, or task completion.
- Rubrics about scene-level attributes (handled by the Environment agent).
- The holistic question "do all characters/objects in the prompt appear?" is
  injected automatically — do NOT generate it here.

Output format example:
{
  "rubric_items": [
    {
      "rubric_id": "obj_01",
      "agent": "object",
      "grounding": "prompt_grounded",
      "question": "Does the red ball remain red and spherical across the video?",
      "rationale": "Passes if late frames are consistent with early frames OR still match the prompt — fails only when both are violated."
    }
  ]
}

Return valid JSON matching the provided schema.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Mandatory holistic rubrics (always injected, not subject to LLM generation)
# ═══════════════════════════════════════════════════════════════════════════

def _make_mandatory_env_coverage_rubric() -> "RubricItem":
    return RubricItem(
        rubric_id="env_00",
        agent="environment",
        grounding="prompt_grounded",
        weight=2.0,
        rubric_text=(
            "Do all environmental elements described in the prompt "
            "appear at some point in the video?"
        ),
    )


def _make_mandatory_obj_coverage_rubric() -> "RubricItem":
    return RubricItem(
        rubric_id="obj_00",
        agent="object",
        grounding="prompt_grounded",
        weight=2.0,
        rubric_text=(
            "Do all characters and primary objects mentioned in the prompt "
            "appear at some point in the video?"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

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


def _encode_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


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


def _clamp_weight(w: float) -> float:
    allowed = sorted(VALID_WEIGHTS)
    return min(allowed, key=lambda a: abs(a - w))


def _normalize_grounding(raw: str) -> str:
    s = raw.strip().lower()
    if s in VALID_GROUNDINGS:
        return s
    if "ref" in s or "image" in s or "gt" in s:
        return "reference_grounded"
    return "prompt_grounded"


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Generate rubrics from one agent
# ═══════════════════════════════════════════════════════════════════════════

def _generate_agent_rubrics(
    agent: Literal["environment", "object"],
    system_prompt: str,
    prompt_text: str,
    scene_name: str,
    provider: str,
    model: str,
    api_key: str,
    gt_frame_path: Optional[str] = None,
) -> List[RubricItem]:
    """Call one agent LLM to generate up to 5 rubrics."""
    mod = _get_provider_module(provider)
    schema = export_candidate_schema(provider)

    has_gt = gt_frame_path and Path(gt_frame_path).is_file()
    images: Optional[List[Tuple[str, str]]] = None
    if has_gt:
        images = [("gt_initial_frame", _encode_image(gt_frame_path))]

    gt_note = (
        "A ground-truth reference frame (gt_initial_frame) is provided. "
        "You may add reference_grounded rubrics for MAJOR visual anchors "
        "clearly visible in the reference image."
        if has_gt
        else "No GT reference frame is provided. All rubrics must be prompt_grounded."
    )

    user_text = (
        f"Scene name: {scene_name}\n\n"
        f"Prompt:\n{prompt_text}\n\n"
        f"{gt_note}\n\n"
        f"Generate up to 5 {agent} rubrics for this scene."
    )

    raw = mod.call_llm(
        system_prompt=system_prompt,
        user_text=user_text,
        schema=schema,
        api_key=api_key,
        model=model,
        images_b64=images,
    )

    vlm_out = VLMCandidateOutput.model_validate(raw)
    items: List[RubricItem] = []
    counter = 0
    prefix = "env" if agent == "environment" else "obj"

    for vi in vlm_out.rubric_items[:5]:
        counter += 1
        rid = f"{prefix}_{counter:02d}"

        grounding = _normalize_grounding(vi.grounding)
        if not has_gt and grounding == "reference_grounded":
            grounding = "prompt_grounded"

        rubric_text = vi.question.strip() or vi.rubric_text.strip()

        items.append(RubricItem(
            rubric_id=rid,
            agent=agent,
            grounding=grounding,
            weight=_clamp_weight(vi.weight),
            rubric_text=rubric_text,
        ))

    return items


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Select final rubrics (cap to max 5 per agent)
# ═══════════════════════════════════════════════════════════════════════════

MAX_PER_AGENT = 5


def select_final_rubrics(items: List[RubricItem]) -> List[RubricItem]:
    """Step 4: Cap to MAX_PER_AGENT per agent, sorted by weight descending."""
    env = sorted(
        [i for i in items if i.agent == "environment"],
        key=lambda x: -x.weight,
    )[:MAX_PER_AGENT]
    obj = sorted(
        [i for i in items if i.agent == "object"],
        key=lambda x: -x.weight,
    )[:MAX_PER_AGENT]
    return env + obj


# ═══════════════════════════════════════════════════════════════════════════
# Public API: generate_rubric
# ═══════════════════════════════════════════════════════════════════════════

def generate_rubric(
    prompt_text: str,
    scene_name: str,
    provider: Literal["openai", "anthropic", "gemini"] = "openai",
    model: Optional[str] = None,
    gt_frame_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Rubric:
    """Dual-agent rubric generation pipeline.

    Calls environment agent and object agent independently, combines them,
    caps to MAX_PER_AGENT, and prepends mandatory holistic coverage rubrics.
    """
    mod = _get_provider_module(provider)
    resolved_key = api_key or _resolve_api_key(provider)
    resolved_model = model or mod.DEFAULT_MODEL

    shared = dict(
        scene_name=scene_name,
        provider=provider,
        model=resolved_model,
        api_key=resolved_key,
        gt_frame_path=gt_frame_path,
    )

    # ── Step 1a: Environment agent ───────────────────────────────────────
    print(f"  Step 1a: Environment agent via {provider}/{resolved_model}...",
          file=sys.stderr)
    env_candidates = _generate_agent_rubrics(
        agent="environment",
        system_prompt=ENVIRONMENT_AGENT_PROMPT,
        prompt_text=prompt_text,
        **shared,
    )
    print(f"    -> {len(env_candidates)} environment candidates", file=sys.stderr)

    # ── Step 1b: Object agent ────────────────────────────────────────────
    print(f"  Step 1b: Object agent via {provider}/{resolved_model}...",
          file=sys.stderr)
    obj_candidates = _generate_agent_rubrics(
        agent="object",
        system_prompt=OBJECT_AGENT_PROMPT,
        prompt_text=prompt_text,
        **shared,
    )
    print(f"    -> {len(obj_candidates)} object candidates", file=sys.stderr)

    all_candidates = env_candidates + obj_candidates
    n_candidates = len(all_candidates)

    # ── Step 2: Select final (max 5 per agent) ────────────────────────────
    final = select_final_rubrics(all_candidates)

    # ── Inject mandatory holistic coverage rubrics ────────────────────────
    final = [
        _make_mandatory_env_coverage_rubric(),
        _make_mandatory_obj_coverage_rubric(),
    ] + final

    # Re-number IDs sequentially per agent
    env_counter = 0
    obj_counter = 0
    for item in final:
        if item.agent == "environment":
            env_counter += 1
            item.rubric_id = f"env_{env_counter:02d}"
        else:
            obj_counter += 1
            item.rubric_id = f"obj_{obj_counter:02d}"

    agent_dist = {
        "environment": sum(1 for i in final if i.agent == "environment"),
        "object": sum(1 for i in final if i.agent == "object"),
    }
    print(f"  Step 2: Selected {len(final)} final rubrics | {agent_dist}",
          file=sys.stderr)

    return Rubric(
        scene_name=scene_name,
        prompt_text=prompt_text,
        rubric_items=final,
        metadata=RubricMetadata(
            generator_provider=provider,
            generator_model=resolved_model,
            generated_at=datetime.now(timezone.utc).isoformat(),
            num_candidates_generated=n_candidates,
        ),
    )
