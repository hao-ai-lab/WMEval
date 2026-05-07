
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import List, Literal, Optional

from .schemas import (
    VALID_WEIGHTS,
    RubricItem,
    RubricMetadata,
    Rubric,
    VLMCandidateOutput,
    export_candidate_schema,
)


ENVIRONMENT_AGENT_PROMPT = """\
You are an expert in evaluation rubric design. Your task is to generate Environment-Consistency rubrics to help a VLM judge score whether a generated video's environment remains consistent with the prompt over time. Each rubric is a binary (yes/no) question answerable from timestamped frames, derived from the text prompt only, and grounded in something the prompt explicitly states or directly implies.

### Important Notes:
1. Consistency: Each rubric tests whether one environmental attribute (e.g., ground material/pattern, sky, lighting, terrain, dominant style/palette) matches the prompt AND remains the same across ALL sampled frames — it must not switch, drift, or contradict the prompt at any timestamp. The camera may rotate, but the environment is assumed static unless the prompt explicitly describes spatial variation or change.
2. Described Variation: If the prompt itself describes an environmental change or contrast (e.g., lighting shifts, terrain changes), do NOT enforce uniform consistency; instead, generate a rubric that checks whether the described variation or contrast is preserved.
3. Atomicity: one environmental attribute per rubric.
4. Scope: Only scene-level attributes (ground, sky, lighting, terrain, style, palette). Salient entities, characters, or main objects are handled by the Object agent.
5. Rubric Construction: Prefer fewer high-confidence criteria over exhaustive lists.

### Output Format:
Return valid JSON matching the provided schema. Each rubric item contains:
- rubric_id: unique ID with the "env_" prefix (env_01, env_02, …).
- agent: always "environment".
- question: the binary yes/no question for the VLM judge.
- rationale: a brief justification explaining which prompt phrase the rubric is grounded in and what it checks.

<example>
{
  "rubric_items": [
    {
      "rubric_id": "env_01",
      "agent": "environment",
      "question": "Does the ground remain a black-and-white chessboard pattern over time?",
      "rationale": "The chessboard ground is the defining environmental attribute and should not change as the camera rotates."
    }
  ]
}
</example>
"""


OBJECT_AGENT_PROMPT = """\
You are an expert in text-to-video evaluation rubric design. Your task is to generate Object-Consistency rubrics to help a VLM judge score whether a generated video realizes the salient entities (characters, vehicles, props, primary objects) described in the text prompt, with their claimed attributes. Each rubric is a binary (yes/no) question answerable from timestamped frames, derived from the text prompt only, and grounded in something the prompt explicitly states.

### Important Notes:
1. Attribute Fidelity: Each rubric targets claimed attributes (shape, texture, size， color, material) of one salient entity.
2. Atomicity: one entity per rubric.
3. Scope: Only salient entities or characters. Scene-level structure, ground, sky, lighting, and palette are handled by the Environment agent.
4. Rubric Construction: Prefer fewer high-confidence criteria over exhaustive lists.

### Output Format:
Return valid JSON matching the provided schema. Each rubric item contains:
- rubric_id: unique ID with the "obj_" prefix (obj_01, obj_02, …).
- agent: always "object".
- question: the binary yes/no question for the VLM judge.
- rationale: a brief justification explaining which prompt phrase the rubric is grounded in and what it checks.

<example>
{
  "rubric_items": [
    {
      "rubric_id": "obj_01",
      "agent": "object",
      "question": "When the sphere is visible and judgeable, does it have the chrome appearance specified in the prompt?",
      "rationale": "The prompt explicitly states the sphere is chrome, so its material appearance should be correct whenever it can be judged."
    }
  ]
}
</example>
"""


def _make_mandatory_env_coverage_rubric() -> "RubricItem":
    return RubricItem(
        rubric_id="env_00",
        agent="environment",
        weight=1.0,
        rubric_text=(
            "Do all scene-level attributes described in the prompt "
            "(e.g., ground type, sky, lighting, terrain, style) "
            "appear at some point in the video?"
        ),
    )


def _make_mandatory_obj_coverage_rubric() -> "RubricItem":
    return RubricItem(
        rubric_id="obj_00",
        agent="object",
        weight=1.0,
        rubric_text=(
            "Do all characters and primary objects mentioned in the prompt "
            "appear in the video?"
        ),
    )


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


def _clamp_weight(w: float) -> float:
    allowed = sorted(VALID_WEIGHTS)
    return min(allowed, key=lambda a: abs(a - w))


def _generate_agent_rubrics(
    agent: Literal["environment", "object"],
    system_prompt: str,
    prompt_text: str,
    scene_name: str,
    provider: str,
    model: str,
    api_key: str,
) -> List[RubricItem]:
    mod = _get_provider_module(provider)
    schema = export_candidate_schema(provider)

    user_text = (
        f"Scene name: {scene_name}\n\n"
        f"Prompt:\n{prompt_text}\n\n"
        f"Generate up to 5 {agent} rubrics for this scene."
    )

    raw = mod.call_llm(
        system_prompt=system_prompt,
        user_text=user_text,
        schema=schema,
        api_key=api_key,
        model=model,
        images_b64=None,
    )

    vlm_out = VLMCandidateOutput.model_validate(raw)
    items: List[RubricItem] = []
    counter = 0
    prefix = "env" if agent == "environment" else "obj"

    for vi in vlm_out.rubric_items[:5]:
        counter += 1
        rid = f"{prefix}_{counter:02d}"

        rubric_text = vi.question.strip() or vi.rubric_text.strip()

        items.append(RubricItem(
            rubric_id=rid,
            agent=agent,
            weight=_clamp_weight(vi.weight),
            rubric_text=rubric_text,
        ))

    return items


MAX_PER_AGENT = 5


def select_final_rubrics(items: List[RubricItem]) -> List[RubricItem]:
    env = sorted(
        [i for i in items if i.agent == "environment"],
        key=lambda x: -x.weight,
    )[:MAX_PER_AGENT]
    obj = sorted(
        [i for i in items if i.agent == "object"],
        key=lambda x: -x.weight,
    )[:MAX_PER_AGENT]
    return env + obj


def generate_rubric(
    prompt_text: str,
    scene_name: str,
    provider: Literal["openai", "anthropic", "gemini"] = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Rubric:
    mod = _get_provider_module(provider)
    resolved_key = api_key or _resolve_api_key(provider)
    resolved_model = model or mod.DEFAULT_MODEL

    shared = dict(
        scene_name=scene_name,
        provider=provider,
        model=resolved_model,
        api_key=resolved_key,
    )

    print(f"  Step 1a: Environment agent via {provider}/{resolved_model}...",
          file=sys.stderr)
    env_candidates = _generate_agent_rubrics(
        agent="environment",
        system_prompt=ENVIRONMENT_AGENT_PROMPT,
        prompt_text=prompt_text,
        **shared,
    )
    print(f"    -> {len(env_candidates)} environment candidates", file=sys.stderr)

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

    final = select_final_rubrics(all_candidates)


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
