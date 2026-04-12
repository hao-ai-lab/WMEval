from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..context import EvaluationContext
from ..video_io import (
    build_default_event_offsets,
    build_vlm_keyframe_indices,
    get_video_metadata,
    load_frames_by_indices,
)

from .lpips import analyze_lpips_from_context, compute_lpips_scores, summarize_scores

PROMPT_VERSION_DEFAULT = "causal_collision_v3"

DEFAULT_DIRECTIONAL_SIDES: FrozenSet[str] = frozenset(
    {"left", "right", "front", "back"}
)
DEFAULT_CONTACT_TYPES: Tuple[str, ...] = ("collision", "impact", "contact")

DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "You are a careful video analyst for physics-style evaluation clips. "
    "You must ground every conclusion in the provided keyframe images. "
    "Respond with a single JSON object only (no markdown fences)."
)

# Placeholders: {idx_str}, {extra_instructions}
DEFAULT_JUDGE_USER_PROMPT_TEMPLATE = """Analyze these keyframes from ONE video. Keyframe order matches image order.
Allowed frame indices (only these may appear in JSON): {idx_str}
{extra_instructions}
Follow this checklist and put concise notes in reasoning_steps (string array):
A) Scene and camera (fixed or moving); note if the view makes depth ambiguous.
B) Main moving object(s), other relevant objects, and static boundaries (walls, floor, props).
C) Scan in time order; flag visually unambiguous contact/impact episodes; cite frame indices from the allowed list only.
D) For each episode assign side (choose the best single label):
   - "left" or "right": contact is primarily lateral in the image (screen-left vs screen-right).
   - "front": contact toward the viewer / foreground (depth into camera).
   - "back": contact away from the viewer / background or rear boundary.
   - "unknown": only if none of the above is clear from the keyframes.
E) Deduplicate: one event per distinct contact episode.
F) Abstain with "events": [] if quality is insufficient.

For EACH event object include:
- type: use one of the contact/interaction types your instructions expect (often "collision").
- side: one of the directional labels above or "unknown"
- frame: integer, must be one of the allowed indices (best contact frame)
- confidence: number 0-1 (optional)
- evidence_frame_indices: non-empty array of allowed integers (typically 2-4 frames around contact)
- evidence_summary: 1-3 sentences describing visible evidence

Output JSON shape:
{{"reasoning_steps": ["..."], "events": [...]}}
"""

DETECTION_ONLY_USER_PROMPT_TEMPLATE = """Analyze these keyframes from ONE video. Keyframe order matches image order.
Allowed frame indices (only these may appear in JSON): {idx_str}
{extra_instructions}
Your only job is to detect distinct contact/impact episodes in chronological order.
Do not spend effort on left/right/front/back semantics in this pass. Use side="unknown" unless the side is completely unavoidable from the local evidence.

Follow this checklist and put concise notes in reasoning_steps (string array):
A) Identify the main moving object and the nearby obstacles/targets it contacts.
B) Scan in time order and enumerate each distinct contact episode once.
C) Keep episodes separate when there is meaningful temporal separation.
   If a visible gap reopens and the moving object later contacts the same or another obstacle again, treat that as a new event rather than merging it into one long episode.
   Do not merge two episodes just because they happen on the same general side of the frame.
D) Cite only allowed frame indices.
E) Abstain with "events": [] if quality is insufficient.

For EACH event object include:
- type: usually "collision"
- side: set to "unknown" unless truly unambiguous
- frame: integer, must be one of the allowed indices (best contact frame)
- confidence: number 0-1 (optional)
- evidence_frame_indices: non-empty array of allowed integers (typically 2-4 frames around contact)
- evidence_summary: 1-3 sentences describing visible contact evidence only

Output JSON shape:
{{"reasoning_steps": ["..."], "events": [...]}}
"""

HIGH_RECALL_DETECTION_USER_PROMPT_TEMPLATE = """Analyze these keyframes from ONE video. Keyframe order matches image order.
Allowed frame indices (only these may appear in JSON): {idx_str}
{extra_instructions}
Your only job is to recover as many distinct contact/impact episodes as are plausibly supported by the keyframes.
Prefer recall over precision in this pass: if two episodes are temporally distinct, list both rather than collapsing them into one summary event.
Do not spend effort on left/right/front/back semantics in this pass. Use side="unknown" unless the side is completely unavoidable from the local evidence.

Follow this checklist and put concise notes in reasoning_steps (string array):
A) Identify the main moving object and the nearby obstacles/targets it contacts.
B) Scan in time order and list every temporally distinct contact episode you can support from the frames.
C) If contact stops and later resumes, treat that as a new event.
D) If a later contact looks brief, grazing, or rebound-like, still include it when it is temporally separate from earlier contact.
E) Do not collapse multiple contacts into one event just because they involve nearby obstacles or similar motion.
F) Cite only allowed frame indices.
G) Abstain with "events": [] only if the frames truly do not support any contact episode.

For EACH event object include:
- type: usually "collision"
- side: set to "unknown" unless truly unambiguous
- frame: integer, must be one of the allowed indices (best contact frame)
- confidence: number 0-1 (optional)
- evidence_frame_indices: non-empty array of allowed integers (prefer 3-8 frames spanning approach/contact/separation when available)
- evidence_summary: 1-3 sentences describing visible contact evidence only

Output JSON shape:
{{"reasoning_steps": ["..."], "events": [...]}}
"""

SIDE_RELABEL_USER_PROMPT_TEMPLATE = """Analyze these keyframes from ONE video around a single already-detected contact episode.
Allowed frame indices (only these may appear in JSON): {idx_str}
Event anchor frame: {anchor_frame}
Current event summary: {event_summary}
{extra_instructions}
Classify the side of this ONE contact episode using motion before/at/after contact.

Choose exactly one side:
- "left" or "right": contact is primarily lateral in image space
- "front": contact is with an obstacle ahead in the moving object's forward/depth direction
- "back": contact is with an obstacle behind/rearward or after the moving object has already passed it
- "unknown": only if the side remains ambiguous from the provided frames

Use temporal cues, not just a single overlap frame:
- compare approach vs departure frames
- if the object is approaching an obstacle head-on in depth, prefer "front"
- if contact happens after the object has already passed an obstacle or with a rearward obstacle, prefer "back"

Output JSON shape:
{{"reasoning_steps": ["..."], "side": "front|back|left|right|unknown", "confidence": 0.0, "evidence_frame_indices": [...], "evidence_summary": "..."}}
"""


def _log_causal(stage: str, **fields: Any) -> None:
    payload = {
        "component": "causal_vlm",
        "stage": stage,
        "ts": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def resolve_directional_sides(artifacts: Mapping[str, Any]) -> FrozenSet[str]:
    raw = artifacts.get("lpips_vlm_directional_sides")
    if raw is None:
        return DEFAULT_DIRECTIONAL_SIDES
    if isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        return frozenset(parts) if parts else DEFAULT_DIRECTIONAL_SIDES
    try:
        seq = [str(x).strip().lower() for x in raw if str(x).strip()]
        return frozenset(seq) if seq else DEFAULT_DIRECTIONAL_SIDES
    except TypeError:
        return DEFAULT_DIRECTIONAL_SIDES


def resolve_contact_types(artifacts: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = artifacts.get("lpips_vlm_contact_types")
    if raw is None:
        return DEFAULT_CONTACT_TYPES
    if isinstance(raw, str):
        parts = tuple(p.strip().lower() for p in raw.split(",") if p.strip())
        return parts if parts else DEFAULT_CONTACT_TYPES
    try:
        parts = tuple(str(x).strip().lower() for x in raw if str(x).strip())
        return parts if parts else DEFAULT_CONTACT_TYPES
    except TypeError:
        return DEFAULT_CONTACT_TYPES


def resolve_judge_system_prompt(artifacts: Mapping[str, Any]) -> str:
    s = artifacts.get("lpips_vlm_system_prompt")
    if isinstance(s, str) and s.strip():
        return s.strip()
    return DEFAULT_JUDGE_SYSTEM_PROMPT


def build_judge_user_prompt(
    allowed_indices: Sequence[int],
    artifacts: Optional[Mapping[str, Any]] = None,
    *,
    max_events: Optional[int] = None,
) -> str:
    artifacts = artifacts or {}
    idx_str = ", ".join(str(i) for i in allowed_indices)
    hints = artifacts.get("lpips_vlm_scene_hints")
    if isinstance(hints, str) and hints.strip():
        extra_instructions = (
            "\n## Scene-specific instructions (follow carefully)\n"
            + hints.strip()
            + "\n"
        )
    else:
        extra_instructions = ""

    if max_events is not None:
        max_events_text = "ONE" if int(max_events) == 1 else str(int(max_events))
        extra_instructions += (
            "Return at most "
            + max_events_text
            + " event"
            + ("" if int(max_events) == 1 else "s")
            + ", and prefer the earliest clearly supported event when uncertain.\n"
        )

    tmpl = artifacts.get("lpips_vlm_judge_user_prompt_template")
    if isinstance(tmpl, str) and tmpl.strip():
        template = tmpl.strip()
    else:
        template = DEFAULT_JUDGE_USER_PROMPT_TEMPLATE

    try:
        return template.format(idx_str=idx_str, extra_instructions=extra_instructions)
    except KeyError as exc:
        raise ValueError(
            "lpips_vlm_judge_user_prompt_template must include placeholders "
            "{idx_str} and {extra_instructions} (extra may be empty)."
        ) from exc


def build_detection_user_prompt(
    allowed_indices: Sequence[int],
    artifacts: Optional[Mapping[str, Any]] = None,
    *,
    max_events: Optional[int] = None,
    high_recall: bool = False,
) -> str:
    local_artifacts = dict(artifacts or {})
    local_artifacts["lpips_vlm_judge_user_prompt_template"] = (
        HIGH_RECALL_DETECTION_USER_PROMPT_TEMPLATE
        if high_recall
        else DETECTION_ONLY_USER_PROMPT_TEMPLATE
    )
    return build_judge_user_prompt(
        allowed_indices,
        local_artifacts,
        max_events=max_events,
    )


def build_side_relabel_user_prompt(
    allowed_indices: Sequence[int],
    event: Mapping[str, Any],
    artifacts: Optional[Mapping[str, Any]] = None,
) -> str:
    artifacts = artifacts or {}
    idx_str = ", ".join(str(i) for i in allowed_indices)
    hints = artifacts.get("lpips_vlm_scene_hints")
    if isinstance(hints, str) and hints.strip():
        extra_instructions = (
            "\n## Scene-specific instructions (follow carefully)\n"
            + hints.strip()
            + "\n"
        )
    else:
        extra_instructions = ""

    return SIDE_RELABEL_USER_PROMPT_TEMPLATE.format(
        idx_str=idx_str,
        anchor_frame=int(event.get("frame", 0)),
        event_summary=str(event.get("evidence_summary", "")).strip() or "n/a",
        extra_instructions=extra_instructions,
    )


def filter_events_by_contact_types(
    events: List[Dict[str, Any]], allowed_types: Sequence[str]
) -> List[Dict[str, Any]]:
    allow = {t.strip().lower() for t in allowed_types if str(t).strip()}
    if not allow:
        allow = set(DEFAULT_CONTACT_TYPES)
    out: List[Dict[str, Any]] = []
    for e in events:
        t = str(e.get("type", "collision")).lower()
        if t in allow:
            out.append(e)
    return out


def filter_events_by_type(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Backward-compatible helper used by older tests and scripts."""
    out: List[Dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("type", "")).strip().lower()
        if (
            event_type == "bounce"
            or "collision" in event_type
            or "impact" in event_type
            or "contact" in event_type
        ):
            out.append(event)
    return out


def resolve_lpips_temporal_pairing(
    artifacts: Mapping[str, Any],
    *,
    event_source: str,
) -> str:
    raw = str(artifacts.get("lpips_causal_lpips_pairing", "") or "").strip().lower()
    if raw and raw != "auto":
        return raw
    return "per_video_contact" if str(event_source).strip().lower() == "both" else "wall_clock"


def paired_frames_per_video_contact(
    reference_contact_frame: int,
    candidate_contact_frame: int,
    offset: int,
    reference_metadata: Any,
    candidate_metadata: Any,
) -> Tuple[int, int]:
    ref_idx = min(
        max(int(reference_contact_frame) + int(offset), 0),
        int(reference_metadata.frame_count) - 1,
    )
    cand_idx = min(
        max(int(candidate_contact_frame) + int(offset), 0),
        int(candidate_metadata.frame_count) - 1,
    )
    return ref_idx, cand_idx


def paired_frames_for_reference_event_offset(
    reference_contact_frame: int,
    offset: int,
    reference_metadata: Any,
    candidate_metadata: Any,
) -> Tuple[int, int]:
    ref_idx = min(
        max(int(reference_contact_frame) + int(offset), 0),
        int(reference_metadata.frame_count) - 1,
    )
    ref_time_sec = float(ref_idx) / float(reference_metadata.fps)
    cand_idx = int(round(ref_time_sec * float(candidate_metadata.fps)))
    cand_idx = min(max(cand_idx, 0), int(candidate_metadata.frame_count) - 1)
    return ref_idx, cand_idx


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("judge output is not a JSON object")
    return obj


def repair_json_object_with_model(
    *,
    client: Any,
    model: str,
    raw_text: str,
    timeout_sec: float,
) -> Dict[str, Any]:
    repair_prompt = (
        "The following model output was intended to be a single JSON object, but it is malformed. "
        "Return only a repaired JSON object with the same intended fields and values. "
        "Do not add markdown fences or commentary.\n\n"
        "Malformed output:\n"
        f"{raw_text}"
    )
    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": DEFAULT_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": repair_prompt},
        ],
    }
    if str(model).startswith("gpt-5"):
        create_kwargs["reasoning_effort"] = "none"
        create_kwargs["temperature"] = 0
    else:
        create_kwargs["temperature"] = 0

    client = client.with_options(timeout=timeout_sec, max_retries=1)
    resp = client.chat.completions.create(**create_kwargs)
    repaired_text = (resp.choices[0].message.content or "").strip()
    return extract_json_object(repaired_text)


def parse_or_repair_judge_output(
    text: str,
    *,
    repair_json: Optional[Any] = None,
    retry_json: Optional[Any] = None,
) -> Dict[str, Any]:
    try:
        return extract_json_object(text)
    except Exception as parse_exc:
        repair_exc: Optional[Exception] = None
        if repair_json is not None:
            try:
                return repair_json(text)
            except Exception as exc:
                repair_exc = exc
        if retry_json is not None:
            return retry_json(text, parse_exc, repair_exc)
        if repair_exc is not None:
            raise RuntimeError(
                f"json_parse_failed: {parse_exc}; json_repair_failed: {repair_exc}"
            ) from repair_exc
        raise


def sanitize_events_to_grid(
    events: List[Dict[str, Any]],
    allowed: set[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for raw in events:
        e = dict(raw)
        try:
            frame = int(e.get("frame", -1))
        except (TypeError, ValueError):
            dropped.append({**e, "drop_reason": "invalid_frame"})
            continue
        if frame not in allowed:
            dropped.append({**e, "drop_reason": "frame_not_in_grid"})
            continue
        ev_list = e.get("evidence_frame_indices")
        if not isinstance(ev_list, list) or not ev_list:
            dropped.append({**e, "drop_reason": "missing_evidence_frame_indices"})
            continue
        clean_ev: List[int] = []
        for x in ev_list:
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if xi in allowed:
                clean_ev.append(xi)
        if not clean_ev:
            dropped.append({**e, "drop_reason": "evidence_not_in_grid"})
            continue
        e["evidence_frame_indices"] = sorted(set(clean_ev))
        e["frame"] = frame
        kept.append(e)
    return kept, dropped


def sanitize_relabel_payload(
    payload: Mapping[str, Any],
    *,
    allowed: set[int],
) -> Dict[str, Any]:
    side = str(payload.get("side", "unknown")).strip().lower() or "unknown"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence_raw = payload.get("evidence_frame_indices")
    evidence: List[int] = []
    if isinstance(evidence_raw, list):
        for value in evidence_raw:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if idx in allowed:
                evidence.append(idx)
    reasoning_raw = payload.get("reasoning_steps")
    reasoning_steps = (
        [str(x) for x in reasoning_raw if str(x).strip()]
        if isinstance(reasoning_raw, list)
        else []
    )
    return {
        "side": side,
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence_frame_indices": sorted(set(evidence)),
        "evidence_summary": str(payload.get("evidence_summary", "")).strip(),
        "reasoning_steps": reasoning_steps,
    }


def resolve_vlm_sampling_strategy(artifacts: Mapping[str, Any]) -> str:
    raw = str(artifacts.get("lpips_vlm_sampling_strategy", "single_grid")).strip().lower()
    if raw in {"single_grid", "coarse_to_fine", "windowed_coarse_to_fine"}:
        return raw
    return "single_grid"


def build_windowed_keyframe_groups(
    *,
    frame_count: int,
    fps: float,
    window_sec: float,
    overlap_sec: float,
    max_frames_per_window: int,
) -> List[List[int]]:
    if frame_count <= 0 or fps <= 0:
        return []
    window_frames = max(2, int(round(window_sec * fps)))
    overlap_frames = max(0, int(round(overlap_sec * fps)))
    step_frames = max(1, window_frames - overlap_frames)
    groups: List[List[int]] = []
    start = 0
    while start < frame_count:
        end = min(frame_count - 1, start + window_frames - 1)
        span = end - start + 1
        sample_count = min(max(2, max_frames_per_window), span)
        raw = np.linspace(start, end, num=sample_count)
        group = sorted(set(min(max(int(round(float(x))), 0), frame_count - 1) for x in raw))
        if group:
            groups.append(group)
        if end >= frame_count - 1:
            break
        start += step_frames
    return groups


def merge_nearby_events(
    events: Sequence[Dict[str, Any]],
    *,
    frame_gap: int,
) -> List[Dict[str, Any]]:
    if not events:
        return []
    ordered = sorted((dict(e) for e in events), key=lambda e: int(e.get("frame", 0)))
    merged: List[Dict[str, Any]] = [ordered[0]]
    for event in ordered[1:]:
        prev = merged[-1]
        if int(event.get("frame", 0)) - int(prev.get("frame", 0)) > int(frame_gap):
            merged.append(event)
            continue
        prev_conf = float(prev.get("confidence", 0.0) or 0.0)
        new_conf = float(event.get("confidence", 0.0) or 0.0)
        chosen = prev
        if new_conf > prev_conf:
            chosen = event
        evidence = set(int(x) for x in prev.get("evidence_frame_indices", []) if isinstance(x, int))
        evidence.update(
            int(x)
            for x in event.get("evidence_frame_indices", [])
            if isinstance(x, int)
        )
        chosen["evidence_frame_indices"] = sorted(evidence)
        merged[-1] = chosen
    return merged


def split_events_by_evidence_gaps(
    events: Sequence[Dict[str, Any]],
    *,
    frame_gap: int,
    min_cluster_size: int = 2,
) -> List[Dict[str, Any]]:
    if not events:
        return []

    split_events: List[Dict[str, Any]] = []
    for raw_event in sorted(events, key=lambda e: int(e.get("frame", 0))):
        event = dict(raw_event)
        evidence_raw = event.get("evidence_frame_indices")
        if not isinstance(evidence_raw, list):
            split_events.append(event)
            continue
        evidence = sorted(
            {
                int(value)
                for value in evidence_raw
                if isinstance(value, (int, np.integer))
            }
        )
        if len(evidence) < max(2, int(min_cluster_size)):
            event["evidence_frame_indices"] = evidence
            split_events.append(event)
            continue

        clusters: List[List[int]] = [[evidence[0]]]
        for idx in evidence[1:]:
            if idx - clusters[-1][-1] > int(frame_gap):
                clusters.append([idx])
            else:
                clusters[-1].append(idx)

        if len(clusters) == 1:
            event["evidence_frame_indices"] = evidence
            split_events.append(event)
            continue

        source_summary = str(event.get("evidence_summary", "")).strip()
        try:
            base_confidence = float(event.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            base_confidence = 0.0

        emitted = False
        for cluster_index, cluster in enumerate(clusters):
            if len(cluster) < int(min_cluster_size):
                continue
            split_event = dict(event)
            split_event["evidence_frame_indices"] = cluster
            anchor = int(event.get("frame", cluster[len(cluster) // 2]))
            if anchor not in cluster:
                anchor = cluster[len(cluster) // 2]
            split_event["frame"] = anchor
            split_event["confidence"] = base_confidence
            if source_summary:
                split_event["evidence_summary"] = (
                    f"{source_summary} Split episode {cluster_index + 1}/{len(clusters)} "
                    f"from frames {cluster[0]}-{cluster[-1]}."
                )
            split_events.append(split_event)
            emitted = True

        if not emitted:
            event["evidence_frame_indices"] = evidence
            split_events.append(event)

    return split_events


def merge_fine_and_coarse_events(
    fine_events: Sequence[Dict[str, Any]],
    coarse_events: Sequence[Dict[str, Any]],
    *,
    frame_gap: int,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = [dict(event) for event in fine_events]
    for coarse in sorted(coarse_events, key=lambda e: int(e.get("frame", 0))):
        coarse_frame = int(coarse.get("frame", 0))
        matched = False
        for index, fine in enumerate(merged):
            fine_frame = int(fine.get("frame", 0))
            if abs(fine_frame - coarse_frame) > int(frame_gap):
                continue
            matched = True
            fine_conf = float(fine.get("confidence", 0.0) or 0.0)
            coarse_conf = float(coarse.get("confidence", 0.0) or 0.0)
            chosen = dict(fine)
            if coarse_conf > fine_conf:
                chosen.update(dict(coarse))
            evidence = {
                int(value)
                for value in fine.get("evidence_frame_indices", [])
                if isinstance(value, (int, np.integer))
            }
            evidence.update(
                int(value)
                for value in coarse.get("evidence_frame_indices", [])
                if isinstance(value, (int, np.integer))
            )
            if evidence:
                chosen["evidence_frame_indices"] = sorted(evidence)
            merged[index] = chosen
            break
        if not matched:
            merged.append(dict(coarse))
    return sorted(merged, key=lambda e: int(e.get("frame", 0)))


def build_refine_window_indices(
    *,
    coarse_events: List[Dict[str, Any]],
    frame_count: int,
    fps: float,
    window_sec: float,
    max_frames_per_window: int,
) -> List[int]:
    if frame_count <= 0 or fps <= 0 or not coarse_events:
        return []

    radius_frames = max(1, int(round(float(window_sec) * float(fps))))
    indices: set[int] = set()
    for event in coarse_events:
        anchors = [int(event.get("frame", 0))]
        evidence = event.get("evidence_frame_indices")
        if isinstance(evidence, list):
            for value in evidence:
                try:
                    anchors.append(int(value))
                except (TypeError, ValueError):
                    continue
        lo = max(0, min(anchors) - radius_frames)
        hi = min(frame_count - 1, max(anchors) + radius_frames)
        span = hi - lo + 1
        sample_count = min(max(1, int(max_frames_per_window)), span)
        raw = np.linspace(lo, hi, num=sample_count)
        for x in raw:
            indices.add(min(max(int(round(float(x))), 0), frame_count - 1))
    return sorted(indices)


def _event_compact(ev: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": ev.get("type"),
        "side": ev.get("side"),
        "frame": ev.get("frame"),
        "confidence": ev.get("confidence"),
        "evidence_summary": ev.get("evidence_summary"),
    }


def build_action_summary(
    events: List[Dict[str, Any]], frame_count: int
) -> List[Dict[str, Any]]:
    denom = max(1, int(frame_count) - 1)
    rows: List[Dict[str, Any]] = []
    for ev in sorted(events, key=lambda e: int(e.get("frame", 0))):
        fi = int(ev.get("frame", 0))
        rows.append(
            {
                **_event_compact(ev),
                "normalized_time": float(fi) / float(denom),
            }
        )
    return rows


def pair_collision_events_strict_with_log(
    ref_events: List[Dict[str, Any]],
    cand_events: List[Dict[str, Any]],
    *,
    directional_sides: FrozenSet[str] = DEFAULT_DIRECTIONAL_SIDES,
) -> Tuple[
    List[Tuple[Dict[str, Any], Dict[str, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    ref_sorted = sorted(ref_events, key=lambda e: int(e.get("frame", 0)))
    cand_sorted = sorted(cand_events, key=lambda e: int(e.get("frame", 0)))
    cand_used = [False] * len(cand_sorted)
    matched: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    unmatched_ref: List[Dict[str, Any]] = []
    pairing_log: List[Dict[str, Any]] = []

    for r in ref_sorted:
        rs = str(r.get("side", "unknown")).lower()
        found: Optional[int] = None
        if rs in directional_sides:
            for j, c in enumerate(cand_sorted):
                if cand_used[j]:
                    continue
                if str(c.get("side", "unknown")).lower() == rs:
                    found = j
                    break
        else:
            for j, c in enumerate(cand_sorted):
                if cand_used[j]:
                    continue
                if str(c.get("side", "unknown")).lower() == "unknown":
                    found = j
                    break
        if found is not None:
            cand_used[found] = True
            matched.append((r, cand_sorted[found]))
            pairing_log.append(
                {
                    "ref_event": _event_compact(r),
                    "status": "matched",
                    "cand_event": _event_compact(cand_sorted[found]),
                    "match_rule": "strict_side",
                }
            )
        else:
            unmatched_ref.append(r)
            if not cand_sorted:
                reason = "no_candidate_events"
                detail: Dict[str, Any] = {}
            elif all(cand_used):
                reason = "no_candidates_remaining"
                detail = {}
            elif rs in directional_sides:
                unused_sides = [
                    str(cand_sorted[j].get("side", "unknown")).lower()
                    for j in range(len(cand_sorted))
                    if not cand_used[j]
                ]
                reason = f"no_unused_candidate_with_side_{rs}"
                detail = {"unused_candidate_sides": unused_sides}
            else:
                reason = "no_unused_candidate_with_side_unknown"
                detail = {
                    "unused_candidate_sides": [
                        str(cand_sorted[j].get("side", "unknown")).lower()
                        for j in range(len(cand_sorted))
                        if not cand_used[j]
                    ]
                }
            pairing_log.append(
                {
                    "ref_event": _event_compact(r),
                    "status": "unmatched",
                    "reason": reason,
                    "detail": detail,
                }
            )

    unmatched_cand = [cand_sorted[j] for j in range(len(cand_sorted)) if not cand_used[j]]
    return matched, unmatched_ref, unmatched_cand, pairing_log


def pair_collision_events_order_with_log(
    ref_events: List[Dict[str, Any]],
    cand_events: List[Dict[str, Any]],
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[Dict[str, Any]]]:
    ref_sorted = sorted(ref_events, key=lambda e: int(e.get("frame", 0)))
    cand_sorted = sorted(cand_events, key=lambda e: int(e.get("frame", 0)))
    matched: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    pairing_log: List[Dict[str, Any]] = []
    n = min(len(ref_sorted), len(cand_sorted))
    for i in range(n):
        r, c = ref_sorted[i], cand_sorted[i]
        matched.append((r, c))
        pairing_log.append(
            {
                "ref_event": _event_compact(r),
                "status": "matched",
                "cand_event": _event_compact(c),
                "match_rule": "order_index",
                "pair_index": i,
            }
        )
    for i in range(n, len(ref_sorted)):
        pairing_log.append(
            {
                "ref_event": _event_compact(ref_sorted[i]),
                "status": "unmatched",
                "reason": "no_candidate_at_order_index",
                "detail": {"pair_index": i},
            }
        )
    return matched, pairing_log


def _normalized_event_time(event: Dict[str, Any], frame_count: int) -> float:
    denom = max(1, int(frame_count) - 1)
    return float(int(event.get("frame", 0))) / float(denom)


def _pair_cost_monotonic(
    ref_event: Dict[str, Any],
    cand_event: Dict[str, Any],
    *,
    ref_frame_count: int,
    cand_frame_count: int,
    directional_sides: FrozenSet[str],
) -> float:
    rs = str(ref_event.get("side", "unknown")).lower()
    cs = str(cand_event.get("side", "unknown")).lower()
    time_cost = abs(
        _normalized_event_time(ref_event, ref_frame_count)
        - _normalized_event_time(cand_event, cand_frame_count)
    )
    if rs == cs:
        side_cost = 0.0
    elif "unknown" in (rs, cs):
        side_cost = 0.12
    elif rs in directional_sides and cs in directional_sides:
        side_cost = 0.35
    else:
        side_cost = 0.2
    return time_cost + side_cost


def pair_collision_events_monotonic_with_log(
    ref_events: List[Dict[str, Any]],
    cand_events: List[Dict[str, Any]],
    *,
    ref_frame_count: int,
    cand_frame_count: int,
    directional_sides: FrozenSet[str] = DEFAULT_DIRECTIONAL_SIDES,
    skip_cost: float = 0.45,
) -> Tuple[
    List[Tuple[Dict[str, Any], Dict[str, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    ref_sorted = sorted(ref_events, key=lambda e: int(e.get("frame", 0)))
    cand_sorted = sorted(cand_events, key=lambda e: int(e.get("frame", 0)))
    n_ref = len(ref_sorted)
    n_cand = len(cand_sorted)

    dp: List[List[float]] = [
        [0.0 for _ in range(n_cand + 1)] for _ in range(n_ref + 1)
    ]
    choice: List[List[str]] = [
        ["" for _ in range(n_cand + 1)] for _ in range(n_ref + 1)
    ]

    for i in range(n_ref - 1, -1, -1):
        dp[i][n_cand] = skip_cost * float(n_ref - i)
        choice[i][n_cand] = "skip_ref"
    for j in range(n_cand - 1, -1, -1):
        dp[n_ref][j] = skip_cost * float(n_cand - j)
        choice[n_ref][j] = "skip_cand"

    for i in range(n_ref - 1, -1, -1):
        for j in range(n_cand - 1, -1, -1):
            match_cost = _pair_cost_monotonic(
                ref_sorted[i],
                cand_sorted[j],
                ref_frame_count=ref_frame_count,
                cand_frame_count=cand_frame_count,
                directional_sides=directional_sides,
            ) + dp[i + 1][j + 1]
            skip_ref_cost = skip_cost + dp[i + 1][j]
            skip_cand_cost = skip_cost + dp[i][j + 1]

            best_cost = match_cost
            best_choice = "match"
            if skip_ref_cost < best_cost:
                best_cost = skip_ref_cost
                best_choice = "skip_ref"
            if skip_cand_cost < best_cost:
                best_cost = skip_cand_cost
                best_choice = "skip_cand"
            dp[i][j] = best_cost
            choice[i][j] = best_choice

    matched: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    unmatched_ref: List[Dict[str, Any]] = []
    unmatched_cand: List[Dict[str, Any]] = []
    pairing_log: List[Dict[str, Any]] = []

    i = 0
    j = 0
    while i < n_ref or j < n_cand:
        current = choice[i][j] if i <= n_ref and j <= n_cand else ""
        if i < n_ref and j < n_cand and current == "match":
            r, c = ref_sorted[i], cand_sorted[j]
            matched.append((r, c))
            pairing_log.append(
                {
                    "ref_event": _event_compact(r),
                    "status": "matched",
                    "cand_event": _event_compact(c),
                    "match_rule": "monotonic_side_time",
                    "pair_index": len(matched) - 1,
                    "pair_cost": _pair_cost_monotonic(
                        r,
                        c,
                        ref_frame_count=ref_frame_count,
                        cand_frame_count=cand_frame_count,
                        directional_sides=directional_sides,
                    ),
                }
            )
            i += 1
            j += 1
            continue
        if i < n_ref and (j == n_cand or current == "skip_ref"):
            unmatched_ref.append(ref_sorted[i])
            pairing_log.append(
                {
                    "ref_event": _event_compact(ref_sorted[i]),
                    "status": "unmatched",
                    "reason": "monotonic_skip_reference",
                    "detail": {"pair_index": len(matched)},
                }
            )
            i += 1
            continue
        if j < n_cand:
            unmatched_cand.append(cand_sorted[j])
            j += 1

    return matched, unmatched_ref, unmatched_cand, pairing_log


def pair_collision_events_ref_first(
    ref_events: List[Dict[str, Any]],
    cand_events: List[Dict[str, Any]],
) -> Tuple[
    List[Tuple[Dict[str, Any], Dict[str, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Backward-compatible wrapper around strict side-aware pairing."""
    matched, unmatched_ref, unmatched_cand, _ = pair_collision_events_strict_with_log(
        ref_events,
        cand_events,
    )
    return matched, unmatched_ref, unmatched_cand


def compute_first_divergence(
    ref_events: List[Dict[str, Any]],
    cand_events: List[Dict[str, Any]],
    ref_frame_count: int,
    cand_frame_count: int,
    temporal_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"event_timeline": {}, "temporal_stream": {}}

    if ref_events and cand_events:
        ref_s = sorted(ref_events, key=lambda e: int(e.get("frame", 0)))
        cand_s = sorted(cand_events, key=lambda e: int(e.get("frame", 0)))
        r0, c0 = ref_s[0], cand_s[0]
        rf, cf = int(r0.get("frame", 0)), int(c0.get("frame", 0))
        denom_r = max(1, int(ref_frame_count) - 1)
        denom_c = max(1, int(cand_frame_count) - 1)
        t_r = float(rf) / float(denom_r)
        t_c = float(cf) / float(denom_c)
        rs = str(r0.get("side", "unknown")).lower()
        cs = str(c0.get("side", "unknown")).lower()
        kth: List[Dict[str, Any]] = []
        for i in range(min(len(ref_s), len(cand_s))):
            ri = int(ref_s[i].get("frame", 0))
            ci = int(cand_s[i].get("frame", 0))
            kth.append(
                {
                    "k": i,
                    "ref_frame": ri,
                    "cand_frame": ci,
                    "ref_norm_t": float(ri) / float(denom_r),
                    "cand_norm_t": float(ci) / float(denom_c),
                    "norm_t_delta_cand_minus_ref": float(ci) / float(denom_c)
                    - float(ri) / float(denom_r),
                    "side_match": str(ref_s[i].get("side", "unknown")).lower()
                    == str(cand_s[i].get("side", "unknown")).lower(),
                }
            )
        out["event_timeline"] = {
            "first_reference_event": _event_compact(r0),
            "first_candidate_event": _event_compact(c0),
            "first_event_time_delta_cand_minus_ref": t_c - t_r,
            "first_event_side_mismatch": rs != cs,
            "reference_side": rs,
            "candidate_side": cs,
            "kth_event_deltas": kth,
        }
    else:
        out["event_timeline"] = {
            "note": "insufficient_events_after_sanitize",
            "num_reference_events": len(ref_events),
            "num_candidate_events": len(cand_events),
        }

    if temporal_result and temporal_result.get("lpips_per_pair"):
        scores = list(temporal_result["lpips_per_pair"])
        ts = list(temporal_result.get("timestamps_sec") or [])
        ridx = list(temporal_result.get("reference_frame_indices") or [])
        cidx = list(temporal_result.get("candidate_frame_indices") or [])
        n = len(scores)
        spike_k: Optional[int] = None
        threshold_rule = "none"
        if n >= 3:
            mu = float(np.mean(scores))
            sigma = float(np.std(scores))
            thr = mu + 2.0 * sigma
            threshold_rule = "first_above_mean_plus_2_std"
            for k in range(n):
                if float(scores[k]) > thr:
                    spike_k = k
                    break
            if spike_k is None:
                spike_k = int(np.argmax(scores))
                threshold_rule = "fallback_argmax_no_threshold_cross"
        elif n >= 1:
            spike_k = int(np.argmax(scores))
            threshold_rule = "argmax_n_lt_3"
        row: Dict[str, Any] = {
            "threshold_rule": threshold_rule,
            "first_lpips_spike_index": spike_k,
        }
        if spike_k is not None:
            row["lpips"] = float(scores[spike_k])
            if spike_k < len(ts):
                row["timestamp_sec"] = float(ts[spike_k])
            if spike_k < len(ridx):
                row["reference_frame_index"] = int(ridx[spike_k])
            if spike_k < len(cidx):
                row["candidate_frame_index"] = int(cidx[spike_k])
        out["temporal_stream"] = row
    else:
        out["temporal_stream"] = {"note": "no_temporal_lpips_per_pair"}

    return out


def _merge_temporal_primary_fields(
    base: Dict[str, Any], temporal_result: Dict[str, Any]
) -> None:
    base["temporal_baseline"] = temporal_result
    for k in ("lpips_mean", "lpips_median", "lpips_std", "lpips_min", "lpips_max"):
        if k in temporal_result:
            base[k] = temporal_result[k]
    base["lpips_per_pair"] = list(temporal_result.get("lpips_per_pair") or [])
    base["num_pairs"] = int(temporal_result.get("num_pairs") or len(base["lpips_per_pair"]))
    base["timestamps_sec"] = list(temporal_result.get("timestamps_sec") or [])
    base["reference_frame_indices"] = list(
        temporal_result.get("reference_frame_indices") or []
    )
    base["candidate_frame_indices"] = list(
        temporal_result.get("candidate_frame_indices") or []
    )
    base["lpips_primary_source"] = "temporal_only"


def bgr_to_data_url_jpeg(bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def call_openai_vision_judge(
    frames_bgr: Sequence[np.ndarray],
    model: str,
    *,
    system_prompt: str,
    user_text: str,
    timeout_sec: float = 120.0,
) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package required for causal_vlm_collision. pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key, timeout=timeout_sec, max_retries=2)
    image_parts: List[Dict[str, Any]] = []
    for bgr in frames_bgr:
        image_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": bgr_to_data_url_jpeg(bgr)},
            }
        )

    def _build_create_kwargs(user_prompt_text: str) -> Dict[str, Any]:
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt_text}, *image_parts],
                },
            ],
        }
        if str(model).startswith("gpt-5"):
            create_kwargs["reasoning_effort"] = "none"
            create_kwargs["temperature"] = 0
        else:
            create_kwargs["temperature"] = 0
        return create_kwargs

    def _run_chat(user_prompt_text: str, *, call_timeout_sec: float) -> str:
        run_client = client.with_options(timeout=call_timeout_sec, max_retries=2)
        resp = run_client.chat.completions.create(**_build_create_kwargs(user_prompt_text))
        return (resp.choices[0].message.content or "").strip()

    text = _run_chat(user_text, call_timeout_sec=timeout_sec)

    def _repair_json(raw_text: str) -> Dict[str, Any]:
        _log_causal(
            "judge_json_repair_start",
            model=model,
            error="malformed_judge_json",
        )
        repaired_payload = repair_json_object_with_model(
            client=client,
            model=model,
            raw_text=raw_text,
            timeout_sec=min(timeout_sec, 120.0),
        )
        _log_causal(
            "judge_json_repair_done",
            model=model,
        )
        return repaired_payload

    def _retry_json(
        raw_text: str,
        parse_exc: Exception,
        repair_exc: Optional[Exception],
    ) -> Dict[str, Any]:
        _log_causal(
            "judge_json_retry_start",
            model=model,
            parse_error=str(parse_exc),
            repair_error=str(repair_exc) if repair_exc is not None else None,
        )
        retry_text = _run_chat(
            (
                "Your previous response was malformed JSON. "
                "Re-answer the original task and return ONLY one valid JSON object. "
                "Do not include markdown fences or commentary.\n\n"
                "Original instructions:\n"
                f"{user_text}"
            ),
            call_timeout_sec=min(timeout_sec, 120.0),
        )
        payload = extract_json_object(retry_text)
        _log_causal(
            "judge_json_retry_done",
            model=model,
        )
        return payload

    return parse_or_repair_judge_output(
        text,
        repair_json=_repair_json,
        retry_json=_retry_json,
    )


def resolve_detection_max_events(
    artifacts: Mapping[str, Any],
    *,
    prompt_mode: str,
) -> Optional[int]:
    key = (
        "lpips_vlm_fine_max_events"
        if prompt_mode == "detection_high_recall"
        else "lpips_vlm_detection_max_events"
    )
    raw = artifacts.get(key)
    if raw is None:
        raw = 12 if prompt_mode == "detection_high_recall" else 6
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def run_vlm_judge_on_video(
    video_path: str,
    max_grid_frames: int,
    model: str,
    artifacts: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[int], List[np.ndarray]]:
    strategy = resolve_vlm_sampling_strategy(artifacts)
    contact_types = resolve_contact_types(artifacts)
    coarse_model = str(artifacts.get("lpips_vlm_coarse_model", model)).strip() or model
    side_relabel_enabled = bool(artifacts.get("lpips_vlm_side_relabel", True))
    side_model = str(artifacts.get("lpips_vlm_side_model", model)).strip() or model
    if strategy in {"coarse_to_fine", "windowed_coarse_to_fine"}:
        meta = get_video_metadata(video_path)
        if meta.frame_count <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")
        coarse_grid_frames = int(
            artifacts.get(
                "lpips_vlm_coarse_grid_frames",
                min(max_grid_frames, 64) if max_grid_frames > 0 else 64,
            )
        )
        windowed_logs: List[Dict[str, Any]] = []
        if strategy == "windowed_coarse_to_fine":
            window_sec = float(artifacts.get("lpips_vlm_window_sec", 2.5))
            overlap_sec = float(artifacts.get("lpips_vlm_window_overlap_sec", 1.25))
            per_window_frames = int(
                artifacts.get("lpips_vlm_window_max_frames", min(coarse_grid_frames, 24))
            )
            split_gap_frames = int(
                artifacts.get("lpips_vlm_event_split_gap_frames", max(8, int(round(meta.fps * 0.5))))
            )
            split_min_cluster_size = int(
                artifacts.get("lpips_vlm_event_split_min_cluster_size", 2)
            )
            coarse_groups = build_windowed_keyframe_groups(
                frame_count=meta.frame_count,
                fps=meta.fps,
                window_sec=window_sec,
                overlap_sec=overlap_sec,
                max_frames_per_window=per_window_frames,
            )
            coarse_indices = sorted({idx for group in coarse_groups for idx in group})
            coarse_events: List[Dict[str, Any]] = []
            coarse_dropped: List[Dict[str, Any]] = []
            for window_index, group in enumerate(coarse_groups):
                payload, _, _ = run_vlm_judge_on_indices(
                    video_path,
                    group,
                    coarse_model,
                    artifacts,
                    phase=f"coarse_window_{window_index}",
                    prompt_mode="detection_only",
                )
                raw_events = payload.get("events")
                if not isinstance(raw_events, list):
                    raw_events = []
                clean_events, clean_dropped = sanitize_events_to_grid(
                    [e for e in raw_events if isinstance(e, dict)],
                    set(group),
                )
                clean_events = filter_events_by_contact_types(clean_events, contact_types)
                coarse_events.extend(clean_events)
                coarse_dropped.extend(clean_dropped)
                windowed_logs.append(
                    {
                        "window_index": window_index,
                        "indices": group,
                        "num_events": len(clean_events),
                    }
                )
            merge_gap_frames = int(
                artifacts.get("lpips_vlm_window_merge_gap_frames", max(6, int(round(meta.fps * 0.6))))
            )
            coarse_events = merge_nearby_events(
                coarse_events,
                frame_gap=merge_gap_frames,
            )
            coarse_events = split_events_by_evidence_gaps(
                coarse_events,
                frame_gap=split_gap_frames,
                min_cluster_size=split_min_cluster_size,
            )
            coarse_payload = {
                "events": coarse_events,
                "reasoning_steps": [],
                "windowed_detection": {
                    "window_sec": window_sec,
                    "overlap_sec": overlap_sec,
                    "per_window_frames": per_window_frames,
                    "merge_gap_frames": merge_gap_frames,
                    "split_gap_frames": split_gap_frames,
                    "split_min_cluster_size": split_min_cluster_size,
                    "windows": windowed_logs,
                },
            }
        else:
            coarse_indices = build_vlm_keyframe_indices(meta.frame_count, coarse_grid_frames)
            coarse_payload, coarse_indices, _ = run_vlm_judge_on_indices(
                video_path,
                coarse_indices,
                coarse_model,
                artifacts,
                phase="coarse",
                prompt_mode="detection_only",
            )
            coarse_events_raw = coarse_payload.get("events")
            if not isinstance(coarse_events_raw, list):
                coarse_events_raw = []
            coarse_events, coarse_dropped = sanitize_events_to_grid(
                [e for e in coarse_events_raw if isinstance(e, dict)],
                set(coarse_indices),
            )
            coarse_events = filter_events_by_contact_types(coarse_events, contact_types)
        window_sec = float(artifacts.get("lpips_vlm_refine_window_sec", 1.0))
        max_frames_per_window = int(artifacts.get("lpips_vlm_refine_max_frames_per_window", 24))
        refine_indices = build_refine_window_indices(
            coarse_events=coarse_events,
            frame_count=meta.frame_count,
            fps=meta.fps,
            window_sec=window_sec,
            max_frames_per_window=max_frames_per_window,
        )
        if not refine_indices:
            coarse_payload["coarse_to_fine"] = {
                "strategy": strategy,
                "coarse_model": coarse_model,
                "coarse_grid_indices": coarse_indices,
                "coarse_events": coarse_events,
                "coarse_dropped_events": coarse_dropped,
                "windowed_detection": coarse_payload.get("windowed_detection"),
                "refine_indices": [],
                "used_fine_pass": False,
            }
            return coarse_payload, coarse_indices, []

        _log_causal(
            "coarse_to_fine_refine_plan",
            video_path=video_path,
            coarse_frames=len(coarse_indices),
            coarse_events=len(coarse_events),
            coarse_model=coarse_model,
            fine_model=model,
            refine_frames=len(refine_indices),
            refine_window_sec=window_sec,
        )
        fine_payload, fine_indices, fine_frames = run_vlm_judge_on_indices(
            video_path,
            refine_indices,
            model,
            artifacts,
            phase="fine",
            prompt_mode="detection_high_recall",
        )
        if side_relabel_enabled:
            fine_events_raw = fine_payload.get("events")
            if isinstance(fine_events_raw, list):
                fine_events_clean, _ = sanitize_events_to_grid(
                    [e for e in fine_events_raw if isinstance(e, dict)],
                    set(fine_indices),
                )
                fine_events_clean = filter_events_by_contact_types(
                    fine_events_clean, contact_types
                )
                fine_events_clean = split_events_by_evidence_gaps(
                    fine_events_clean,
                    frame_gap=int(
                        artifacts.get(
                            "lpips_vlm_event_split_gap_frames",
                            max(8, int(round(meta.fps * 0.5))),
                        )
                    ),
                    min_cluster_size=int(
                        artifacts.get("lpips_vlm_event_split_min_cluster_size", 2)
                    ),
                )
                fine_events_clean = merge_fine_and_coarse_events(
                    fine_events_clean,
                    coarse_events,
                    frame_gap=int(
                        artifacts.get("lpips_vlm_fine_coarse_merge_gap_frames", 18)
                    ),
                )
                relabeled_events, relabel_logs = relabel_event_sides_on_video(
                    video_path,
                    fine_events_clean,
                    model=side_model,
                    artifacts=artifacts,
                )
                fine_payload["events"] = relabeled_events
                fine_payload["side_relabel"] = {
                    "enabled": True,
                    "model": side_model,
                    "logs": relabel_logs,
                }
        fine_payload["coarse_to_fine"] = {
            "strategy": strategy,
            "coarse_model": coarse_model,
            "fine_model": model,
            "side_model": side_model if side_relabel_enabled else None,
            "coarse_grid_indices": coarse_indices,
            "coarse_events": coarse_events,
            "coarse_dropped_events": coarse_dropped,
            "windowed_detection": coarse_payload.get("windowed_detection"),
            "refine_indices": fine_indices,
            "used_fine_pass": True,
        }
        return fine_payload, fine_indices, fine_frames

    indices = build_vlm_keyframe_indices(
        get_video_metadata(video_path).frame_count,
        max_grid_frames,
    )
    payload, indices, frames = run_vlm_judge_on_indices(
        video_path,
        indices,
        model,
        artifacts,
        phase="single_grid",
        prompt_mode="detection_only",
    )
    if side_relabel_enabled:
        events_raw = payload.get("events")
        if isinstance(events_raw, list):
            events_clean, _ = sanitize_events_to_grid(
                [e for e in events_raw if isinstance(e, dict)],
                set(indices),
            )
            events_clean = filter_events_by_contact_types(events_clean, contact_types)
            relabeled_events, relabel_logs = relabel_event_sides_on_video(
                video_path,
                events_clean,
                model=side_model,
                artifacts=artifacts,
            )
            payload["events"] = relabeled_events
            payload["side_relabel"] = {
                "enabled": True,
                "model": side_model,
                "logs": relabel_logs,
            }
    return payload, indices, frames


def run_vlm_judge_on_indices(
    video_path: str,
    indices: Sequence[int],
    model: str,
    artifacts: Mapping[str, Any],
    *,
    phase: str,
    prompt_mode: str = "joint",
) -> Tuple[Dict[str, Any], List[int], List[np.ndarray]]:
    _log_causal(
        "judge_prepare_start",
        video_path=video_path,
        phase=phase,
        requested_frames=len(indices),
    )
    prep_start = perf_counter()
    meta = get_video_metadata(video_path)
    if meta.frame_count <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    _log_causal(
        "judge_frame_load_start",
        video_path=video_path,
        phase=phase,
        requested_keyframes=len(indices),
        frame_count=meta.frame_count,
    )
    frames = load_frames_by_indices(video_path, indices)
    if len(frames) != len(indices):
        indices = indices[: len(frames)]
    _log_causal(
        "judge_frame_load_done",
        video_path=video_path,
        phase=phase,
        loaded_keyframes=len(frames),
        elapsed_sec=round(perf_counter() - prep_start, 3),
    )
    system_prompt = resolve_judge_system_prompt(artifacts)
    if prompt_mode in {"detection_only", "detection_high_recall"}:
        user_text = build_detection_user_prompt(
            indices,
            artifacts,
            max_events=resolve_detection_max_events(
                artifacts,
                prompt_mode=prompt_mode,
            ),
            high_recall=(prompt_mode == "detection_high_recall"),
        )
    else:
        user_text = build_judge_user_prompt(indices, artifacts)
    raw_timeout = artifacts.get("lpips_vlm_timeout_sec")
    if raw_timeout is not None:
        timeout_sec = float(raw_timeout)
    else:
        n = len(frames)
        timeout_sec = max(120.0, min(3600.0, 5.0 * float(n) + 180.0))
    _log_causal(
        "judge_api_start",
        video_path=video_path,
        phase=phase,
        model=model,
        num_frames=len(frames),
        timeout_sec=round(timeout_sec, 3),
    )
    judge_start = perf_counter()
    payload = call_openai_vision_judge(
        frames,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        timeout_sec=timeout_sec,
    )
    _log_causal(
        "judge_api_done",
        video_path=video_path,
        phase=phase,
        model=model,
        num_frames=len(frames),
        elapsed_sec=round(perf_counter() - judge_start, 3),
    )
    return payload, list(indices), frames


def relabel_event_sides_on_video(
    video_path: str,
    events: Sequence[Dict[str, Any]],
    *,
    model: str,
    artifacts: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not events:
        return [], []

    meta = get_video_metadata(video_path)
    if meta.frame_count <= 0:
        return list(events), []

    window_sec = float(artifacts.get("lpips_vlm_side_window_sec", 0.75))
    max_frames = int(artifacts.get("lpips_vlm_side_max_frames", 12))
    radius_frames = max(1, int(round(window_sec * float(meta.fps))))
    relabeled: List[Dict[str, Any]] = []
    relabel_logs: List[Dict[str, Any]] = []

    for index, event in enumerate(sorted(events, key=lambda e: int(e.get("frame", 0)))):
        anchors = [int(event.get("frame", 0))]
        evidence = event.get("evidence_frame_indices")
        if isinstance(evidence, list):
            for value in evidence:
                try:
                    anchors.append(int(value))
                except (TypeError, ValueError):
                    continue
        lo = max(0, min(anchors) - radius_frames)
        hi = min(meta.frame_count - 1, max(anchors) + radius_frames)
        sample_count = min(max_frames, hi - lo + 1)
        indices = sorted(
            set(int(round(float(x))) for x in np.linspace(lo, hi, num=sample_count))
        )
        frames = load_frames_by_indices(video_path, indices)
        if len(frames) != len(indices):
            indices = indices[: len(frames)]
        if not indices:
            relabeled.append(dict(event))
            relabel_logs.append(
                {
                    "event_index": index,
                    "status": "skipped",
                    "reason": "no_local_frames_loaded",
                    "original_side": str(event.get("side", "unknown")).lower(),
                }
            )
            continue

        system_prompt = resolve_judge_system_prompt(artifacts)
        user_text = build_side_relabel_user_prompt(indices, event, artifacts)
        raw_timeout = artifacts.get("lpips_vlm_side_timeout_sec")
        if raw_timeout is not None:
            timeout_sec = float(raw_timeout)
        else:
            timeout_sec = max(90.0, min(600.0, 4.0 * float(len(indices)) + 90.0))
        _log_causal(
            "side_relabel_start",
            video_path=video_path,
            model=model,
            event_index=index,
            anchor_frame=int(event.get("frame", 0)),
            num_frames=len(indices),
        )
        started = perf_counter()
        try:
            payload = call_openai_vision_judge(
                frames,
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                timeout_sec=timeout_sec,
            )
            clean = sanitize_relabel_payload(payload, allowed=set(indices))
            updated = dict(event)
            updated["side"] = clean["side"]
            if clean["evidence_frame_indices"]:
                updated["evidence_frame_indices"] = clean["evidence_frame_indices"]
            if clean["evidence_summary"]:
                updated["evidence_summary"] = clean["evidence_summary"]
            if clean["confidence"] > 0.0:
                updated["confidence"] = clean["confidence"]
            relabeled.append(updated)
            relabel_logs.append(
                {
                    "event_index": index,
                    "status": "ok",
                    "original_side": str(event.get("side", "unknown")).lower(),
                    "relabeled_side": clean["side"],
                    "elapsed_sec": round(perf_counter() - started, 3),
                    "reasoning_steps": clean["reasoning_steps"],
                }
            )
            _log_causal(
                "side_relabel_done",
                video_path=video_path,
                model=model,
                event_index=index,
                relabeled_side=clean["side"],
                elapsed_sec=round(perf_counter() - started, 3),
            )
        except Exception as exc:
            relabeled.append(dict(event))
            relabel_logs.append(
                {
                    "event_index": index,
                    "status": "error",
                    "reason": str(exc),
                    "original_side": str(event.get("side", "unknown")).lower(),
                }
            )
            _log_causal(
                "side_relabel_error",
                video_path=video_path,
                model=model,
                event_index=index,
                error=str(exc),
            )

    return relabeled, relabel_logs


def resolve_event_offsets(artifacts: Mapping[str, Any]) -> List[int]:
    if "lpips_event_offsets" in artifacts and artifacts["lpips_event_offsets"] is not None:
        raw = artifacts["lpips_event_offsets"]
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            return [int(p) for p in parts]
        return [int(x) for x in raw]
    before = int(artifacts.get("lpips_event_frames_before", 2))
    after = int(artifacts.get("lpips_event_frames_after", 2))
    return build_default_event_offsets(before, after)


def analyze_causal_vlm_lpips(
    context: EvaluationContext,
    device_obj: Any,
    artifacts: Mapping[str, Any],
) -> Dict[str, Any]:
    model = str(artifacts.get("lpips_vlm_model", "gpt-5.4"))
    max_grid = int(artifacts.get("lpips_vlm_max_grid_frames", 24))
    prompt_version = str(artifacts.get("lpips_vlm_prompt_version", PROMPT_VERSION_DEFAULT))
    offsets = resolve_event_offsets(artifacts)
    match_mode = str(artifacts.get("causal_match_mode", "monotonic_side_time")).strip().lower()
    contact_types = resolve_contact_types(artifacts)
    directional_sides = resolve_directional_sides(artifacts)

    base: Dict[str, Any] = {
        "module": "lpips",
        "lpips_sampling": "causal_vlm_collision",
        "reference_video": context.reference_video,
        "candidate_video": context.candidate_video,
        "reference_metadata": context.reference.metadata.to_dict(),
        "candidate_metadata": context.candidate.metadata.to_dict(),
        "shared_duration_sec": context.shared_duration_sec,
        "eval_fps": context.eval_fps,
        "max_pairs": context.max_pairs,
        "alignment_mode": context.alignment_mode,
        "device": str(device_obj),
        "lpips_vlm_model": model,
        "lpips_vlm_prompt_version": prompt_version,
        "lpips_event_offsets": offsets,
        "causal_match_mode_requested": match_mode,
        "lpips_vlm_effective_contact_types": list(contact_types),
        "lpips_vlm_effective_directional_sides": sorted(directional_sides),
    }

    if context.num_pairs == 0:
        base["error"] = "no overlapping timestamps available"
        base.update(summarize_scores([]))
        return base

    event_source = str(
        artifacts.get("lpips_causal_event_source", "both")
    ).strip().lower()
    _log_causal(
        "analysis_start",
        model=model,
        event_source=event_source,
        max_grid=max_grid,
        offsets=offsets,
    )
    vgg16_weights_cached = os.path.exists(
        os.path.expanduser("~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
    )

    if event_source in ("reference", "unity") and not vgg16_weights_cached:
        temporal_baseline = {
            "module": "lpips",
            "lpips_sampling": "temporal",
            "lpips_per_pair": [],
            "timestamps_sec": [],
            "reference_frame_indices": [],
            "candidate_frame_indices": [],
            "num_pairs": 0,
            "skipped": True,
            "skip_reason": "lpips_vgg16_weights_not_cached",
        }
    else:
        try:
            temporal_context = EvaluationContext(
                reference_video=context.reference_video,
                candidate_video=context.candidate_video,
                reference=context.reference,
                candidate=context.candidate,
                shared_duration_sec=context.shared_duration_sec,
                eval_fps=context.eval_fps,
                max_pairs=context.max_pairs,
                alignment_mode=context.alignment_mode,
                text_prompt=context.text_prompt,
                artifacts={**(context.artifacts or {}), "lpips_sampling": "temporal"},
            )
            temporal_baseline = analyze_lpips_from_context(
                temporal_context, device=str(device_obj)
            )
        except Exception as exc:
            temporal_baseline = {
                "module": "lpips",
                "lpips_sampling": "temporal",
                "lpips_per_pair": [],
                "timestamps_sec": [],
                "reference_frame_indices": [],
                "candidate_frame_indices": [],
                "num_pairs": 0,
                "error": f"temporal_lpips_failed: {exc}",
            }

    def _finalize_inspection(result: Dict[str, Any]) -> Dict[str, Any]:
        if artifacts.get("save_inspection_artifacts"):
            try:
                from ..inspection_artifacts import write_causal_inspection

                write_causal_inspection(
                    artifacts=artifacts,
                    context=context,
                    result=result,
                )
            except Exception as exc:
                result["inspection_write_warning"] = str(exc)
        return result

    try:
        _log_causal("reference_judge_start", video_path=context.reference_video)
        ref_j_raw, ref_grid_idx, _ = run_vlm_judge_on_video(
            context.reference_video,
            max_grid,
            model,
            artifacts=artifacts,
        )
        if event_source in ("reference", "unity"):
            cand_j_raw = {
                "reasoning_steps": [],
                "events": [],
                "skipped": True,
                "skip_reason": "reference_only_event_source",
            }
            cand_grid_idx = []
        else:
            _log_causal("candidate_judge_start", video_path=context.candidate_video)
            cand_j_raw, cand_grid_idx, _ = run_vlm_judge_on_video(
                context.candidate_video,
                max_grid,
                model,
                artifacts=artifacts,
            )
    except Exception as exc:
        base["error"] = f"vlm_judge_failed: {exc}"
        _merge_temporal_primary_fields(base, temporal_baseline)
        base["first_divergence"] = compute_first_divergence(
            [], [], context.reference.metadata.frame_count, context.candidate.metadata.frame_count, temporal_baseline
        )
        return _finalize_inspection(base)

    allowed_ref = set(ref_grid_idx)
    allowed_cand = set(cand_grid_idx)
    _log_causal(
        "judge_outputs_ready",
        ref_frames=len(ref_grid_idx),
        cand_frames=len(cand_grid_idx),
    )

    ref_events_raw = ref_j_raw.get("events")
    cand_events_raw = cand_j_raw.get("events")
    if not isinstance(ref_events_raw, list):
        ref_events_raw = []
    if not isinstance(cand_events_raw, list):
        cand_events_raw = []

    ref_events_f, ref_dropped = sanitize_events_to_grid(
        [e for e in ref_events_raw if isinstance(e, dict)], allowed_ref
    )
    cand_events_f, cand_dropped = sanitize_events_to_grid(
        [e for e in cand_events_raw if isinstance(e, dict)], allowed_cand
    )
    ref_events_f = filter_events_by_contact_types(ref_events_f, contact_types)
    cand_events_f = filter_events_by_contact_types(cand_events_f, contact_types)
    if event_source in ("reference", "unity"):
        ref_meta = context.reference.metadata
        cand_meta = context.candidate.metadata
        mapped_candidate_events: List[Dict[str, Any]] = []
        for event in ref_events_f:
            ref_frame = int(event.get("frame", 0))
            ref_time_sec = float(ref_frame) / float(ref_meta.fps)
            cand_frame = int(round(ref_time_sec * float(cand_meta.fps)))
            cand_frame = min(max(cand_frame, 0), cand_meta.frame_count - 1)
            mapped = dict(event)
            mapped["frame"] = cand_frame
            mapped["mapped_from_reference"] = True
            mapped_candidate_events.append(mapped)
        cand_events_f = mapped_candidate_events

    _, _, _, strict_log = pair_collision_events_strict_with_log(
        ref_events_f, cand_events_f, directional_sides=directional_sides
    )

    monotonic_matched, monotonic_unmatched_ref, monotonic_unmatched_cand, monotonic_log = (
        pair_collision_events_monotonic_with_log(
            ref_events_f,
            cand_events_f,
            ref_frame_count=context.reference.metadata.frame_count,
            cand_frame_count=context.candidate.metadata.frame_count,
            directional_sides=directional_sides,
        )
    )

    matched = monotonic_matched
    event_pairing_log = monotonic_log
    causal_match_mode_used = "monotonic_side_time"
    unmatched_ref = monotonic_unmatched_ref
    unmatched_cand = monotonic_unmatched_cand

    if match_mode in ("order", "side_agnostic"):
        matched, event_pairing_log = pair_collision_events_order_with_log(
            ref_events_f, cand_events_f
        )
        causal_match_mode_used = match_mode
        ref_sorted = sorted(ref_events_f, key=lambda e: int(e.get("frame", 0)))
        cand_sorted = sorted(cand_events_f, key=lambda e: int(e.get("frame", 0)))
        n_pair = min(len(ref_sorted), len(cand_sorted))
        unmatched_ref = ref_sorted[n_pair:]
        unmatched_cand = cand_sorted[n_pair:]
    elif match_mode == "strict":
        matched, unmatched_ref, unmatched_cand, event_pairing_log = (
            pair_collision_events_strict_with_log(
                ref_events_f, cand_events_f, directional_sides=directional_sides
            )
        )
        causal_match_mode_used = "strict"
    else:
        # backward-compatible aliases
        if match_mode in ("strict_fallback_order", "auto"):
            matched, unmatched_ref, unmatched_cand, event_pairing_log = (
                pair_collision_events_monotonic_with_log(
                    ref_events_f,
                    cand_events_f,
                    ref_frame_count=context.reference.metadata.frame_count,
                    cand_frame_count=context.candidate.metadata.frame_count,
                    directional_sides=directional_sides,
                )
            )
            causal_match_mode_used = "monotonic_side_time"
        elif len(monotonic_matched) == 0 and ref_events_f and cand_events_f:
            matched, event_pairing_log = pair_collision_events_order_with_log(
                ref_events_f, cand_events_f
            )
            causal_match_mode_used = "order_fallback"
            ref_sorted = sorted(ref_events_f, key=lambda e: int(e.get("frame", 0)))
            cand_sorted = sorted(cand_events_f, key=lambda e: int(e.get("frame", 0)))
            n_pair = min(len(ref_sorted), len(cand_sorted))
            unmatched_ref = ref_sorted[n_pair:]
            unmatched_cand = cand_sorted[n_pair:]

    num_ref = len(ref_events_f)
    num_cand = len(cand_events_f)
    num_matched = len(matched)
    ref_meta = context.reference.metadata
    cand_meta = context.candidate.metadata

    base.update(
        {
            "judge_reference": {
                "reasoning_steps": ref_j_raw.get("reasoning_steps"),
                "events": ref_events_f,
                "grid_frame_indices": ref_grid_idx,
                "dropped_events": ref_dropped,
                "coarse_to_fine": ref_j_raw.get("coarse_to_fine"),
                "side_relabel": ref_j_raw.get("side_relabel"),
            },
            "judge_candidate": {
                "reasoning_steps": cand_j_raw.get("reasoning_steps"),
                "events": cand_events_f,
                "grid_frame_indices": cand_grid_idx,
                "dropped_events": cand_dropped,
                "coarse_to_fine": cand_j_raw.get("coarse_to_fine"),
                "side_relabel": cand_j_raw.get("side_relabel"),
                "skipped": bool(cand_j_raw.get("skipped")),
                "skip_reason": cand_j_raw.get("skip_reason"),
            },
            "reference_action_summary": build_action_summary(
                ref_events_f, ref_meta.frame_count
            ),
            "candidate_action_summary": build_action_summary(
                cand_events_f, cand_meta.frame_count
            ),
            "strict_pairing_log": strict_log,
            "event_pairing_log": event_pairing_log,
            "causal_match_mode_used": causal_match_mode_used,
            "num_reference_events": num_ref,
            "num_candidate_events": num_cand,
            "num_matched_events": num_matched,
            "num_unmatched_reference_events": len(unmatched_ref),
            "num_unmatched_candidate_events": len(unmatched_cand),
            "unmatched_reference_events": unmatched_ref,
            "unmatched_candidate_events": unmatched_cand,
            "matched_reference_event_fraction": (
                float(num_matched) / float(max(1, num_ref))
            ),
        }
    )

    base["first_divergence"] = compute_first_divergence(
        ref_events_f,
        cand_events_f,
        ref_meta.frame_count,
        cand_meta.frame_count,
        temporal_baseline,
    )
    base["temporal_baseline"] = temporal_baseline

    if event_source in ("reference", "unity") and not vgg16_weights_cached:
        base["causal_lpips_skipped_reason"] = "lpips_vgg16_weights_not_cached"
        base["lpips_per_matched_event"] = [
            {
                "match_index": mi,
                "side": ev_r.get("side"),
                "e_ref": int(ev_r["frame"]),
                "e_cand": int(ev_c["frame"]),
                "e_cand_contact_ref_frame": int(ev_r["frame"]),
                "offsets": offsets,
                "lpips_per_offset": [],
                "evaluation_interval": {
                    "mode": "reference_only_wall_clock",
                    "lpips_skipped": True,
                },
            }
            for mi, (ev_r, ev_c) in enumerate(matched)
        ]
        base["causal_lpips_per_pair"] = []
        _merge_temporal_primary_fields(base, temporal_baseline)
        return _finalize_inspection(base)

    if num_matched == 0:
        base["causal_lpips_skipped_reason"] = "no_matched_events"
        base["lpips_per_matched_event"] = []
        base["causal_lpips_per_pair"] = []
        _merge_temporal_primary_fields(base, temporal_baseline)
        return _finalize_inspection(base)

    per_event_rows: List[Dict[str, Any]] = []
    flat_scores: List[float] = []

    for mi, (ev_r, ev_c) in enumerate(matched):
        e_ref = int(ev_r["frame"])
        e_cand = int(ev_c["frame"])
        row: Dict[str, Any] = {
            "match_index": mi,
            "side": ev_r.get("side"),
            "e_ref": e_ref,
            "e_cand": e_cand,
            "offsets": offsets,
            "lpips_per_offset": [],
        }
        for delta in offsets:
            ri = min(max(e_ref + int(delta), 0), ref_meta.frame_count - 1)
            ci = min(max(e_cand + int(delta), 0), cand_meta.frame_count - 1)
            fr = load_frames_by_indices(context.reference_video, [ri])
            fc = load_frames_by_indices(context.candidate_video, [ci])
            if not fr or not fc:
                continue
            pair_scores = compute_lpips_scores([fr[0]], [fc[0]], device_obj)
            s = pair_scores[0] if pair_scores else None
            if s is not None:
                row["lpips_per_offset"].append(
                    {"offset": int(delta), "ref_frame": ri, "cand_frame": ci, "lpips": s}
                )
                flat_scores.append(s)
        per_event_rows.append(row)

    base["lpips_per_matched_event"] = per_event_rows
    base["causal_lpips_per_pair"] = flat_scores
    base["lpips_per_pair"] = flat_scores
    base["num_causal_lpips_scores"] = len(flat_scores)
    base["num_pairs"] = len(flat_scores)
    if flat_scores:
        base["reference_frame_indices"] = [
            r["ref_frame"] for ev in per_event_rows for r in ev["lpips_per_offset"]
        ]
        base["candidate_frame_indices"] = [
            r["cand_frame"] for ev in per_event_rows for r in ev["lpips_per_offset"]
        ]
    base.update(summarize_scores(flat_scores))
    base["lpips_primary_source"] = "causal"
    causal_summ = summarize_scores(flat_scores)
    base["causal_lpips_aggregates"] = causal_summ

    temp_summ = {
        k: temporal_baseline.get(k)
        for k in ("lpips_mean", "lpips_median", "lpips_std", "lpips_min", "lpips_max")
    }
    base["temporal_lpips_aggregates"] = temp_summ

    return _finalize_inspection(base)
