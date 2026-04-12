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


def resolve_vlm_sampling_strategy(artifacts: Mapping[str, Any]) -> str:
    raw = str(artifacts.get("lpips_vlm_sampling_strategy", "single_grid")).strip().lower()
    if raw in {"single_grid", "coarse_to_fine"}:
        return raw
    return "single_grid"


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
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for bgr in frames_bgr:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": bgr_to_data_url_jpeg(bgr)},
            }
        )

    # GPT-5.x: temperature is only valid with reasoning_effort "none" (OpenAI API).
    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    }
    if str(model).startswith("gpt-5"):
        create_kwargs["reasoning_effort"] = "none"
        create_kwargs["temperature"] = 0
    else:
        create_kwargs["temperature"] = 0

    resp = client.chat.completions.create(**create_kwargs)
    text = (resp.choices[0].message.content or "").strip()
    return extract_json_object(text)


def run_vlm_judge_on_video(
    video_path: str,
    max_grid_frames: int,
    model: str,
    artifacts: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[int], List[np.ndarray]]:
    strategy = resolve_vlm_sampling_strategy(artifacts)
    contact_types = resolve_contact_types(artifacts)
    coarse_model = str(artifacts.get("lpips_vlm_coarse_model", model)).strip() or model
    if strategy == "coarse_to_fine":
        meta = get_video_metadata(video_path)
        if meta.frame_count <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")
        coarse_grid_frames = int(
            artifacts.get("lpips_vlm_coarse_grid_frames", min(max_grid_frames, 24) if max_grid_frames > 0 else 24)
        )
        coarse_indices = build_vlm_keyframe_indices(meta.frame_count, coarse_grid_frames)
        coarse_payload, coarse_indices, _ = run_vlm_judge_on_indices(
            video_path,
            coarse_indices,
            coarse_model,
            artifacts,
            phase="coarse",
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
        )
        fine_payload["coarse_to_fine"] = {
            "strategy": strategy,
            "coarse_model": coarse_model,
            "fine_model": model,
            "coarse_grid_indices": coarse_indices,
            "coarse_events": coarse_events,
            "coarse_dropped_events": coarse_dropped,
            "refine_indices": fine_indices,
            "used_fine_pass": True,
        }
        return fine_payload, fine_indices, fine_frames

    indices = build_vlm_keyframe_indices(
        get_video_metadata(video_path).frame_count,
        max_grid_frames,
    )
    return run_vlm_judge_on_indices(
        video_path,
        indices,
        model,
        artifacts,
        phase="single_grid",
    )


def run_vlm_judge_on_indices(
    video_path: str,
    indices: Sequence[int],
    model: str,
    artifacts: Mapping[str, Any],
    *,
    phase: str,
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
    match_mode = str(artifacts.get("causal_match_mode", "strict_fallback_order")).strip().lower()
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

    strict_matched, strict_unmatched_ref, strict_unmatched_cand, strict_log = (
        pair_collision_events_strict_with_log(
            ref_events_f, cand_events_f, directional_sides=directional_sides
        )
    )

    matched = strict_matched
    event_pairing_log = strict_log
    causal_match_mode_used = "strict"
    unmatched_ref = strict_unmatched_ref
    unmatched_cand = strict_unmatched_cand

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
        pass
    else:
        # strict_fallback_order, auto, unknown → strict then order if no pairs
        if len(strict_matched) == 0 and ref_events_f and cand_events_f:
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
            },
            "judge_candidate": {
                "reasoning_steps": cand_j_raw.get("reasoning_steps"),
                "events": cand_events_f,
                "grid_frame_indices": cand_grid_idx,
                "dropped_events": cand_dropped,
                "coarse_to_fine": cand_j_raw.get("coarse_to_fine"),
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
