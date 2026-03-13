#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# -----------------------------
# API endpoints
# -----------------------------
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"

# -----------------------------
# Judge schema (OpenAI: strict json_schema supported in text.format)
# Anthropic: use a simplified schema (avoid min/max constraints)
# -----------------------------
JUDGE_SCHEMA_OPENAI: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "keep": {"type": "boolean"},
        "label": {
            "type": "string",
            "enum": [
                "explicit_mechanical_motion",
                "not_explicit_mechanical_motion",
            ],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "groundingdino_identifiable": {"type": "boolean"},
        "bytetrack_trackable": {"type": "boolean"},
        "visible_objects": {"type": "array", "items": {"type": "string"}},
        "motion_targets": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": [
        "keep",
        "label",
        "confidence",
        "groundingdino_identifiable",
        "bytetrack_trackable",
        "visible_objects",
        "motion_targets",
        "rationale",
    ],
}

# Safer for Anthropic structured outputs: omit min/max (some schema features are limited).
JUDGE_SCHEMA_ANTHROPIC: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "keep": {"type": "boolean"},
        "label": {
            "type": "string",
            "enum": [
                "explicit_mechanical_motion",
                "not_explicit_mechanical_motion",
            ],
        },
        "confidence": {"type": "number"},
        "groundingdino_identifiable": {"type": "boolean"},
        "bytetrack_trackable": {"type": "boolean"},
        "visible_objects": {"type": "array", "items": {"type": "string"}},
        "motion_targets": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": [
        "keep",
        "label",
        "confidence",
        "groundingdino_identifiable",
        "bytetrack_trackable",
        "visible_objects",
        "motion_targets",
        "rationale",
    ],
}

SYSTEM_PROMPT = """
You are a strict dataset filter for physical-phenomenon video evaluation.

Decide whether a candidate should be kept for evaluation of EXPLICIT MECHANICAL MOTION.

Keep ONLY if ALL are true:
1) The phenomenon is primarily judged through visible mechanical motion/deformation of physical objects over time.
2) At least one concrete visible object or object-part is plausible for a detector like GroundingDINO.
3) There is temporal motion/deformation a tracker like ByteTrack could plausibly follow across frames.
4) The motion is central to the phenomenon, not incidental.

Usually KEEP:
- falling, rolling, bouncing, swinging, spinning, colliding
- oscillation, pendulum motion, projectile motion
- chain fountain, slinky drop, Newton’s cradle
- visible deformation, fracture, collapse, crumpling
- buoyancy-driven rise/sink if the moving object is explicit and trackable
- obvious object/fluid motion when the moving structure is visible and central

Usually REJECT:
- pure color change/glow/fluorescence/phosphorescence
- chemistry-only reactions without explicit trackable motion
- pure optics: refraction/diffraction/interference/polarization/reflection
- heating/cooling where the main signal is temperature/fog/appearance rather than trackable motion
- static before/after state changes without meaningful temporal motion
- cases with no concrete localizable object or motion target too diffuse/abstract

Be conservative: prefer false negatives over false positives.
If groundingdino_identifiable is false OR bytetrack_trackable is false, keep MUST be false.

Return JSON only that matches the provided schema.
""".strip()


# -----------------------------
# JSONL IO helpers
# -----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path == "-":
        for line in sys.stdin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_ids(path: str, ids: List[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{x}\n" for x in ids), encoding="utf-8")


def get_row_id(row: Dict[str, Any]) -> str:
    for k in ("pid", "vid", "id"):
        if k in row:
            return str(row[k])
    raise KeyError(f"Row missing pid/vid/id: {row}")


def maybe_load_ids(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    s: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                s.add(x)
    return s


def select_rows(
    rows: List[Dict[str, Any]],
    only_ids: Optional[Set[str]],
    offset: int,
    limit: Optional[int],
    sample_size: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if only_ids is not None:
        rows = [r for r in rows if get_row_id(r) in only_ids]

    if offset:
        rows = rows[offset:]

    if limit is not None:
        rows = rows[:limit]

    if sample_size is not None and sample_size < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_size)

    return rows


# -----------------------------
# Prompt construction
# -----------------------------
def build_user_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": get_row_id(row),
        "field": row.get("field", ""),
        "keywords": row.get("keywords", []),
        "prompt": row.get("prompt", ""),
        "expected_phenomenon": row.get("expected phenomenon", ""),
    }


def build_user_prompt(row: Dict[str, Any]) -> str:
    return json.dumps(build_user_payload(row), ensure_ascii=False, indent=2)


# -----------------------------
# Parsing / validation helpers
# -----------------------------
def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not find JSON object in model output:\n{text}")
    return json.loads(m.group(0))


def normalize_judgment(obj: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "keep",
        "label",
        "confidence",
        "groundingdino_identifiable",
        "bytetrack_trackable",
        "visible_objects",
        "motion_targets",
        "rationale",
    ]
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"Model output missing required keys {missing}. Output: {obj}")

    out = {
        "keep": bool(obj["keep"]),
        "label": str(obj["label"]),
        "confidence": float(obj["confidence"]),
        "groundingdino_identifiable": bool(obj["groundingdino_identifiable"]),
        "bytetrack_trackable": bool(obj["bytetrack_trackable"]),
        "visible_objects": [str(x) for x in (obj.get("visible_objects") or [])],
        "motion_targets": [str(x) for x in (obj.get("motion_targets") or [])],
        "rationale": str(obj["rationale"]).strip(),
    }

    # Enforce consistency: if any hard requirement fails, keep must be false.
    if (
        not out["groundingdino_identifiable"]
        or not out["bytetrack_trackable"]
        or out["label"] != "explicit_mechanical_motion"
    ):
        out["keep"] = False

    # Clamp confidence into [0,1] defensively
    if out["confidence"] < 0.0:
        out["confidence"] = 0.0
    if out["confidence"] > 1.0:
        out["confidence"] = 1.0

    return out


# -----------------------------
# OpenAI Responses API
# Structured Outputs via text.format json_schema strict
# -----------------------------
def extract_openai_output_text(resp_json: Dict[str, Any]) -> str:
    # Preferred: output_text convenience field (docs show this pattern)
    ot = resp_json.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot

    # Fallback: parse output items
    out_items = resp_json.get("output", [])
    texts: List[str] = []
    for item in out_items:
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    texts.append(c["text"])
    if texts:
        return "\n".join(texts)

    raise ValueError(f"Could not extract OpenAI text from response: {resp_json}")


def call_openai(
    api_key: str,
    model: str,
    row: Dict[str, Any],
    timeout: int,
    max_retries: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "mechanics_motion_judgment",
                "strict": True,
                "schema": JUDGE_SCHEMA_OPENAI,
            }
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                OPENAI_RESPONSES_URL,
                headers=headers,
                json=body,
                timeout=timeout,
            )
            r.raise_for_status()
            resp_json = r.json()

            text = extract_openai_output_text(resp_json)
            parsed = normalize_judgment(extract_json_object(text))
            return parsed, resp_json

        except Exception as e:
            last_err = e
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)
            else:
                break

    raise RuntimeError(f"OpenAI request failed after {max_retries} attempts: {last_err}")


# -----------------------------
# Anthropic Messages API
# Structured Outputs via output_config.format json_schema
# Claude returns valid JSON in response.content[0].text
# -----------------------------
def extract_anthropic_text(resp_json: Dict[str, Any]) -> str:
    parts = resp_json.get("content", [])
    if not parts:
        raise ValueError(f"Anthropic response has no content: {resp_json}")

    # Docs: valid JSON returned in response.content[0].text
    # content items typically look like: {"type":"text","text":"..."}
    first = parts[0]
    if first.get("type") == "text" and isinstance(first.get("text"), str):
        return first["text"]

    # Fallback: concatenate any text parts
    texts: List[str] = []
    for p in parts:
        if p.get("type") == "text" and p.get("text"):
            texts.append(p["text"])
    if texts:
        return "\n".join(texts)

    raise ValueError(f"Could not extract text from Anthropic response: {resp_json}")


def call_anthropic(
    api_key: str,
    model: str,
    row: Dict[str, Any],
    timeout: int,
    max_retries: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body = {
        "model": model,
        "max_tokens": 800,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": build_user_prompt(row)},
        ],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": JUDGE_SCHEMA_ANTHROPIC,
            }
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                ANTHROPIC_MESSAGES_URL,
                headers=headers,
                json=body,
                timeout=timeout,
            )
            r.raise_for_status()
            resp_json = r.json()

            text = extract_anthropic_text(resp_json)
            parsed = normalize_judgment(extract_json_object(text))
            return parsed, resp_json

        except Exception as e:
            last_err = e
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)
            else:
                break

    raise RuntimeError(f"Anthropic request failed after {max_retries} attempts: {last_err}")


def judge_row(
    provider: str,
    model: str,
    row: Dict[str, Any],
    timeout: int,
    max_retries: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return call_openai(api_key, model, row, timeout, max_retries)

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return call_anthropic(api_key, model, row, timeout, max_retries)

    raise ValueError(f"Unsupported provider: {provider}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "LLM-judge a subset of mechanics_candidates.jsonl and keep only items "
            "with explicit mechanical motion suitable for GroundingDINO + ByteTrack evaluation."
        )
    )
    ap.add_argument("--input_jsonl", required=True, help="mechanics_candidates.jsonl (or - for stdin)")
    ap.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    ap.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults: openai=gpt-5.2, anthropic=claude-3-7-sonnet-latest",
    )

    ap.add_argument("--output_jsonl", default="mechanics_candidates_judged.jsonl", help="Kept rows only")
    ap.add_argument("--audit_jsonl", default=None, help="All judged rows (kept + rejected)")
    ap.add_argument("--output_ids", default=None, help="Kept ids, one per line")

    ap.add_argument("--only_ids_file", default=None, help="Optional file with ids, one per line")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--min_confidence", type=float, default=0.6)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument(
        "--keep_raw_provider_response",
        action="store_true",
        help="Store raw provider response in _judge.raw_provider_response (can be large)",
    )

    args = ap.parse_args()

    if args.model is None:
        args.model = "gpt-5.2" if args.provider == "openai" else "claude-3-7-sonnet-latest"

    rows = load_jsonl(args.input_jsonl)
    selected = select_rows(
        rows=rows,
        only_ids=maybe_load_ids(args.only_ids_file),
        offset=args.offset,
        limit=args.limit,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    kept_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(selected, start=1):
        row_id = get_row_id(row)
        print(f"[{i}/{len(selected)}] judging id={row_id}", file=sys.stderr)

        judgment, raw_resp = judge_row(
            provider=args.provider,
            model=args.model,
            row=row,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )

        keep_after_threshold = (
            judgment["keep"]
            and judgment["confidence"] >= args.min_confidence
            and judgment["groundingdino_identifiable"]
            and judgment["bytetrack_trackable"]
            and judgment["label"] == "explicit_mechanical_motion"
        )

        out_row = dict(row)
        out_row["_judge"] = {
            "provider": args.provider,
            "model": args.model,
            "keep_after_threshold": keep_after_threshold,
            **judgment,
        }
        if args.keep_raw_provider_response:
            out_row["_judge"]["raw_provider_response"] = raw_resp

        audit_rows.append(out_row)
        if keep_after_threshold:
            kept_rows.append(out_row)

    write_jsonl(args.output_jsonl, kept_rows)

    if args.audit_jsonl:
        write_jsonl(args.audit_jsonl, audit_rows)

    if args.output_ids:
        ids = [get_row_id(r) for r in kept_rows]
        write_ids(args.output_ids, ids)

    print(
        json.dumps(
            {
                "provider": args.provider,
                "model": args.model,
                "input_count": len(rows),
                "judged_count": len(selected),
                "kept_count": len(kept_rows),
                "output_jsonl": args.output_jsonl,
                "audit_jsonl": args.audit_jsonl,
                "output_ids": args.output_ids,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()