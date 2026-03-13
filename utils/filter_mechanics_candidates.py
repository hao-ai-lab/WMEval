#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# ---------------------------------------------------------------------
# Heuristic vocabularies
# ---------------------------------------------------------------------

# Mechanics concepts we want to keep.
MECHANICS_TERMS = {
    "gravity",
    "inertia",
    "friction",
    "momentum conservation",
    "elastic collision",
    "inelastic collision",
    "kinetic energy loss",
    "impulse",
    "centripetal force",
    "centrifugal force",
    "angular momentum",
    "gyroscopic precession",
    "rotational dynamics",
    "projectile motion",
    "newton's laws",
    "hooke’s law",
    "hooke's law",
    "simple harmonic motion",
    "pendulum",
    "impact mechanics",
    "stress-concentration",
    "stress concentration",
    "fracture",
    "deformation",
    "structural physics",
    "wave propagation",
    "buoyancy",
    "density and buoyancy",
    "pressure equilibrium",
    "hydrostatic",
    "fluid dynamics",
    "fluid flow",
    "bernoulli effect",
    "coanda effect",
    "viscosity",
    "non-newtonian fluid",
    "standing waves",
    "energy transfer",
    "energy storage & release",
    "energy storage and release",
    "energy conversion",
    "air pocket",
    "thermal expansion",
    "pressure",
}

# Domains that are often NOT object-tracking-first mechanics tasks.
# These are not hard excludes by themselves; they mostly hurt the score
# if there are no mechanics hits.
NON_MECHANICS_DOMINANT_TERMS = {
    "refraction",
    "reflection",
    "dispersion",
    "polarization",
    "diffraction",
    "interference",
    "fluorescence",
    "phosphorescence",
    "light spectrum",
    "chromatic aberration",
    "acid-base reaction",
    "acid base reaction",
    "precipitation reaction",
    "redox reaction",
    "electrolysis",
    "chemical reaction",
    "indicator color change",
    "plasma physics",
    "gas ionization",
}

# Concrete visible objects that GroundingDINO is likely to localize.
# Put longer phrases first to avoid weaker matches like "ball" before "billiard ball".
OBJECT_TERMS = [
    "prince rupert's drop",
    "ball chain",
    "newton's cradle",
    "double cone",
    "turntable",
    "billiard ball",
    "eyedropper",
    "tea bag",
    "wine glass",
    "tuning fork",
    "soap bubble",
    "water stream",
    "water jet",
    "metal ring",
    "metal blade",
    "razor blade",
    "pendulum",
    "slinky",
    "spaghetti",
    "egg",
    "coin",
    "card",
    "can",
    "bottle",
    "beaker",
    "glass",
    "cup",
    "chain",
    "ball",
    "marble",
    "top",
    "carts",
    "cart",
    "magnet",
    "block",
    "ramp",
    "rails",
    "cone",
    "bottle",
    "projectile",
    "candle",
    "nickel piece",
    "blade",
    "spoon",
    "bowl",
    "speaker",
    "tube",
    "apple",
    "pole",
    "rod",
    "disk",
    "penny",
    "bottles",
    "plate",
    "flask",
    "compass",
    "fork",
    "magnet",
    "balloon",
    "paper towel",
    "toilet paper",
    "lead musket ball",
    "copper wire",
    "battery",
    "copper pipe",
    "comb",
    "balloon",
    "violin bow",
    "violin string",
    "mirror",
    "tray",
    "water droplet",
    "drop",
]

# Motion / trackability cues: if these appear, tracking is more plausible.
MOTION_TERMS = {
    "move",
    "moving",
    "moves",
    "motion",
    "fall",
    "falls",
    "falling",
    "drop",
    "drops",
    "dropping",
    "rise",
    "rises",
    "rising",
    "sink",
    "sinks",
    "sinking",
    "float",
    "floats",
    "floating",
    "swing",
    "swings",
    "swinging",
    "roll",
    "rolls",
    "rolling",
    "spin",
    "spins",
    "spinning",
    "rotate",
    "rotates",
    "rotating",
    "bounce",
    "bounces",
    "bouncing",
    "jump",
    "jumps",
    "jumping",
    "slide",
    "slides",
    "sliding",
    "glide",
    "glides",
    "flow",
    "flows",
    "flowing",
    "crawl",
    "crawling",
    "lift",
    "lifts",
    "lifting",
    "pull",
    "pulls",
    "pulling",
    "propel",
    "propelled",
    "accelerate",
    "accelerates",
    "decelerate",
    "decelerates",
    "collide",
    "collides",
    "collision",
    "oscillate",
    "oscillates",
    "oscillation",
    "vibrate",
    "vibrates",
    "vibration",
    "crumple",
    "crumples",
    "collapse",
    "collapses",
    "shatter",
    "shatters",
}


# ---------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------
def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("‐", "-").replace("-", "-").replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if path == "-":
        for line in sys.stdin:
            line = line.strip()
            if line:
                yield json.loads(line)
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_id_value(row: Dict[str, Any]) -> str:
    for key in ("pid", "vid", "id"):
        if key in row:
            return str(row[key])
    raise KeyError(f"No pid/vid/id key found in row: {row}")


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize_text(phrase) in text


def collect_hits(text: str, phrases: Iterable[str]) -> List[str]:
    hits: List[str] = []
    seen = set()
    for phrase in phrases:
        p = normalize_text(phrase)
        if p and p not in seen and p in text:
            hits.append(phrase)
            seen.add(p)
    return hits


def looks_trackable(prompt_text: str, phenomenon_text: str) -> Tuple[List[str], List[str]]:
    merged = normalize_text(prompt_text + " " + phenomenon_text)
    object_hits = collect_hits(merged, OBJECT_TERMS)
    motion_hits = collect_hits(merged, MOTION_TERMS)
    return object_hits, motion_hits


def build_grounding_prompt(object_hits: List[str], max_terms: int = 5) -> str:
    """
    Return a GroundingDINO-friendly category prompt, joined with dots.
    """
    cleaned: List[str] = []
    seen = set()
    for x in object_hits:
        k = normalize_text(x)
        if k not in seen:
            cleaned.append(k)
            seen.add(k)
        if len(cleaned) >= max_terms:
            break
    return " . ".join(cleaned)


# ---------------------------------------------------------------------
# Core filtering logic
# ---------------------------------------------------------------------
def classify_row(
    row: Dict[str, Any],
    min_score: int,
    require_visual_object: bool,
    require_motion: bool,
) -> Dict[str, Any]:
    field = str(row.get("field", "")).strip()
    keywords = [str(x) for x in row.get("keywords", [])]
    prompt = str(row.get("prompt", ""))
    phenomenon = str(row.get("expected phenomenon", ""))

    merged_keywords = normalize_text(" ".join(keywords))
    merged_text = normalize_text(" ".join(keywords) + " " + prompt + " " + phenomenon)

    mechanics_hits = collect_hits(merged_text, MECHANICS_TERMS)
    non_mech_hits = collect_hits(merged_text, NON_MECHANICS_DOMINANT_TERMS)
    object_hits, motion_hits = looks_trackable(prompt, phenomenon)

    score = 0
    reasons: List[str] = []

    if field.lower() == "physics":
        score += 2
        reasons.append("field=Physics")

    if mechanics_hits:
        score += 3 + len(mechanics_hits)
        reasons.append(f"mechanics_hits={mechanics_hits}")

    if object_hits:
        score += 2
        reasons.append(f"object_hits={object_hits}")

    if motion_hits:
        score += 2
        reasons.append(f"motion_hits={motion_hits}")

    # Penalize optics/chemistry-dominant items if mechanics evidence is weak.
    if non_mech_hits and not mechanics_hits:
        score -= 4
        reasons.append(f"non_mechanics_dominant={non_mech_hits}")
    elif non_mech_hits:
        score -= 1
        reasons.append(f"mixed_domain={non_mech_hits}")

    keep = True
    if field.lower() != "physics":
        keep = False
        reasons.append("rejected_non_physics")

    if not mechanics_hits:
        keep = False
        reasons.append("rejected_no_mechanics_hit")

    if require_visual_object and not object_hits:
        keep = False
        reasons.append("rejected_no_visual_object_hit")

    if require_motion and not motion_hits:
        keep = False
        reasons.append("rejected_no_motion_hit")

    if score < min_score:
        keep = False
        reasons.append(f"rejected_score<{min_score}")

    return {
        "id": get_id_value(row),
        "keep": keep,
        "score": score,
        "field": field,
        "keywords": keywords,
        "mechanics_hits": mechanics_hits,
        "non_mechanics_hits": non_mech_hits,
        "object_hits": object_hits,
        "motion_hits": motion_hits,
        "groundingdino_prompt": build_grounding_prompt(object_hits),
        "row": row,
        "reasons": reasons,
    }


def sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
    id_str = str(item["id"])
    return (0, int(id_str)) if id_str.isdigit() else (1, id_str)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Filter a JSONL of physics/chemistry prompts and keep IDs that are "
            "mechanics-related and likely suitable for GroundingDINO + ByteTrack."
        )
    )
    ap.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to input JSONL, or '-' for stdin",
    )
    ap.add_argument(
        "--output_jsonl",
        default=None,
        help="Optional path to write kept rows as JSONL with metadata",
    )
    ap.add_argument(
        "--output_ids",
        default=None,
        help="Optional path to write kept IDs, one per line",
    )
    ap.add_argument(
        "--min_score",
        type=int,
        default=6,
        help="Minimum heuristic score to keep a row",
    )
    ap.add_argument(
        "--no_require_visual_object",
        action="store_true",
        help="Allow rows even if no obvious visible object term is found",
    )
    ap.add_argument(
        "--no_require_motion",
        action="store_true",
        help="Allow rows even if no motion/trackability term is found",
    )
    ap.add_argument(
        "--print_mode",
        choices=("ids", "json", "pretty"),
        default="pretty",
        help="stdout format",
    )
    args = ap.parse_args()

    require_visual_object = not args.no_require_visual_object
    require_motion = not args.no_require_motion

    kept: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for row in load_jsonl(args.input_jsonl):
        item = classify_row(
            row=row,
            min_score=args.min_score,
            require_visual_object=require_visual_object,
            require_motion=require_motion,
        )
        all_rows.append(item)
        if item["keep"]:
            kept.append(item)

    kept.sort(key=sort_key)

    # stdout
    if args.print_mode == "ids":
        for item in kept:
            print(item["id"])
    elif args.print_mode == "json":
        print(
            json.dumps(
                [
                    {
                        "id": x["id"],
                        "score": x["score"],
                        "mechanics_hits": x["mechanics_hits"],
                        "object_hits": x["object_hits"],
                        "motion_hits": x["motion_hits"],
                        "groundingdino_prompt": x["groundingdino_prompt"],
                    }
                    for x in kept
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"Kept {len(kept)} / {len(all_rows)} rows\n")
        for x in kept:
            print(
                f"[{x['id']}] score={x['score']}\n"
                f"  keywords={x['keywords']}\n"
                f"  mechanics_hits={x['mechanics_hits']}\n"
                f"  object_hits={x['object_hits']}\n"
                f"  motion_hits={x['motion_hits']}\n"
                f"  groundingdino_prompt={x['groundingdino_prompt'] or '<empty>'}\n"
            )

    # optional outputs
    if args.output_ids:
        out_ids = Path(args.output_ids)
        out_ids.parent.mkdir(parents=True, exist_ok=True)
        out_ids.write_text("".join(f"{x['id']}\n" for x in kept), encoding="utf-8")

    if args.output_jsonl:
        out_jsonl = Path(args.output_jsonl)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for x in kept:
                payload = dict(x["row"])
                payload["_filter_meta"] = {
                    "score": x["score"],
                    "mechanics_hits": x["mechanics_hits"],
                    "object_hits": x["object_hits"],
                    "motion_hits": x["motion_hits"],
                    "groundingdino_prompt": x["groundingdino_prompt"],
                    "reasons": x["reasons"],
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()