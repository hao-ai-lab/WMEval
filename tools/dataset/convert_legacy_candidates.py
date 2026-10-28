#!/usr/bin/env python3
"""Convert pre-suite candidate JSONL rows to the unified strict format.

Old rows (PR #16 style) named a dimension and a loose test-case id with many
aliases::

    {"dimension":"causal_lpips","test_case_id":"mirror_ball_third_person",
     "world_model_name":"genie","run_id":"run_01","world_model_video":"/path.mp4"}

New rows are dimension-free (the suite fans a case out to every dimension that
references it) and use the strict field set::

    {"model":"genie","case_id":"mirror_ball/third_person","seed":"run_01",
     "candidate_video":"/path.mp4"}

Case ids are resolved via suite.yaml's legacy_case_aliases. Rows that only
differ by dimension collapse into one output row (deduplicated).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_jsonl", help="Old-format candidate JSONL")
    parser.add_argument("output_jsonl", help="Where to write the strict JSONL")
    parser.add_argument("--suite", default="lite")
    args = parser.parse_args(argv)

    from evaluation.lite.suite_loader import load_suite

    suite = load_suite(args.suite)

    # legacy id -> case_id across all dimensions (aliases + current ids).
    alias_map: dict[str, str] = {}
    for dimension_id, aliases in suite.legacy_case_aliases.items():
        for legacy_id, case_id in aliases.items():
            existing = alias_map.get(legacy_id)
            if existing is not None and existing != case_id:
                raise SystemExit(
                    f"legacy id {legacy_id!r} maps to both {existing!r} and "
                    f"{case_id!r}; convert by hand"
                )
            alias_map[legacy_id] = case_id
    for case_id in suite.all_case_ids():
        alias_map.setdefault(case_id, case_id)

    seen: set[tuple[str, str, str]] = set()
    converted: list[dict] = []
    source = Path(args.legacy_jsonl)
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        legacy_case = str(
            row.get("test_case_id")
            or row.get("scene_id")
            or row.get("sample_id")
            or row.get("id")
            or ""
        ).strip()
        case_id = alias_map.get(legacy_case)
        if case_id is None:
            raise SystemExit(
                f"{source}: line {line_number}: unknown test case {legacy_case!r}"
            )
        video = str(
            row.get("world_model_video")
            or row.get("candidate_video")
            or row.get("genie3_video")
            or row.get("pred_video")
            or ""
        ).strip()
        if not video:
            raise SystemExit(f"{source}: line {line_number}: no candidate video")
        model = str(
            row.get("world_model_name") or row.get("model_name") or row.get("model") or "model"
        ).strip() or "model"
        seed = str(row.get("seed") or row.get("run_id") or "0001").strip() or "0001"

        key = (model, case_id, seed)
        if key in seen:
            continue  # rows differing only by dimension collapse
        seen.add(key)
        converted.append(
            {
                "model": model,
                "case_id": case_id,
                "seed": seed,
                "candidate_video": video,
            }
        )

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted),
        encoding="utf-8",
    )
    print(f"wrote {len(converted)} row(s) to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
