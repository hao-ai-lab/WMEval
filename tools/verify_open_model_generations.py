#!/usr/bin/env python3
"""Fail closed on missing, mismatched, or duplicate open-model generations."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("lingbot_world_cam", "matrix_game_2", "matrix_game_3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(model: str, repeat_count: int, selected: set[str]) -> dict[str, Any]:
    queue = ROOT / f"runs/tables_4_7/model_inputs/{model}/generation_queue.jsonl"
    rows = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines() if line]
    rows = [
        row for row in rows
        if (not selected or row["case_id"] in selected)
        and int(str(row["generation_repeat"]).removeprefix("run")) < repeat_count
    ]
    errors: list[str] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        case_id = str(row["case_id"])
        repeat = str(row["generation_repeat"])
        counts[case_id] += 1
        candidate = ROOT / str(row["candidate_video"])
        receipt_path = candidate.parent / "generation_receipt.json"
        if not candidate.is_file() or not receipt_path.is_file():
            errors.append(f"{case_id}/{repeat}: missing candidate or receipt")
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        digest = sha256(candidate)
        expected = {
            "status": "complete",
            "model": model,
            "case_id": case_id,
            "generation_repeat": repeat,
            "candidate_video_sha256": digest,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                errors.append(f"{case_id}/{repeat}: receipt {key} mismatch")
        hashes[case_id].append(digest)
    expected_cases = selected or set(counts)
    for case_id in sorted(expected_cases):
        if counts.get(case_id) != repeat_count:
            errors.append(f"{case_id}: expected {repeat_count} queued repeats, got {counts.get(case_id, 0)}")
        if len(hashes.get(case_id, [])) == repeat_count and len(set(hashes[case_id])) != repeat_count:
            errors.append(f"{case_id}: candidate hashes are not all distinct")
    return {
        "schema_version": 1,
        "status": "complete" if not errors else "failed",
        "model": model,
        "repeat_count": repeat_count,
        "case_count": len(expected_cases),
        "generation_count": len(rows),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    report = verify(args.model, args.repeat_count, set(args.case))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
