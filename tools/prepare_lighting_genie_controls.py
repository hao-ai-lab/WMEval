#!/usr/bin/env python3
"""Export the frozen hosted Genie key program; this does not drive a browser."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepare(program: Path, output: Path, case: Path) -> dict:
    raw = program.read_bytes()
    spec = json.loads(raw)
    if spec["freeze_id"] != "fine40-native10-v1":
        raise ValueError("unexpected frozen program")
    if spec["cycles"] != 40 or spec["canonical_duration_sec"] != 10:
        raise ValueError("expected forty steps over the ten-second target")
    segments = spec["segments"]
    if segments != [{"keys": "d+Left", "duration_ms": 114},
                    {"keys": "d", "duration_ms": 112}]:
        raise ValueError("key program differs from the executed frozen repeats")
    if sum(row["duration_ms"] for row in segments) >= spec["cycle_wall_ms"]:
        raise ValueError("cycle needs a released-key remainder")
    rows = []
    for cycle in range(spec["cycles"]):
        cursor = cycle * spec["cycle_wall_ms"]
        for row in segments:
            rows.append({"cycle": cycle, "wall_deadline_ms": cursor, **row})
            cursor += row["duration_ms"]
        rows.append({"cycle": cycle, "keys": "",
                     "wall_deadline_ms": cursor,
                     "duration_ms": (cycle + 1) * spec["cycle_wall_ms"] - cursor})
    packet = {"schema_version": 1, "program_sha256": hashlib.sha256(raw).hexdigest(),
              "status": "diagnostic_timing_not_qualified", "program": spec,
              "operations": rows,
              "execution": "Anchor all cycle deadlines to the first input timestamp; "
                           "subtract actual input-call overhead from the released remainder. "
                           "Never adjust controls using generated image quality."}
    output.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    source = case / "reference/ref.png"
    seed_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if seed_hash != "2128a19a4b8c63c39831ab1614e8dac05c5563beda4bde60624c8b2fbd57bdfa":
        raise ValueError("unexpected Unity reference image")
    image = Image.open(source).convert("RGB")
    if image.size != (1024, 1024):
        raise ValueError("expected the square Unity seed")
    image = image.resize((704, 704), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1280, 704))
    canvas.paste(image, (288, 0))
    seed_path = output / "seed_letterbox.png"
    canvas.save(seed_path)
    upload_hash = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    if upload_hash != "08a7914e26d488efcb8e68d44e7ed1525077a0ab4c3d34b6f2ad0248b1742b81":
        raise ValueError("letterboxed image differs from the executed upload")
    packet["conditioning"] = {"source_sha256": seed_hash,
                              "upload_sha256": upload_hash,
                              "file": "seed_letterbox.png",
                              "transform": "uniform 704/1024 Lanczos; 288px black side bars"}
    (output / "controls.json").write_text(json.dumps(packet, indent=2) + "\n")
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=Path(__file__).resolve().parents[1] /
                        "experiments/lighting_orbit_v2/genie_controls_frozen.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=Path, default=Path(__file__).resolve().parents[1] /
                        "eval_sets/wmeval_lite_v1/cases/SunRisesAndSets/lighting_orbit_v2")
    args = parser.parse_args()
    packet = prepare(args.program, args.output, args.case)
    print(json.dumps({"status": packet["status"], "operations": len(packet["operations"]),
                      "program_sha256": packet["program_sha256"]}))


if __name__ == "__main__":
    main()
