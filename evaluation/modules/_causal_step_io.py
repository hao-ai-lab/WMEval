"""I/O helpers for the causal_step module: load segments, load/write answers,
and bootstrap the vendored CausalStep directory (WMEval/imports/CausalStep) onto sys.path."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


def ensure_causalstep_importable(causalstep_root: str | None) -> None:
    """Add the vendored CausalStep directory to sys.path so `import causalstep` works.

    If `causalstep_root` is None, default to `<WMEval>/imports/CausalStep`.
    """
    if causalstep_root:
        root = Path(causalstep_root).resolve()
    else:
        # This file is WMEval/evaluation/modules/_causal_step_io.py
        # parents[0]=modules, [1]=evaluation, [2]=WMEval
        root = Path(__file__).resolve().parents[2] / "imports" / "CausalStep"
    if not (root / "causalstep" / "__init__.py").is_file():
        raise RuntimeError(
            f"CausalStep package not found at {root}. "
            "Set artifact 'causalstep_root' to its absolute path."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_segments(path: str) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    required = {"trajectory_id", "source", "events"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"segments.json missing keys: {sorted(missing)}")
    for ev in data["events"]:
        for k in ("index", "name", "start_sec", "end_sec", "trigger"):
            if k not in ev:
                raise ValueError(f"segment event missing key {k}: {ev}")
    return data


def write_answers(
    path: str,
    answers: Mapping[Tuple[str, int, str], str],
    *,
    video_id: str,
) -> None:
    """Write answers in CausalStep's PrecomputedModel-loadable format."""
    entries = []
    for (vid, seg_idx, qa_type), letter in answers.items():
        entries.append(
            {
                "video_id": vid,
                "segment_index": int(seg_idx),
                "qa_type": qa_type,
                "answer": letter,
            }
        )
    payload = {"video_id": video_id, "answers": entries}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_answers_as_map(path: str) -> Dict[Tuple[str, int, str], str]:
    with open(path) as f:
        data = json.load(f)
    out: Dict[Tuple[str, int, str], str] = {}
    for entry in data["answers"]:
        out[(entry["video_id"], int(entry["segment_index"]), entry["qa_type"])] = entry["answer"]
    return out


def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
