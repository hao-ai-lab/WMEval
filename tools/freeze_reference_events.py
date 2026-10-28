#!/usr/bin/env python3
"""Judge a case's Unity reference once and freeze its contact events.

Causal LPIPS pairs contact events found in the reference with the candidate's.
The hosted judge is non-deterministic -- the same reference can yield 0, 1 or 3
events across runs -- so re-judging per candidate makes the reference a moving
target. This tool judges the reference N times with exactly the settings the
runtime uses, keeps every pass in ``reference/artifacts/contact_events.json``,
and selects the events that a majority of passes agree on. Runtime then reuses
that selection for every candidate (``lpips_reference_events_json``).

Requires OPENAI_API_KEY (explicit opt-in to hosted-judge calls)::

    python tools/freeze_reference_events.py MirrorRollingBall/lpips_collision --passes 5
    python tools/build_lite_case.py lite/cases/MirrorRollingBall/lpips_collision
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUITE_ROOT = ROOT / "lite"
EVENTS_FILENAME = "contact_events.json"
CASE_PARAM = "lpips_reference_events_json"
DEFAULT_CLUSTER_GAP_FRAMES = 12


def select_consensus(
    passes: Sequence[Mapping[str, Any]],
    *,
    min_support: int,
    cluster_gap_frames: int = DEFAULT_CLUSTER_GAP_FRAMES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Events supported by at least ``min_support`` distinct passes.

    Events are clustered by frame proximity and side; a cluster is kept when
    events from at least ``min_support`` different passes fall in it, and is
    represented by its highest-confidence member (raw, unmodified). Returns
    ``(selected_events, supports)`` where ``supports`` lists every cluster with
    its support, kept or not, so a reviewer can see what was dropped.
    """
    tagged: list[tuple[int, dict[str, Any]]] = []
    for index, one in enumerate(passes):
        for event in one["payload"].get("events") or []:
            if isinstance(event, dict) and isinstance(event.get("frame"), (int, float)):
                tagged.append((index, event))
    tagged.sort(key=lambda item: (str(item[1].get("side")), float(item[1]["frame"])))

    clusters: list[list[tuple[int, dict[str, Any]]]] = []
    for item in tagged:
        last = clusters[-1][-1][1] if clusters else None
        if (
            last is not None
            and str(last.get("side")) == str(item[1].get("side"))
            and float(item[1]["frame"]) - float(last["frame"]) <= cluster_gap_frames
        ):
            clusters[-1].append(item)
        else:
            clusters.append([item])

    selected: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []
    for cluster in clusters:
        support = len({pass_index for pass_index, _ in cluster})
        representative = max(cluster, key=lambda item: float(item[1].get("confidence") or 0.0))[1]
        kept = support >= min_support
        supports.append(
            {
                "frame": int(representative["frame"]),
                "side": representative.get("side"),
                "support": support,
                "kept": kept,
            }
        )
        if kept:
            selected.append(representative)
    selected.sort(key=lambda event: float(event["frame"]))
    supports.sort(key=lambda entry: entry["frame"])
    return selected, supports


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def judge_artifacts(case: Any, dimension: Any) -> dict[str, Any]:
    """The artifacts dict the runtime would build for this case's lpips run."""
    from evaluation.lite.adapters.registry_adapter import merge_metric_config
    from evaluation.runner import _merge_lpips_artifacts, run_world_model_evaluation

    config = dict(merge_metric_config(dimension, case, SimpleNamespace(params={})))
    config.pop(CASE_PARAM, None)  # always judge live here
    run_params = inspect.signature(run_world_model_evaluation).parameters
    kwargs: dict[str, Any] = {}
    for name, parameter in inspect.signature(_merge_lpips_artifacts).parameters.items():
        if name == "context":
            continue
        if name in config:
            kwargs[name] = config[name]
        elif name in run_params and run_params[name].default is not inspect.Parameter.empty:
            kwargs[name] = run_params[name].default
        elif parameter.default is inspect.Parameter.empty:
            raise SystemExit(f"cannot resolve judge setting {name!r}")
    shim = SimpleNamespace(artifacts={})
    _merge_lpips_artifacts(shim, **kwargs)
    shim.artifacts.pop(CASE_PARAM, None)
    return shim.artifacts


def freeze(
    case_id: str, *, passes: int, min_support: int | None, allow_empty: bool, workers: int = 5
) -> Path:
    from evaluation.lite.case_loader import load_case
    from evaluation.lite.suite_loader import load_suite

    suite = load_suite(SUITE_ROOT)
    case = load_case(suite.cases_root / case_id, allowed_cases=suite.all_case_ids())
    return freeze_case(
        case, suite.dimensions["lpips"], passes=passes, min_support=min_support,
        allow_empty=allow_empty, workers=workers,
    )


def freeze_case(
    case: Any,
    dimension: Any,
    *,
    passes: int,
    min_support: int | None,
    allow_empty: bool,
    workers: int = 5,
    output: Path | None = None,
) -> Path:
    """Judge ``case.reference_video`` ``passes`` times; write the consensus to
    ``output`` (default: the case's ``reference/artifacts/contact_events.json``)."""
    from evaluation.modules.lpips_causal_vlm import detect_event_candidates

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required: freezing runs the hosted judge")
    case_id = case.case_id
    artifacts = judge_artifacts(case, dimension)
    reference = Path(case.reference_video)
    needed = min_support if min_support is not None else passes // 2 + 1

    print(
        f"[freeze] {case_id}: {passes} passes, model={artifacts['lpips_vlm_model']}, "
        f"coarse={artifacts.get('lpips_vlm_coarse_model')}, min_support={needed}",
        flush=True,
    )
    def one_pass(number: int) -> dict[str, Any]:
        started = perf_counter()
        payload, indices, _frames = detect_event_candidates(
            str(reference),
            max_grid_frames=int(artifacts["lpips_vlm_max_grid_frames"]),
            model=str(artifacts["lpips_vlm_model"]),
            artifacts=artifacts,
        )
        events = payload.get("events") or []
        print(f"[freeze]   pass {number}: {len(events)} event(s) in {perf_counter() - started:.0f}s", flush=True)
        return {
            "pass": number,
            "num_events": len(events),
            "grid_frame_indices": [int(i) for i in indices],
            "payload": json.loads(json.dumps(payload, default=_jsonable)),
        }

    # Passes are independent judge calls; run them concurrently.
    with ThreadPoolExecutor(max_workers=max(1, min(workers, passes))) as pool:
        recorded = list(pool.map(one_pass, range(1, passes + 1)))

    selected, supports = select_consensus(recorded, min_support=needed)
    if not selected and not allow_empty:
        raise SystemExit(
            f"no event reached {needed}/{passes} support; per-pass counts "
            f"{[p['num_events'] for p in recorded]}. Not writing an empty reference "
            "(re-run with more passes, a lower --min-support, or --allow-empty)."
        )
    source = max(recorded, key=lambda p: sum(1 for e in p["payload"].get("events") or [] if e in selected))
    selected_payload = {
        key: value for key, value in source["payload"].items() if key != "events"
    }
    selected_payload["events"] = selected

    document = {
        "schema_version": 1,
        "case_id": case_id,
        "reference_video_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": {
            "provider": "openai",
            **({"transport": os.environ["WMEVAL_JUDGE_TRANSPORT"]} if os.environ.get("WMEVAL_JUDGE_TRANSPORT") else {}),
            "model": artifacts["lpips_vlm_model"],
            "coarse_model": artifacts.get("lpips_vlm_coarse_model"),
            "side_model": artifacts.get("lpips_vlm_side_model"),
            "prompt_version": artifacts["lpips_vlm_prompt_version"],
            "sampling_strategy": artifacts["lpips_vlm_sampling_strategy"],
            "max_grid_frames": artifacts["lpips_vlm_max_grid_frames"],
        },
        "consensus": {
            "rule": "events supported by at least min_support distinct passes (frame-proximity clusters per side)",
            "passes": passes,
            "min_support": needed,
            "cluster_gap_frames": DEFAULT_CLUSTER_GAP_FRAMES,
            "clusters": supports,
        },
        "passes": recorded,
        "selected": {
            "payload": selected_payload,
            "grid_frame_indices": sorted({i for p in recorded for i in p["grid_frame_indices"]}),
        },
    }
    output = output or case.artifacts_dir / EVENTS_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[freeze] kept {len(selected)} event(s) -> {output}", flush=True)
    return output


def register_in_case(case_id: str) -> None:
    """Point the case's lpips module params at the frozen file (keeps other keys)."""
    case_yaml = SUITE_ROOT / "cases" / case_id / "case.yaml"
    raw = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
    modules = raw.setdefault("module_params", {})
    modules.setdefault("lpips", {})[CASE_PARAM] = f"reference/artifacts/{EVENTS_FILENAME}"
    case_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case_id")
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=None, help="default: majority of passes")
    parser.add_argument("--workers", type=int, default=5, help="concurrent judge passes")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--register", action="store_true", help="also add lpips_reference_events_json to case.yaml")
    args = parser.parse_args(argv)

    from tools.run_wmeval_lite import _load_dotenv

    _load_dotenv(ROOT)
    freeze(
        args.case_id,
        passes=args.passes,
        min_support=args.min_support,
        allow_empty=args.allow_empty,
        workers=args.workers,
    )
    if args.register:
        register_in_case(args.case_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
