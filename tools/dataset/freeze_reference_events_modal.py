#!/usr/bin/env python3
"""Run tools/dataset/freeze_reference_events.py on Modal, one container per case.

The hosted judge sends tens of MB of frames per pass; from a laptop that is
unreliable (stalled connections), from a Modal container it is not. Each case
is judged in its own container with the repo baked into the image.

Each finished case is stored server-side in a ``modal.Dict`` the moment it
completes, so a dropped client (sleep, network) never loses paid-for judge work;
run with ``--detach`` and re-run with ``--fetch-only`` to collect results.

OPENAI_API_KEY is read from the environment or the repo's local ``.env`` and
forwarded as an ephemeral Secret (never baked into the image)::

    modal run --detach tools/dataset/freeze_reference_events_modal.py \
        --cases MirrorRollingBall/lpips_collision,BeamDomino/causal,BallRoller/ctrl --register
    modal run tools/dataset/freeze_reference_events_modal.py --fetch-only --register   # collect later
    python tools/dataset/build_lite_case.py lite/cases/<Scene>/<variant>
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[2]
# Locally the repo is two levels up; in the container Modal mounts this file at
# the filesystem root, so the baked-in repo copy is used instead.
for _root in (REPO_ROOT, Path("/workspace/WMEval")):
    if (_root / "tools").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tools.run_wmeval_lite_modal import REMOTE_REPO, _load_dotenv_local, image  # noqa: E402

DEFAULT_CASES = "MirrorRollingBall/lpips_collision,BeamDomino/causal,BallRoller/ctrl"
app = modal.App("wmeval-lite-freeze-events")
results = modal.Dict.from_name("wmeval-lite-freeze-events", create_if_missing=True)


@app.function(image=image, timeout=3600, cpu=2.0)
def freeze_case(case_id: str, passes: int, workers: int) -> str:
    """Judge one case's reference; store and return the contact_events.json text."""
    os.chdir(REMOTE_REPO)
    if REMOTE_REPO not in sys.path:
        sys.path.insert(0, REMOTE_REPO)
    from tools.dataset.freeze_reference_events import freeze

    path = freeze(case_id, passes=passes, min_support=None, allow_empty=False, workers=workers)
    text = Path(path).read_text(encoding="utf-8")
    results[case_id] = text
    return text


@app.local_entrypoint()
def main(
    cases: str = DEFAULT_CASES,
    passes: int = 5,
    workers: int = 5,
    register: bool = False,
    fetch_only: bool = False,
    wait_sec: int = 3600,
):
    from tools.dataset.freeze_reference_events import EVENTS_FILENAME, SUITE_ROOT, register_in_case

    case_ids = [item.strip() for item in cases.split(",") if item.strip()]
    if not fetch_only:
        _load_dotenv_local(REPO_ROOT)
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY is required: freezing runs the hosted judge")
        fn = freeze_case.with_options(secrets=[modal.Secret.from_dict({"OPENAI_API_KEY": key})])
        for case_id in case_ids:
            results.pop(case_id, None)  # never mistake an older run's result for this one
            fn.spawn(case_id, passes, workers)
        print(f"spawned {len(case_ids)} case(s) x {passes} passes: {case_ids}", flush=True)

    pending = set(case_ids)
    deadline = time.time() + wait_sec
    while pending and time.time() < deadline:
        for case_id in sorted(pending):
            text = results.get(case_id)
            if not text:
                continue
            out = SUITE_ROOT / "cases" / case_id / "reference" / "artifacts" / EVENTS_FILENAME
            out.write_text(text, encoding="utf-8")
            print(f"wrote {out}", flush=True)
            if register:
                register_in_case(case_id)
            pending.discard(case_id)
        if pending:
            time.sleep(10)
    if pending:
        print(f"still pending: {sorted(pending)} (re-run with --fetch-only to collect)", flush=True)
