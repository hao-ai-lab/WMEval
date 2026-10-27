#!/usr/bin/env python3
"""Generate WMEval Lite candidates with the hosted Odyssey world model, headless.

Odyssey (https://experience.odyssey.systems/) is an interactive world model in
the browser: a session starts from a prompt + optional image + starting camera,
is driven with W/A/S/D, Space/F/E and the arrow keys (look), and every session
is saved under *You → My Simulations*. This tool drives that public UI with a
dedicated, persistent Chrome profile in **headless** mode (no window, no focus
stealing), replays each case's key program, then downloads the saved recording.

Pipeline::

    python tools/odyssey_agent.py status                     # sign-in + remaining budget
    python tools/odyssey_agent.py plan                       # jobs.jsonl for every supported case
    python tools/odyssey_agent.py run --seeds run0           # one 60 s session per case
    python tools/odyssey_agent.py manifest                   # candidates.jsonl (excludes *.raw.mp4)
    python tools/run_wmeval_lite.py run --suite eval_sets/wmeval_lite_v1 \\
        --candidate-manifest runs/odyssey/candidates.jsonl --output-dir reports/wmeval_lite/odyssey --resume

Layout written under ``runs/odyssey/`` (the runner's ``<Scene>/<variant>/<seed>.mp4``):

* ``<Scene>/<variant>/<seed>.mp4``      evaluated candidate: the action timeline,
                                         trimmed from the recording (resampled by the
                                         case's normalization token at evaluation time)
* ``<Scene>/<variant>/<seed>.raw.mp4``   the full saved Odyssey recording
* ``<Scene>/<variant>/<seed>.json``      receipt: inputs, hashes, action trace, timing
* ``jobs.jsonl`` / ``unsupported.jsonl`` the plan, and cases Odyssey cannot express

Key programs come from the case bundle (``reference/artifacts/actions.jsonl`` for the
Table 4/7 cells, otherwise an executable ``actions.txt``) or a declared supplemental
trajectory. BeamDomino's ball strike (an object force) has no key mapping and is
recorded as ``unsupported`` -- never substituted with another control. Where the case
prompt does not name the controlled entity, PLAYER_PROMPTS appends it (the bottle).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

URL = "https://experience.odyssey.systems/"
SUITE = ROOT / "eval_sets/wmeval_lite_v1"
PROFILE = ROOT / ".local/odyssey-profile"
OUT = ROOT / "runs/odyssey"
CAMERA = {"first_person": "First-Person View", "third_person": "Third-Person View"}
SUPPLEMENTAL = {
    # Cases without a native key program: the declared supplemental trajectory.
    "ChessBoard/walk_third": "experiments/trajectory_reconstructions/compiled/chess_walk_third.actions.jsonl",
    "SunRisesAndSets/lighting": "experiments/trajectory_reconstructions/compiled/sun_rises_and_sets_lighting.actions.jsonl",
    # case.yaml: deterministic in-place 360-degree yaw sweep from a fixed standing position.
    "WanderingGarden/scene_first": "experiments/tables_4_7/compiled/camera_360_in.actions.jsonl",
}
# Odyssey has one prompt and no separate character field, so the controlled entity is
# named in it, as BallRoller's case prompt does for its ball (the role Genie's character
# prompt plays). Checked on LiquidContainer/fluid_chirp_strafe: the bottle moves and the
# camera holds still.
PLAYER_PROMPTS = {
    "LiquidContainer": ("The player controls only the glass bottle: W/A/S/D slide the bottle across the floor. "
                        "The camera stays fixed in place and never moves; only the bottle moves."),
}
IDLE = {
    # Fixed top-down billiards: the cue strike is a scene event, not a player action.
    "BilliardGame/prompt_first": "cue strike is a scene event; idle input for the reference duration",
}
UNSUPPORTED_SCENES = {
    "BeamDomino": "action is a 2-channel force on the ball; Odyssey has no object-force control",
}


# ── planning ─────────────────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ops(text: str) -> list[dict[str, Any]]:
    ops = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, list):
            ops.extend(x for x in item if isinstance(x, dict) and x.get("op"))
        elif isinstance(item, dict) and item.get("op"):
            ops.append(item)
    if not ops:  # a single JSON array spanning several lines
        try:
            item = json.loads(text)
            if isinstance(item, list):
                ops = [x for x in item if isinstance(x, dict) and x.get("op")]
        except json.JSONDecodeError:
            pass
    return ops


def key_program(case_id: str, case_dir: Path) -> tuple[list[dict[str, Any]] | None, str]:
    """(ops, source) — ops is None when the case cannot be expressed with keys."""
    scene = case_id.split("/")[0]
    if scene in UNSUPPORTED_SCENES:
        return None, UNSUPPORTED_SCENES[scene]
    artifacts = case_dir / "reference" / "artifacts"
    if (artifacts / "actions.jsonl").is_file():
        return _json_ops((artifacts / "actions.jsonl").read_text(encoding="utf-8")), "bundle:actions.jsonl"
    if case_id in SUPPLEMENTAL:
        path = ROOT / SUPPLEMENTAL[case_id]
        return _json_ops(path.read_text(encoding="utf-8")), f"supplemental:{SUPPLEMENTAL[case_id]}"
    if (artifacts / "actions.txt").is_file():
        ops = _json_ops((artifacts / "actions.txt").read_text(encoding="utf-8"))
        if ops:
            return ops, "bundle:actions.txt"
    if case_id in IDLE:
        return [], f"idle:{IDLE[case_id]}"
    return None, "no executable key program"


def plan(dimensions: list[str] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    suite = yaml.safe_load((SUITE / "suite.yaml").read_text(encoding="utf-8"))
    wanted = dimensions or list(suite["dimensions"])
    cases: dict[str, list[str]] = {}
    for dim in wanted:
        for case_id in suite["dimensions"][dim]["cases"]:
            cases.setdefault(case_id, []).append(dim)
    jobs, unsupported = [], []
    for case_id, dims in sorted(cases.items()):
        case_dir = SUITE / "cases" / case_id
        case = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))
        ops, source = key_program(case_id, case_dir)
        row = {"case_id": case_id, "dimensions": dims, "program_source": source}
        if ops is None:
            unsupported.append({**row, "status": "unsupported", "reason": source})
            continue
        ref = case.get("reference", {})
        jobs.append({
            **row,
            "prompt": " ".join(filter(None, [str(case.get("prompt") or "").strip(),
                                             PLAYER_PROMPTS.get(case_id.split("/")[0], "")])),
            "first_frame": ref.get("first_frame"),
            "reference_video": ref.get("video"),
            "camera": CAMERA.get(str(case.get("perspective")), "World View"),
            "ops": ops,
            "program_ms": sum(int(op.get("ms", 0)) for op in ops if op.get("op") in {"hold", "wait"}),
        })
    return jobs, unsupported


# ── browser session ──────────────────────────────────────────────────────────
TIMER = re.compile(r"^(\d+):(\d\d)$")


def _budget_seconds(page) -> int | None:
    for line in page.inner_text("body").splitlines()[:6]:
        match = TIMER.match(line.strip())
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
    return None


def _open(playwright, headless: bool = True):
    PROFILE.mkdir(parents=True, exist_ok=True)
    ctx = playwright.chromium.launch_persistent_context(
        str(PROFILE), channel="chrome", headless=headless,
        viewport={"width": 1440, "height": 900}, args=["--autoplay-policy=no-user-gesture-required"],
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto(URL, wait_until="networkidle", timeout=90000)
    page.wait_for_timeout(1500)
    # Right after a session, the server may still hold the previous connection and
    # show "Odyssey might be open in another tab" with a Reload button.
    for _ in range(6):
        reload = page.get_by_role("button", name="Reload")
        if not reload.count():
            break
        page.wait_for_timeout(10000)
        reload.first.click()
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)
    return ctx, page


def _simulation_sources(ctx, page) -> list[str]:
    page.get_by_role("button", name="You", exact=True).click()
    page.wait_for_timeout(700)
    page.get_by_role("button", name="My Simulations", exact=True).click()
    page.wait_for_timeout(3000)
    srcs = [v.get_attribute("src") or "" for v in page.locator("video").all()]
    page.goto(URL, wait_until="networkidle", timeout=90000)  # back to the prompt view
    page.wait_for_timeout(1500)
    return [s for s in srcs if "experience-user-recordings" in s]


def _recording_id(src: str) -> str:
    match = re.search(r"experience-user-recordings/([0-9a-f-]+)/", src)
    return match.group(1) if match else src


def _first_frame(job: dict[str, Any], workdir: Path) -> Path:
    case_dir = SUITE / "cases" / job["case_id"]
    if job.get("first_frame") and (case_dir / job["first_frame"]).is_file():
        return case_dir / job["first_frame"]
    frame = workdir / "first_frame.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(case_dir / job["reference_video"]),
                    "-frames:v", "1", str(frame)], check=True)
    return frame


def _replay(page, ops: list[dict[str, Any]], start: float) -> list[dict[str, Any]]:
    trace = []
    for op in ops:
        kind = str(op.get("op"))
        keys = [str(k) for k in op.get("keys", [])] + ([str(op["key"])] if op.get("key") else [])
        row = {"op": kind, "keys": keys, "ms": op.get("ms"), "requested_sec": round(time.monotonic() - start, 3)}
        if kind == "hold":
            for key in keys:
                page.keyboard.down(key)
            row["accepted_sec"] = round(time.monotonic() - start, 3)
            page.wait_for_timeout(int(op["ms"]))
            for key in reversed(keys):
                page.keyboard.up(key)
        elif kind == "wait":
            row["accepted_sec"] = round(time.monotonic() - start, 3)
            page.wait_for_timeout(int(op["ms"]))
        elif kind == "press":
            page.keyboard.press(keys[0])
            row["accepted_sec"] = round(time.monotonic() - start, 3)
        else:  # setview etc.: the starting camera is chosen before the session
            row["skipped"] = True
        row["completed_sec"] = round(time.monotonic() - start, 3)
        trace.append(row)
    return trace


def run_job(playwright, job: dict[str, Any], seed: str, *, session_sec: float, min_budget_sec: int) -> dict[str, Any]:
    out_dir = OUT / job["case_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema_version": 1, "backend": "odyssey_web", "url": URL, "case_id": job["case_id"],
        "seed": seed, "dimensions": job["dimensions"], "program_source": job["program_source"],
        "camera": job["camera"], "started_at": datetime.now(timezone.utc).isoformat(),
        "seed_control": "unavailable_in_web_ui",
    }
    with tempfile.TemporaryDirectory() as tmp:
        ctx, page = _open(playwright)
        try:
            budget = _budget_seconds(page)
            receipt["budget_sec_before"] = budget
            if budget is not None and budget < min_budget_sec:
                receipt["status"] = "budget_exhausted"
                return receipt
            before = {_recording_id(s) for s in _simulation_sources(ctx, page)}
            image = _first_frame(job, Path(tmp))
            receipt.update({"first_frame_sha256": _sha256(image), "prompt": job["prompt"],
                            "actions_sha256": hashlib.sha256(json.dumps(job["ops"], sort_keys=True).encode()).hexdigest()})
            page.locator("input[type=file]").set_input_files(str(image))
            page.wait_for_timeout(1500)
            page.locator("textarea").fill(job["prompt"])
            camera = page.locator("button[aria-label^='Starting camera']")
            if job["camera"] not in (camera.get_attribute("aria-label") or ""):
                camera.click()
                page.wait_for_timeout(400)
                page.locator("button:visible", has_text=job["camera"]).first.click()
                page.wait_for_timeout(400)
            receipt["camera_selected"] = page.locator("button[aria-label^='Starting camera']").get_attribute("aria-label")
            page.get_by_role("button", name="Start Simulating").click()
            page.wait_for_selector("text=REC", timeout=60000)      # recording has started
            t_rec = time.monotonic()
            page.locator("body").click(position={"x": 720, "y": 450})  # keyboard focus on the stage
            page.wait_for_timeout(300)
            receipt["actions_offset_sec"] = round(time.monotonic() - t_rec, 3)
            receipt["action_trace"] = _replay(page, job["ops"], t_rec)
            remaining = session_sec - (time.monotonic() - t_rec)
            if remaining > 0:
                page.wait_for_timeout(int(remaining * 1000))
            receipt["session_sec"] = round(time.monotonic() - t_rec, 3)
            receipt["budget_sec_after"] = _budget_seconds(page)
            close = page.get_by_role("button", name="Close")
            if close.count():
                close.first.click()
            else:  # the session already ended (e.g. the budget ran out mid-session)
                receipt["session_ended_before_close"] = True
            page.wait_for_timeout(4000)
            for _ in range(10):  # the saved recording can take a few seconds to appear
                after = {(_recording_id(s), s) for s in _simulation_sources(ctx, page)}
                new = [s for rid, s in after if rid not in before]
                if new:
                    break
                page.wait_for_timeout(3000)
            if not new:
                receipt["status"] = "recording_not_found"
                return receipt
            raw = out_dir / f"{seed}.raw.mp4"
            response = ctx.request.get(new[0])
            raw.write_bytes(response.body())
            receipt.update({"recording_id": _recording_id(new[0]), "raw_video": str(raw.relative_to(ROOT)),
                            "raw_sha256": _sha256(raw)})
        except Exception:
            try:
                page.screenshot(path=str(out_dir / f"{seed}.error.png"))
            except Exception:
                pass
            raise
        finally:
            ctx.close()
    # Evaluated candidate = the action timeline, aligned to the recording start.
    duration = max(job["program_ms"] / 1000.0, 1.0)
    candidate = out_dir / f"{seed}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{receipt['actions_offset_sec']:.3f}", "-i", str(raw),
                    "-t", f"{duration:.3f}", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    str(candidate)], check=True)
    receipt.update({"status": "complete", "candidate_video": str(candidate.relative_to(ROOT)),
                    "candidate_sha256": _sha256(candidate), "candidate_duration_sec": duration})
    return receipt


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p_man = sub.add_parser("manifest", help="write a candidates manifest from completed receipts")
    p_man.add_argument("--dimensions", help="only cases scored by these dimensions (the runner rejects others)")
    p_man.add_argument("--output", type=Path, default=OUT / "candidates.jsonl")
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--dimensions", help="comma-separated (default: all)")
    p_run = sub.add_parser("run")
    p_run.add_argument("--dimensions")
    p_run.add_argument("--case", action="append", default=[])
    p_run.add_argument("--seeds", default="run0", help="comma-separated repeat ids")
    p_run.add_argument("--session-sec", type=float, default=60.0)
    p_run.add_argument("--min-budget-sec", type=int, default=75)
    p_run.add_argument("--force", action="store_true", help="redo completed jobs")
    p_run.add_argument("--wait-for-budget-min", type=float, default=0.0,
                       help="when the session budget is exhausted, re-check every 2 min for up to this many minutes")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    if args.cmd == "status":
        with sync_playwright() as p:
            ctx, page = _open(p)
            try:
                page.get_by_role("button", name="You", exact=True).click()
                page.wait_for_timeout(700)
                signed_in = page.get_by_role("button", name="Sign In").count() == 0
                page.keyboard.press("Escape")
                print(json.dumps({"budget_sec": _budget_seconds(page), "signed_in": signed_in,
                                  "saved_simulations": len(_simulation_sources(ctx, page))}, indent=2))
            finally:
                ctx.close()
        return 0

    if args.cmd == "manifest":
        allowed = None
        if args.dimensions:
            suite = yaml.safe_load((SUITE / "suite.yaml").read_text(encoding="utf-8"))
            allowed = {c for d in args.dimensions.split(",") for c in suite["dimensions"][d]["cases"]}
        rows = []
        for receipt_path in sorted(OUT.glob("*/*/*.json")):
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") == "complete" and (allowed is None or receipt["case_id"] in allowed):
                rows.append({"model": "odyssey", "case_id": receipt["case_id"], "seed": receipt["seed"],
                             "candidate_video": str((ROOT / receipt["candidate_video"]).resolve())})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(json.dumps({"candidates": len(rows), "manifest": str(args.output)}, indent=2))
        return 0

    dims = args.dimensions.split(",") if args.dimensions else None
    jobs, unsupported = plan(dims)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jobs.jsonl").write_text("".join(json.dumps({k: v for k, v in j.items() if k != "ops"}) + "\n" for j in jobs))
    (OUT / "unsupported.jsonl").write_text("".join(json.dumps(u) + "\n" for u in unsupported))
    if args.cmd == "plan":
        print(json.dumps({"jobs": len(jobs), "unsupported": [u["case_id"] for u in unsupported]}, indent=2))
        return 0

    if args.case:
        jobs = [j for j in jobs if j["case_id"] in set(args.case)]
    summary = []
    with sync_playwright() as p:
        for seed in args.seeds.split(","):
            for job in jobs:
                receipt_path = OUT / job["case_id"] / f"{seed}.json"
                if receipt_path.is_file() and not args.force and json.loads(receipt_path.read_text()).get("status") == "complete":
                    continue
                deadline = time.monotonic() + args.wait_for_budget_min * 60
                attempts = 0
                while True:
                    attempts += 1
                    try:
                        receipt = run_job(p, job, seed, session_sec=args.session_sec, min_budget_sec=args.min_budget_sec)
                    except Exception as error:  # keep going; the receipt records the failure
                        receipt = {"case_id": job["case_id"], "seed": seed, "status": "error", "error": repr(error)[:500]}
                    receipt["attempts"] = attempts
                    if receipt.get("status") == "error" and attempts < 3:
                        print(json.dumps({"retrying": job["case_id"], "error": receipt["error"][:120]}), flush=True)
                        time.sleep(20)
                        continue
                    if receipt.get("status") != "budget_exhausted" or time.monotonic() > deadline:
                        break
                    print(json.dumps({"waiting_for_budget": receipt.get("budget_sec_before"), "case_id": job["case_id"]}), flush=True)
                    time.sleep(120)
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                summary.append({k: receipt.get(k) for k in ("case_id", "seed", "status", "budget_sec_after")})
                print(json.dumps(summary[-1]), flush=True)
                if receipt.get("status") == "budget_exhausted":
                    print("stopping: Odyssey session budget exhausted (sign in or wait for it to reset)", flush=True)
                    return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
