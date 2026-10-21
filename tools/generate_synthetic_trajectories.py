#!/usr/bin/env python3
"""Generate the WMEval Lite synthetic action-trajectory suite (``test_full``).

Text first: every synthetic trajectory starts as an action description -- a
short sequence of steps from a small per-target vocabulary (``turn left at 30
degrees per second for 2 seconds``, ``brakes hard for 1 second``, ``shakes left
and right at 2 hertz for 4 seconds`` ...) -- which is then

* rendered into the second part of the prompt (the first part is the scene
  description of the bound core case, with any sentence about the core case's
  own action removed), and
* compiled into the action sequence: a schema-v1 ``trajectory.yaml`` (the same
  format as the curated cases) and its compiled forms.

Both come from the same steps, so the prompt always describes the actions, and
``parse_description`` reads a sentence back into its steps (a round-trip check
the tests enforce). Steps are sampled with a fixed seed; for diversity each
family keeps only distinct action sequences and picks them by farthest-point
selection on the compiled control signal (integer arithmetic, so the choice is
the same on every platform), stratified by the number of steps.

Each trajectory is bound to a core case that supplies the scene, conditioning
frame and action target; steps start from that case's initial state, so the
core conditioning frame stays valid. Re-running reproduces the files in
``synthetic/manifest.json``, which ``--verify`` checks (byte-identical, except
orbit poses, which may differ in the last float bits across platforms).

Each trajectory directory gets ``trajectory.yaml``, the compiled ``action.npz``
(timestamps, values, channel names; camera poses for look-at orbits),
``actions.jsonl`` (Project Genie browser program) and ``meta.json`` (prompt,
action description and steps, and -- once ``tools/render_synthetic_references.py
publish`` has listed it in ``synthetic/references.json`` -- its Unity reference,
which sits in the trajectory's ``reference/`` folder). The generated files and
references are published to the Hub dataset (split ``test_full``); this
repository keeps the generator, the manifest and references.json.

    python tools/generate_synthetic_trajectories.py            # write eval_sets/wmeval_lite_v1/synthetic/
    python tools/generate_synthetic_trajectories.py --verify   # regenerate and compare with the manifest

Sign conventions follow the core trajectories and the browser key rules:
positive yaw / steer / lateral / x is to the right, positive forward / y is
away from the camera, and a positive orbit rate is counter-clockwise (the core
``SunRisesAndSets/lighting`` reference orbit runs at +36 deg/s and its prompt
calls that counter-clockwise).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.trajectories import load_trajectory, write_compiled_trajectory  # noqa: E402

SUITE_ROOT = ROOT / "eval_sets/wmeval_lite_v1"
OUT_ROOT = SUITE_ROOT / "synthetic"
MANIFEST = OUT_ROOT / "manifest.json"
REFERENCES = OUT_ROOT / "references.json"  # written by tools/render_synthetic_references.py publish
GENERATOR_VERSION = 2
SEED = 20261019
POOL_FACTOR = 20  # candidate plans sampled per selected trajectory

Seg = tuple[float, dict[str, float]]  # (duration_sec, channel values)
Step = dict[str, Any]                 # {"kind": ..., field: value, ...}


# --------------------------------------------------------------------------- helpers
def segments(parts: list[Seg], duration: float, hz: int, rest: dict[str, float]) -> list[dict[str, Any]]:
    """Contiguous segments on the control grid; pads with ``rest`` to ``duration``."""
    out, t = [], 0
    total = round(duration * hz)
    for seconds, values in parts:
        steps = max(1, round(seconds * hz))
        steps = min(steps, total - t)
        if steps <= 0:
            break
        out.append({"start_sec": t / hz, "end_sec": (t + steps) / hz, "values": dict(values)})
        t += steps
    if t < total:
        out.append({"start_sec": t / hz, "end_sec": total / hz, "values": dict(rest)})
    return out


def square_wave(channel: str, amp: float, f0: float, f1: float, start: float, end: float, other: dict) -> list[Seg]:
    """Square wave on ``channel`` sweeping linearly from f0 to f1 Hz between start and end."""
    parts: list[Seg] = [(start, {**other, channel: 0.0})] if start > 0 else []
    t, sign = start, 1.0
    while t < end - 1e-9:
        f = f0 + (f1 - f0) * (t - start) / max(end - start, 1e-9)
        half = min(0.5 / f, end - t)
        parts.append((half, {**other, channel: sign * amp}))
        t += half
        sign = -sign
    return parts


def core_case(case_id: str) -> dict[str, Any]:
    return yaml.safe_load((SUITE_ROOT / "cases" / case_id / "case.yaml").read_text(encoding="utf-8"))


def first_frame(case_id: str) -> str:
    ref = core_case(case_id).get("reference") or {}
    rel = ref.get("first_frame") or ref.get("video")
    return f"cases/{case_id}/{rel}"


def fmt(x: float) -> str:
    return f"{float(x):g}"


# --------------------------------------------------------------------------- action vocabulary
@dataclass(frozen=True)
class Clause:
    """One step type: its sentence template, field domains and compiled control parts.

    ``fields`` maps a template field to its allowed words (a dict word -> value)
    or to ``None`` for a number. ``derived`` adds numbers computed from the step
    (e.g. total degrees) that the sentence states and the parser re-checks.
    """

    kind: str
    text: str
    fields: dict[str, dict[str, Any] | None]
    parts: Callable[[Step], list[Seg]]
    derived: Callable[[Step], dict[str, float]] = lambda step: {}

    def render(self, step: Step) -> str:
        values = {k: (fmt(step[k]) if dom is None else step[k]) for k, dom in self.fields.items()}
        values.update({k: fmt(v) for k, v in self.derived(step).items()})
        return self.text.format(**values)

    @property
    def regex(self) -> re.Pattern[str]:
        out, pos = "", 0
        for m in re.finditer(r"\{(\w+)\}", self.text):
            out += re.escape(self.text[pos:m.start()])
            dom = self.fields.get(m.group(1))
            if dom is None:
                out += rf"(?P<{m.group(1)}>-?\d+(?:\.\d+)?)"
            else:
                words = sorted(dom, key=len, reverse=True)
                out += f"(?P<{m.group(1)}>" + "|".join(re.escape(w) for w in words) + ")"
            pos = m.end()
        return re.compile(out + re.escape(self.text[pos:]))

    def parse(self, text: str) -> Step | None:
        m = self.regex.fullmatch(text)
        if m is None:
            return None
        step: Step = {"kind": self.kind}
        for k, dom in self.fields.items():
            step[k] = float(m.group(k)) if dom is None else m.group(k)
        for k, v in self.derived(step).items():
            if m.group(k) != fmt(v):
                return None
        return step

    def seconds(self, step: Step) -> float:
        return sum(sec for sec, _ in self.parts(step))


@dataclass
class Grammar:
    prefix: str                    # sentence opening, e.g. "From this starting view, the camera "
    suffix: str                    # appended when the steps end before the clip does
    clauses: dict[str, Clause]
    sample: Callable[[random.Random, int], list[Step] | None]  # (rng, stratum) -> steps
    strata: dict[int, float]       # stratum (usually the number of steps) -> share of the family
    rest: dict[str, float] = field(default_factory=dict)
    filler: tuple[str, ...] = ()   # connective step kinds left out of the step-kind balance

    def describe(self, plan: list[Step], duration: float) -> str:
        parts = [self.clauses[s["kind"]].render(s) for s in plan]
        if self.seconds(plan) < duration - 1e-9:
            parts.append(self.suffix)
        return _singular(self.prefix + ", then ".join(parts) + ".")

    def seconds(self, plan: list[Step]) -> float:
        return sum(self.clauses[s["kind"]].seconds(s) for s in plan)

    def compile(self, plan: list[Step]) -> list[Seg]:
        return [part for s in plan for part in self.clauses[s["kind"]].parts(s)]


def parse_description(grammar: Grammar, sentence: str, duration: float) -> list[Step]:
    """Read a description back into its steps (the inverse of ``Grammar.describe``)."""
    if not sentence.startswith(grammar.prefix) or not sentence.endswith("."):
        raise ValueError(f"not a description for this grammar: {sentence!r}")
    items = _plural(sentence)[len(grammar.prefix):-1].split(", then ")
    padded = items[-1] == grammar.suffix
    if padded:
        items = items[:-1]
    plan = []
    for item in items:
        matches = [step for clause in grammar.clauses.values() if (step := clause.parse(item)) is not None]
        if len(matches) != 1:
            raise ValueError(f"{len(matches)} clause(s) match {item!r}")
        plan.append(matches[0])
    if padded != (grammar.seconds(plan) < duration - 1e-9):
        raise ValueError(f"end-of-clip phrase inconsistent with step durations: {sentence!r}")
    return plan


def _singular(text: str) -> str:
    return re.sub(r"(?<![\d.])1 seconds\b", "1 second", text)


def _plural(text: str) -> str:
    return re.sub(r"(?<![\d.])1 second\b", "1 seconds", text)


def _pick(rng: random.Random, dom: dict[str, Any] | tuple) -> Any:
    return rng.choice(sorted(dom) if isinstance(dom, dict) else list(dom))


def _sample_plan(rng: random.Random, make: Callable[[random.Random], list[Step] | None],
                 grammar_ref: list[Grammar], duration: float, attempts: int = 200) -> list[Step] | None:
    for _ in range(attempts):
        plan = make(rng)
        if plan and grammar_ref[0].seconds(plan) <= duration + 1e-9 and not any(map(_mergeable, plan, plan[1:])):
            return plan
    return None


def _mergeable(a: Step, b: Step) -> bool:
    """Consecutive steps that differ only in duration read (and act) as one step."""
    strip = lambda s: {k: v for k, v in s.items() if k != "sec"}  # noqa: E731
    return strip(a) == strip(b) and "sec" in a


D = round(1 / math.sqrt(2), 4)
DIR8 = {"forward": (0.0, 1.0), "backward": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0),
        "forward-left": (-D, D), "forward-right": (D, D), "backward-left": (-D, -D), "backward-right": (D, -D)}
SIDE = {"left": -1.0, "right": 1.0}


def _no_repeats(kinds: list[str], *once: str) -> bool:
    """No two consecutive steps of a kind listed in ``once``."""
    return all(not (a == b and a in once) for a, b in zip(kinds, kinds[1:]))


# camera yaw -----------------------------------------------------------------
YAW = "yaw_rate_deg_s"
TURN_SPEEDS, TURN_SECS = (15, 20, 30, 45, 60, 90), (1, 1.5, 2, 2.5, 3, 4, 5)
SWEEPS = [(a, s) for a in (30, 45, 60, 90) for s in (30, 45, 60, 90) if (a / s * 20).is_integer() and a / s <= 2]


def camera_yaw_grammar() -> Grammar:
    y = lambda v: {YAW: float(v)}  # noqa: E731
    clauses = {
        "turn": Clause("turn", "turns {dir} at {speed} degrees per second for {sec} seconds ({deg} degrees in total)",
                       {"dir": SIDE, "speed": None, "sec": None},
                       lambda s: [(s["sec"], y(SIDE[s["dir"]] * s["speed"]))],
                       lambda s: {"deg": s["speed"] * s["sec"]}),
        "hold": Clause("hold", "holds still for {sec} seconds", {"sec": None}, lambda s: [(s["sec"], y(0))]),
        "sweep": Clause("sweep", "sweeps {amp} degrees to the {dir} and back at {speed} degrees per second",
                        {"amp": None, "dir": SIDE, "speed": None},
                        lambda s: [(s["amp"] / s["speed"], y(SIDE[s["dir"]] * s["speed"])),
                                   (s["amp"] / s["speed"], y(-SIDE[s["dir"]] * s["speed"]))]),
        "look_back": Clause("look_back", "turns {dir} to look directly behind (180 degrees at {speed} degrees per second)",
                            {"dir": SIDE, "speed": None},
                            lambda s: [(180 / s["speed"], y(SIDE[s["dir"]] * s["speed"]))]),
    }

    def step(rng: random.Random, kind: str) -> Step:
        if kind == "turn":
            return {"kind": kind, "dir": _pick(rng, SIDE), "speed": float(rng.choice(TURN_SPEEDS)),
                    "sec": float(rng.choice(TURN_SECS))}
        if kind == "hold":
            return {"kind": kind, "sec": float(rng.choice((0.5, 1, 1.5, 2, 3)))}
        if kind == "sweep":
            amp, speed = rng.choice(SWEEPS)
            return {"kind": kind, "amp": float(amp), "dir": _pick(rng, SIDE), "speed": float(speed)}
        return {"kind": kind, "dir": _pick(rng, SIDE), "speed": float(rng.choice((60, 90)))}

    def sample(rng: random.Random, n: int) -> list[Step] | None:
        def make(r: random.Random) -> list[Step] | None:
            kinds = [r.choice(("turn", "turn", "hold", "sweep", "look_back")) for _ in range(n)]
            if kinds[-1] == "hold" or not _no_repeats(kinds, "hold") or set(kinds) == {"hold"}:
                return None
            return [step(r, k) for k in kinds]
        return _sample_plan(rng, make, ref, 10.0)

    ref: list[Grammar] = []
    g = Grammar("From this starting view, the camera ", "holds still until the end of the 10-second clip",
                clauses, sample, {1: 0.15, 2: 0.3, 3: 0.3, 4: 0.25}, rest={YAW: 0.0})
    ref.append(g)
    return g


# camera orbit ---------------------------------------------------------------
RATE = "orbit_phase_rate_deg_s"
ORBIT_DIR = {"counter-clockwise": 1.0, "clockwise": -1.0}


def camera_orbit_grammar() -> Grammar:
    clauses = {
        "orbit": Clause("orbit", "circles {dir} at {speed} degrees per second for {sec} seconds ({deg} degrees of arc)",
                        {"dir": ORBIT_DIR, "speed": None, "sec": None},
                        lambda s: [(s["sec"], {RATE: ORBIT_DIR[s["dir"]] * s["speed"]})],
                        lambda s: {"deg": s["speed"] * s["sec"]}),
        "pause": Clause("pause", "pauses for {sec} seconds", {"sec": None}, lambda s: [(s["sec"], {RATE: 0.0})]),
    }

    def sample(rng: random.Random, n: int) -> list[Step] | None:
        def make(r: random.Random) -> list[Step] | None:
            kinds = [r.choice(("orbit", "orbit", "pause")) for _ in range(n)]
            if kinds[-1] == "pause" or not _no_repeats(kinds, "pause"):
                return None
            return [{"kind": "orbit", "dir": _pick(r, ORBIT_DIR), "speed": float(r.choice((9, 18, 27, 36, 45, 54, 72))),
                     "sec": float(r.choice((1, 2, 2.5, 3, 4, 5, 10)))} if k == "orbit"
                    else {"kind": "pause", "sec": float(r.choice((0.5, 1, 1.5, 2)))} for k in kinds]
        return _sample_plan(rng, make, ref, 10.0)

    ref: list[Grammar] = []
    g = Grammar("The camera ", "stays in place until the end of the 10-second clip", clauses, sample,
                {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.2}, rest={RATE: 0.0})
    ref.append(g)
    return g


# ego vehicle ----------------------------------------------------------------
THROTTLE = {"half": 0.5, "three-quarter": 0.75, "full": 1.0}
STEER = {"gently": 0.3, "moderately": 0.6, "sharply": 1.0}
LANE_STEER = {"gentle": 0.3, "moderate": 0.6, "sharp": 1.0}


def ego_vehicle_grammar() -> Grammar:
    def car(steer: float = 0.0, throttle: float = 0.0, brake: int = 0) -> dict[str, float]:
        return {"steer": float(steer), "throttle": float(throttle), "brake": brake}

    clauses = {
        "wait": Clause("wait", "waits at the start line for {sec} seconds", {"sec": None}, lambda s: [(s["sec"], car())]),
        "drive": Clause("drive", "drives straight at {level} throttle for {sec} seconds", {"level": THROTTLE, "sec": None},
                        lambda s: [(s["sec"], car(0, THROTTLE[s["level"]]))]),
        "steer": Clause("steer", "steers {amount} {dir} at {level} throttle for {sec} seconds",
                        {"amount": STEER, "dir": SIDE, "level": THROTTLE, "sec": None},
                        lambda s: [(s["sec"], car(SIDE[s["dir"]] * STEER[s["amount"]], THROTTLE[s["level"]]))]),
        "lane": Clause("lane", "changes lanes to the {dir} with {amount} steering over {sec} seconds",
                       {"dir": SIDE, "amount": LANE_STEER, "sec": None},
                       lambda s: [(s["sec"] / 2, car(SIDE[s["dir"]] * LANE_STEER[s["amount"]], 0.75)),
                                  (s["sec"] / 2, car(-SIDE[s["dir"]] * LANE_STEER[s["amount"]], 0.75))]),
        "coast": Clause("coast", "coasts for {sec} seconds", {"sec": None}, lambda s: [(s["sec"], car())]),
        "brake": Clause("brake", "brakes hard for {sec} seconds", {"sec": None}, lambda s: [(s["sec"], car(brake=1))]),
    }

    def step(r: random.Random, kind: str) -> Step:
        if kind == "drive":
            return {"kind": kind, "level": _pick(r, THROTTLE), "sec": float(r.choice((1, 1.5, 2, 2.5, 3, 4)))}
        if kind == "steer":
            return {"kind": kind, "amount": _pick(r, STEER), "dir": _pick(r, SIDE), "level": _pick(r, THROTTLE),
                    "sec": float(r.choice((0.5, 1, 1.5, 2)))}
        if kind == "lane":
            return {"kind": kind, "dir": _pick(r, SIDE), "amount": _pick(r, LANE_STEER), "sec": float(r.choice((1, 2, 3)))}
        secs = {"wait": (0.5, 1, 1.5, 2), "coast": (1, 1.5, 2), "brake": (0.5, 1, 1.5, 2)}[kind]
        return {"kind": kind, "sec": float(r.choice(secs))}

    def sample(rng: random.Random, n: int) -> list[Step] | None:
        def make(r: random.Random) -> list[Step] | None:
            # n moving steps; the car must be driving before it steers, coasts or brakes
            kinds = ["drive"] + [r.choice(("drive", "steer", "steer", "lane", "lane", "coast", "brake")) for _ in range(n - 1)]
            if kinds[-1] == "coast" or not _no_repeats(kinds, "coast", "brake", "drive"):
                return None
            plan = [step(r, k) for k in kinds]
            return ([step(r, "wait")] if r.random() < 0.4 else []) + plan
        return _sample_plan(rng, make, ref, 10.0)

    ref: list[Grammar] = []
    g = Grammar("The car ", "coasts until the end of the 10-second clip", clauses, sample,
                {1: 0.1, 2: 0.3, 3: 0.3, 4: 0.3}, rest=car(), filler=("wait", "coast"))
    ref.append(g)
    return g


# cue ball -------------------------------------------------------------------
CUE_DIR = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "to the left": (0.0, -1.0), "to the right": (0.0, 1.0),
           "forward-left": (D, -D), "forward-right": (D, D), "backward-left": (-D, -D), "backward-right": (-D, D)}
STRENGTH = {"light": 0.35, "medium": 0.6, "hard": 0.9}
REPEATS = {"twice": 2, "three times": 3}


def cue_ball_grammar() -> Grammar:
    # The Unity replay fires one impulse when a non-zero input starts and again every 0.5 s while held.
    def hit(s: Step) -> dict[str, float]:
        f, lat = CUE_DIR[s["dir"]]
        k = STRENGTH[s["strength"]]
        return {"forward": round(f * k, 4), "lateral": round(lat * k, 4)}

    clauses = {
        "wait": Clause("wait", "waits {sec} seconds", {"sec": None},
                       lambda s: [(s["sec"], {"forward": 0.0, "lateral": 0.0})]),
        "strike": Clause("strike", "is struck {dir} at {strength} strength", {"dir": CUE_DIR, "strength": STRENGTH},
                         lambda s: [(0.5, hit(s))]),
        "repeat": Clause("repeat", "is struck {dir} at {strength} strength {n}, half a second apart",
                         {"dir": CUE_DIR, "strength": STRENGTH, "n": REPEATS}, lambda s: [(0.5 * REPEATS[s["n"]], hit(s))]),
    }

    def strike(r: random.Random) -> Step:
        base = {"dir": _pick(r, CUE_DIR), "strength": _pick(r, STRENGTH)}
        if r.random() < 0.3:
            return {"kind": "repeat", **base, "n": _pick(r, REPEATS)}
        return {"kind": "strike", **base}

    def sample(rng: random.Random, n: int) -> list[Step] | None:
        def make(r: random.Random) -> list[Step] | None:
            plan: list[Step] = [{"kind": "wait", "sec": float(r.choice((0.5, 1, 1.5, 2)))}, strike(r)]
            for _ in range(n - 1):  # n strikes, separated by waits
                plan += [{"kind": "wait", "sec": float(r.choice((1, 1.5, 2, 3)))}, strike(r)]
            return plan
        return _sample_plan(rng, make, ref, 10.0)

    ref: list[Grammar] = []
    g = Grammar("The cue ball ", "rolls freely until the end of the 10-second clip", clauses, sample,
                {1: 0.3, 2: 0.4, 3: 0.3}, rest={"forward": 0.0, "lateral": 0.0}, filler=("wait",))
    ref.append(g)
    return g


# rolling ball (BallRoller, MirrorRollingBall) --------------------------------
FORCE = {"half": 0.5, "full": 1.0}


def ball_force_grammar() -> Grammar:
    def push(s: Step) -> dict[str, float]:
        x, y = DIR8[s["dir"]]
        return {"horizontal": round(x * FORCE[s["strength"]], 4), "vertical": round(y * FORCE[s["strength"]], 4)}

    clauses = {
        "push": Clause("push", "is pushed {dir} at {strength} force for {sec} seconds",
                       {"dir": DIR8, "strength": FORCE, "sec": None}, lambda s: [(s["sec"], push(s))]),
        "roll": Clause("roll", "rolls without input for {sec} seconds", {"sec": None},
                       lambda s: [(s["sec"], {"horizontal": 0.0, "vertical": 0.0})]),
    }

    def sample(rng: random.Random, n: int) -> list[Step] | None:
        def make(r: random.Random) -> list[Step] | None:
            plan: list[Step] = [{"kind": "roll", "sec": 0.5}] if r.random() < 0.3 else []
            for i in range(n):  # n pushes, optionally separated by free rolling
                if i and r.random() < 0.4:
                    plan.append({"kind": "roll", "sec": float(r.choice((0.5, 1, 1.5)))})
                plan.append({"kind": "push", "dir": _pick(r, DIR8), "strength": _pick(r, FORCE),
                             "sec": float(r.choice((0.5, 1, 1.5, 2, 2.5)))})
            return plan
        return _sample_plan(rng, make, ref, 6.0)

    ref: list[Grammar] = []
    g = Grammar("The ball ", "rolls freely until the end of the 6-second clip", clauses, sample,
                {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.2}, rest={"horizontal": 0.0, "vertical": 0.0}, filler=("roll",))
    ref.append(g)
    return g


# beam domino ----------------------------------------------------------------
BEAM_DIR = {"straight forward": (1.0, 0.0), "forward and slightly left": (1.0, -0.25),
            "forward and to the left": (1.0, -0.5), "forward and slightly right": (1.0, 0.25),
            "forward and to the right": (1.0, 0.5), "sideways to the left": (0.0, -1.0),
            "sideways to the right": (0.0, 1.0), "backward": (-1.0, 0.0)}


def beam_domino_grammar() -> Grammar:
    def force(s: Step) -> dict[str, float]:
        z, x = BEAM_DIR[s["dir"]]
        return {"z_force": round(z * FORCE[s["strength"]], 4), "x_force": round(x * FORCE[s["strength"]], 4)}

    clauses = {
        "wait": Clause("wait", "waits {sec} seconds", {"sec": None}, lambda s: [(s["sec"], {"z_force": 0.0, "x_force": 0.0})]),
        "roll": Clause("roll", "rolls {dir} under {strength} force for {sec} seconds",
                       {"dir": BEAM_DIR, "strength": FORCE, "sec": None}, lambda s: [(s["sec"], force(s))]),
    }

    def sample(rng: random.Random, n: int) -> list[Step] | None:
        def make(r: random.Random) -> list[Step] | None:
            plan: list[Step] = [{"kind": "wait", "sec": float(r.choice((1, 1.5, 2, 3, 4)))}] if r.random() < 0.6 else []
            for i in range(n):  # n rolls, optionally separated by waits
                if i and r.random() < 0.3:
                    plan.append({"kind": "wait", "sec": float(r.choice((1, 1.5, 2)))})
                plan.append({"kind": "roll", "dir": _pick(r, BEAM_DIR), "strength": _pick(r, FORCE),
                             "sec": float(r.choice((1, 1.5, 2, 3, 4, 5)))})
            # the ball has to head for the structure at some point for a causal chain to be possible
            if not any(s["kind"] == "roll" and s["dir"].startswith(("straight", "forward")) and s["sec"] >= 2 for s in plan):
                return None
            return plan
        return _sample_plan(rng, make, ref, 12.2)

    ref: list[Grammar] = []
    g = Grammar("The blue ball ", "receives no further input until the end of the 12.2-second clip", clauses, sample,
                {1: 0.3, 2: 0.35, 3: 0.35}, rest={"z_force": 0.0, "x_force": 0.0}, filler=("wait",))
    ref.append(g)
    return g


# liquid container -----------------------------------------------------------
LIQ_DIR = {name: (y, x, 0.0) for name, (x, y) in DIR8.items()}  # (forward, right, up)
LIQ_JOLT = {**LIQ_DIR, "upward": (0.0, 0.0, 1.0)}
SHAKE_AXIS = {"left and right": "right", "forward and back": "forward"}


def liquid_container_grammar() -> Grammar:
    z = {"forward": 0.0, "right": 0.0, "up": 0.0}

    def move(vec: tuple[float, float, float]) -> dict[str, float]:
        return {"forward": vec[0], "right": vec[1], "up": vec[2]}

    clauses = {
        "rest": Clause("rest", "rests for {sec} seconds", {"sec": None}, lambda s: [(s["sec"], dict(z))]),
        "jolt": Clause("jolt", "is jolted {dir}", {"dir": LIQ_JOLT}, lambda s: [(0.25, move(LIQ_JOLT[s["dir"]]))]),
        "slide": Clause("slide", "slides {dir} for {sec} seconds", {"dir": LIQ_DIR, "sec": None},
                        lambda s: [(s["sec"], move(LIQ_DIR[s["dir"]]))]),
        "shake": Clause("shake", "shakes {axis} at {freq} hertz for {sec} seconds",
                        {"axis": SHAKE_AXIS, "freq": None, "sec": None},
                        lambda s: square_wave(SHAKE_AXIS[s["axis"]], 1.0, s["freq"], s["freq"], 0, s["sec"], z)),
        "chirp": Clause("chirp", "shakes {axis} with the rate rising from {f0} to {f1} hertz over {sec} seconds",
                        {"axis": SHAKE_AXIS, "f0": None, "f1": None, "sec": None},
                        lambda s: square_wave(SHAKE_AXIS[s["axis"]], 1.0, s["f0"], s["f1"], 0, s["sec"], z)),
    }

    def motion(r: random.Random) -> Step:
        kind = r.choice(("jolt", "jolt", "slide", "shake", "shake", "chirp"))
        if kind == "jolt":
            return {"kind": kind, "dir": _pick(r, LIQ_JOLT)}
        if kind == "slide":
            return {"kind": kind, "dir": _pick(r, LIQ_DIR), "sec": float(r.choice((1, 2, 3, 4, 6)))}
        if kind == "shake":
            return {"kind": kind, "axis": _pick(r, SHAKE_AXIS), "freq": float(r.choice((0.5, 1, 1.5, 2, 3))),
                    "sec": float(r.choice((2, 3, 4, 6)))}
        return {"kind": kind, "axis": _pick(r, SHAKE_AXIS), "f0": float(r.choice((0.3, 0.5))),
                "f1": float(r.choice((1.5, 2, 3))), "sec": float(r.choice((4, 6, 8)))}

    def sample(rng: random.Random, n: int) -> list[Step] | None:
        def make(r: random.Random) -> list[Step] | None:
            plan: list[Step] = [{"kind": "rest", "sec": float(r.choice((1, 2, 3)))}]
            for i in range(n):  # n motions, separated by rests so each one can ring down
                if i:
                    plan.append({"kind": "rest", "sec": float(r.choice((1, 2, 3, 4, 5)))})
                plan.append(motion(r))
            return plan
        # leave at least 3 s at the end for the liquid to settle
        return _sample_plan(rng, make, ref, 17.0)

    ref: list[Grammar] = []
    g = Grammar("The bottle ", "rests until the end of the 20-second clip", clauses, sample,
                {1: 0.3, 2: 0.4, 3: 0.3}, rest=dict(z), filler=("rest",))
    ref.append(g)
    return g


# character ------------------------------------------------------------------
SPRINT_DIR = {k: DIR8[k] for k in ("forward", "forward-left", "forward-right")}


def character_grammar() -> Grammar:
    def c(x: float = 0.0, y: float = 0.0, jump: int = 0, sprint: int = 0) -> dict[str, float]:
        return {"move_x": float(x), "move_y": float(y), "jump": jump, "sprint": sprint}

    clauses = {
        "walk": Clause("walk", "walks {dir} for {sec} seconds", {"dir": DIR8, "sec": None},
                       lambda s: [(s["sec"], c(*DIR8[s["dir"]]))]),
        "sprint": Clause("sprint", "sprints {dir} for {sec} seconds", {"dir": SPRINT_DIR, "sec": None},
                         lambda s: [(s["sec"], c(*SPRINT_DIR[s["dir"]], sprint=1))]),
        "stand": Clause("stand", "stands still for {sec} seconds", {"sec": None}, lambda s: [(s["sec"], c())]),
        "jump": Clause("jump", "jumps in place", {}, lambda s: [(1 / 30, c(jump=1))]),
    }

    def step(r: random.Random, kind: str) -> Step:
        if kind == "walk":
            return {"kind": kind, "dir": _pick(r, DIR8), "sec": float(r.choice((0.5, 1, 1.5, 2, 3)))}
        if kind == "sprint":
            return {"kind": kind, "dir": _pick(r, SPRINT_DIR), "sec": float(r.choice((1, 1.5, 2, 3)))}
        if kind == "stand":
            return {"kind": kind, "sec": float(r.choice((0.5, 1)))}
        return {"kind": kind}

    def sample(rng: random.Random, n: int) -> list[Step] | None:
        def make(r: random.Random) -> list[Step] | None:
            kinds = [r.choice(("walk", "walk", "walk", "sprint", "stand", "jump")) for _ in range(n)]
            if kinds[-1] == "stand" or not _no_repeats(kinds, "stand", "jump") or set(kinds) <= {"stand", "jump"}:
                return None
            return [step(r, k) for k in kinds]
        return _sample_plan(rng, make, ref, 6.0)

    ref: list[Grammar] = []
    g = Grammar("The character ", "stands still until the end of the 6-second clip", clauses, sample,
                {1: 0.15, 2: 0.3, 3: 0.3, 4: 0.25}, rest=c())
    ref.append(g)
    return g


# --------------------------------------------------------------------------- families
GENIE_YAW = [{"channel": "yaw_rate_deg_s", "threshold": 1.0, "positive_key": "ArrowRight", "negative_key": "ArrowLeft"}]
CAMERA_YAW_CONTEXTS = [  # (core case supplying frame + scene description)
    "WanderingGarden/scene_360_in", "WanderingGarden/scene_360_out", "BilliardGame/scene_360_in",
    "MirrorRollingBall/prompt_first", "MirrorRollingBall/prompt_third",
    "ChessBoard/prompt_first", "ChessBoard/prompt_third",
    "RacingCar/prompt_first", "RacingCar/prompt_third",
]

# Sentences of a core prompt that describe the core case's own action, and what
# replaces them in the scene description of a synthetic trajectory.
_360 = ("From that orientation, the camera performs the controlled full 360-degree camera rotation used by the paper.", "")
SCENE_EDITS: dict[str, list[tuple[str, str]]] = {
    "WanderingGarden/scene_360_in": [_360],
    "WanderingGarden/scene_360_out": [_360],
    "BilliardGame/scene_360_in": [_360],
    "SunRisesAndSets/lighting": [(
        "The viewer walks slowly and steadily counter-clockwise around a wooden round dining table at a constant "
        "2-metre radius from the table centre, completing one full loop over the duration of the clip while "
        "continuously keeping the table centred in the field of view. Smooth steady gait, no head bobbing, no "
        "jumping, no looking up at the sky, no looking down at the feet.",
        "The viewer stays on a circle around a wooden round dining table, continuously keeping the table in the "
        "field of view. Smooth steady gait, no head bobbing, no jumping, no looking up at the sky, no looking down "
        "at the feet.")],
    "BeamDomino/causal": [(
        "After a brief pause it rolls straight forward into the nearest front-right pillar, then takes no further "
        "steering. Expected chain: the ball rolls, strikes a supporting pillar, the roof tips and the gold ball "
        "drops, and the domino ring cascades.",
        "If the ball strikes a supporting pillar, the roof can tip, the gold ball can drop and the domino ring can "
        "cascade.")],
    "ChessBoard/walk_third": [(
        "A gray humanoid mannequin character walks and jogs along the platform in third-person view, turning and "
        "briefly jumping.",
        "A gray humanoid mannequin character stands on the platform, seen in third-person view.")],
    "LiquidContainer/fluid_rest": [
        ("it rotates in response to control inputs but does not deform.",
         "it moves in response to control inputs but does not deform."),
        ("The bottle is not commanded to move at all for the entire clip: no strafe, no jump, no rotation. Any "
         "bottle or liquid motion in a generation is hallucinated.",
         "The bottle never rotates; it only translates as described below."),
    ],
}


def scene_description(case_id: str) -> str:
    """The core prompt with sentences about the core case's own action replaced."""
    paragraphs = re.split(r"\n\s*\n", core_case(case_id)["prompt"].strip())
    flat = "\n\n".join(" ".join(p.split()) for p in paragraphs)
    for old, new in SCENE_EDITS.get(case_id, []):
        if old not in flat:
            raise ValueError(f"{case_id}: action sentence not found in the core prompt: {old[:60]!r}")
        flat = flat.replace(old, new)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r" {2,}", " ", re.sub(r" +\n", "\n", flat))).strip()


# (family, target, core case(s), count, duration, control_hz, channels, genie key rules, grammar)
def families() -> list[dict[str, Any]]:
    cont = lambda unit="normalized": {"kind": "continuous", "min": -1.0, "max": 1.0, "unit": unit}  # noqa: E731
    boolean = {"kind": "discrete", "min": 0, "max": 1, "unit": "boolean"}
    return [
        {"family": "camera_yaw", "target": "camera", "cases": CAMERA_YAW_CONTEXTS, "count": 45, "duration": 10.0, "hz": 20,
         "channels": {"yaw_rate_deg_s": {"kind": "continuous", "min": -360.0, "max": 360.0, "unit": "deg_per_sec"}},
         "keys": GENIE_YAW, "grammar": camera_yaw_grammar(),
         "dimensions": ["environment_object_consistency", "prompt_scene_consistency"]},
        {"family": "camera_orbit", "target": "camera", "cases": ["SunRisesAndSets/lighting"], "count": 20,
         "duration": 10.0, "hz": 60, "orbit": True, "grammar": camera_orbit_grammar(), "dimensions": ["lighting_shading"]},
        {"family": "ego_vehicle", "target": "ego_vehicle", "cases": ["RacingCar/kinematics_third"], "count": 28,
         "duration": 10.0, "hz": 30, "channels": {"steer": cont(), "throttle": cont(), "brake": boolean},
         "keys": [{"channel": "steer", "threshold": 0.25, "positive_key": "KeyD", "negative_key": "KeyA"},
                  {"channel": "throttle", "threshold": 0.25, "positive_key": "KeyW", "negative_key": "KeyS"},
                  {"channel": "brake", "threshold": 0.5, "positive_key": "KeyS"}],
         "grammar": ego_vehicle_grammar(), "dimensions": ["kinematics_mechanics", "environment_object_consistency"]},
        {"family": "cue_ball", "target": "cue_ball", "cases": ["BilliardGame/kinematics_first"], "count": 25,
         "duration": 10.0, "hz": 30,
         "channels": {"forward": cont("normalized_ball_impulse"), "lateral": cont("normalized_ball_impulse")},
         "keys": [{"channel": "forward", "threshold": 0.25, "positive_key": "KeyW", "negative_key": "KeyS"},
                  {"channel": "lateral", "threshold": 0.15, "positive_key": "KeyD", "negative_key": "KeyA"}],
         "grammar": cue_ball_grammar(), "dimensions": ["kinematics_mechanics"]},
        {"family": "ball_force", "target": "ball", "cases": ["BallRoller/ctrl", "MirrorRollingBall/lpips_collision"],
         "count": 28, "duration": 6.0, "hz": 20,
         "channels": {"horizontal": cont("normalized_force_input"), "vertical": cont("normalized_force_input")},
         "keys": [{"channel": "horizontal", "threshold": 0.25, "positive_key": "KeyD", "negative_key": "KeyA"},
                  {"channel": "vertical", "threshold": 0.25, "positive_key": "KeyW", "negative_key": "KeyS"}],
         "grammar": ball_force_grammar(), "dimensions": ["action_controllability", "lpips"]},
        {"family": "beam_domino", "target": "rolling_ball", "cases": ["BeamDomino/causal"], "count": 25,
         "duration": 12.2, "hz": 10, "channels": {"z_force": cont(), "x_force": cont()},
         "keys": [{"channel": "z_force", "threshold": 0.25, "positive_key": "KeyW", "negative_key": "KeyS"},
                  {"channel": "x_force", "threshold": 0.25, "positive_key": "KeyD", "negative_key": "KeyA"}],
         "grammar": beam_domino_grammar(), "dimensions": ["physical_causality", "action_controllability", "lpips"]},
        {"family": "liquid_container", "target": "liquid_container", "cases": ["LiquidContainer/fluid_rest"], "count": 25,
         "duration": 20.0, "hz": 12, "channels": {"forward": cont(), "right": cont(), "up": cont()},
         "keys": [{"channel": "forward", "threshold": 0.5, "positive_key": "KeyW", "negative_key": "KeyS"},
                  {"channel": "right", "threshold": 0.5, "positive_key": "KeyD", "negative_key": "KeyA"},
                  {"channel": "up", "threshold": 0.5, "positive_key": "Space"}],
         "grammar": liquid_container_grammar(), "dimensions": ["fluid_mechanics", "lpips"]},
        {"family": "character", "target": "character", "cases": ["ChessBoard/walk_third"], "count": 24,
         "duration": 6.0, "hz": 30, "channels": {"move_x": cont(), "move_y": cont(), "jump": boolean, "sprint": boolean},
         "keys": [{"channel": "move_x", "threshold": 0.4, "positive_key": "KeyD", "negative_key": "KeyA"},
                  {"channel": "move_y", "threshold": 0.4, "positive_key": "KeyW", "negative_key": "KeyS"},
                  {"channel": "jump", "threshold": 0.5, "positive_key": "Space"},
                  {"channel": "sprint", "threshold": 0.5, "positive_key": "Shift"}],
         "grammar": character_grammar(), "dimensions": ["environment_object_consistency"]},
    ]


# --------------------------------------------------------------------------- diverse selection
def control_signal(fam: dict[str, Any], plan: list[Step]) -> np.ndarray:
    """The compiled per-step control values (x1000, int64) used to compare plans exactly."""
    g: Grammar = fam["grammar"]
    segs = segments(g.compile(plan), fam["duration"], fam["hz"], g.rest)
    names = sorted(g.rest)
    rows = []
    for seg in segs:
        n = round((seg["end_sec"] - seg["start_sec"]) * fam["hz"])
        rows += [[round(float(seg["values"][k]) * 1000) for k in names]] * n
    return np.asarray(rows, dtype=np.int64).ravel()


def _quotas(strata: dict[int, float], total: int) -> dict[int, int]:
    raw = {k: v * total for k, v in strata.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def select_plans(fam: dict[str, Any], index: int) -> list[list[Step]]:
    """Distinct, mutually far-apart plans for a family (deterministic on every platform)."""
    g: Grammar = fam["grammar"]
    rng = random.Random(SEED + index)
    pool: list[tuple[int, list[Step]]] = []
    seen: set[bytes] = set()
    quotas = _quotas(g.strata, fam["count"])
    for stratum, quota in quotas.items():
        for _ in range(quota * POOL_FACTOR):
            plan = g.sample(rng, stratum)
            if plan is None:
                continue
            key = control_signal(fam, plan).tobytes()
            if key not in seen:
                seen.add(key)
                pool.append((stratum, plan))
    signals = np.stack([control_signal(fam, plan) for _, plan in pool])
    kinds = [sorted({st["kind"] for st in plan} - set(g.filler)) for _, plan in pool]
    used: dict[str, int] = {k: 0 for k in g.clauses}
    chosen: list[int] = []
    nearest = np.full(len(pool), np.iinfo(np.int64).max, dtype=np.int64)
    for stratum, quota in quotas.items():
        members = [i for i, (s, _) in enumerate(pool) if s == stratum]
        if len(members) < quota:
            raise RuntimeError(f"{fam['family']}: only {len(members)} distinct plans for stratum {stratum}")
        for _ in range(quota):
            # Farthest point, damped for plans whose rarest step kind is already common, so short
            # steps (a jolt, a jump) are not crowded out by long high-amplitude ones. Integer
            # arithmetic keeps the choice platform-independent; ties -> lowest index.
            def score(i: int) -> tuple[int, int]:
                rarest = min((used[k] for k in kinds[i]), default=0)
                return (int(nearest[i]) // (1 + rarest) ** 2, -i)
            pick = members[0] if not chosen else max(members, key=score)
            chosen.append(pick)
            members.remove(pick)
            for k in kinds[pick]:
                used[k] += 1
            d = signals - signals[pick]
            nearest = np.minimum(nearest, np.einsum("ij,ij->i", d, d))
    return [pool[i][1] for i in chosen]


# --------------------------------------------------------------------------- build
def trajectory_doc(fam: dict[str, Any], case_id: str, slug: str, plan: list[Step], sentence: str) -> dict[str, Any]:
    scene, variant = case_id.split("/")
    g: Grammar = fam["grammar"]
    tid = f"synthetic.{fam['family']}.{scene.lower()}_{variant}.{slug}_v{GENERATOR_VERSION}"
    description = (f"Synthetic {fam['family'].replace('_', ' ')} trajectory '{slug}' on {case_id}, generated by "
                   f"tools/generate_synthetic_trajectories.py (v{GENERATOR_VERSION}) from its action description: "
                   f"{sentence}")
    parts = g.compile(plan)
    if fam.get("orbit"):
        base = load_trajectory(SUITE_ROOT / "cases" / case_id / "reference/artifacts/trajectory.yaml")
        doc = {
            "schema_version": 1, "trajectory_id": tid, "description": description,
            "kind": "camera_look_at_orbit", "target": "camera", "coordinates": base["coordinates"],
            "duration_sec": fam["duration"], "control_hz": fam["hz"], "geometry": base["geometry"],
            "channels": {RATE: base["channels"][RATE]},
            "segments": segments(parts, fam["duration"], fam["hz"], g.rest),
            "environment": base.get("environment"),
        }
        if "adapters" in base:
            doc["adapters"] = base["adapters"]
        doc["paper_alignment"] = {"status": "synthetic", "note": "Not in the paper; procedurally generated."}
        return doc
    clean = [(d, {k: (int(v) if fam["channels"][k]["kind"] == "discrete" else float(v)) for k, v in vals.items()})
             for d, vals in parts]
    return {
        "schema_version": 1, "trajectory_id": tid, "description": description,
        "duration_sec": fam["duration"], "control_hz": fam["hz"], "channels": fam["channels"],
        "segments": segments(clean, fam["duration"], fam["hz"], g.rest),
        "paper_alignment": {"status": "synthetic", "matched": [], "missing": "Not in the paper; procedurally generated.",
                            "source": f"cases/{case_id} (initial state)"},
        "adapters": {"unity": {"status": "pending_render", "channels": list(fam["channels"])},
                     "project_genie": {"status": "nominal", "key_rules": fam["keys"]}},
    }


def load_references() -> dict[str, Any]:
    if not REFERENCES.is_file():
        return {}
    return json.loads(REFERENCES.read_text(encoding="utf-8"))["references"]


def generate(out_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    references = load_references()
    for index, fam in enumerate(families()):
        g: Grammar = fam["grammar"]
        for i, plan in enumerate(select_plans(fam, index)):
            case_id = fam["cases"][i % len(fam["cases"])]
            case = core_case(case_id)
            scene, variant = case_id.split("/")
            slug = f"s{i:02d}_" + "_".join(dict.fromkeys(s["kind"] for s in plan))
            sentence = g.describe(plan, fam["duration"])
            assert parse_description(g, sentence, fam["duration"]) == plan, sentence
            doc = trajectory_doc(fam, case_id, slug, plan, sentence)
            rel = f"{fam['family']}/{scene}__{variant}/{slug}"
            out = out_root / rel
            out.mkdir(parents=True, exist_ok=True)
            traj = out / "trajectory.yaml"
            traj.write_text(yaml.safe_dump(doc, sort_keys=False, width=100000), encoding="utf-8")
            # Compile next to the source so the recorded source path is location-independent.
            write_compiled_trajectory(traj, out)
            (out / "trajectory.action.npz").replace(out / "action.npz")
            if (out / "trajectory.actions.jsonl").is_file():
                (out / "trajectory.actions.jsonl").replace(out / "actions.jsonl")
            if (out / "trajectory.poses.npy").is_file():  # camera poses of look-at orbits
                (out / "trajectory.poses.npy").replace(out / "poses.npy")
            for extra in ("trajectory.action.npy", "trajectory.actions.txt", "trajectory.manifest.json"):
                (out / extra).unlink(missing_ok=True)
            meta = {
                "schema_version": 2, "synthetic_id": f"synthetic/{rel}", "trajectory_id": doc["trajectory_id"],
                "family": fam["family"], "pattern": slug, "action_target": fam["target"],
                "scene": scene, "conditioning_case": case_id, "perspective": case.get("perspective"),
                "prompt": f"{scene_description(case_id)}\n\n{sentence}",
                "action_description": sentence, "action_steps": plan,
                "first_frame": first_frame(case_id),
                "duration_sec": fam["duration"], "control_hz": fam["hz"], "dimensions": fam["dimensions"],
                "reference_status": "pending_unity_render", "generator_version": GENERATOR_VERSION,
            }
            ref = references.get(f"synthetic/{rel}")
            if ref:
                meta.update({"reference_status": "unity_rendered",
                             "reference_video": f"synthetic/{rel}/{ref['video']}",
                             "reference_quality": ref["quality"],
                             "reference_first_frame_psnr_db": ref["first_frame_psnr_db"]})
            (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
            rows.append({**meta, "path": f"synthetic/{rel}", "files": files})
    return rows


# --------------------------------------------------------------------------- verify
FLOAT_FILES = ("action.npz", "poses.npy", "actions.jsonl")


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    if isinstance(loaded, np.ndarray):
        return {"": loaded}
    with loaded:
        return {key: loaded[key] for key in loaded.files}


def _json_close(x: Any, y: Any) -> bool:
    if isinstance(x, float) or isinstance(y, float):
        return (isinstance(x, (int, float)) and isinstance(y, (int, float))
                and not isinstance(x, bool) and not isinstance(y, bool) and abs(x - y) <= 1e-9)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_json_close(u, v) for u, v in zip(x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_json_close(x[k], y[k]) for k in x)
    return x == y


def _numerically_equal(a: Path, b: Path) -> bool:
    if a.suffix == ".jsonl":
        xs, ys = a.read_text(encoding="utf-8").splitlines(), b.read_text(encoding="utf-8").splitlines()
        return len(xs) == len(ys) and all(_json_close(json.loads(u), json.loads(v)) for u, v in zip(xs, ys))
    xs, ys = _load_arrays(a), _load_arrays(b)
    if xs.keys() != ys.keys():
        return False
    for key, u in xs.items():
        v = ys[key]
        if u.shape != v.shape or u.dtype != v.dtype:
            return False
        if np.issubdtype(u.dtype, np.floating):
            if not np.allclose(u, v, rtol=0.0, atol=1e-9):
                return False
        elif not np.array_equal(u, v):
            return False
    return True


def compare(rows: list[dict[str, Any]], out_root: Path) -> list[str]:
    """synthetic_ids whose regenerated files (under ``out_root``) differ from the manifest."""
    want = {r["synthetic_id"]: r["files"] for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["trajectories"]}
    diff = sorted(set(want) - {r["synthetic_id"] for r in rows})
    for r in rows:
        expected = want.get(r["synthetic_id"])
        if expected is None or expected.keys() != r["files"].keys():
            diff.append(r["synthetic_id"])
            continue
        for name, digest in r["files"].items():
            if digest == expected[name]:
                continue
            published = SUITE_ROOT / r["path"] / name
            if (name in FLOAT_FILES and published.is_file()
                    and hashlib.sha256(published.read_bytes()).hexdigest() == expected[name]
                    and _numerically_equal(out_root / r["path"] / name, published)):
                continue
            diff.append(r["synthetic_id"])
            break
    return diff


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="regenerate in a temp dir and compare with the manifest")
    args = parser.parse_args(argv)
    if args.verify:
        with tempfile.TemporaryDirectory() as tmp:
            rows = generate(Path(tmp) / "synthetic")
            diff = compare(rows, Path(tmp))
        if diff:
            print(f"MISMATCH: {len(diff)} trajectory(ies) differ, e.g. {diff[:3]}")
            return 1
        print(f"verified {len(rows)} synthetic trajectories against {MANIFEST.relative_to(ROOT)}")
        return 0
    rows = generate(OUT_ROOT)
    # Drop trajectory folders that are no longer generated (each one's reference/ is kept
    # with it otherwise: references are published into the same folder).
    keep = {Path(r["path"]).relative_to("synthetic") for r in rows}
    for traj in sorted(OUT_ROOT.glob("*/*/*")):
        if traj.is_dir() and traj.relative_to(OUT_ROOT) not in keep:
            shutil.rmtree(traj)
    by_family: dict[str, int] = {}
    for r in rows:
        by_family[r["family"]] = by_family.get(r["family"], 0) + 1
    MANIFEST.write_text(json.dumps({"schema_version": 2, "generator_version": GENERATOR_VERSION,
                                    "count": len(rows), "by_family": by_family, "trajectories": rows},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(rows), "by_family": by_family}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
