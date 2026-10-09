"""Canonical action trajectories shared by Unity and browser backends.

The semantic YAML is authoritative.  Repository-native ``action.npy`` arrays,
self-describing ``action.npz`` bundles, and Project Genie keyboard programs
are compiled derivatives so their timing and provenance can be audited instead
of being maintained by hand.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


class TrajectoryError(ValueError):
    """Raised when a semantic trajectory violates the schema contract."""


@dataclass(frozen=True)
class CompiledTrajectory:
    trajectory_id: str
    channel_names: tuple[str, ...]
    timestamps_sec: np.ndarray
    values: np.ndarray
    semantic_sha256: str
    alignment_status: str
    project_genie_ops: tuple[dict[str, Any], ...]
    project_genie_status: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trajectory(path: str | Path) -> dict[str, Any]:
    trajectory_path = Path(path)
    raw = yaml.safe_load(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TrajectoryError(f"{trajectory_path}: root must be a mapping")
    return raw


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrajectoryError(f"{label} must be numeric")
    return float(value)


def validate_trajectory(raw: Mapping[str, Any], *, source: str = "trajectory") -> None:
    if raw.get("schema_version") != 1:
        raise TrajectoryError(f"{source}: schema_version must be 1")
    trajectory_id = raw.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise TrajectoryError(f"{source}: trajectory_id is required")

    duration = _number(raw.get("duration_sec"), f"{source}: duration_sec")
    control_hz = _number(raw.get("control_hz"), f"{source}: control_hz")
    if duration <= 0 or control_hz <= 0:
        raise TrajectoryError(f"{source}: duration_sec and control_hz must be positive")
    steps_exact = duration * control_hz
    if abs(steps_exact - round(steps_exact)) > 1e-8:
        raise TrajectoryError(f"{source}: duration_sec * control_hz must be integral")

    channels = raw.get("channels")
    if not isinstance(channels, dict) or not channels:
        raise TrajectoryError(f"{source}: channels must be a non-empty mapping")
    for name, spec in channels.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise TrajectoryError(f"{source}: invalid channel declaration")
        if spec.get("kind") not in {"continuous", "discrete"}:
            raise TrajectoryError(f"{source}: channel {name} has invalid kind")
        lower = _number(spec.get("min"), f"{source}: channels.{name}.min")
        upper = _number(spec.get("max"), f"{source}: channels.{name}.max")
        if lower > upper:
            raise TrajectoryError(f"{source}: channel {name} min exceeds max")

    segments = raw.get("segments")
    if not isinstance(segments, list) or not segments:
        raise TrajectoryError(f"{source}: segments must be a non-empty list")
    cursor = 0.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise TrajectoryError(f"{source}: segment {index} must be a mapping")
        start = _number(segment.get("start_sec"), f"{source}: segment {index} start_sec")
        end = _number(segment.get("end_sec"), f"{source}: segment {index} end_sec")
        if abs(start - cursor) > 1e-8:
            raise TrajectoryError(
                f"{source}: segment {index} starts at {start}, expected contiguous {cursor}"
            )
        if end <= start or end > duration + 1e-8:
            raise TrajectoryError(f"{source}: segment {index} has invalid interval")
        if abs(start * control_hz - round(start * control_hz)) > 1e-8:
            raise TrajectoryError(f"{source}: segment {index} start is not on a control step")
        if abs(end * control_hz - round(end * control_hz)) > 1e-8:
            raise TrajectoryError(f"{source}: segment {index} end is not on a control step")
        values = segment.get("values")
        if not isinstance(values, dict) or set(values) != set(channels):
            raise TrajectoryError(
                f"{source}: segment {index} values must name every channel exactly once"
            )
        for name, value in values.items():
            numeric = _number(value, f"{source}: segment {index}.{name}")
            spec = channels[name]
            if not float(spec["min"]) <= numeric <= float(spec["max"]):
                raise TrajectoryError(f"{source}: segment {index}.{name} is out of range")
            if spec["kind"] == "discrete" and not numeric.is_integer():
                raise TrajectoryError(f"{source}: segment {index}.{name} must be integral")
        cursor = end
    if abs(cursor - duration) > 1e-8:
        raise TrajectoryError(f"{source}: segments end at {cursor}, expected {duration}")

    alignment = raw.get("paper_alignment")
    if not isinstance(alignment, dict) or alignment.get("status") not in {
        "exact", "reconstructed", "unresolved"
    }:
        raise TrajectoryError(
            f"{source}: paper_alignment.status must be exact, reconstructed, or unresolved"
        )


def _compress_keyboard_steps(
    key_steps: Sequence[frozenset[str]], control_hz: float
) -> tuple[dict[str, Any], ...]:
    if not key_steps:
        return ()
    ops: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(key_steps) + 1):
        if index < len(key_steps) and key_steps[index] == key_steps[start]:
            continue
        duration_ms = int(round((index - start) * 1000.0 / control_hz))
        keys = sorted(key_steps[start])
        if keys:
            ops.append({"op": "hold", "keys": keys, "ms": duration_ms})
        else:
            ops.append({"op": "wait", "ms": duration_ms})
        start = index
    return tuple(ops)


def _project_genie_ops(
    raw: Mapping[str, Any], channel_names: Sequence[str], values: np.ndarray
) -> tuple[tuple[dict[str, Any], ...], str]:
    adapter = raw.get("adapters", {}).get("project_genie", {})
    status = str(adapter.get("status", "unsupported"))
    if status == "unsupported":
        return (), status
    key_rules = adapter.get("key_rules")
    if not isinstance(key_rules, list):
        raise TrajectoryError("project_genie adapter requires key_rules")
    channel_index = {name: index for index, name in enumerate(channel_names)}
    steps: list[frozenset[str]] = []
    for row in values:
        keys: set[str] = set()
        for rule in key_rules:
            channel = str(rule.get("channel", ""))
            if channel not in channel_index:
                raise TrajectoryError(f"project_genie rule names unknown channel {channel!r}")
            value = float(row[channel_index[channel]])
            threshold = float(rule.get("threshold", 0.5))
            if value >= threshold and rule.get("positive_key"):
                keys.add(str(rule["positive_key"]))
            if value <= -threshold and rule.get("negative_key"):
                keys.add(str(rule["negative_key"]))
        steps.append(frozenset(keys))
    control_hz = float(raw["control_hz"])
    ops = [dict(op) for op in _compress_keyboard_steps(steps, control_hz)]
    # Preserve the semantic yaw displacement in the browser derivative.  A
    # hosted world chooses its own key-to-yaw gain, so elapsed milliseconds are
    # only a nominal transport value; the target angle is what a calibrated
    # browser adapter must satisfy.
    yaw_index = channel_index.get("yaw_rate_deg_s")
    if yaw_index is not None:
        cursor = 0
        for op in ops:
            step_count = int(round(float(op["ms"]) * control_hz / 1000.0))
            segment = values[cursor : cursor + step_count, yaw_index]
            target_yaw = float(segment.sum(dtype=np.float64) / control_hz)
            if op["op"] == "hold" and abs(target_yaw) > 1e-8:
                op["target_yaw_deg"] = round(target_yaw, 6)
            cursor += step_count
        if cursor != len(values):
            raise TrajectoryError(
                f"project_genie operation timeline covers {cursor} steps, expected {len(values)}"
            )
    return tuple(ops), status


def compile_trajectory(path: str | Path) -> CompiledTrajectory:
    source_path = Path(path).resolve()
    raw = load_trajectory(source_path)
    validate_trajectory(raw, source=str(source_path))
    channel_names = tuple(raw["channels"].keys())
    control_hz = float(raw["control_hz"])
    step_count = int(round(float(raw["duration_sec"]) * control_hz))
    timestamps = np.arange(step_count, dtype=np.float64) / control_hz
    values = np.zeros((step_count, len(channel_names)), dtype=np.float32)
    for index, timestamp in enumerate(timestamps):
        segment = next(
            segment
            for segment in raw["segments"]
            if float(segment["start_sec"]) <= timestamp < float(segment["end_sec"])
        )
        values[index] = [float(segment["values"][name]) for name in channel_names]
    ops, genie_status = _project_genie_ops(raw, channel_names, values)
    return CompiledTrajectory(
        trajectory_id=str(raw["trajectory_id"]),
        channel_names=channel_names,
        timestamps_sec=timestamps,
        values=values,
        semantic_sha256=_sha256(source_path),
        alignment_status=str(raw["paper_alignment"]["status"]),
        project_genie_ops=ops,
        project_genie_status=genie_status,
    )


def write_compiled_trajectory(path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_path = Path(path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiled = compile_trajectory(source_path)
    stem = source_path.stem
    npy_path = output / f"{stem}.action.npy"
    npz_path = output / f"{stem}.action.npz"
    summary_path = output / f"{stem}.actions.txt"
    actions_path = output / f"{stem}.actions.jsonl"
    manifest_path = output / f"{stem}.manifest.json"
    metadata = {
        "schema_version": 1,
        "trajectory_id": compiled.trajectory_id,
        "source": os.path.relpath(source_path, output),
        "semantic_sha256": compiled.semantic_sha256,
        "alignment_status": compiled.alignment_status,
        "project_genie_status": compiled.project_genie_status,
        "channel_names": list(compiled.channel_names),
        "step_count": int(len(compiled.timestamps_sec)),
    }
    arrays = {
        "timestamps_sec": compiled.timestamps_sec,
        "values": compiled.values,
        "channel_names": np.asarray(compiled.channel_names),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    # Keep the same plain float32 matrix convention as existing WMEval
    # reference/artifacts/action.npy files. The NPZ below carries the richer
    # timestamps/channel metadata.
    with npy_path.open("wb") as handle:
        np.save(handle, compiled.values, allow_pickle=False)
    # numpy.savez_compressed stamps ZIP members with wall-clock time. Write the
    # same standard NPZ representation with a fixed timestamp so content hashes
    # are stable across devices and reruns.
    with zipfile.ZipFile(npz_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(array), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    actions_path.write_text(
        "".join(json.dumps(op, sort_keys=True) + "\n" for op in compiled.project_genie_ops),
        encoding="utf-8",
    )
    raw = load_trajectory(source_path)
    summary_lines = [
        f"# {compiled.trajectory_id} -- generated action trajectory.",
        f"# Provenance: {source_path.name} (semantic SHA-256: {compiled.semantic_sha256})",
        f"# Alignment: {compiled.alignment_status}; {len(compiled.timestamps_sec)} steps; "
        f"channels: {', '.join(compiled.channel_names)}",
        "# start_s end_s " + " ".join(compiled.channel_names),
    ]
    for segment in raw["segments"]:
        values = " ".join(str(segment["values"][name]) for name in compiled.channel_names)
        summary_lines.append(f"{segment['start_sec']} {segment['end_sec']} {values}")
    if compiled.project_genie_ops:
        summary_lines.extend(["", "# Project Genie browser actions:"])
        summary_lines.extend(json.dumps(op, sort_keys=True) for op in compiled.project_genie_ops)
    else:
        summary_lines.extend(["", "# Project Genie browser actions: unsupported for this action space."])
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    metadata["action_npy"] = npy_path.name
    metadata["action_npy_sha256"] = _sha256(npy_path)
    metadata["action_npz"] = npz_path.name
    metadata["action_npz_sha256"] = _sha256(npz_path)
    metadata["actions_txt"] = summary_path.name
    metadata["actions_txt_sha256"] = _sha256(summary_path)
    metadata["project_genie_actions"] = actions_path.name
    metadata["project_genie_actions_sha256"] = _sha256(actions_path)
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_actions_jsonl(path: str | Path) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for lineno, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            action = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrajectoryError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(action, dict) or action.get("op") not in {"hold", "wait", "press"}:
            raise TrajectoryError(f"{path}:{lineno}: invalid browser action")
        actions.append(action)
    return actions


# ── Backend-neutral sampling ────────────────────────────────────────────────
# Model adapters run on their own native timelines (e.g. 129 frames at 12 fps,
# 161 poses at 16 fps).  These helpers are the only sanctioned ways to move a
# semantic trajectory onto such a timeline, so every backend shares one
# integration/hold convention.


def integrate_rate_channel(
    trajectory: Mapping[str, Any], channel: str, fps: float, frames: int
) -> np.ndarray:
    """Integrate a piecewise-constant rate channel over each native frame interval.

    Frame ``k`` covers ``[k / fps, (k + 1) / fps)``; the result is the channel's
    integral over that interval (e.g. degrees of yaw for ``yaw_rate_deg_s``).
    """
    increments = np.zeros(frames, dtype=np.float32)
    for frame in range(frames):
        start = frame / fps
        end = (frame + 1) / fps
        for segment in trajectory["segments"]:
            overlap = max(0.0, min(end, float(segment["end_sec"])) - max(start, float(segment["start_sec"])))
            increments[frame] += overlap * float(segment["values"].get(channel, 0.0))
    return increments


def integrated_total(trajectory: Mapping[str, Any], channel: str) -> float:
    """Exact integral of a piecewise-constant rate channel over the whole trajectory."""
    return float(
        sum(
            (float(segment["end_sec"]) - float(segment["start_sec"])) * float(segment["values"].get(channel, 0.0))
            for segment in trajectory["segments"]
        )
    )


def sample_held_values(
    compiled_npz: str | Path, fps: float, frames: int, *, end_sec: float
) -> tuple[np.ndarray, list[str]]:
    """Zero-order-hold the compiled control values at ``k / fps``; zero after ``end_sec``."""
    data = np.load(Path(compiled_npz), allow_pickle=False)
    timestamps = np.asarray(data["timestamps_sec"], dtype=np.float64)
    values = np.asarray(data["values"], dtype=np.float32)
    indices = np.searchsorted(timestamps, np.arange(frames) / fps, side="right") - 1
    indices = np.clip(indices, 0, len(timestamps) - 1)
    sampled = values[indices]
    sampled[np.arange(frames) / fps >= end_sec] = 0.0
    return sampled, [str(value) for value in data["channel_names"].tolist()]
