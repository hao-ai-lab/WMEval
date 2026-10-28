#!/usr/bin/env python3
"""Resumable computer-use controller for the hosted Project Genie UI.

This deliberately uses the public browser UI and a dedicated Chrome profile;
it does not call or reverse-engineer private endpoints. Authentication and
subscription/age confirmation remain user-controlled. Upload-policy
acknowledgement is automated only with explicit CLI opt-in. UI actions are selected
from accessible roles/labels first, with coordinate fallback disabled by
default. Every run writes screenshots, an action timing trace, hashes, and a
receipt suitable for an evaluation audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import parse_qs, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.trajectories import TrajectoryError, load_actions_jsonl
from evaluation.project_genie_alignment import (
    NativeAlignmentError,
    alignment_receipt_fields,
    align_native_video,
    update_receipt_with_alignment,
)

GENIE_URL = "https://labs.google/fx/projectgenie"
DEFAULT_CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
DEFAULT_PROFILE = REPO_ROOT / ".local" / "project-genie-profile"


class UserActionRequired(RuntimeError):
    """The browser needs a user-owned login, consent, or entitlement action."""


class InvalidConfiguration(RuntimeError):
    """The generated world does not match the requested evaluation settings."""


def world_view_from_url(url: str) -> str | None:
    """Return the camera perspective encoded by a live Project Genie URL."""
    parsed = urlsplit(url)
    if not parsed.path.endswith("/tools/projectgenie/w"):
        return None
    value = parse_qs(parsed.query).get("use3p", [None])[0]
    if value == "true":
        return "third_person"
    if value == "false":
        return "first_person"
    return None


def world_prompt_from_url(url: str) -> str | None:
    """Return the environment prompt encoded by a live Project Genie URL."""
    parsed = urlsplit(url)
    if not parsed.path.endswith("/tools/projectgenie/w"):
        return None
    value = parse_qs(parsed.query).get("w", [None])[0]
    return str(value) if value is not None else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_artifact_status(
    *,
    has_observed_video: bool,
    has_turn_calibration: bool,
    closed_loop_yaw: bool,
    trajectory_claim_scope: str,
) -> str:
    action_timing_is_declared = (
        has_turn_calibration
        or closed_loop_yaw
        or trajectory_claim_scope == "reconstructed_nominal_timing"
    )
    return (
        "action_aligned_capture_ready"
        if has_observed_video and action_timing_is_declared
        else "not_ready"
    )


def recover_action_checkpoint(
    output: Path,
    actions: list[dict[str, Any]],
    capture_fps: float,
    *,
    closed_loop_yaw: bool = False,
) -> dict[str, Any] | None:
    """Recover a completed replay after interruption during download.

    Every action and captured JPEG is revalidated before the replay is skipped.
    Closed-loop trajectories additionally require their recorded departure and
    visual-return gates to be present and validated.
    """
    trace_path = output / "action_trace.json"
    manifest_path = output / "observed_frames.json"
    video_path = output / "observed.mp4"
    if not all(path.is_file() for path in (trace_path, manifest_path, video_path)):
        return None
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(trace, list) or len(trace) != len(actions):
        return None
    for expected, actual in zip(actions, trace, strict=True):
        if not isinstance(actual, dict) or actual.get("op") != expected.get("op"):
            return None
        if list(actual.get("keys", [])) != list(expected.get("keys", [])):
            return None
        if expected.get("op") == "turn_until_return":
            duration_ms = int(actual.get("duration_ms", -1))
            if not (
                closed_loop_yaw
                and actual.get("visual_return_status") == "validated"
                and actual.get("departure_observed") is True
                and int(expected.get("min_duration_ms", 0))
                <= duration_ms
                <= int(expected.get("max_duration_ms", -1))
            ):
                return None
        elif int(actual.get("duration_ms", -1)) != int(expected.get("ms", 0)):
            return None
        if actual.get("key") != expected.get("key"):
            return None
    frames = manifest.get("frames") if isinstance(manifest, dict) else None
    if not isinstance(frames, list) or not frames:
        return None
    frame_root = (output / "observed_frames").resolve()
    for index, row in enumerate(frames):
        if not isinstance(row, dict) or row.get("index") != index:
            return None
        try:
            frame = Path(str(row["path"])).resolve()
            frame.relative_to(frame_root)
        except (KeyError, ValueError):
            return None
        if not frame.is_file() or sha256(frame) != row.get("sha256"):
            return None
    return {
        "observed_frames": str(manifest_path),
        "observed_frames_sha256": sha256(manifest_path),
        "observed_frame_count": len(frames),
        "observed_capture_fps": capture_fps,
        "observed_duration_sec": float(frames[-1].get("elapsed_sec", 0.0)),
        "observed_video": str(video_path),
        "observed_video_sha256": sha256(video_path),
        "observed_timeline_alignment": "action_replay_start",
        "action_trace": str(trace_path),
        "action_trace_sha256": sha256(trace_path),
    }


def recover_nominal_action_checkpoint(
    output: Path, actions: list[dict[str, Any]], capture_fps: float
) -> dict[str, Any] | None:
    """Backward-compatible nominal-only recovery entry point."""
    return recover_action_checkpoint(output, actions, capture_fps)
def load_turn_calibration(
    path: Path, *, expected_view: str, allow_view_mismatch: bool = False
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "validated_periodicity":
        raise InvalidConfiguration(
            f"{path}: turn calibration must have status validated_periodicity"
        )
    if report.get("claim_scope") != "visual_yaw_cycle_timing":
        raise InvalidConfiguration(f"{path}: unsupported calibration claim scope")
    if report.get("view") != expected_view and not allow_view_mismatch:
        raise InvalidConfiguration(
            f"{path}: calibration view {report.get('view')!r} does not match {expected_view!r}"
        )
    key = report.get("key")
    period = report.get("full_turn_duration_sec")
    if not isinstance(key, str) or not key.startswith("Arrow"):
        raise InvalidConfiguration(f"{path}: calibration key is missing or invalid")
    if isinstance(period, bool) or not isinstance(period, (int, float)) or not 0.1 < period < 60:
        raise InvalidConfiguration(f"{path}: full_turn_duration_sec is missing or invalid")
    return report


def apply_turn_calibration(
    actions: Sequence[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    allow_opposite_key: bool = False,
) -> list[dict[str, Any]]:
    """Replace nominal yaw-key durations while preserving the semantic angle.

    The semantic compiler writes ``target_yaw_deg`` into yaw operations.  This
    function converts that target through one frozen, independently validated
    key period and adjusts idle time to retain the nominal evaluation horizon
    whenever the calibrated turn fits within it.
    """
    result = [dict(action) for action in actions]
    nominal_total_ms = sum(int(action.get("ms", 0)) for action in result)
    calibrated_key = str(calibration["key"])
    full_turn_ms = float(calibration["full_turn_duration_sec"]) * 1000.0
    calibrated_count = 0
    for action in result:
        if "target_yaw_deg" not in action:
            continue
        if action.get("op") != "hold":
            raise InvalidConfiguration("target_yaw_deg is only valid on a hold operation")
        keys = [str(key) for key in action.get("keys", [])]
        opposite_key = (
            len(keys) == 1
            and {keys[0], calibrated_key} == {"ArrowLeft", "ArrowRight"}
        )
        if keys != [calibrated_key] and not (allow_opposite_key and opposite_key):
            raise InvalidConfiguration(
                f"turn action keys {keys!r} do not match calibration key {calibrated_key!r}"
            )
        target = float(action["target_yaw_deg"])
        if target == 0:
            raise InvalidConfiguration("calibrated yaw target cannot be zero")
        action_key = keys[0]
        if (action_key == "ArrowLeft" and target > 0) or (
            action_key == "ArrowRight" and target < 0
        ):
            raise InvalidConfiguration(
                f"yaw target {target} has the wrong sign for {action_key}"
            )
        nominal_ms = int(action["ms"])
        action["nominal_ms"] = nominal_ms
        action["ms"] = int(round(full_turn_ms * abs(target) / 360.0))
        action["calibration_status"] = (
            "opposite_key_symmetric_period_assumption"
            if opposite_key
            else "validated_periodicity"
        )
        if opposite_key:
            action["calibration_source_key"] = calibrated_key
        calibrated_count += 1
    if calibrated_count == 0:
        raise InvalidConfiguration("action program has no semantic target_yaw_deg operation")

    calibrated_total_ms = sum(int(action.get("ms", 0)) for action in result)
    difference = nominal_total_ms - calibrated_total_ms
    if difference > 0:
        waits = [action for action in result if action.get("op") == "wait"]
        if waits:
            waits[-1]["ms"] = int(waits[-1].get("ms", 0)) + difference
        else:
            result.append({"op": "wait", "ms": difference, "reason": "preserve_nominal_horizon"})
    elif difference < 0:
        remaining = -difference
        for action in reversed(result):
            if action.get("op") != "wait" or remaining <= 0:
                continue
            removable = min(int(action.get("ms", 0)), remaining)
            action["ms"] = int(action.get("ms", 0)) - removable
            remaining -= removable
        result = [action for action in result if action.get("op") != "wait" or action.get("ms")]
    return result


def apply_closed_loop_yaw(actions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compile a full-yaw semantic action into a visual return controller."""
    result: list[dict[str, Any]] = []
    converted = 0
    for source in actions:
        action = dict(source)
        if "target_yaw_deg" not in action:
            result.append(action)
            continue
        target = float(action["target_yaw_deg"])
        keys = [str(key) for key in action.get("keys", [])]
        if action.get("op") != "hold" or len(keys) != 1:
            raise InvalidConfiguration("closed-loop yaw requires one held arrow key")
        if abs(target) < 330.0:
            raise InvalidConfiguration(
                "closed-loop visual return currently supports full 360-degree yaw only"
            )
        key = keys[0]
        if (key == "ArrowLeft" and target > 0) or (key == "ArrowRight" and target < 0):
            raise InvalidConfiguration(f"yaw target {target} has the wrong sign for {key}")
        result.append(
            {
                "op": "turn_until_return",
                "keys": keys,
                "target_yaw_deg": target,
                "nominal_horizon_ms": int(action.get("ms", 0)),
                "max_duration_ms": 50_000,
                "min_duration_ms": 2_000,
                "departure_match_max": 0.015,
                "return_match_min": 0.030,
                "return_window_prominence_min": 0.015,
                "controller_version": "visual_return_sift_v2",
            }
        )
        converted += 1
    if converted != 1:
        raise InvalidConfiguration(
            f"closed-loop yaw requires exactly one semantic turn; found {converted}"
        )
    return result


class VisualReturnDetector:
    """Match revisited views despite moderate generative warp and translation."""

    def __init__(self, reference: Path) -> None:
        import cv2

        self.cv2 = cv2
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher()
        _, self.reference_descriptors = self._describe(reference)
        if self.reference_descriptors is None or len(self.reference_descriptors) < 100:
            raise ValueError(f"not enough reference features in {reference}")

    def _describe(self, path: Path) -> tuple[Any, Any]:
        image = self.cv2.imread(str(path), self.cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not read captured viewport {path}")
        height, width = image.shape
        crop = image[
            int(0.09 * height) : int(0.78 * height),
            int(0.05 * width) : int(0.95 * width),
        ]
        return self.sift.detectAndCompute(crop, None)

    def score(self, path: Path) -> float:
        _, descriptors = self._describe(path)
        if descriptors is None or len(descriptors) < 2:
            return 0.0
        pairs = self.matcher.knnMatch(self.reference_descriptors, descriptors, k=2)
        good = sum(first.distance < 0.72 * second.distance for first, second in pairs)
        return good / len(self.reference_descriptors)


def visual_return_feature(path: Path) -> Any:
    """Legacy normalized-pixel descriptor retained for offline diagnostics."""
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read captured viewport {path}")
    height, width = image.shape
    crop = image[
        int(0.09 * height) : int(0.78 * height),
        int(0.05 * width) : int(0.95 * width),
    ]
    small = cv2.resize(crop, (192, 108), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= float(small.mean())
    norm = float(np.linalg.norm(small))
    if norm <= 1e-9:
        raise ValueError(f"captured viewport is uninformative: {path}")
    return (small / norm).ravel()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_for_port(port: int, timeout_sec: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if port_is_open(port):
            return
        time.sleep(0.2)
    raise RuntimeError(f"Chrome DevTools port {port} did not open")


def launch_chrome(chrome: Path, profile: Path, port: int) -> subprocess.Popen[bytes]:
    if not chrome.is_file():
        raise FileNotFoundError(f"Chrome executable not found: {chrome}")
    profile.mkdir(parents=True, exist_ok=True)
    browser_args = [
        f"--remote-debugging-port={port}",
        "--remote-debugging-address=127.0.0.1",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--disable-default-apps",
        GENIE_URL,
    ]
    headless = os.environ.get("WMEVAL_GENIE_HEADLESS") == "1"
    if headless:
        # No visible window; the viewport matches the headed MacBook capture size.
        browser_args[:0] = ["--headless=new", "--window-size=1728,1000", "--use-angle=metal"]
    command = (
        ["open", "-na", "Google Chrome", "--args", *browser_args]
        if platform.system() == "Darwin" and not headless
        else [str(chrome), *browser_args]
    )
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def playwright_module() -> Any:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is missing; install the repository's requirements/macos.lock"
        ) from exc
    return sync_playwright


def visible_controls(page: Any, limit: int = 80) -> list[dict[str, str]]:
    script = """
    (limit) => {
      const nodes = [...document.querySelectorAll(
        'button, input, textarea, [role="button"], [role="textbox"], a, canvas'
      )];
      return nodes.filter((node) => {
        const r = node.getBoundingClientRect();
        const s = getComputedStyle(node);
        return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none';
      }).slice(0, limit).map((node) => {
        const r = node.getBoundingClientRect();
        return {
        tag: node.tagName.toLowerCase(),
        role: node.getAttribute('role') || '',
        name: node.getAttribute('aria-label') || node.getAttribute('placeholder') ||
              node.getAttribute('name') || (node.innerText || '').trim().slice(0, 160),
        type: node.getAttribute('type') || '',
        box: {x: r.x, y: r.y, width: r.width, height: r.height}
      }});
    }
    """
    return list(page.evaluate(script, limit))


def text_target_diagnostics(page: Any, text: str) -> list[dict[str, Any]]:
    script = """
    (wanted) => [...document.querySelectorAll('*')]
      .filter((node) => (node.innerText || '').trim().toLowerCase() === wanted.toLowerCase())
      .slice(0, 12)
      .map((node) => {
        const r = node.getBoundingClientRect();
        const s = getComputedStyle(node);
        const chain = [];
        let current = node;
        for (let i = 0; current && i < 5; i++, current = current.parentElement) {
          chain.push({
            tag: current.tagName.toLowerCase(),
            className: String(current.className || '').slice(0, 240),
            role: current.getAttribute('role') || '',
            tabindex: current.getAttribute('tabindex') || '',
            ariaLabel: current.getAttribute('aria-label') || ''
          });
        }
        return {
          tag: node.tagName.toLowerCase(),
          className: String(node.className || '').slice(0, 240),
          cursor: s.cursor,
          pointerEvents: s.pointerEvents,
          box: {x: r.x, y: r.y, width: r.width, height: r.height},
          ancestors: chain
        };
      });
    """
    return list(page.evaluate(script, text))


@dataclass
class RunRecorder:
    output_dir: Path
    receipt: dict[str, Any]

    def checkpoint(self, page: Any, stage: str, **extra: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        screenshot = self.output_dir / f"{stage}.png"
        page.screenshot(path=str(screenshot), full_page=True)
        self.receipt.update(
            {
                "stage": stage,
                "updated_at": utc_now(),
                "page_url": page.url,
                "page_title": page.title(),
            }
        )
        self.receipt.setdefault("screenshots", {})[stage] = {
            "path": str(screenshot),
            "sha256": sha256(screenshot),
        }
        self.receipt.update(extra)
        (self.output_dir / "receipt.json").write_text(
            json.dumps(self.receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def fail(self, page: Any, exc: BaseException) -> None:
        if isinstance(exc, UserActionRequired):
            status = "user_action_required"
        elif isinstance(exc, InvalidConfiguration):
            status = "invalid_configuration"
        else:
            status = "error"
        self.receipt["status"] = status
        self.receipt["error"] = f"{type(exc).__name__}: {exc}"
        self.receipt["visible_controls"] = visible_controls(page)
        self.checkpoint(page, "blocked")


class GenieAgent:
    def __init__(
        self,
        page: Any,
        recorder: RunRecorder,
        *,
        timeout_ms: int,
        allow_canvas_entry: bool = False,
        auto_accept_upload_policy: bool = False,
        filesystem_download_dir: Path | None = None,
        filesystem_download_grace_ms: int = 7_500,
    ) -> None:
        self.page = page
        self.recorder = recorder
        self.timeout_ms = timeout_ms
        self.allow_canvas_entry = allow_canvas_entry
        self.auto_accept_upload_policy = auto_accept_upload_policy
        self.filesystem_download_dir = (
            filesystem_download_dir.resolve()
            if filesystem_download_dir is not None
            else None
        )
        self.filesystem_download_grace_ms = filesystem_download_grace_ms
        self.last_capture_manifest: Path | None = None
        self.last_download_attempts: list[dict[str, Any]] = []

    def _filesystem_download_snapshot(self) -> dict[Path, tuple[int, int]]:
        directory = self.filesystem_download_dir
        if directory is None or not directory.is_dir():
            return {}
        snapshot: dict[Path, tuple[int, int]] = {}
        for path in directory.glob("Genie*.mp4"):
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[path.resolve()] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    def _wait_for_filesystem_download(
        self,
        baseline: dict[Path, tuple[int, int]],
    ) -> Path | None:
        """Recover a late Chrome download whose Playwright event was missed."""
        if self.filesystem_download_dir is None:
            return None
        deadline = time.monotonic() + self.filesystem_download_grace_ms / 1000.0
        stable: dict[Path, int] = {}
        while time.monotonic() < deadline:
            current = self._filesystem_download_snapshot()
            candidates = [
                path
                for path, signature in current.items()
                if signature[1] > 0 and baseline.get(path) != signature
            ]
            candidates.sort(key=lambda path: current[path][0], reverse=True)
            for path in candidates:
                size = current[path][1]
                if stable.get(path) == size:
                    return path
                stable[path] = size
            self.page.wait_for_timeout(250)
        return None

    def _first_visible(self, locators: Iterable[Any]) -> Any | None:
        for locator in locators:
            try:
                count = min(locator.count(), 10)
                for index in range(count):
                    item = locator.nth(index)
                    if item.is_visible():
                        return item
            except Exception:
                continue
        return None

    def _named(self, role: str, names: Sequence[str]) -> Any | None:
        import re

        return self._first_visible(
            self.page.get_by_role(role, name=re.compile(name, re.I)) for name in names
        )

    def click_named(self, role: str, names: Sequence[str], purpose: str) -> None:
        locator = self._named(role, names)
        if locator is None:
            raise UserActionRequired(
                f"Could not locate {purpose}; inspect blocked.png and receipt.json"
            )
        locator.click(timeout=self.timeout_ms)

    def enter_create_flow(self) -> None:
        import re

        self.page.goto(GENIE_URL, wait_until="domcontentloaded", timeout=self.timeout_ms)
        self.page.wait_for_timeout(1500)
        # Project Genie currently shows product-announcement modals before the
        # landing controls. Only dismiss a specifically labelled modal close;
        # do not click a generic page-level Close button.
        modal_close = self._named("button", (r"^close this modal$",))
        if modal_close is not None:
            modal_close.click()
            self.page.wait_for_timeout(500)
        if self._named("button", (r"create with project genie", r"create", r"get started")):
            self.click_named(
                "button", (r"create with project genie", r"create", r"get started"), "create entry"
            )
            self.page.wait_for_timeout(1000)
        elif self._named("link", (r"create with project genie", r"launch genie", r"create")):
            self.click_named(
                "link", (r"create with project genie", r"launch genie", r"create"), "create entry"
            )
            self.page.wait_for_timeout(1000)
        else:
            # The authenticated gallery currently renders the central entry as
            # visible text inside a custom/canvas-backed card, without button
            # or link semantics. Keep this fallback exact so unrelated gallery
            # cards are never selected.
            create_text = self._first_visible(
                [self.page.get_by_text(re.compile(r"^create your own$", re.I), exact=True)]
            )
            if self.allow_canvas_entry:
                # The signed-in gallery is currently a single full-page canvas
                # whose central sphere is the documented "Create your own"
                # entry. Use a viewport-relative center only when explicitly
                # enabled, and checkpoint immediately after the transition.
                canvas = self._first_visible([self.page.locator("canvas")])
                box = canvas.bounding_box() if canvas is not None else None
                if not box or box["width"] < 640 or box["height"] < 360:
                    raise UserActionRequired("Could not identify the full-page Project Genie gallery canvas")
                canvas.click(position={"x": box["width"] / 2, "y": box["height"] / 2})
                self.page.wait_for_timeout(5000)
                # Some WebGL builds treat the first pointer activation as
                # focus/intro only. If no creation controls materialized, send
                # Enter to the now-focused central card and wait once more.
                has_form_control = self._first_visible(
                    [self.page.locator('input[type="file"]'), self.page.get_by_role("textbox")]
                )
                if has_form_control is None:
                    self.page.keyboard.press("Enter")
                    self.page.wait_for_timeout(5000)
                    has_form_control = self._first_visible(
                        [self.page.locator('input[type="file"]'), self.page.get_by_role("textbox")]
                    )
                if has_form_control is None:
                    canvas.dblclick(position={"x": box["width"] / 2, "y": box["height"] / 2})
                    self.page.wait_for_timeout(5000)
            elif create_text is not None:
                create_text.click(timeout=self.timeout_ms)
                self.page.wait_for_timeout(1000)
        controls = visible_controls(self.page)
        joined = " ".join(item.get("name", "") for item in controls).lower()
        if "sign in" in joined or "subscribe" in joined or "upgrade" in joined:
            raise UserActionRequired(
                "Project Genie requires user-owned sign-in/entitlement confirmation in the open browser"
            )
        self.recorder.checkpoint(self.page, "create_flow")

    def upload_conditioning_frame(self, first_frame: Path) -> None:
        file_inputs = self.page.locator('input[type="file"]')
        upload = file_inputs.first if file_inputs.count() else self._first_visible(
            [
                self.page.get_by_label("Upload image", exact=False),
                self.page.get_by_label("Image", exact=False),
            ]
        )
        if upload is None:
            button = self._named("button", (r"upload", r"add image", r"image", r"^photo$"))
            if button is None:
                raise UserActionRequired("Could not locate the conditioning-image upload control")
            with self.page.expect_file_chooser(timeout=self.timeout_ms) as chooser:
                button.click()
            chooser.value.set_files(str(first_frame))
        else:
            upload.set_input_files(str(first_frame))
        self.page.wait_for_timeout(1000)
        self.recorder.checkpoint(self.page, "frame_uploaded")

    def require_user_consent_if_present(self) -> bool:
        """Handle only the exact upload-policy dialog, or fail for user action."""
        deadline = time.monotonic() + (5.0 if self.auto_accept_upload_policy else 0.0)
        while True:
            policy = self._first_visible(
                [
                    self.page.get_by_role(
                        "link", name="Prohibited Use Policy", exact=True
                    )
                ]
            )
            agreement = self._first_visible(
                [self.page.get_by_role("button", name="I agree", exact=True)]
            )
            if policy is not None and agreement is not None:
                if self.auto_accept_upload_policy:
                    agreement.click(timeout=self.timeout_ms)
                    self.page.wait_for_timeout(500)
                    if agreement.is_visible():
                        raise UserActionRequired(
                            "Project Genie upload-policy dialog remained visible after "
                            "the authorized automated I agree click"
                        )
                    event = {
                        "accepted_at": utc_now(),
                        "button_label": "I agree",
                        "mode": "automated_explicit_cli_opt_in",
                        "policy_label": "Prohibited Use Policy",
                    }
                    self.recorder.receipt.setdefault(
                        "upload_policy_acceptances", []
                    ).append(event)
                    self.recorder.checkpoint(
                        self.page,
                        "upload_policy_auto_accepted",
                        upload_policy_acceptance_mode="automated_explicit_cli_opt_in",
                    )
                    return True
                raise UserActionRequired(
                    "Google's upload-rights / Prohibited Use Policy notice requires the user "
                    "to review and click I agree in the open browser"
                )
            if time.monotonic() >= deadline:
                return False
            self.page.wait_for_timeout(250)

    def fill_prompt(self, text: str, *, character: bool = False) -> None:
        import re

        names = (r"character", r"who") if character else (r"environment", r"world", r"describe")
        locators = []
        for name in names:
            pattern = re.compile(name, re.I)
            locators.extend(
                [self.page.get_by_label(pattern), self.page.get_by_placeholder(pattern)]
            )
        target = self._first_visible(locators)
        if target is None:
            textboxes = self.page.get_by_role("textbox")
            visible = [textboxes.nth(i) for i in range(min(textboxes.count(), 8)) if textboxes.nth(i).is_visible()]
            index = 1 if character else 0
            if len(visible) <= index:
                raise UserActionRequired("Could not locate Project Genie prompt fields")
            target = visible[index]
        target.fill(text)

    def choose_view(self, view: str) -> None:
        import re

        desired = r"first.person" if view.startswith("first") else r"third.person"
        desired_button = self._first_visible(
            [self.page.locator("button").filter(has_text=re.compile(desired, re.I))]
        )
        if desired_button is not None:
            return
        toggle = self._first_visible(
            [
                self.page.locator("button").filter(
                    has_text=re.compile(r"(?:first|third).person", re.I)
                )
            ]
        )
        if toggle is None:
            raise UserActionRequired(f"Could not locate the {view} perspective control")
        toggle.click()
        self.page.wait_for_timeout(500)
        # The current build toggles the button label in place. Some builds may
        # instead expose an option through a radio/menu item.
        verified = self._first_visible(
            [self.page.locator("button").filter(has_text=re.compile(desired, re.I))]
        )
        if verified is not None:
            return
        locator = self._named("radio", (desired,)) or self._named("menuitem", (desired,))
        if locator is None:
            raise UserActionRequired(f"Could not select the {view} perspective")
        locator.click()
        self.page.wait_for_timeout(300)
        verified = self._first_visible(
            [self.page.locator("button").filter(has_text=re.compile(desired, re.I))]
        )
        if verified is None:
            raise UserActionRequired(f"Project Genie did not retain the requested {view} perspective")

    def generate_world(self) -> None:
        self.recorder.checkpoint(self.page, "prompts_ready")
        self.click_named(
            "button", (r"^create world", r"^create sketch", r"^create$", r"^generate$"),
            "Create world/sketch",
        )
        self.recorder.checkpoint(self.page, "generation_requested")
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        while time.monotonic() < deadline:
            too_small = self._first_visible(
                [self.page.get_by_text("This image is too small", exact=False)]
            )
            if too_small is not None:
                raise InvalidConfiguration(
                    "Project Genie rejected the conditioning image: minimum "
                    "dimensions are 1280x704"
                )
            if world_view_from_url(self.page.url) is not None:
                return
            continuation = self._named(
                "button",
                (
                    r"^continue generating$",
                    r"^generate world$",
                    r"^continue$",
                    r"^explore your world$",
                ),
            )
            if continuation is not None:
                self.recorder.checkpoint(self.page, "sketch_ready")
                continuation.click()
                self.recorder.checkpoint(self.page, "world_requested")
                return
            self.page.wait_for_timeout(1000)
        raise UserActionRequired("Timed out waiting for Project Genie to start the world")

    def wait_for_world(self, scene_wait_ms: int, expected_view: str) -> Any:
        deadline = time.monotonic() + scene_wait_ms / 1000.0
        target = None
        while time.monotonic() < deadline:
            actual_view = world_view_from_url(self.page.url)
            active_control = self._named("button", (r"ESC$",))
            creating = self._first_visible(
                [self.page.get_by_text("Creating your world", exact=False)]
            )
            if creating is not None:
                active_control = None
            if actual_view is not None and active_control is not None:
                candidates = self.page.locator("canvas, video")
                visible: list[tuple[float, Any]] = []
                for index in range(min(candidates.count(), 12)):
                    candidate = candidates.nth(index)
                    if not candidate.is_visible():
                        continue
                    box = candidate.bounding_box()
                    if box:
                        visible.append((float(box["width"] * box["height"]), candidate))
                target = max(visible, key=lambda item: item[0])[1] if visible else None
            if target is not None and actual_view is not None:
                box = target.bounding_box()
                if box and box["width"] >= 320 and box["height"] >= 180:
                    if actual_view != expected_view:
                        raise InvalidConfiguration(
                            f"requested {expected_view}, but generated URL encodes {actual_view}: "
                            f"{self.page.url}"
                        )
                    break
            target = None
            self.page.wait_for_timeout(1000)
        if target is None:
            raise UserActionRequired("Timed out waiting for the interactive world viewport")
        # First-run instructional overlays can sit above the renderer and
        # intercept pointer clicks. Give the renderer DOM focus directly so
        # keyboard events still reach the world without clicking through or
        # dismissing user-facing UI by coordinates.
        target.evaluate(
            "node => { node.setAttribute('tabindex', '0'); node.focus({preventScroll: true}); }"
        )
        self.recorder.checkpoint(self.page, "world_ready")
        return target

    def replay(
        self,
        actions: Sequence[dict[str, Any]],
        *,
        capture_target: Any | None = None,
        capture_dir: Path | None = None,
        capture_fps: float = 15.0,
    ) -> list[dict[str, Any]]:
        if capture_target is not None and capture_dir is None:
            raise ValueError("capture_dir is required when capture_target is supplied")
        if capture_fps <= 0:
            raise ValueError("capture_fps must be positive")
        trace: list[dict[str, Any]] = []
        capture_rows: list[dict[str, Any]] = []
        start = time.monotonic()
        next_capture = 0.0
        held: set[str] = set()

        def capture_due(*, force: bool = False) -> dict[str, Any] | None:
            nonlocal next_capture
            if capture_target is None or capture_dir is None:
                return None
            elapsed = time.monotonic() - start
            if not force and elapsed + 1e-9 < next_capture:
                return None
            capture_dir.mkdir(parents=True, exist_ok=True)
            path = capture_dir / f"frame_{len(capture_rows):06d}.jpg"
            capture_target.screenshot(
                path=str(path), type="jpeg", quality=88, scale="css"
            )
            row = {
                "index": len(capture_rows),
                "elapsed_sec": elapsed,
                "path": str(path),
                "sha256": sha256(path),
            }
            capture_rows.append(row)
            next_capture += 1.0 / capture_fps
            captured = time.monotonic() - start
            while next_capture <= captured:
                next_capture += 1.0 / capture_fps
            return row

        def wait_with_capture(duration_ms: int) -> None:
            if capture_target is None:
                self.page.wait_for_timeout(duration_ms)
                return
            deadline = time.monotonic() + duration_ms / 1000.0
            while True:
                capture_due()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self.page.wait_for_timeout(max(1, min(10, int(remaining * 1000))))

        try:
            capture_due(force=True)
            # Frame zero is the pre-action observation. Encoding its screenshot
            # must not consume any of the requested action timeline.
            start = time.monotonic()
            if capture_rows:
                capture_rows[0]["elapsed_sec"] = 0.0
            next_capture = 1.0 / capture_fps
            # Only closed-loop yaw needs the start-view matcher; building it for
            # every trajectory fails on low-texture scenes (e.g. flat arenas).
            return_detector = (
                VisualReturnDetector(Path(capture_rows[0]["path"]))
                if capture_rows
                and any(str(a.get("op")) == "turn_until_return" for a in actions)
                else None
            )
            for index, action in enumerate(actions):
                op = str(action["op"])
                requested = time.monotonic() - start
                keys = [str(key) for key in action.get("keys", [])]
                if action.get("key"):
                    keys.append(str(action["key"]))
                dispatched = time.monotonic() - start
                action_extra: dict[str, Any] = {}
                if op == "wait":
                    accepted = time.monotonic() - start
                    wait_with_capture(int(action["ms"]))
                elif op == "press":
                    self.page.keyboard.press(keys[0], delay=int(action.get("delay_ms", 0)))
                    accepted = time.monotonic() - start
                elif op == "hold":
                    for key in keys:
                        self.page.keyboard.down(key)
                        held.add(key)
                    accepted = time.monotonic() - start
                    wait_with_capture(int(action["ms"]))
                    for key in reversed(keys):
                        self.page.keyboard.up(key)
                        held.discard(key)
                elif op == "turn_until_return":
                    if capture_target is None or return_detector is None:
                        raise InvalidConfiguration(
                            "closed-loop yaw requires the action-aligned viewport capture"
                        )
                    if len(keys) != 1:
                        raise InvalidConfiguration("closed-loop yaw requires exactly one key")
                    key = keys[0]
                    self.page.keyboard.down(key)
                    held.add(key)
                    accepted = time.monotonic() - start
                    deadline = time.monotonic() + int(action["max_duration_ms"]) / 1000.0
                    departed = False
                    return_row: dict[str, Any] | None = None
                    similarity_window: list[tuple[float, float]] = []
                    while time.monotonic() < deadline:
                        row = capture_due()
                        if row is not None:
                            match_score = return_detector.score(Path(str(row["path"])))
                            row["start_view_sift_match"] = match_score
                            elapsed_turn = float(row["elapsed_sec"]) - accepted
                            similarity_window.append((elapsed_turn, match_score))
                            similarity_window = [
                                item
                                for item in similarity_window
                                if item[0] >= elapsed_turn - 1.0
                            ]
                            if match_score <= float(action["departure_match_max"]):
                                departed = True
                            prominence = match_score - min(
                                value for _, value in similarity_window
                            )
                            if (
                                departed
                                and elapsed_turn >= int(action["min_duration_ms"]) / 1000.0
                                and match_score >= float(action["return_match_min"])
                                and prominence
                                >= float(action["return_window_prominence_min"])
                            ):
                                return_row = row
                                action_extra = {
                                    "controller_version": action["controller_version"],
                                    "visual_return_status": "validated",
                                    "visual_return_elapsed_sec": elapsed_turn,
                                    "visual_return_sift_match": match_score,
                                    "visual_return_prominence": prominence,
                                    "departure_observed": departed,
                                }
                                break
                        self.page.wait_for_timeout(10)
                    self.page.keyboard.up(key)
                    held.discard(key)
                    if return_row is None:
                        raise InvalidConfiguration(
                            "closed-loop yaw did not observe a validated return to the start view"
                        )
                    actual_turn_ms = int(
                        round(float(action_extra["visual_return_elapsed_sec"]) * 1000.0)
                    )
                    action_extra["actual_turn_duration_ms"] = actual_turn_ms
                    action_extra["visual_return_global_elapsed_sec"] = float(
                        return_row["elapsed_sec"]
                    )
                    action_extra["normalization_target_duration_ms"] = int(
                        action.get("nominal_horizon_ms", 0)
                    )
                    # A closed-loop trajectory ends at the validated visual
                    # return.  Waiting to pad the browser session to the paper's
                    # nominal ten-second horizon lets the generative world drift
                    # after key release and changes the semantic endpoint.
                    action_extra["post_turn_wait_ms"] = 0
                else:
                    raise TrajectoryError(f"unsupported browser op: {op}")
                trace.append(
                    {
                        "index": index,
                        "op": op,
                        "keys": keys,
                        "duration_ms": int(
                            action_extra.get("actual_turn_duration_ms", action.get("ms", 0))
                        ),
                        "requested_at_sec": requested,
                        "dispatched_at_sec": dispatched,
                        "accepted_at_sec": accepted,
                        "completed_at_sec": time.monotonic() - start,
                        **(
                            {"target_yaw_deg": float(action["target_yaw_deg"])}
                            if "target_yaw_deg" in action
                            else {}
                        ),
                        **action_extra,
                    }
                )
            capture_due(force=True)
        finally:
            for key in held:
                self.page.keyboard.up(key)
            if capture_rows and capture_dir is not None:
                manifest = capture_dir.parent / "observed_frames.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "timeline_origin": "action_replay_start",
                            "nominal_capture_fps": capture_fps,
                            "frames": capture_rows,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self.last_capture_manifest = manifest
        return trace

    def download(self, timeout_ms: int, *, max_attempts: int = 2) -> Path:
        """Download the native video, retrying explicit Genie export failures."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.last_download_attempts = []
        filesystem_baseline = self._filesystem_download_snapshot()
        for attempt in range(1, max_attempts + 1):
            button_deadline = time.monotonic() + timeout_ms / 1000.0
            download_button = None
            while time.monotonic() < button_deadline:
                download_button = self._named(
                    "button", (r"download", r"save video")
                )
                if download_button is not None:
                    break
                self.page.wait_for_timeout(500)
            if download_button is None:
                raise UserActionRequired(
                    "Timed out waiting for Project Genie's download button"
                )

            downloads: list[Any] = []

            def receive_download(item: Any) -> None:
                downloads.append(item)

            self.page.on("download", receive_download)
            started = time.monotonic()
            explicit_failure = False
            try:
                download_button.click()
                event_deadline = time.monotonic() + timeout_ms / 1000.0
                while time.monotonic() < event_deadline and not downloads:
                    failure = self._first_visible(
                        [self.page.get_by_text("Download failed", exact=True)]
                    )
                    if failure is not None:
                        explicit_failure = True
                        dismiss = self._named("button", (r"^dismiss$",))
                        if dismiss is not None:
                            dismiss.click()
                        break
                    self.page.wait_for_timeout(250)
            finally:
                self.page.remove_listener("download", receive_download)
            attempt_row = {
                "attempt": attempt,
                "elapsed_sec": time.monotonic() - started,
                "status": (
                    "downloaded"
                    if downloads
                    else ("explicit_export_failure" if explicit_failure else "timeout")
                ),
            }
            self.last_download_attempts.append(attempt_row)
            if downloads:
                download = downloads[0]
                suffix = Path(download.suggested_filename).suffix or ".mp4"
                path = self.recorder.output_dir / f"native{suffix}"
                download.save_as(str(path))
                return path
            if explicit_failure:
                late_download = self._wait_for_filesystem_download(
                    filesystem_baseline
                )
                if late_download is not None:
                    path = self.recorder.output_dir / "native.mp4"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(late_download, path)
                    attempt_row["status"] = "downloaded_via_filesystem_fallback"
                    attempt_row["source"] = str(late_download)
                    return path
        statuses = ", ".join(
            str(row["status"]) for row in self.last_download_attempts
        )
        raise UserActionRequired(
            f"Project Genie native download failed after {max_attempts} attempts: {statuses}"
        )


def connect_page(playwright: Any, port: int) -> tuple[Any, Any]:
    browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
    if not browser.contexts:
        raise RuntimeError("Chrome has no browser context")
    context = browser.contexts[0]
    for page in context.pages:
        if "labs.google/fx" in (page.url or ""):
            return browser, page
    return browser, context.new_page()


def encode_observed_video(manifest: Path, output: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the action-aligned viewport capture")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not isinstance(frames, list) or len(frames) < 2:
        raise RuntimeError(f"{manifest}: at least two observed frames are required")
    concat = manifest.with_name("observed_concat.txt")
    lines: list[str] = []
    for index, frame in enumerate(frames[:-1]):
        path = Path(str(frame["path"])).resolve()
        escaped = str(path).replace("'", "'\\''")
        duration = max(
            1e-6,
            float(frames[index + 1]["elapsed_sec"]) - float(frame["elapsed_sec"]),
        )
        lines.extend([f"file '{escaped}'", f"duration {duration:.9f}"])
    final_path = Path(str(frames[-1]["path"])).resolve()
    escaped_final = str(final_path).replace("'", "'\\''")
    lines.extend([f"file '{escaped_final}'", f"file '{escaped_final}'"])
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-vf",
        f"fps={fps:g},scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(command, check=True)


def doctor(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "chrome": str(args.chrome),
        "chrome_exists": args.chrome.is_file(),
        "profile": str(args.profile),
        "cdp_port": args.port,
        "cdp_reachable": port_is_open(args.port),
        "genie_url": GENIE_URL,
    }
    try:
        import importlib.metadata

        report["playwright_version"] = importlib.metadata.version("playwright")
    except Exception as exc:
        report["playwright_error"] = str(exc)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["chrome_exists"] and "playwright_version" in report else 2


def launch_browser(args: argparse.Namespace) -> int:
    if port_is_open(args.port):
        report = {"status": "already_running", "cdp_port": args.port, "profile": str(args.profile)}
    else:
        process = launch_chrome(args.chrome, args.profile, args.port)
        wait_for_port(args.port)
        report = {
            "status": "launched",
            "pid": process.pid,
            "cdp_port": args.port,
            "profile": str(args.profile),
            "url": GENIE_URL,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def download_current(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    receipt_path = output / "receipt.json"
    receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.is_file()
        else {"schema_version": 1, "status": "download_only"}
    )
    sync_playwright = playwright_module()
    with sync_playwright() as playwright:
        _, page = connect_page(playwright, args.port)
        page.bring_to_front()
        recorder = RunRecorder(output, receipt)
        agent = GenieAgent(
            page,
            recorder,
            timeout_ms=args.timeout_ms,
            filesystem_download_dir=Path.home() / "Downloads",
        )
        try:
            native = agent.download(args.download_timeout_ms)
            receipt.update(
                {
                    "native_video": str(native),
                    "native_video_sha256": sha256(native),
                    "native_video_status": "downloaded",
                    "native_video_error": None,
                    "native_download_attempts": agent.last_download_attempts,
                }
            )
            recorder.checkpoint(page, "native_downloaded")
            if receipt.get("trajectory_validation_status") == "closed_loop_visual_phase":
                observed_manifest = output / "observed_frames.json"
                action_trace = output / "action_trace.json"
                alignment = align_native_video(
                    native, observed_manifest, action_trace, output
                )
                updated = update_receipt_with_alignment(receipt_path, alignment)
                receipt.clear()
                receipt.update(updated)
                recorder.checkpoint(page, "complete")
        except BaseException as exc:
            recorder.fail(page, exc)
            raise
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def inspect_browser(args: argparse.Namespace) -> int:
    sync_playwright = playwright_module()
    with sync_playwright() as playwright:
        browser, page = connect_page(playwright, args.port)
        page.bring_to_front()
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        transition_events: list[dict[str, str]] = []
        page.on(
            "console",
            lambda message: transition_events.append(
                {"kind": "console", "type": message.type, "text": message.text[:500]}
            ),
        )
        page.on(
            "pageerror",
            lambda error: transition_events.append(
                {"kind": "pageerror", "type": type(error).__name__, "text": str(error)[:500]}
            ),
        )
        page.on(
            "requestfailed",
            lambda request: transition_events.append(
                {
                    "kind": "requestfailed",
                    "type": request.method,
                    "text": f"{request.url} {request.failure}"[:500],
                }
            ),
        )
        if args.press_escape:
            page.keyboard.press("Escape")
            page.wait_for_timeout(2000)
        if args.enter_create:
            recorder = RunRecorder(output, {"schema_version": 1, "status": "inspection"})
            GenieAgent(
                page,
                recorder,
                timeout_ms=args.timeout_ms,
                allow_canvas_entry=args.allow_canvas_entry,
            ).enter_create_flow()
        if args.open_perspective:
            import re

            modal_close = page.get_by_role("button", name=re.compile(r"^close this modal$", re.I))
            if modal_close.count() and modal_close.first.is_visible():
                modal_close.first.click()
                page.wait_for_timeout(300)
            perspective = page.locator("button").filter(
                has_text=re.compile(r"(?:first|third).person", re.I)
            )
            target = None
            for index in range(min(perspective.count(), 10)):
                if perspective.nth(index).is_visible():
                    target = perspective.nth(index)
                    break
            if target is None:
                raise UserActionRequired("Could not locate the current perspective selector")
            target.click()
            page.wait_for_timeout(500)
        shot = output / "inspect.png"
        page.screenshot(path=str(shot), full_page=True)
        report = {
            "url": page.url,
            "title": page.title(),
            "screenshot": str(shot),
            "visible_controls": visible_controls(page),
            "create_target_diagnostics": text_target_diagnostics(page, "Create your own"),
            "canvas_boxes": page.locator("canvas").evaluate_all(
                "nodes => nodes.map(node => { const r = node.getBoundingClientRect(); "
                "return {x:r.x,y:r.y,width:r.width,height:r.height}; })"
            ),
            "context_pages": [candidate.url for candidate in browser.contexts[0].pages],
            "transition_events": transition_events[-100:],
        }
        (output / "inspect.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def calibrate_turn(args: argparse.Namespace) -> int:
    """Replay a saved world and record a timed camera-turn screenshot series."""
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "backend": "project_genie_web",
        "key": args.key,
        "requested_duration_sec": args.duration_sec,
        "sample_interval_sec": args.sample_interval_sec,
        "view": args.view,
    }
    recorder = RunRecorder(output, receipt)
    sync_playwright = playwright_module()
    with sync_playwright() as playwright:
        _, page = connect_page(playwright, args.port)
        page.bring_to_front()
        agent = GenieAgent(
            page,
            recorder,
            timeout_ms=args.timeout_ms,
            filesystem_download_dir=Path.home() / "Downloads",
        )
        try:
            active = agent._named("button", (r"ESC$",))
            if active is not None and args.restart_world:
                page.keyboard.press("Escape")
                page.wait_for_timeout(1500)
                active = None
            if active is None:
                explore = agent._named("button", (r"explore$", r"revisit world$"))
                if explore is not None:
                    explore.click()
                elif world_view_from_url(page.url) is None:
                    raise UserActionRequired(
                        "Open a completed Project Genie world so its Explore/Revisit control is visible"
                    )
            agent.wait_for_world(args.timeout_ms, args.view)
            frames: list[dict[str, Any]] = []
            started = time.monotonic()
            next_sample = 0.0
            controls_missing_since: float | None = None
            page.keyboard.down(args.key)
            try:
                while True:
                    elapsed = time.monotonic() - started
                    if elapsed >= args.duration_sec:
                        break
                    if elapsed >= next_sample:
                        path = output / f"turn_{len(frames):04d}.png"
                        page.screenshot(path=str(path), full_page=False)
                        frames.append(
                            {
                                "index": len(frames),
                                "elapsed_sec": elapsed,
                                "path": str(path),
                                "sha256": sha256(path),
                            }
                        )
                        next_sample += args.sample_interval_sec
                    active = agent._named("button", (r"ESC$",))
                    if active is None:
                        if controls_missing_since is None:
                            controls_missing_since = time.monotonic()
                        elif time.monotonic() - controls_missing_since >= 5.0:
                            break
                    else:
                        controls_missing_since = None
                    page.wait_for_timeout(100)
            finally:
                page.keyboard.up(args.key)
            actual_duration = time.monotonic() - started
            frames_path = output / "turn_samples.json"
            frames_path.write_text(
                json.dumps(frames, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            recorder.checkpoint(
                page,
                "turn_complete",
                actual_duration_sec=actual_duration,
                sample_count=len(frames),
                turn_samples=str(frames_path),
                turn_samples_sha256=sha256(frames_path),
            )
            receipt.update(
                {
                    "status": "complete",
                    "completed_at": utc_now(),
                }
            )
            if args.download:
                native = agent.download(args.download_timeout_ms)
                receipt.update(
                    {
                        "native_video": str(native),
                        "native_video_sha256": sha256(native),
                        "native_download_attempts": agent.last_download_attempts,
                    }
                )
            recorder.checkpoint(page, "complete")
        except BaseException as exc:
            recorder.fail(page, exc)
            raise
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    first_frame = args.first_frame.resolve()
    actions_file = args.actions.resolve()
    if not first_frame.is_file():
        raise FileNotFoundError(first_frame)
    if not actions_file.is_file():
        raise FileNotFoundError(actions_file)
    actions = load_actions_jsonl(actions_file)
    if not actions:
        raise TrajectoryError(f"{actions_file}: no executable Project Genie actions")
    has_semantic_yaw = any("target_yaw_deg" in action for action in actions)
    turn_calibration: dict[str, Any] | None = None
    calibration_path: Path | None = None
    closed_loop_yaw = bool(args.closed_loop_yaw)
    if closed_loop_yaw and args.turn_calibration:
        raise InvalidConfiguration(
            "choose either --closed-loop-yaw or --turn-calibration, not both"
        )
    if closed_loop_yaw:
        actions = apply_closed_loop_yaw(actions)
    elif args.turn_calibration:
        calibration_path = args.turn_calibration.resolve()
        turn_calibration = load_turn_calibration(
            calibration_path,
            expected_view=args.view,
            allow_view_mismatch=args.allow_cross_view_calibration,
        )
        actions = apply_turn_calibration(
            actions,
            turn_calibration,
            allow_opposite_key=args.allow_opposite_turn_calibration,
        )
    elif has_semantic_yaw:
        raise InvalidConfiguration(
            "semantic yaw actions require --turn-calibration; nominal key-hold time "
            "cannot certify a 45- or 360-degree Project Genie trajectory"
        )
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"{output} is non-empty; pass --resume or choose another run directory")
    existing_receipt: dict[str, Any] = {}
    existing_receipt_path = output / "receipt.json"
    if args.resume and existing_receipt_path.is_file():
        try:
            existing_receipt = json.loads(existing_receipt_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_receipt = {}
    reuse_receipt: dict[str, Any] = {}
    reuse_receipt_path: Path | None = None
    if args.reuse_current_world_from:
        reuse_receipt_path = args.reuse_current_world_from.resolve()
        reuse_receipt = json.loads(reuse_receipt_path.read_text(encoding="utf-8"))
        if not isinstance(reuse_receipt, dict):
            raise InvalidConfiguration(f"{reuse_receipt_path}: expected a receipt object")

    if args.launch and not port_is_open(args.port):
        launch_chrome(args.chrome, args.profile, args.port)
        wait_for_port(args.port)
    elif not port_is_open(args.port):
        raise UserActionRequired(
            f"No Chrome CDP endpoint on port {args.port}; run the launch command first"
        )

    receipt = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "backend": "project_genie_web",
        "backend_url": GENIE_URL,
        "repeat_id": args.repeat_id,
        "seed_control": "unavailable_in_web_ui",
        "first_frame": str(first_frame),
        "first_frame_sha256": sha256(first_frame),
        "actions": str(actions_file),
        "actions_sha256": sha256(actions_file),
        "view": args.view,
        "environment_prompt": args.environment_prompt,
        "character_prompt": args.character_prompt,
        "prompt_provenance_status": args.prompt_provenance_status,
        "trajectory_claim_scope": args.trajectory_claim_scope,
        "upload_policy_handling_mode": (
            "automated_explicit_cli_opt_in"
            if args.auto_accept_upload_policy
            else "manual_if_required"
        ),
        "generation_mode": (
            "revisit_existing_world"
            if args.reuse_current_world_from
            else "new_world_generation"
        ),
        "table_repeat_eligible": not bool(args.reuse_current_world_from),
        "trajectory_validation_status": (
            "closed_loop_visual_phase_pending"
            if closed_loop_yaw
            else (
                "calibrated_visual_periodicity"
                if turn_calibration
                else (
                    "reconstructed_nominal_timing"
                    if args.trajectory_claim_scope == "reconstructed_nominal_timing"
                    else "unstandardized"
                )
            )
        ),
    }
    if turn_calibration is not None and calibration_path is not None:
        receipt["turn_calibration"] = {
            "path": str(calibration_path),
            "sha256": sha256(calibration_path),
            "status": turn_calibration["status"],
            "claim_scope": turn_calibration["claim_scope"],
            "key": turn_calibration["key"],
            "full_turn_duration_sec": turn_calibration["full_turn_duration_sec"],
            "calibration_view": turn_calibration.get("view"),
            "requested_view": args.view,
            "view_match": turn_calibration.get("view") == args.view,
            "cross_view_override": bool(args.allow_cross_view_calibration),
            "opposite_key_symmetric_period_override": bool(
                args.allow_opposite_turn_calibration
            ),
        }
        receipt["calibrated_actions"] = actions
    if existing_receipt:
        for key, expected in {
            "first_frame_sha256": receipt["first_frame_sha256"],
            "actions_sha256": receipt["actions_sha256"],
            "repeat_id": receipt["repeat_id"],
        }.items():
            if existing_receipt.get(key) != expected:
                raise ValueError(
                    f"cannot resume: prior receipt {key}={existing_receipt.get(key)!r}, "
                    f"current input={expected!r}"
                )
        receipt["resumed_from"] = {
            "started_at": existing_receipt.get("started_at"),
            "stage": existing_receipt.get("stage"),
            "status": existing_receipt.get("status"),
            "error": existing_receipt.get("error"),
        }
    if reuse_receipt:
        for key, expected in {
            "first_frame_sha256": receipt["first_frame_sha256"],
            "view": receipt["view"],
            "environment_prompt": receipt["environment_prompt"],
            "character_prompt": receipt["character_prompt"],
        }.items():
            if reuse_receipt.get(key) != expected:
                raise InvalidConfiguration(
                    f"cannot reuse current world: source receipt {key}="
                    f"{reuse_receipt.get(key)!r}, requested={expected!r}"
                )
        assert reuse_receipt_path is not None
        receipt["reused_world_from"] = {
            "receipt": str(reuse_receipt_path),
            "receipt_sha256": sha256(reuse_receipt_path),
            "source_status": reuse_receipt.get("status"),
            "source_native_video_sha256": reuse_receipt.get("native_video_sha256"),
        }
    recovered_checkpoint = None
    needs_post_action_finalize = existing_receipt.get("status") != "complete" or (
        closed_loop_yaw
        and existing_receipt.get("trajectory_validation_status")
        == "closed_loop_visual_phase_pending"
    )
    if existing_receipt and needs_post_action_finalize:
        may_recover = (
            args.trajectory_claim_scope == "reconstructed_nominal_timing"
            or turn_calibration is not None
            or closed_loop_yaw
        )
        if may_recover:
            recovered_checkpoint = recover_action_checkpoint(
                output,
                actions,
                args.capture_fps,
                closed_loop_yaw=closed_loop_yaw,
            )
    if recovered_checkpoint is not None and args.allow_missing_native:
        receipt.update(recovered_checkpoint)
        receipt.update(
            {
                "status": "complete",
                "stage": "complete",
                "completed_at": utc_now(),
                "updated_at": utc_now(),
                "resume_mode": "offline_hash_validated_post_action_finalize",
                "native_video_status": "unavailable_after_export_failure",
                "native_video_error": existing_receipt.get("native_video_error")
                or existing_receipt.get("error"),
                "native_download_attempts": existing_receipt.get(
                    "native_download_attempts", []
                ),
                "native_video_timeline_alignment": "unresolved",
                "trajectory_validation_status": (
                    "closed_loop_visual_phase"
                    if closed_loop_yaw
                    else receipt["trajectory_validation_status"]
                ),
                "evaluation_artifact_status": evaluation_artifact_status(
                    has_observed_video=True,
                    has_turn_calibration=turn_calibration is not None,
                    closed_loop_yaw=closed_loop_yaw,
                    trajectory_claim_scope=args.trajectory_claim_scope,
                ),
            }
        )
        for key in (
            "page_title",
            "page_url",
            "screenshots",
            "upload_policy_handling_mode",
            "upload_policy_acceptance_mode",
            "upload_policy_acceptances",
        ):
            if key in existing_receipt:
                receipt[key] = existing_receipt[key]
        output.mkdir(parents=True, exist_ok=True)
        existing_receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    recorder = RunRecorder(output, receipt)
    sync_playwright = playwright_module()
    with sync_playwright() as playwright:
        browser, page = connect_page(playwright, args.port)
        page.bring_to_front()
        agent = GenieAgent(
            page,
            recorder,
            timeout_ms=args.timeout_ms,
            allow_canvas_entry=args.allow_canvas_entry,
            auto_accept_upload_policy=args.auto_accept_upload_policy,
            filesystem_download_dir=Path.home() / "Downloads",
        )
        try:
            live_world_resume = (
                bool(existing_receipt)
                and existing_receipt.get("status") != "complete"
                and world_view_from_url(page.url) == args.view
                and world_prompt_from_url(page.url) == args.environment_prompt
            )
            reuse_current_world = bool(reuse_receipt)
            if recovered_checkpoint is not None:
                actual_view = world_view_from_url(page.url)
                actual_prompt = world_prompt_from_url(page.url)
                if actual_view != args.view or actual_prompt != args.environment_prompt:
                    raise InvalidConfiguration(
                        "post-action resume page does not match the interrupted run"
                    )
                receipt.update(recovered_checkpoint)
                recorder.checkpoint(
                    page,
                    "resumed_after_actions",
                    observed_frame_count=receipt["observed_frame_count"],
                )
            elif reuse_current_world:
                actual_view = world_view_from_url(page.url)
                actual_prompt = world_prompt_from_url(page.url)
                if actual_view != args.view or actual_prompt != args.environment_prompt:
                    raise InvalidConfiguration(
                        "the open Project Genie page is not the world named by the source receipt"
                    )
                active = agent._named("button", (r"ESC$",))
                if active is None:
                    explore = agent._named("button", (r"explore$", r"revisit world$"))
                    if explore is None:
                        raise UserActionRequired(
                            "The source world is open, but no Explore/Revisit control is visible"
                        )
                    explore.click()
                recorder.checkpoint(page, "reusing_current_world")
            consent_resume = (
                bool(existing_receipt)
                and "Prohibited Use Policy notice" in str(existing_receipt.get("error", ""))
                and page.url.endswith("/tools/projectgenie/creation")
            )
            if recovered_checkpoint is not None:
                pass
            elif reuse_current_world:
                pass
            elif live_world_resume:
                recorder.checkpoint(page, "resumed_live_world")
            elif consent_resume:
                # The rights notice is shown for every new upload. Preserve the
                # current browser form and uploaded bytes after the user accepts
                # it; navigating/re-uploading would create an endless notice loop.
                agent.require_user_consent_if_present()
                loaded_image = agent._first_visible(
                    [page.locator('img[role="button"]'), page.get_by_role("button", name="Create world")]
                )
                if loaded_image is None:
                    raise UserActionRequired(
                        "The live form no longer shows the uploaded conditioning image; "
                        "restart the run to upload it again"
                    )
                recorder.checkpoint(page, "resumed_after_upload_consent")
            else:
                agent.enter_create_flow()
                agent.upload_conditioning_frame(first_frame)
                agent.require_user_consent_if_present()
            if recovered_checkpoint is None and not live_world_resume and not reuse_current_world:
                agent.fill_prompt(args.environment_prompt)
                if args.character_prompt:
                    agent.fill_prompt(args.character_prompt, character=True)
                agent.choose_view(args.view)
                agent.generate_world()
            observed_video: Path | None = None
            if recovered_checkpoint is not None:
                observed_video = Path(recovered_checkpoint["observed_video"])
            else:
                viewport = agent.wait_for_world(args.scene_wait_ms, args.view)
                observed_frames_dir = output / "observed_frames"
                trace = agent.replay(
                    actions,
                    capture_target=None if args.no_observed_capture else viewport,
                    capture_dir=None if args.no_observed_capture else observed_frames_dir,
                    capture_fps=args.capture_fps,
                )
                (output / "action_trace.json").write_text(
                    json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                if closed_loop_yaw:
                    if not any(row.get("visual_return_status") == "validated" for row in trace):
                        raise InvalidConfiguration(
                            "closed-loop yaw finished without a validated visual return"
                        )
                    receipt["trajectory_validation_status"] = "closed_loop_visual_phase"
                if agent.last_capture_manifest is not None:
                    observed_video = output / "observed.mp4"
                    encode_observed_video(
                        agent.last_capture_manifest, observed_video, args.capture_fps
                    )
                    observed_payload = json.loads(
                        agent.last_capture_manifest.read_text(encoding="utf-8")
                    )
                    observed_rows = observed_payload["frames"]
                    receipt.update(
                        {
                            "observed_frames": str(agent.last_capture_manifest),
                            "observed_frames_sha256": sha256(agent.last_capture_manifest),
                            "observed_frame_count": len(observed_rows),
                            "observed_capture_fps": args.capture_fps,
                            "observed_duration_sec": (
                                float(observed_rows[-1]["elapsed_sec"])
                                if observed_rows
                                else 0.0
                            ),
                            "observed_video": str(observed_video),
                            "observed_video_sha256": sha256(observed_video),
                            "observed_timeline_alignment": "action_replay_start",
                        }
                    )
                recorder.checkpoint(
                    page,
                    "actions_complete",
                    action_count=len(trace),
                    observed_frame_count=receipt.get("observed_frame_count", 0),
                )
            native: Path | None = None
            native_error: str | None = None
            try:
                native = agent.download(args.download_timeout_ms)
            except Exception as exc:
                if not args.allow_missing_native or observed_video is None:
                    raise
                native_error = f"{type(exc).__name__}: {exc}"
            receipt.update(
                {
                    "status": "complete",
                    "completed_at": utc_now(),
                    "native_video_status": "downloaded" if native else "unavailable_on_revisit",
                    "native_video_error": native_error,
                    "native_download_attempts": agent.last_download_attempts,
                    "action_trace": str(output / "action_trace.json"),
                    "action_trace_sha256": sha256(output / "action_trace.json"),
                    "native_video_timeline_alignment": "unresolved",
                    "evaluation_artifact_status": (
                        evaluation_artifact_status(
                            has_observed_video=observed_video is not None,
                            has_turn_calibration=turn_calibration is not None,
                            closed_loop_yaw=closed_loop_yaw,
                            trajectory_claim_scope=args.trajectory_claim_scope,
                        )
                    ),
                }
            )
            if native is not None:
                receipt.update(
                    {
                        "native_video": str(native),
                        "native_video_sha256": sha256(native),
                    }
                )
            if native is not None and (
                closed_loop_yaw
                or turn_calibration is not None
                or args.trajectory_claim_scope == "reconstructed_nominal_timing"
            ):
                if agent.last_capture_manifest is None:
                    if closed_loop_yaw:
                        raise InvalidConfiguration(
                            "closed-loop native alignment requires observed browser frames"
                        )
                    receipt["native_alignment_error"] = (
                        "observed browser frames unavailable for native alignment"
                    )
                    recorder.checkpoint(page, "complete")
                    print(json.dumps(receipt, indent=2, sort_keys=True))
                    return 0
                try:
                    alignment = align_native_video(
                        native,
                        agent.last_capture_manifest,
                        output / "action_trace.json",
                        output,
                        endpoint_mode=(
                            "closed_loop_visual_return"
                            if closed_loop_yaw
                            else "action_trace_end"
                        ),
                    )
                except NativeAlignmentError as exc:
                    if closed_loop_yaw:
                        raise InvalidConfiguration(str(exc)) from exc
                    receipt["native_alignment_error"] = str(exc)
                else:
                    receipt.update(alignment_receipt_fields(alignment))
            recorder.checkpoint(page, "complete")
        except BaseException as exc:
            recorder.fail(page, exc)
            raise
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", type=Path, default=DEFAULT_CHROME)
    ap.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    ap.add_argument("--port", type=int, default=9222)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("launch")

    download = sub.add_parser("download-current")
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--timeout-ms", type=int, default=60_000)
    download.add_argument("--download-timeout-ms", type=int, default=90_000)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "project_genie" / "inspect")
    inspect.add_argument("--enter-create", action="store_true")
    inspect.add_argument("--allow-canvas-entry", action="store_true")
    inspect.add_argument("--timeout-ms", type=int, default=60_000)
    inspect.add_argument("--press-escape", action="store_true")
    inspect.add_argument("--open-perspective", action="store_true")

    calibrate = sub.add_parser("calibrate-turn")
    calibrate.add_argument("--output-dir", type=Path, required=True)
    calibrate.add_argument("--key", default="ArrowLeft")
    calibrate.add_argument("--duration-sec", type=float, default=55.0)
    calibrate.add_argument("--sample-interval-sec", type=float, default=1.0)
    calibrate.add_argument("--view", choices=("first_person", "third_person"), default="first_person")
    calibrate.add_argument("--restart-world", action="store_true")
    calibrate.add_argument("--download", action="store_true")
    calibrate.add_argument("--timeout-ms", type=int, default=60_000)
    calibrate.add_argument("--download-timeout-ms", type=int, default=90_000)

    execute = sub.add_parser("run")
    execute.add_argument("--launch", action="store_true")
    execute.add_argument("--allow-canvas-entry", action="store_true")
    execute.add_argument(
        "--auto-accept-upload-policy",
        action="store_true",
        help=(
            "detect the exact Prohibited Use Policy upload dialog and click "
            "I agree; records the explicit opt-in in receipt.json"
        ),
    )
    execute.add_argument("--resume", action="store_true")
    execute.add_argument(
        "--reuse-current-world-from",
        type=Path,
        help="restart the currently open saved world after validating lineage from this receipt",
    )
    execute.add_argument("--first-frame", type=Path, required=True)
    execute.add_argument("--actions", type=Path, required=True)
    execute.add_argument("--turn-calibration", type=Path)
    execute.add_argument(
        "--allow-cross-view-calibration",
        action="store_true",
        help=(
            "explicitly reuse validated browser yaw timing across first/third-person "
            "views and record the mismatch as reconstructed calibration provenance"
        ),
    )
    execute.add_argument(
        "--allow-opposite-turn-calibration",
        action="store_true",
        help=(
            "explicitly reuse a validated left/right full-turn period for the "
            "opposite arrow key and record the symmetry assumption"
        ),
    )
    execute.add_argument("--closed-loop-yaw", action="store_true")
    execute.add_argument("--environment-prompt", required=True)
    execute.add_argument("--character-prompt", default="")
    execute.add_argument("--prompt-provenance-status", default="unspecified")
    execute.add_argument(
        "--trajectory-claim-scope",
        choices=("calibrated_or_closed_loop", "reconstructed_nominal_timing"),
        default="calibrated_or_closed_loop",
    )
    execute.add_argument("--view", choices=("first_person", "third_person"), required=True)
    execute.add_argument("--repeat-id", default="run0")
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--timeout-ms", type=int, default=180_000)
    execute.add_argument("--scene-wait-ms", type=int, default=180_000)
    execute.add_argument("--download-timeout-ms", type=int, default=90_000)
    execute.add_argument("--capture-fps", type=float, default=15.0)
    execute.add_argument("--no-observed-capture", action="store_true")
    execute.add_argument("--allow-missing-native", action="store_true")
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.command == "doctor":
        return doctor(args)
    if args.command == "launch":
        return launch_browser(args)
    if args.command == "download-current":
        return download_current(args)
    if args.command == "inspect":
        return inspect_browser(args)
    if args.command == "calibrate-turn":
        return calibrate_turn(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
