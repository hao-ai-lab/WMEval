from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "generation" / "project_genie_agent.py"
SPEC = importlib.util.spec_from_file_location("project_genie_agent", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeKeyboard:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def down(self, key: str) -> None:
        self.events.append(("down", key))

    def up(self, key: str) -> None:
        self.events.append(("up", key))

    def press(self, key: str, delay: int = 0) -> None:
        self.events.append(("press", key))


class FakePage:
    def __init__(self) -> None:
        self.keyboard = FakeKeyboard()
        self.waits: list[int] = []

    def wait_for_timeout(self, value: int) -> None:
        self.waits.append(value)


def test_world_view_from_url_is_fail_closed() -> None:
    assert MODULE.world_view_from_url(
        "https://labs.google/fx/projectgenie/tools/projectgenie/w?id=abc&use3p=true"
    ) == "third_person"
    assert MODULE.world_view_from_url(
        "https://labs.google/fx/projectgenie/tools/projectgenie/w?use3p=false&id=abc"
    ) == "first_person"
    assert MODULE.world_view_from_url(
        "https://labs.google/fx/projectgenie/tools/projectgenie/creation?use3p=false"
    ) is None
    assert MODULE.world_view_from_url(
        "https://labs.google/fx/projectgenie/tools/projectgenie/w?id=abc"
    ) is None
    assert MODULE.world_prompt_from_url(
        "https://labs.google/fx/projectgenie/tools/projectgenie/w?w=A+garden&use3p=false"
    ) == "A garden"
    assert MODULE.world_prompt_from_url(
        "https://labs.google/fx/projectgenie/tools/projectgenie/creation?w=A+garden"
    ) is None


def test_reconstructed_nominal_timing_marks_observed_capture_ready() -> None:
    assert MODULE.evaluation_artifact_status(
        has_observed_video=True,
        has_turn_calibration=False,
        closed_loop_yaw=False,
        trajectory_claim_scope="reconstructed_nominal_timing",
    ) == "action_aligned_capture_ready"
    assert MODULE.evaluation_artifact_status(
        has_observed_video=True,
        has_turn_calibration=False,
        closed_loop_yaw=False,
        trajectory_claim_scope="calibrated_or_closed_loop",
    ) == "not_ready"


def test_recover_nominal_action_checkpoint_revalidates_files(tmp_path: Path) -> None:
    frames = tmp_path / "observed_frames"
    frames.mkdir()
    frame = frames / "frame_000000.jpg"
    frame.write_bytes(b"jpeg")
    manifest = {
        "frames": [
            {
                "index": 0,
                "elapsed_sec": 1.25,
                "path": str(frame),
                "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
            }
        ]
    }
    (tmp_path / "observed_frames.json").write_text(json.dumps(manifest))
    (tmp_path / "observed.mp4").write_bytes(b"video")
    trace = [{"op": "hold", "keys": ["KeyW"], "key": None, "duration_ms": 1000}]
    (tmp_path / "action_trace.json").write_text(json.dumps(trace))
    recovered = MODULE.recover_nominal_action_checkpoint(
        tmp_path, [{"op": "hold", "keys": ["KeyW"], "ms": 1000}], 15.0
    )
    assert recovered is not None
    assert recovered["observed_frame_count"] == 1
    frame.write_bytes(b"tampered")
    assert MODULE.recover_nominal_action_checkpoint(
        tmp_path, [{"op": "hold", "keys": ["KeyW"], "ms": 1000}], 15.0
    ) is None


def test_recover_closed_loop_checkpoint_requires_validated_visual_return(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "observed_frames"
    frames.mkdir()
    frame = frames / "frame_000000.jpg"
    frame.write_bytes(b"jpeg")
    (tmp_path / "observed_frames.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "index": 0,
                        "elapsed_sec": 4.0,
                        "path": str(frame),
                        "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )
    (tmp_path / "observed.mp4").write_bytes(b"video")
    trace = [
        {
            "op": "turn_until_return",
            "keys": ["ArrowRight"],
            "duration_ms": 4000,
            "visual_return_status": "validated",
            "departure_observed": True,
        }
    ]
    (tmp_path / "action_trace.json").write_text(json.dumps(trace))
    action = {
        "op": "turn_until_return",
        "keys": ["ArrowRight"],
        "min_duration_ms": 2000,
        "max_duration_ms": 50000,
    }
    assert MODULE.recover_action_checkpoint(
        tmp_path, [action], 15.0, closed_loop_yaw=True
    ) is not None
    trace[0]["visual_return_status"] = "not_found"
    (tmp_path / "action_trace.json").write_text(json.dumps(trace))
    assert MODULE.recover_action_checkpoint(
        tmp_path, [action], 15.0, closed_loop_yaw=True
    ) is None


def test_turn_calibration_view_mismatch_requires_explicit_override(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "status": "validated_periodicity",
                "claim_scope": "visual_yaw_cycle_timing",
                "view": "first_person",
                "key": "ArrowLeft",
                "full_turn_duration_sec": 8.0,
            }
        )
    )
    with pytest.raises(MODULE.InvalidConfiguration, match="does not match"):
        MODULE.load_turn_calibration(calibration, expected_view="third_person")
    loaded = MODULE.load_turn_calibration(
        calibration,
        expected_view="third_person",
        allow_view_mismatch=True,
    )
    assert loaded["view"] == "first_person"


def test_replay_supports_simultaneous_keys_and_releases_them(tmp_path: Path) -> None:
    page = FakePage()
    recorder = MODULE.RunRecorder(tmp_path, {})
    agent = MODULE.GenieAgent(page, recorder, timeout_ms=1000)
    trace = agent.replay(
        [
            {"op": "hold", "keys": ["KeyW", "KeyD"], "ms": 250},
            {"op": "wait", "ms": 100},
            {"op": "press", "key": "Space"},
        ]
    )
    assert page.keyboard.events == [
        ("down", "KeyW"),
        ("down", "KeyD"),
        ("up", "KeyD"),
        ("up", "KeyW"),
        ("press", "Space"),
    ]
    assert page.waits == [250, 100]
    assert [row["op"] for row in trace] == ["hold", "wait", "press"]


def test_turn_calibration_preserves_target_angle_and_nominal_horizon() -> None:
    actions = [
        {"op": "hold", "keys": ["ArrowLeft"], "ms": 1250, "target_yaw_deg": -45.0},
        {"op": "wait", "ms": 8750},
    ]
    calibrated = MODULE.apply_turn_calibration(
        actions,
        {"key": "ArrowLeft", "full_turn_duration_sec": 7.0},
    )
    assert calibrated[0]["nominal_ms"] == 1250
    assert calibrated[0]["ms"] == 875
    assert calibrated[0]["target_yaw_deg"] == -45.0
    assert calibrated[1]["ms"] == 9125
    assert sum(action["ms"] for action in calibrated) == 10_000


def test_turn_calibration_fails_on_wrong_key() -> None:
    with pytest.raises(MODULE.InvalidConfiguration, match="do not match"):
        MODULE.apply_turn_calibration(
            [{"op": "hold", "keys": ["ArrowRight"], "ms": 1000, "target_yaw_deg": 45}],
            {"key": "ArrowLeft", "full_turn_duration_sec": 7.0},
        )


def test_turn_calibration_can_explicitly_assume_opposite_key_symmetry() -> None:
    calibrated = MODULE.apply_turn_calibration(
        [{"op": "hold", "keys": ["ArrowRight"], "ms": 1250, "target_yaw_deg": 45}],
        {"key": "ArrowLeft", "full_turn_duration_sec": 7.0},
        allow_opposite_key=True,
    )
    assert calibrated[0]["ms"] == 875
    assert calibrated[0]["calibration_source_key"] == "ArrowLeft"
    assert calibrated[0]["calibration_status"] == "opposite_key_symmetric_period_assumption"


def test_closed_loop_yaw_accepts_full_turn_and_rejects_perturbation() -> None:
    compiled = MODULE.apply_closed_loop_yaw(
        [{"op": "hold", "keys": ["ArrowLeft"], "ms": 10_000, "target_yaw_deg": -360}]
    )
    assert compiled[0]["op"] == "turn_until_return"
    assert compiled[0]["controller_version"] == "visual_return_sift_v2"
    assert compiled[0]["nominal_horizon_ms"] == 10_000
    with pytest.raises(MODULE.InvalidConfiguration, match="full 360-degree"):
        MODULE.apply_closed_loop_yaw(
            [{"op": "hold", "keys": ["ArrowLeft"], "ms": 1250, "target_yaw_deg": -45}]
        )


class ConsentElement:
    def __init__(self, *, visible: bool = True) -> None:
        self.visible = visible
        self.clicks = 0

    def count(self) -> int:
        return 1

    def nth(self, index: int) -> "ConsentElement":
        return self

    def is_visible(self) -> bool:
        return self.visible

    def click(self, timeout: int) -> None:
        self.clicks += 1
        self.visible = False


class ConsentPage:
    def __init__(self) -> None:
        self.policy = ConsentElement()
        self.agreement = ConsentElement()
        self.waits: list[int] = []

    def get_by_role(self, role: str, *, name: str, exact: bool) -> ConsentElement:
        assert exact is True
        if role == "link" and name == "Prohibited Use Policy":
            return self.policy
        if role == "button" and name == "I agree":
            return self.agreement
        return ConsentElement(visible=False)

    def wait_for_timeout(self, value: int) -> None:
        self.waits.append(value)


class ConsentRecorder:
    def __init__(self) -> None:
        self.receipt: dict[str, object] = {}
        self.checkpoints: list[tuple[str, dict[str, object]]] = []

    def checkpoint(self, page: object, stage: str, **extra: object) -> None:
        self.receipt.update(extra)
        self.checkpoints.append((stage, extra))


def test_upload_policy_auto_accept_requires_explicit_opt_in() -> None:
    page = ConsentPage()
    recorder = ConsentRecorder()
    agent = MODULE.GenieAgent(page, recorder, timeout_ms=1000)
    with pytest.raises(MODULE.UserActionRequired, match="review and click I agree"):
        agent.require_user_consent_if_present()
    assert page.agreement.clicks == 0
    assert recorder.receipt == {}


def test_upload_policy_auto_accept_is_exact_and_audited() -> None:
    page = ConsentPage()
    recorder = ConsentRecorder()
    agent = MODULE.GenieAgent(
        page,
        recorder,
        timeout_ms=1000,
        auto_accept_upload_policy=True,
    )
    assert agent.require_user_consent_if_present() is True
    assert page.agreement.clicks == 1
    assert page.waits == [500]
    assert recorder.checkpoints[0][0] == "upload_policy_auto_accepted"
    events = recorder.receipt["upload_policy_acceptances"]
    assert isinstance(events, list)
    assert events[0]["mode"] == "automated_explicit_cli_opt_in"


def test_upload_policy_auto_accept_polls_for_delayed_dialog() -> None:
    class DelayedConsentPage(ConsentPage):
        def __init__(self) -> None:
            super().__init__()
            self.policy.visible = False
            self.agreement.visible = False

        def wait_for_timeout(self, value: int) -> None:
            super().wait_for_timeout(value)
            if value == 250:
                self.policy.visible = True
                self.agreement.visible = True

    page = DelayedConsentPage()
    recorder = ConsentRecorder()
    agent = MODULE.GenieAgent(
        page,
        recorder,
        timeout_ms=1000,
        auto_accept_upload_policy=True,
    )
    assert agent.require_user_consent_if_present() is True
    assert page.agreement.clicks == 1
    assert page.waits == [250, 500]


def test_download_retries_explicit_export_failure(tmp_path: Path) -> None:
    class FakeDownload:
        suggested_filename = "world.mp4"

        def save_as(self, path: str) -> None:
            Path(path).write_bytes(b"video")

    class Visible:
        def __init__(self, page: "DownloadPage") -> None:
            self.page = page

        def count(self) -> int:
            return 1

        def nth(self, index: int) -> "Visible":
            return self

        def is_visible(self) -> bool:
            return self.page.failed

    class DownloadButton:
        def __init__(self, page: "DownloadPage") -> None:
            self.page = page

        def click(self) -> None:
            self.page.clicks += 1
            if self.page.clicks == 1:
                self.page.failed = True
            else:
                for callback in list(self.page.listeners):
                    callback(FakeDownload())

    class DismissButton:
        def __init__(self, page: "DownloadPage") -> None:
            self.page = page

        def click(self) -> None:
            self.page.failed = False

    class DownloadPage:
        def __init__(self) -> None:
            self.failed = False
            self.clicks = 0
            self.listeners: list[object] = []

        def on(self, event: str, callback: object) -> None:
            assert event == "download"
            self.listeners.append(callback)

        def remove_listener(self, event: str, callback: object) -> None:
            assert event == "download"
            self.listeners.remove(callback)

        def get_by_text(self, text: str, exact: bool = False) -> Visible:
            assert text == "Download failed"
            assert exact is True
            return Visible(self)

        def wait_for_timeout(self, value: int) -> None:
            pass

    page = DownloadPage()
    agent = MODULE.GenieAgent(page, MODULE.RunRecorder(tmp_path, {}), timeout_ms=1000)
    download = DownloadButton(page)
    dismiss = DismissButton(page)
    agent._named = lambda role, names: (
        dismiss if any("dismiss" in name for name in names) and page.failed else download
    )
    path = agent.download(1000)
    assert path.read_bytes() == b"video"
    assert [row["status"] for row in agent.last_download_attempts] == [
        "explicit_export_failure",
        "downloaded",
    ]


def test_download_recovers_late_filesystem_export(tmp_path: Path) -> None:
    download_dir = tmp_path / "Downloads"
    download_dir.mkdir()

    class Visible:
        def count(self) -> int:
            return 1

        def nth(self, index: int) -> "Visible":
            return self

        def is_visible(self) -> bool:
            return True

    class DownloadButton:
        def click(self) -> None:
            (download_dir / "Genie-late.mp4").write_bytes(b"late-video")

    class DownloadPage:
        def on(self, event: str, callback: object) -> None:
            assert event == "download"

        def remove_listener(self, event: str, callback: object) -> None:
            assert event == "download"

        def get_by_text(self, text: str, exact: bool = False) -> Visible:
            assert text == "Download failed"
            assert exact is True
            return Visible()

        def wait_for_timeout(self, value: int) -> None:
            pass

    output = tmp_path / "run"
    agent = MODULE.GenieAgent(
        DownloadPage(),
        MODULE.RunRecorder(output, {}),
        timeout_ms=1000,
        filesystem_download_dir=download_dir,
        filesystem_download_grace_ms=100,
    )
    button = DownloadButton()
    agent._named = lambda role, names: button
    path = agent.download(1000)
    assert path == output / "native.mp4"
    assert path.read_bytes() == b"late-video"
    assert agent.last_download_attempts == [
        {
            "attempt": 1,
            "elapsed_sec": agent.last_download_attempts[0]["elapsed_sec"],
            "source": str((download_dir / "Genie-late.mp4").resolve()),
            "status": "downloaded_via_filesystem_fallback",
        }
    ]
