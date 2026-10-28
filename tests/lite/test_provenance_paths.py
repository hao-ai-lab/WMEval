from pathlib import Path

from evaluation.provenance_paths import resolve_recorded_path


def test_resolve_recorded_path_relocates_repo_suffix(tmp_path: Path) -> None:
    artifact = tmp_path / "runs/project_genie/example/run0/video.mp4"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"video")

    resolved = resolve_recorded_path(
        "/old/host/WMEval/runs/project_genie/example/run0/video.mp4",
        repo_root=tmp_path,
    )

    assert resolved == artifact.resolve()


def test_resolve_recorded_path_prefers_existing_original(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")

    assert resolve_recorded_path(artifact, repo_root=tmp_path) == artifact.resolve()
