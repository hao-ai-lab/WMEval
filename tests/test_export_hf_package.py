from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tools.dataset import export_hf_package as exporter
REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SUITE = REPO_ROOT / "lite"


def _write(path: Path, text: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _case(root: Path, case_id: str, *, video: str = "reference/reference.mp4") -> Path:
    case_root = root / "cases" / case_id
    _write(
        case_root / "case.yaml",
        yaml.safe_dump(
            {
                "schema_version": 2,
                "case_id": case_id,
                "scene": case_id.split("/")[0],
                "display_name": case_id,
                "modules": ["lpips"],
                "reference": {
                    "video": video,
                    "checksums": "reference/checksums.sha256",
                    "artifacts": {"events": "reference/artifacts/events.json"},
                },
                "protocol": {
                    "candidate_normalization": "none",
                    "min_duration_sec": 1.0,
                    "min_duration_ratio": 0.5,
                    "duration_threshold_mode": "max",
                },
                "evaluator_lock": {
                    "release_id": "toy-release-v9",
                    "code_revision": "abc",
                    "evaluator_fingerprint": "def",
                    "module_config": {},
                    "artifact_schema_versions": {"events": 3},
                    "dependency_versions": {},
                },
            },
            sort_keys=False,
        ),
    )
    _write(case_root / "reference" / "reference.mp4", b"\x00fake video")
    _write(case_root / "reference" / "checksums.sha256", "")
    _write(
        case_root / "reference" / "artifacts" / "events.json",
        json.dumps({"schema_version": 1, "events": [1, 2]}),
    )
    return case_root


def _toy_suite(tmp_path: Path) -> Path:
    root = tmp_path / "suite"
    _write(
        root / "suite.yaml",
        "# comment banner: data came from Modal volume WMEval-lite-candidates\n"
        + yaml.safe_dump(
            {
                "suite_name": "Toy Suite",
                "version": "v9",
                "dimensions": {
                    "lpips": {
                        "evaluator": "registry",
                        "requires_reference": True,
                        "modules": ["lpips"],
                        "metric_config": {"some_cache_dir": "runs/toy_cache"},
                        "cases": ["SceneA/one", "SceneB/two"],
                    },
                    "other": {
                        "evaluator": "registry",
                        "requires_reference": True,
                        "modules": ["lpips"],
                        "cases": ["SceneB/two"],
                    },
                },
                "legacy_case_aliases": {"lpips": {"old_a": "SceneA/one"}},
            },
            sort_keys=False,
        ),
    )
    _case(root, "SceneA/one")
    _case(root, "SceneB/two")
    _case(root, "SceneC/orphan")  # on disk, listed by no suite
    _write(root / "PENDING.md", "maintainer notes\n")
    _write(root / "DATASET_CARD.md", "---\npretty_name: Toy\n---\n# Toy\n")
    _write(root / "human_alignment" / "study.json", json.dumps({"schema_version": 1}))
    _write(root / "baselines" / "scores.json", json.dumps({"score": 1.0}))
    _write(root / "cases" / "SceneA" / "one" / "reference" / ".DS_Store", b"junk")
    _write(root / "cases" / "SceneA" / "one" / "__pycache__" / "x.pyc", b"junk")
    # companion suite
    companion = root / "companion"
    _write(
        companion / "suite.yaml",
        yaml.safe_dump(
            {
                "schema_version": 1,
                "suite_name": "Companion",
                "version": "native",
                "dimensions": {
                    "lpips": {
                        "evaluator": "registry",
                        "requires_reference": True,
                        "modules": ["lpips"],
                        "cases": ["SceneA/one"],
                    }
                },
            }
        ),
    )
    _case(companion, "SceneA/one")
    return root


def test_export_selects_listed_cases_and_companions(tmp_path: Path) -> None:
    suite = _toy_suite(tmp_path)
    out = tmp_path / "pkg"
    manifest = exporter.export_package(suite, out)

    files = set(manifest["files"])
    assert "suite.yaml" in files
    assert "cases/SceneA/one/case.yaml" in files
    assert "cases/SceneB/two/reference/artifacts/events.json" in files
    assert "companion/suite.yaml" in files
    assert "companion/cases/SceneA/one/case.yaml" in files
    assert "human_alignment/study.json" in files
    assert "baselines/scores.json" in files
    assert "README.md" in files  # DATASET_CARD.md renamed for the Hub
    assert not any("SceneC" in f for f in files)
    assert not any(f.endswith((".DS_Store", ".pyc")) for f in files)
    assert "PENDING.md" not in files
    # suite.yaml copied verbatim (comments kept) when no subset is requested
    assert (out / "suite.yaml").read_bytes() == (suite / "suite.yaml").read_bytes()

    assert manifest["excluded"]["unlisted_case_dirs"] == ["cases/SceneC/orphan"]
    assert "PENDING.md" in manifest["excluded"]["suite_root_files"]
    assert manifest["dataset"]["id"] == "toy-release-v9"
    assert manifest["dataset"]["version"] == "v9"
    assert manifest["exporter"]["version"] == exporter.EXPORTER_VERSION
    assert "revision" in manifest["evaluator_code"]
    schema = manifest["schema_versions"]
    assert schema["case_yaml_schema_version"] == {"2": 3}
    assert schema["suite_yaml_schema_version"] == {".": None, "companion": 1}
    assert schema["evaluator_lock_artifact_schema_versions"] == {"events": [3]}
    assert schema["artifact_json_schema_fields"]["events.json"] == {"schema_version": [1]}
    assert [s["path"] for s in manifest["suites"]] == [".", "companion"]
    assert manifest["suites"][0]["dimensions"] == {"lpips": 2, "other": 1}

    row = manifest["files"]["cases/SceneA/one/reference/reference.mp4"]
    assert row["bytes"] == len(b"\x00fake video")
    assert manifest["totals"]["files"] == len(files)

    on_disk = json.loads((out / "MANIFEST.json").read_text())
    assert on_disk["files"] == manifest["files"]


def test_export_refuses_nonempty_output_and_escaping_case_paths(tmp_path: Path) -> None:
    suite = _toy_suite(tmp_path)
    out = tmp_path / "pkg"
    _write(out / "stale.txt", "x")
    with pytest.raises(exporter.ExportError, match="not empty"):
        exporter.export_package(suite, out)

    _case(suite, "SceneB/two", video="../../../outside.mp4")
    with pytest.raises(exporter.ExportError, match="escapes the case bundle"):
        exporter.export_package(suite, tmp_path / "pkg2")


def test_subset_export_filters_suite_yaml(tmp_path: Path) -> None:
    suite = _toy_suite(tmp_path)
    out = tmp_path / "pkg"
    manifest = exporter.export_package(
        suite, out, cases=["SceneB/two"], include_baselines=False
    )
    doc = yaml.safe_load((out / "suite.yaml").read_text())
    assert {k: v["cases"] for k, v in doc["dimensions"].items()} == {
        "lpips": ["SceneB/two"],
        "other": ["SceneB/two"],
    }
    assert doc["legacy_case_aliases"] == {}
    assert [s["path"] for s in manifest["suites"]] == ["."]  # companion has no SceneB
    assert not any(f.startswith(("baselines/", "companion/")) for f in manifest["files"])
    assert manifest["subset_cases"] == ["SceneB/two"]

    with pytest.raises(exporter.ExportError, match="not listed"):
        exporter.export_package(suite, tmp_path / "pkg3", cases=["Nope/none"])


def test_verify_reports_missing_extra_and_mismatched(tmp_path: Path) -> None:
    suite = _toy_suite(tmp_path)
    out = tmp_path / "pkg"
    exporter.export_package(suite, out)

    clean = exporter.verify_package(out, run_validators=False)
    assert clean.integrity_ok
    assert clean.ok()

    (out / "cases/SceneA/one/reference/reference.mp4").write_bytes(b"tampered")
    (out / "baselines/scores.json").unlink()
    _write(out / "cases/SceneB/two/reference/new.bin", b"extra")
    _write(out / ".DS_Store", b"ignored cache file")
    report = exporter.verify_package(out, run_validators=False)
    assert report.missing == ["baselines/scores.json"]
    assert report.extra == ["cases/SceneB/two/reference/new.bin"]
    assert [row["file"] for row in report.mismatched] == [
        "cases/SceneA/one/reference/reference.mp4"
    ]
    assert not report.ok()


def test_scan_classifies_load_bearing_versus_provenance(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _write(
        pkg / "cases/S/v/reference/artifacts/meta.yaml",
        "# built from /Users/someone/WorldModelBenchScenePrivate/Scenes/x.cs\n"
        "description: >-\n"
        "  Sampled from WorldModelBenchScenePrivate/GameScenes/S/controller.py\n"
        "  at 10 Hz.\n"
        "source: /Users/someone/raw/meta.json\n"
        "poses: /Users/someone/data/poses.npy\n"
        "events: reference/artifacts/missing.json\n"
        "results_cache_dir: runs/cache\n",
    )
    _write(
        pkg / "baselines/receipt.json",
        json.dumps(
            {
                "command": ["/workspace/env/bin/python", "gen.py"],
                "video": "rollouts/a.mp4",
                "log": "/workspace/out/gen.log",
            },
            indent=2,
        ),
    )
    _write(pkg / "baselines/README.md", "Build into /tmp/release first.\n")
    hits = exporter.scan_package(
        pkg,
        ["cases/S/v/reference/artifacts/meta.yaml", "baselines/receipt.json",
         "baselines/README.md"],
    )
    by = {(h.file.split("/")[-1], h.line, h.kind): (h.category, h.context) for h in hits}

    assert by[("meta.yaml", 1, "absolute_path")] == ("provenance", "comment")
    assert by[("meta.yaml", 3, "private_repo")] == ("provenance", "description")
    assert by[("meta.yaml", 5, "absolute_path")] == ("provenance", "source")
    assert by[("meta.yaml", 6, "absolute_path")] == ("load_bearing", "poses")
    assert by[("meta.yaml", 7, "unresolved_relative_path")] == ("load_bearing", "events")
    assert by[("meta.yaml", 8, "runs_dir")] == ("runtime_cache", "results_cache_dir")
    assert by[("receipt.json", 3, "absolute_path")][0] == "provenance"  # command argv
    assert by[("receipt.json", 6, "unresolved_relative_path")] == ("record_reference", "video")
    assert by[("receipt.json", 7, "absolute_path")] == ("record_reference", "log")
    assert by[("README.md", 1, "absolute_path")] == ("provenance", "prose")
    # the private repo inside the comment's absolute path is not double-reported
    assert ("meta.yaml", 1, "private_repo") not in by


def test_case_relative_paths_in_artifacts_resolve(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _write(pkg / "cases/S/v/reference/unity.mp4", b"v")
    _write(
        pkg / "cases/S/v/reference/artifacts/scores.json",
        json.dumps({"reference_video": "reference/unity.mp4"}),
    )
    assert exporter.scan_package(pkg, ["cases/S/v/reference/artifacts/scores.json"]) == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_git_ignored_files_are_not_exported(tmp_path: Path) -> None:
    suite = _toy_suite(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _write(tmp_path / ".gitignore", "*.scratch\n")
    _write(suite / "cases/SceneA/one/reference/notes.scratch", "local only")
    manifest = exporter.export_package(suite, tmp_path / "pkg")
    assert "cases/SceneA/one/reference/notes.scratch" not in manifest["files"]
    assert manifest["excluded"]["git_ignored"] == [
        "cases/SceneA/one/reference/notes.scratch"
    ]
    # nothing is committed in this throwaway repo, so every file is flagged
    assert "cases/SceneA/one/case.yaml" in manifest["excluded"]["untracked_included"]


@pytest.mark.integration("case_data")
@pytest.mark.skipif(not (REAL_SUITE / "suite.yaml").is_file(), reason="suite not present")
def test_real_suite_subset_round_trip_validates(tmp_path: Path) -> None:
    out = tmp_path / "pkg"
    cases = ["BallRoller/ctrl", "ChessBoard/walk_third"]
    assert exporter.main(
        ["export", "--suite", str(REAL_SUITE), "--output", str(out),
         "--cases", ",".join(cases), "--no-baselines"]
    ) == 0
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["suites"][0]["cases"] == cases
    report = exporter.verify_package(out)
    assert report.integrity_ok
    assert report.by_category("load_bearing") == []
    assert [row["returncode"] for row in report.validator] == [0]
    assert report.ok()
