from __future__ import annotations

import hashlib
import importlib.metadata as md
import platform
from pathlib import Path

import pytest

from tests.lite_suite_test_utils import build_default_suite_dir
from tools import preflight_lite as pf
from tools.lock_case_dependencies import load_assets_lock


def _statuses(rows: list[pf.Row]) -> dict[str, str]:
    return {row.check: row.status for row in rows}


def test_dependency_rows_pass_warn_and_fail() -> None:
    numpy = md.version("numpy")
    locks = {
        "a/ok": {"python": platform.python_version(), "numpy": numpy, "lock_source": "x", "asset:raft_large": "y"},
        "a/drift": {"numpy": "0.0.1", "gpu": "Some GPU", "not-a-real-dist-wmeval": "1.0"},
        "a/empty": {},
    }
    rows = pf.check_dependencies(locks, hardware=lambda: {"gpu": None})
    assert [r.check for r in rows if r.status == pf.WARN and "no dependency lock" in r.detail] == ["deps"]
    by_detail = {(r.check, r.detail.split(" [")[0]): r.status for r in rows}
    assert by_detail[("dep numpy", numpy)] == pf.PASS
    assert by_detail[("dep numpy", f"installed {numpy}, locked 0.0.1")] == pf.WARN
    assert by_detail[("hw gpu", "have n/a, locked Some GPU")] == pf.WARN
    assert by_detail[("dep not-a-real-dist-wmeval", "not installed; locked 1.0")] == pf.FAIL
    assert not any(r.check.startswith("dep asset") or r.check == "dep lock_source" for r in rows)

    strict = pf.check_dependencies({"a/drift": {"numpy": "0.0.1"}}, strict=True)
    assert _statuses(strict)["dep numpy"] == pf.FAIL
    assert _statuses(pf.check_python({"a": {"python": "2.7"}}))["python"] == pf.WARN
    assert _statuses(pf.check_python({"a": {"python": "2.7"}}, strict=True))["python"] == pf.FAIL


def test_asset_checks_verify_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assets = load_assets_lock()
    assert _statuses(pf.check_asset("ptlflow_synthetic", assets["ptlflow_synthetic"]))["asset ptlflow_synthetic"] == pf.PASS

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    weight = b"fake weights"
    entry = {"model_id": "org/model", "revision": "a" * 40, "weight": "model.safetensors",
             "sha256": hashlib.sha256(weight).hexdigest()}
    assert _statuses(pf.check_asset("m", entry))["asset m"] == pf.WARN  # not cached
    snapshot = tmp_path / "models--org--model" / "snapshots" / entry["revision"]
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(weight)
    assert _statuses(pf.check_asset("m", entry))["asset m"] == pf.PASS
    (snapshot / "model.safetensors").write_bytes(b"tampered")
    assert _statuses(pf.check_asset("m", entry))["asset m"] == pf.FAIL

    stale = pf.check_assets({"c": {"asset:raft_large": "0" * 64}}, {}, assets, hashes=False)
    assert any(r.status == pf.FAIL and "re-run lock_case_dependencies" in r.detail for r in stale)


def test_ffmpeg_and_judge_key_checks() -> None:
    assert _statuses(pf.check_ffmpeg(["c/needs"], which=lambda _: None))["ffmpeg"] == pf.FAIL
    assert _statuses(pf.check_ffmpeg([], which=lambda _: None))["ffprobe"] == pf.WARN
    needs = {"OPENAI_API_KEY": ["lpips"]}
    assert _statuses(pf.check_judge_keys(needs, environ={}))["judge OPENAI_API_KEY"] == pf.FAIL
    assert _statuses(pf.check_judge_keys(needs, environ={"OPENAI_API_KEY": "sk-x"}))["judge OPENAI_API_KEY"] == pf.PASS
    assert _statuses(pf.check_device("cpu"))["device"] == pf.PASS
    assert _statuses(pf.check_device("tpu"))["device"] == pf.FAIL


def test_main_renders_table_and_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    suite = build_default_suite_dir(tmp_path / "suite")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert pf.main(["--suite", str(suite), "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "STATUS" in out and "suite" in out and " fail" in out
    assert "PASS    judge OPENAI_API_KEY" in out  # toy vlm dimension; presence only

    monkeypatch.delenv("OPENAI_API_KEY")
    import tools.run_wmeval_lite as runner

    monkeypatch.setattr(runner, "_load_dotenv", lambda root: None)  # ignore a local .env
    assert pf.main(["--suite", str(suite), "--device", "cpu", "--dimensions", "toy_promptcons"]) == 1


def test_lighting_checks_the_sam2_cuda_extension() -> None:
    present = lambda name: object()  # noqa: E731
    def broken() -> None:
        raise ImportError("libc10.so: cannot open shared object file")
    assert pf.check_lighting_sam2_extension(["fluid_mechanics"], find_spec=present, import_extension=broken) == []
    assert pf.check_lighting_sam2_extension(["lighting_shading"], find_spec=lambda name: None,
                                            import_extension=broken) == []
    assert _statuses(pf.check_lighting_sam2_extension(
        ["lighting_shading"], find_spec=present, import_extension=lambda: None)) == {"sam2 CUDA extension": pf.PASS}
    rows = pf.check_lighting_sam2_extension(["lighting_shading"], find_spec=present, import_extension=broken)
    assert _statuses(rows) == {"sam2 CUDA extension": pf.WARN}
    assert pf.LIGHTING_DOC in rows[0].detail and "ImportError" in rows[0].detail


def test_source_revision_falls_back_to_the_distribution_of_an_import_name(monkeypatch) -> None:
    class Dist:
        def read_text(self, name):
            return '{"url": "https://github.com/facebookresearch/sam2.git", "vcs_info": {"commit_id": "2b90b9f"}}'

    def distribution(name):
        if name != "SAM-2":
            raise md.PackageNotFoundError(name)
        return Dist()

    monkeypatch.setattr(pf.importlib_metadata, "distribution", distribution)
    monkeypatch.setattr(pf.importlib_metadata, "packages_distributions", lambda: {"sam2": ["SAM-2"]})
    assert pf._source_revision("sam2") == "2b90b9f"
    assert pf._source_revision("other") is None
