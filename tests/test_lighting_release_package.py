import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from tools.build_lighting_release_package import build, verify, archive, MODELS, SEEDS, UNSUPPORTED


def fixture_release(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    case = repo / "eval_sets/wmeval_lite_v1/cases/SunRisesAndSets/lighting_orbit_v2"
    case.mkdir(parents=True)
    (case / "case.yaml").write_text("schema_version: 2\n")
    suite = repo / "eval_sets/wmeval_lite_v1/suite.yaml"
    suite.write_text(yaml.safe_dump({"dimensions": {"lighting_shading": {
        "cases": ["SunRisesAndSets/lighting"], "metric_config": {"eval_fps": 24}}}}))
    decisions = {name: {"status": "supported" if name in MODELS.values() else "unsupported",
                        "reason": "Audited fixture interface"}
                 for name in (*MODELS.values(), *UNSUPPORTED)}
    for name in (
        "experiments/tables_4_7/action_backends.yaml",
        "experiments/lighting_orbit_v2/genie_controls_frozen.json",
        "experiments/lighting_orbit_v2/results/capability_audit.json",
        "experiments/lighting_orbit_v2/results/run_summary.json",
        "experiments/lighting_orbit_v2/results/genie_diagnostic_summary.json",
        "experiments/lighting_orbit_v2/results/genie_calibration.json",
        "experiments/lighting_orbit_v2/results/waypoint_round2_calibration.json",
        "experiments/lighting_orbit_v2/results/generation_case.yaml",
        "experiments/lighting_orbit_v2/hy_runtime_inventory.json",
        "experiments/lighting_orbit_v2/hy_glyph_inventory.json",
    ):
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    (repo / "experiments/tables_4_7/action_backends.yaml").write_text(yaml.safe_dump({
        "trajectory_support": {"lighting.camera.look_at_orbit_v2": {"backends": decisions}}}))
    monkeypatch.setattr("tools.build_lighting_release_package.subprocess.check_output",
                        lambda *a, **kw: "")
    runs = []
    for model in MODELS:
        for seed in SEEDS:
            p = tmp_path / "inputs" / model / str(seed)
            p.mkdir(parents=True)
            video = p / "video.mp4"
            video.write_bytes(f"verified fixture {model} {seed}".encode())
            sha = hashlib.sha256(video.read_bytes()).hexdigest()
            receipt = p / "receipt.json"
            receipt.write_text(json.dumps({"status": "completed", "video_sha256": sha,
                "lighting_time_mapping": {"schema_version": 1, "native_fps": 12,
                                         "canonical_zero_frame": 0, "video_sha256": sha}}))
            runs.append({"model": model, "seed": seed, "video": str(video),
                         "receipt": str(receipt), "metrics": {"fixture_metric": seed}})
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"runs": runs}))
    return repo, inputs


def test_package_relocates_and_is_byte_reproducible(tmp_path, monkeypatch):
    repo, inputs = fixture_release(tmp_path, monkeypatch)
    built = tmp_path / "original"
    assert build(inputs, built, repo)["formal_jobs"] == 15
    relocated = tmp_path / "different" / "location"
    relocated.parent.mkdir()
    built.rename(relocated)
    assert verify(relocated)["formal_jobs"] == 15
    for row in (relocated / "candidates.jsonl").read_text().splitlines():
        assert not Path(json.loads(row)["candidate_video"]).is_absolute()
    first = archive(relocated, tmp_path / "first.zip")
    generated = relocated / 'tools/__pycache__/build_lighting_release_package.cpython-312.pyc'
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b'Python runtime cache')
    results = relocated / 'results/per_case_results.jsonl'
    results.parent.mkdir(); results.write_text('runtime outputs')
    assert verify(relocated)['formal_jobs'] == 15
    second = archive(relocated, tmp_path / "second.zip")
    assert first["sha256"] == second["sha256"]
    video = next((relocated / "candidates").rglob("video.mp4"))
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(relocated)


def test_package_rejects_missing_required_seed(tmp_path, monkeypatch):
    repo, inputs = fixture_release(tmp_path, monkeypatch)
    rows = json.loads(inputs.read_text())
    rows["runs"].pop()
    inputs.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="fifteen"):
        build(inputs, tmp_path / "out", repo)


def test_package_cannot_hide_an_additional_supported_backend(tmp_path, monkeypatch):
    repo, inputs = fixture_release(tmp_path, monkeypatch)
    p = repo / "experiments/tables_4_7/action_backends.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["trajectory_support"]["lighting.camera.look_at_orbit_v2"]["backends"]["waypoint"]["status"] = "supported"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="population"):
        build(inputs, tmp_path / "out", repo)


def test_package_build_inputs_resolve_relative_to_manifest(tmp_path, monkeypatch):
    repo, inputs = fixture_release(tmp_path, monkeypatch)
    inventory = json.loads(inputs.read_text())
    for row in inventory['runs']:
        for field in ('video', 'receipt'):
            row[field] = str(Path(row[field]).relative_to(inputs.parent))
    inputs.write_text(json.dumps(inventory))
    monkeypatch.chdir(repo)
    assert build(inputs, tmp_path / 'relative-inputs', repo)['formal_jobs'] == 15


def test_transport_metadata_refreshes_stale_case_lock_without_changing_inputs(tmp_path):
    from evaluation.lite.authoring import _fingerprint
    from tools.build_lighting_orbit_case import write_table_cell
    source = Path(__file__).resolve().parents[1] / 'eval_sets/wmeval_lite_v1/cases' / 'SunRisesAndSets/lighting_orbit_v2'
    case = tmp_path / 'case'
    shutil.copytree(source, case)
    inputs = {p.relative_to(case): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (case / 'reference').rglob('*') if p.is_file()
              and p.name not in {'checksums.sha256', 'table_cell.json'}}
    raw = yaml.safe_load((case / 'case.yaml').read_text())
    raw['evaluator_lock']['evaluator_fingerprint'] = 'stale'
    (case / 'case.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    write_table_cell(case)
    raw = yaml.safe_load((case / 'case.yaml').read_text())
    assert raw['evaluator_lock']['evaluator_fingerprint'] == _fingerprint(case, raw)
    for relative, expected in inputs.items():
        assert hashlib.sha256((case / relative).read_bytes()).hexdigest() == expected


def test_strict_score_verification_rejects_hardware_variation_before_inference(tmp_path):
    from tools.verify_lighting_release_run import check_pinned_hardware
    case = tmp_path / 'eval_sets/wmeval_lite_v1/cases/SunRisesAndSets/lighting_orbit_v2/case.yaml'
    case.parent.mkdir(parents=True)
    case.write_text(yaml.safe_dump({'evaluator_lock': {'dependency_versions': {
        'gpu': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'nvidia_driver': '595.91.07'}}}))
    valid = 'NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.91.07\n'
    assert check_pinned_hardware(tmp_path, 1, valid)['policy'] == 'pinned_gpu_and_driver'
    with pytest.raises(ValueError, match='pinned GPU'):
        check_pinned_hardware(tmp_path, 1, 'NVIDIA RTX PRO 4500 Blackwell, 595.91.07\n')
    with pytest.raises(ValueError, match='not enough GPUs'):
        check_pinned_hardware(tmp_path, 2, valid)
