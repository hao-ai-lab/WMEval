"""Tests for case-bundle authoring (checksums + evaluator lock refresh).

Regression focus: the evaluator fingerprint is a derived value of the full
case contract, reference checksums, and evaluator source hashes and must be
recomputed on EVERY re-author — a bundle edit
followed by a default (non --relock) re-author must never leave a stale
fingerprint stamped into evaluation rows.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from evaluation.lite.authoring import author_case
from evaluation.lite.case_loader import load_case
from evaluation.lite.validation import validate_case_bundle
from tests.lite_suite_test_utils import write_reference_free_case


def _lock(case_dir: Path) -> dict:
    raw = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))
    return raw["evaluator_lock"]


def test_author_case_always_recomputes_fingerprint(tmp_path: Path) -> None:
    case_dir = write_reference_free_case(tmp_path / "cases")
    assert _lock(case_dir)["evaluator_fingerprint"] == "test-fingerprint"

    author_case(case_dir)  # default: no --relock

    first = _lock(case_dir)
    # derived value replaced even though it was already set...
    assert first["evaluator_fingerprint"] != "test-fingerprint"
    # ...while the deliberate code_revision declaration is preserved
    assert first["code_revision"] == "test"

    # the empirically-reported defect scenario: edit the bundle, re-author
    # with the DEFAULT command (the documented re-lock step)
    (case_dir / "reference" / "ref.png").write_text(
        "edited first frame\n", encoding="utf-8"
    )
    author_case(case_dir)

    second = _lock(case_dir)
    assert second["evaluator_fingerprint"] != first["evaluator_fingerprint"]
    validate_case_bundle(load_case(case_dir))


def test_author_case_is_idempotent_without_content_changes(tmp_path: Path) -> None:
    case_dir = write_reference_free_case(tmp_path / "cases")
    author_case(case_dir)
    first = _lock(case_dir)["evaluator_fingerprint"]

    author_case(case_dir)

    assert _lock(case_dir)["evaluator_fingerprint"] == first


def test_author_case_fingerprint_covers_module_params(tmp_path: Path) -> None:
    case_dir = write_reference_free_case(
        tmp_path / "cases",
        module_params={"environment_object_consistency": {"eoc_text_prompt": "tree."}},
    )
    author_case(case_dir)
    first = _lock(case_dir)["evaluator_fingerprint"]

    case_yaml = case_dir / "case.yaml"
    raw = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
    raw["module_params"]["environment_object_consistency"]["eoc_text_prompt"] = (
        "tree. lamp post."
    )
    case_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    author_case(case_dir)

    assert _lock(case_dir)["evaluator_fingerprint"] != first


def test_author_case_fingerprint_covers_evaluator_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evaluation.lite.authoring as authoring

    case_dir = write_reference_free_case(tmp_path / "cases")
    evaluator_source = tmp_path / "evaluation" / "module.py"
    evaluator_source.parent.mkdir()
    evaluator_source.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(authoring, "REPO_ROOT", tmp_path)

    author_case(case_dir)
    first = _lock(case_dir)["evaluator_fingerprint"]
    evaluator_source.write_text("VERSION = 2\n", encoding="utf-8")
    author_case(case_dir)

    assert _lock(case_dir)["evaluator_fingerprint"] != first


def test_relock_refreshes_code_revision(tmp_path: Path) -> None:
    case_dir = write_reference_free_case(tmp_path / "cases")
    assert _lock(case_dir)["code_revision"] == "test"

    author_case(case_dir, relock=True)

    revision = _lock(case_dir)["code_revision"]
    assert revision != "test"
    assert revision  # current git HEAD (or "unknown" outside a repo)
