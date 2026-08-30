"""Tests for case-bundle authoring (checksums + evaluator lock refresh).

Regression focus: the evaluator fingerprint is a derived value of
checksums+protocol and must be recomputed on EVERY re-author — a bundle edit
followed by a default (non --relock) re-author must never leave a stale
fingerprint stamped into evaluation rows.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from evaluation.lite.authoring import author_case
from evaluation.lite.case_loader import load_case
from evaluation.lite.validation import validate_case_bundle
from lite_suite_test_utils import write_reference_free_case


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
    # with the DEFAULT command (what PENDING.md step 3 prescribes)
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


def test_relock_refreshes_code_revision(tmp_path: Path) -> None:
    case_dir = write_reference_free_case(tmp_path / "cases")
    assert _lock(case_dir)["code_revision"] == "test"

    author_case(case_dir, relock=True)

    revision = _lock(case_dir)["code_revision"]
    assert revision != "test"
    assert revision  # current git HEAD (or "unknown" outside a repo)
