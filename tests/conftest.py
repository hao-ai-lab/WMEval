from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# External (non-repo) test inputs.
#
# A few tests exercise the Table 4/7 reproduction against data that is not in
# git: the git-ignored ``runs/`` tree produced by local Unity captures /
# reproduction runs, and the sibling paper checkout ``../_arxiv__WorldModel_Eval``
# pinned by experiments/tables_4_7/paper_baseline.yaml.  Such tests are marked
#
#     @pytest.mark.integration("case_data")   # or "synthetic_data", "table_runs", "paper_checkout"
#
# and are skipped (with the first missing path in the reason) when the named
# input is absent, so ``pytest tests`` stays green in a fresh clone/CI, and
# ``pytest -m "not integration"`` deselects them entirely.
# --------------------------------------------------------------------------

TABLE_ROOT = ROOT / "experiments" / "tables_4_7"
RUNS_ROOT = ROOT / "runs"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _paper_checkout_paths() -> list[Path]:
    """Paper sources pinned by paper_baseline.yaml (resolved like load_baseline)."""
    baseline_path = TABLE_ROOT / "paper_baseline.yaml"
    sources = _load_yaml(baseline_path).get("sources", {})
    return [
        (baseline_path.parent / str(source["path"])).resolve()
        for source in sources.values()
    ]


def _manifest_path(raw: str) -> Path:
    # normpath (not resolve) so a symlinked runs/ still counts as being under runs/.
    return Path(os.path.normpath(TABLE_ROOT / raw))


def _under_runs(path: Path) -> bool:
    try:
        path.relative_to(RUNS_ROOT)
    except ValueError:
        return False
    return True


def _table_runs_paths() -> list[Path]:
    """runs/ inputs referenced by the Table 4/7 manifests.

    * every ``first_frame`` of an active cell in cases.yaml,
    * every conditioning-source / calibration file in conditioning_sources.yaml,
    * the ``evaluation_artifacts/`` of every Unity capture those reference.
    """
    paths: list[Path] = []
    cases = _load_yaml(TABLE_ROOT / "cases.yaml").get("cases", [])
    for case in cases:
        if case.get("execution_scope", "active") != "active":
            continue
        if case.get("first_frame"):
            paths.append(_manifest_path(str(case["first_frame"])))

    def walk(node: object) -> Iterable[str]:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"path", "video", "poses", "receipt"} and isinstance(value, str):
                    yield value
                else:
                    yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    conditioning = _load_yaml(TABLE_ROOT / "conditioning_sources.yaml")
    for key in ("sources", "backend_calibrations"):
        for raw in walk(conditioning.get(key, {})):
            paths.append(_manifest_path(raw))

    paths = [path for path in paths if _under_runs(path)]
    captures = RUNS_ROOT / "tables_4_7" / "unity_scene_runs" / "captures"
    capture_dirs = {
        captures / path.relative_to(captures).parts[0]
        for path in paths
        if path.is_relative_to(captures)
    }
    paths.extend(sorted(d / "evaluation_artifacts" for d in capture_dirs))
    return list(dict.fromkeys(paths))


def _case_data_paths() -> list[Path]:
    """Every case-data file listed in the suites' checksum manifests (Hugging Face download)."""
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.sync_lite_data import case_dirs, manifest

    return [case_dir / rel for case_dir in case_dirs() for rel in manifest(case_dir)]


def _synthetic_data_paths() -> list[Path]:
    """Every synthetic-trajectory file listed in synthetic/manifest.json (``--split full`` download)."""
    import json

    suite = ROOT / "lite"
    manifest = json.loads((suite / "synthetic" / "manifest.json").read_text(encoding="utf-8"))
    return [suite / t["path"] / name for t in manifest["trajectories"] for name in t["files"]]


EXTERNAL_INPUTS: dict[str, tuple[str, Callable[[], list[Path]]]] = {
    "case_data": (
        "WMEval Lite case data from the Hugging Face Hub "
        "(python tools/sync_lite_data.py download)",
        _case_data_paths,
    ),
    "synthetic_data": (
        "WMEval Lite synthetic trajectories from the Hugging Face Hub "
        "(python tools/sync_lite_data.py download --split full)",
        _synthetic_data_paths,
    ),
    "table_runs": (
        "git-ignored runs/ Table 4/7 reproduction outputs (Unity captures, "
        "prepared assets, backend calibration)",
        _table_runs_paths,
    ),
    "paper_checkout": (
        "sibling paper checkout ../_arxiv__WorldModel_Eval "
        "(sources pinned by experiments/tables_4_7/paper_baseline.yaml)",
        _paper_checkout_paths,
    ),
}


@lru_cache(maxsize=None)
def missing_external_input(name: str) -> tuple[str, ...]:
    """Return the missing paths for a named external input (empty = present)."""
    _, resolver = EXTERNAL_INPUTS[name]
    return tuple(str(path) for path in resolver() if not path.exists())


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration(*inputs): needs data outside the repository; skipped when "
        "any named input is missing (inputs: " + ", ".join(sorted(EXTERNAL_INPUTS)) + "). "
        "Deselect with -m 'not integration'.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for marker in item.iter_markers("integration"):
            unknown = [name for name in marker.args if name not in EXTERNAL_INPUTS]
            if unknown:
                raise pytest.UsageError(
                    f"{item.nodeid}: unknown integration input(s) {unknown}; "
                    f"known: {sorted(EXTERNAL_INPUTS)}"
                )


def pytest_runtest_setup(item: pytest.Item) -> None:
    for marker in item.iter_markers("integration"):
        for name in marker.args:
            missing = missing_external_input(name)
            if missing:
                description, _ = EXTERNAL_INPUTS[name]
                more = f" (+{len(missing) - 1} more)" if len(missing) > 1 else ""
                pytest.skip(f"integration input '{name}' missing — {description}: {missing[0]}{more}")
