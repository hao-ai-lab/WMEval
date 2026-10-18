#!/usr/bin/env python3
"""Wait for, audit, and export the requested open-model Table 4/7 results."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_open_model_table_completion import ACTIVE_CASES, audit
from tools.export_open_model_tables_latex import rows_for, table4_block, table7_block


FINAL = ROOT / "reports/tables_4_7/final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readiness_errors() -> list[str]:
    errors: list[str] = []
    for model, cases in ACTIVE_CASES.items():
        for case_id in cases:
            try:
                rows_for(model, case_id)
            except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{model}/{case_id}: {exc}")
    try:
        table4_block("matrix_game_2")
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"Table 4: {exc}")
    for model in ("lingbot_world_cam", "matrix_game_3", "matrix_game_2"):
        try:
            table7_block(model)
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Table 7 {model}: {exc}")
    return errors


def finalize() -> dict[str, Any]:
    FINAL.mkdir(parents=True, exist_ok=True)
    audit_report = audit()
    audit_path = FINAL / "completion_audit.json"
    audit_path.write_text(
        json.dumps(audit_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit_report["status"] != "complete":
        raise ValueError("full completion audit failed: " + "; ".join(audit_report["errors"]))
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/export_open_model_tables_latex.py"),
            "--output-dir",
            str(FINAL),
        ],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"LaTeX export failed with exit code {completed.returncode}")
    stats = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/summarize_open_model_statistical_tests.py"),
            "--output-dir",
            str(FINAL),
        ],
        cwd=ROOT,
        check=False,
    )
    if stats.returncode:
        raise RuntimeError(f"statistical summary failed with exit code {stats.returncode}")
    artifacts = [
        audit_path,
        FINAL / "table4.tex",
        FINAL / "table7.tex",
        FINAL / "export_receipt.json",
        FINAL / "statistical_comparisons.json",
        FINAL / "statistical_comparisons.csv",
    ]
    if any(not path.is_file() for path in artifacts):
        raise ValueError("finalizer did not produce every required artifact")
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "scope_case_count": 38,
        "artifacts": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
    }
    (FINAL / "finalization_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    while True:
        errors = readiness_errors()
        if not errors:
            break
        if not args.wait:
            print(json.dumps({"status": "incomplete", "errors": errors}, indent=2))
            return 2
        time.sleep(args.poll_seconds)
    try:
        receipt = finalize()
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
