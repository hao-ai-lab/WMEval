from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".githooks"


def test_hooks_are_executable() -> None:
    for name in ("check-lite-data", "post-checkout", "post-merge"):
        assert os.access(HOOKS / name, os.X_OK), f"{name} must be executable for git to run it"


def test_hooks_never_block_git() -> None:
    # Whatever the arguments (unknown commits, file checkouts), the hooks only warn.
    for cmd in ([HOOKS / "check-lite-data", "not-a-commit", "HEAD"],
                [HOOKS / "post-checkout", "HEAD", "HEAD", "0"],
                [HOOKS / "post-checkout", "HEAD", "HEAD", "1"]):
        assert subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True).returncode == 0
