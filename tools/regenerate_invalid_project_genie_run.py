#!/usr/bin/env python3
"""Archive one invalid Table 4/7 repeat and regenerate it through Project Genie."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _prepared_command(case_id: str, repeat_id: str) -> list[str]:
    queue = ROOT / "runs/tables_4_7/prepared_assets/generation_queue.jsonl"
    if not queue.is_file():
        raise FileNotFoundError(
            f"{queue} is missing; run tools/prepare_table_assets.py first"
        )
    for line in queue.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("case_id") != case_id or row.get("repeat_id") != repeat_id:
            continue
        if row.get("status") != "ready" or not isinstance(row.get("command"), list):
            raise ValueError(f"{case_id}/{repeat_id} is not runnable in {queue}")
        command = [str(value) for value in row["command"]]
        if "--repeat" not in command:
            command.extend(["--repeat", repeat_id])
        return command
    raise ValueError(f"{case_id}/{repeat_id} is absent from {queue}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--repeat", required=True, choices=[f"run{i}" for i in range(5)])
    parser.add_argument("--reason", default="required_metric_null")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.case.split("/")) != 3 or not args.case.startswith(("table4/", "table7/")):
        parser.error("--case must be a canonical table4/... or table7/... ID")

    source = ROOT / "runs/project_genie" / args.case / args.repeat
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_parent = ROOT / "runs/project_genie/invalid_archive" / args.case
    archive = archive_parent / f"{args.repeat}_{stamp}"
    source_already_archived = False
    if not source.is_dir():
        prior_archives = sorted(archive_parent.glob(f"{args.repeat}_*"))
        if not prior_archives:
            raise FileNotFoundError(source)
        archive = prior_archives[-1]
        source_already_archived = True
    command = _prepared_command(args.case, args.repeat)
    plan = {"case_id": args.case, "repeat_id": args.repeat, "reason": args.reason,
            "source": str(source), "archive": str(archive),
            "source_already_archived": source_already_archived, "command": command}
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    launch = subprocess.run(
        [command[0], str(ROOT / "tools/project_genie_agent.py"), "launch"],
        cwd=ROOT,
        check=False,
    )
    if launch.returncode:
        return launch.returncode
    if not source_already_archived:
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise FileExistsError(archive)
        shutil.move(str(source), str(archive))
        (archive / "invalid_reason.json").write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8"
        )
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
