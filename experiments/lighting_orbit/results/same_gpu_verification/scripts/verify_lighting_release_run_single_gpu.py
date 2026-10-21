#!/usr/bin/env python3
"""Run the standalone Lite release, compare all scores, then resume."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(root, output, deadline, *, gpu_count=2):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from tools.build_lighting_release_package import verify
    rows = [json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()]
    workers = []
    status = {"schema_version": 1, "status": "running", "deadline_utc": deadline.isoformat(),
              "bundle_manifest_sha256": sha(root / "MANIFEST.json"), "required_jobs": 15}
    report = output / "verification.json"
    def save():
        report.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    save()
    try:
        verify(root)
        for index in range(gpu_count):
            manifest = output / f"worker_{index}.jsonl"
            subset = rows[index::gpu_count]
            # The split manifest lives outside the bundle; fix relative input location.
            manifest.write_text("".join(json.dumps({**r, "candidate_video": str(
                root / r["candidate_video"])}) + "\n" for r in subset))
            log = (output / f"worker_{index}.log").open("w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(index),
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            cmd = [sys.executable, str(root / "tools/run_wmeval_lite.py"), "run",
                   "--suite", str(root / "eval_sets/wmeval_lite_v1"),
                   "--dimensions", "lighting_shading", "--candidate-manifest", str(manifest),
                   "--output-dir", str(output / f"worker_{index}"), "--device", "cuda",
                   "--run-id", "lighting-release-fresh"]
            process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            workers.append((process, log))
        while any(p.poll() is None for p, _ in workers):
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("release verification worker deadline reached")
            completed = sum(len((output / f"worker_{i}/per_case_results.jsonl").read_text().splitlines())
                            for i in range(gpu_count)
                            if (output / f"worker_{i}/per_case_results.jsonl").exists())
            status["completed_rows_observed"] = completed
            save()
            time.sleep(2)
        if any(p.returncode != 0 for p, _ in workers):
            raise RuntimeError("a Lite worker failed; retain logs and partial rows")
        all_rows = []
        for i in range(gpu_count):
            all_rows += [json.loads(line) for line in
                         (output / f"worker_{i}/per_case_results.jsonl").read_text().splitlines()]
        wanted = {(r["model"], str(r["seed"])): r["metrics"]
                  for r in json.loads((root / "expected_scores.json").read_text())["rows"]}
        actual = {(r["model"], str(r["seed"])): r for r in all_rows}
        if len(all_rows) != 15 or set(actual) != set(wanted):
            raise ValueError("missing or duplicate formal score rows")
        differences = []
        max_error = 0.
        count = 0
        for identity, expected in wanted.items():
            row = actual[identity]
            if row["status"] != "completed" or set(row["metrics"]) != set(expected):
                raise ValueError(f"incomplete metrics: {identity}")
            for metric, value in expected.items():
                if not math.isfinite(float(row['metrics'][metric])) or not math.isfinite(float(value)):
                    raise ValueError(f'non-finite metric: {identity}/{metric}')
                error = abs(float(row["metrics"][metric]) - float(value))
                max_error = max(max_error, error); count += 1
                if error > 1e-6:
                    differences.append({"model": identity[0], "seed": identity[1],
                                        "metric": metric, "absolute_error": error})
        from evaluation.lite.aggregate import build_aggregate, write_aggregate_json
        from evaluation.lite.suite_loader import load_suite
        combined = output / "combined"
        combined.mkdir()
        path = combined / "per_case_results.jsonl"
        all_rows.sort(key=lambda r: (r["model"], str(r["seed"])))
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in all_rows))
        write_aggregate_json(combined / "aggregate_summary.json",
                             build_aggregate(load_suite(root / "eval_sets/wmeval_lite_v1"), all_rows))
        before = sha(path)
        command = [sys.executable, str(root / "tools/run_wmeval_lite.py"), "run",
                   "--suite", str(root / "eval_sets/wmeval_lite_v1"),
                   "--dimensions", "lighting_shading", "--candidate-manifest",
                   str(root / "candidates.jsonl"), "--output-dir", str(combined),
                   "--device", "cuda", "--resume", "--run-id", "lighting-release-resume"]
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("no time left for actual Lite resume verification")
        with (output / "resume.log").open("w") as log:
            subprocess.run(command, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=remaining)
        after = sha(path)
        if before != after:
            raise ValueError("resume changed the fifteen fresh score rows")
        status.update(status="passed" if not differences else "failed_metric_comparison",
            completed_rows=15, compared_metrics=count, comparison_tolerance=1e-6,
            maximum_absolute_metric_error=max_error, differences=differences,
            resume_rows_unchanged=True, fresh_results_sha256=before,
            resume_results_sha256=after, completed_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        if differences:
            raise ValueError("fresh release metrics differ from verified original scores")
    except BaseException as error:
        for process, _ in workers:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, _ in workers:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
        status.update(status=status.get("status") if status.get("status", "").startswith("failed") else "failed",
                      error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        for _, log in workers:
            log.close()
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--gpu-count", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
        parser.error("requires a future timezone-aware deadline")
    print(json.dumps(run(args.root, args.output, deadline, gpu_count=args.gpu_count)), flush=True)


if __name__ == "__main__":
    main()
