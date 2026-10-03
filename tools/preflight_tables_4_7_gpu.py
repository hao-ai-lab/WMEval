#!/usr/bin/env python3
"""Fail-closed preflight for the portable Table 4/7 CUDA rerun."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_table_gpu_handoff import DEFAULT_MANIFEST, verify_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--allow-no-cuda", action="store_true",
        help="development-only: verify assets/imports without requiring CUDA",
    )
    args = parser.parse_args()
    checks: dict[str, object] = {}
    errors: list[str] = []

    if not args.manifest.is_file():
        errors.append(f"handoff manifest is missing: {args.manifest}")
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest_errors = verify_manifest(manifest)
        checks["handoff"] = {
            "file_count": manifest.get("file_count"),
            "total_bytes": manifest.get("total_bytes"),
            "active_generation_count": manifest.get("active_generation_count"),
        }
        errors.extend(manifest_errors)

    ffmpeg = shutil.which("ffmpeg")
    checks["ffmpeg"] = ffmpeg
    if ffmpeg is None:
        errors.append("ffmpeg is not on PATH")

    required_imports = (
        "cv2", "numpy", "yaml", "transformers", "cjm_byte_track", "sam2"
    )
    missing_imports = [
        name for name in required_imports if importlib.util.find_spec(name) is None
    ]
    checks["required_imports"] = {
        "required": list(required_imports), "missing": missing_imports
    }
    errors.extend(f"missing Python package: {name}" for name in missing_imports)

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        checks["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ] if cuda_available else [],
        }
        if not cuda_available and not args.allow_no_cuda:
            errors.append("torch.cuda.is_available() is false")
    except Exception as exc:
        checks["torch"] = {"error": str(exc)}
        errors.append(f"torch import failed: {exc}")

    checks["runtime_model_artifacts"] = {
        "status": "network_or_prepopulated_cache_required",
        "models": [
            "IDEA-Research/grounding-dino-tiny",
            "facebook/sam2.1-hiera-large",
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
            "facebookresearch/dinov2:dinov2_vits14",
        ],
    }
    payload = {
        "status": "ready" if not errors else "blocked",
        "checks": checks,
        "errors": errors,
        "run_command": (
            "scripts/run_tables_4_7_gpu.sh" if not errors else None
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
