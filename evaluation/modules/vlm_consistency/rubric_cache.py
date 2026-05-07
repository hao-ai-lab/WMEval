
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .schemas import Rubric


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)


def _rubric_path(
    rubric_dir: str,
    scene_name: str,
    provider: str,
    model: str,
    prompt_version: str,
) -> Path:
    fname = f"{_sanitize(provider)}_{_sanitize(model)}_{_sanitize(prompt_version)}.json"
    return Path(rubric_dir) / _sanitize(scene_name) / fname


def save_rubric(rubric: Rubric, rubric_dir: str, prompt_version: str = "v1") -> Path:
    meta = rubric.metadata
    path = _rubric_path(
        rubric_dir,
        rubric.scene_name,
        meta.generator_provider,
        meta.generator_model,
        prompt_version,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rubric.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_rubric(
    scene_name: str,
    provider: str,
    model: str,
    prompt_version: str,
    rubric_dir: str,
) -> Rubric:
    path = _rubric_path(rubric_dir, scene_name, provider, model, prompt_version)
    if not path.is_file():
        raise FileNotFoundError(f"No cached rubric at {path}")
    return Rubric.model_validate_json(path.read_text(encoding="utf-8"))


def find_rubric(
    scene_name: str,
    prompt_version: str,
    rubric_dir: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Path]:
    scene_dir = Path(rubric_dir) / _sanitize(scene_name)
    if not scene_dir.is_dir():
        return None
    ver = _sanitize(prompt_version)
    for p in sorted(scene_dir.glob("*.json")):
        if ver not in p.stem:
            continue
        if provider and _sanitize(provider) not in p.stem:
            continue
        if model and _sanitize(model) not in p.stem:
            continue
        return p
    return None

