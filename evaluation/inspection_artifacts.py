from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import cv2
import numpy as np

from .context import EvaluationContext


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _hconcat_ref_cand(ref_bgr: np.ndarray, cand_bgr: np.ndarray, max_width: int = 1024) -> np.ndarray:
    h = min(ref_bgr.shape[0], cand_bgr.shape[0])
    if ref_bgr.shape[0] != h or cand_bgr.shape[0] != h:
        ref_bgr = cv2.resize(ref_bgr, (int(ref_bgr.shape[1] * h / ref_bgr.shape[0]), h))
        cand_bgr = cv2.resize(cand_bgr, (int(cand_bgr.shape[1] * h / cand_bgr.shape[0]), h))
    pair = np.hstack([ref_bgr, cand_bgr])
    if pair.shape[1] > max_width:
        scale = max_width / pair.shape[1]
        pair = cv2.resize(
            pair,
            (int(pair.shape[1] * scale), int(pair.shape[0] * scale)),
        )
    return pair


def inspection_base_dir(artifacts: Mapping[str, Any]) -> Path:
    root = Path(
        str(artifacts.get("inspection_root", "reports/world_model_lpips/qa"))
    )
    run_id = str(artifacts.get("inspection_run_id") or "run").strip() or "run"
    sample_id = str(artifacts.get("inspection_sample_id") or "sample").strip() or "sample"
    return root / run_id / sample_id


def write_readme(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_temporal_inspection(
    *,
    artifacts: Mapping[str, Any],
    context: EvaluationContext,
    fractions: Sequence[float],
    ref_indices: Sequence[int],
    cand_indices: Sequence[int],
    scores: Sequence[float],
    ref_frames: Sequence[np.ndarray],
    cand_frames: Sequence[np.ndarray],
) -> None:
    base = inspection_base_dir(artifacts)
    tdir = base / "temporal"
    _safe_mkdir(tdir)
    rows: List[str] = [
        "| anchor | fraction | ref_frame | cand_frame | LPIPS |",
        "|--------|----------|-----------|------------|-------|",
    ]
    for i, f in enumerate(fractions):
        if i >= len(scores):
            break
        sub = tdir / f"anchor_{i:02d}_f{float(f):.4f}".replace(".", "p")
        _safe_mkdir(sub)
        ridx = ref_indices[i]
        cidx = cand_indices[i]
        cv2.imwrite(str(sub / "ref.png"), ref_frames[i])
        cv2.imwrite(str(sub / "cand.png"), cand_frames[i])
        pair = _hconcat_ref_cand(ref_frames[i], cand_frames[i])
        cv2.imwrite(str(sub / "pair.png"), pair)
        (sub / "score.txt").write_text(f"{scores[i]}\n", encoding="utf-8")
        rows.append(f"| {i} | {f} | {ridx} | {cidx} | {scores[i]:.6f} |")

    meta = {
        "lpips_sampling": "temporal",
        "lpips_temporal_fractions": list(map(float, fractions)),
        "reference_video": context.reference_video,
        "candidate_video": context.candidate_video,
        "reference_frame_indices": list(map(int, ref_indices)),
        "candidate_frame_indices": list(map(int, cand_indices)),
        "lpips_per_pair": list(map(float, scores)),
    }
    (tdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    readme = base / "README.md"
    write_readme(
        readme,
        [
            "# LPIPS inspection (temporal anchors)",
            f"Reference: `{context.reference_video}`",
            f"Candidate: `{context.candidate_video}`",
            "",
            *rows,
        ],
    )


def write_causal_inspection(
    *,
    artifacts: Mapping[str, Any],
    context: EvaluationContext,
    result: Mapping[str, Any],
) -> None:
    from .video_io import load_frames_by_indices

    base = inspection_base_dir(artifacts)
    cdir = base / "causal"
    _safe_mkdir(cdir)
    (cdir / "judge_reference.json").write_text(
        json.dumps(result.get("judge_reference"), indent=2, default=str),
        encoding="utf-8",
    )
    (cdir / "judge_candidate.json").write_text(
        json.dumps(result.get("judge_candidate"), indent=2, default=str),
        encoding="utf-8",
    )

    matched = result.get("lpips_per_matched_event") or []
    mdir = cdir / "matched"
    _safe_mkdir(mdir)
    rows: List[str] = [
        "| match | side | e_ref | e_cand | offset | LPIPS |",
        "|-------|------|-------|--------|--------|-------|",
    ]
    for block in matched:
        if not isinstance(block, dict):
            continue
        mi = int(block.get("match_index", 0))
        side = block.get("side", "")
        e_ref = block.get("e_ref")
        e_cand = block.get("e_cand")
        evdir = mdir / f"evt_{mi:02d}_{side}"
        _safe_mkdir(evdir)
        for row in block.get("lpips_per_offset") or []:
            if not isinstance(row, dict):
                continue
            off = int(row["offset"])
            if off < 0:
                off_label = f"off_m{abs(off)}"
            else:
                off_label = f"off_p{off}"
            odir = evdir / off_label
            _safe_mkdir(odir)
            ri = int(row["ref_frame"])
            ci = int(row["cand_frame"])
            fr = load_frames_by_indices(context.reference_video, [ri])
            fc = load_frames_by_indices(context.candidate_video, [ci])
            if fr and fc:
                cv2.imwrite(str(odir / "ref.png"), fr[0])
                cv2.imwrite(str(odir / "cand.png"), fc[0])
                cv2.imwrite(str(odir / "pair.png"), _hconcat_ref_cand(fr[0], fc[0]))
            (odir / "score.txt").write_text(f"{row.get('lpips')}\n", encoding="utf-8")
            rows.append(
                f"| {mi} | {side} | {e_ref} | {e_cand} | {off} | {row.get('lpips')} |"
            )

    ur = cdir / "unmatched_reference.json"
    ur.write_text(json.dumps(result.get("unmatched_reference_events"), indent=2), encoding="utf-8")
    uc = cdir / "unmatched_candidate.json"
    uc.write_text(json.dumps(result.get("unmatched_candidate_events"), indent=2), encoding="utf-8")

    (cdir / "summary.json").write_text(json.dumps(dict(result), indent=2, default=str), encoding="utf-8")

    readme = base / "README.md"
    write_readme(
        readme,
        [
            "# LPIPS inspection (causal VLM)",
            f"Reference: `{context.reference_video}`",
            f"Candidate: `{context.candidate_video}`",
            f"matched_reference_event_fraction: {result.get('matched_reference_event_fraction')}",
            "",
            *rows,
        ],
    )
