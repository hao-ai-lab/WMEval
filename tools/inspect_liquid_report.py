#!/usr/bin/env python3
"""Pretty-print a LiquidContainer eval report with sanity flags.

Usage:
  python tools/inspect_liquid_report.py reports/liquid/lingbot_step_hold.json
  python tools/inspect_liquid_report.py reports/liquid/*.json   # multi-file table

Surfaces the four sloshing metrics + LPIPS in a compact form, marks
problematic diagnostics (low PIV validity, near-DC dominant frequency,
NaN divergence) so you don't have to hand-scan a 96 KB JSON.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _g(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def _fmt(x: Any, prec: int = 4) -> str:
    if x is None:
        return "—"
    if _is_nan(x):
        return "NaN"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return f"{x:.{prec}g}"
    return str(x)


def _flags(report: dict) -> list[str]:
    flags: list[str] = []
    res = report.get("results", {})

    flow_est = _g(res, "liquid_velocity.diagnostics.flow_estimator", "openpiv")
    for side in ("unity", "wm"):
        validity = _g(res, f"liquid_velocity.diagnostics.piv_validity_{side}")
        mask_frac = _g(res, f"liquid_velocity.diagnostics.liquid_mask_{side}.mean_area_frac")
        if not isinstance(validity, (int, float)):
            continue
        # Validity is bounded by the mask's frame coverage (cells outside the mask
        # are NaN by design). RAFT/PIV is "failing" only when validity is much
        # lower than the mask-fraction ceiling. For OpenPIV we keep the absolute
        # 10% guard as a fallback because cross-correlation can also fail INSIDE
        # the mask on low-texture regions.
        if isinstance(mask_frac, (int, float)) and mask_frac > 0:
            ratio = validity / float(mask_frac)
            # ratio < 0.5: clearly broken (RAFT/PIV failing inside the mask).
            # ratio in [0.5, ~4]: depends on PIV overlap; treat as OK.
            if ratio < 0.5:
                flags.append(
                    f"flow validity {side} = {validity:.1%} vs mask coverage {mask_frac:.1%} "
                    f"(ratio {ratio:.2f}, estimator={flow_est}): estimator is rejecting cells "
                    f"INSIDE the mask"
                    + (" — try --liquid_flow_estimator raft" if flow_est == "openpiv" else
                       " — check SAM2 mask QA, may be tracking the wrong region")
                )
        elif validity < 0.05:
            flags.append(f"flow validity {side} = {validity:.1%} (<5%, estimator={flow_est}): "
                         f"liquid signal too small to evaluate")

    div = _g(res, "liquid_velocity.divergence_residual")
    if _is_nan(div):
        flags.append("divergence_residual is NaN: every PIV stencil had at least one invalid neighbour "
                     "(usually downstream of low PIV validity)")

    for side in ("unity", "wm"):
        pre = _g(res, f"liquid_velocity.diagnostics.mean_speed_{side}_pre_comp_px_per_frame")
        bulk = _g(res, f"liquid_velocity.diagnostics.bottle_bulk_speed_{side}_px_per_frame")
        post = _g(res, f"liquid_velocity.diagnostics.mean_speed_{side}_px_per_frame")
        if not all(isinstance(x, (int, float)) for x in (pre, bulk, post)):
            continue
        # If compensation worked, post ≈ |pre - bulk|. A big residual means either
        # the RAFT signal isn't actually the bottle's translation (RAFT-noise on
        # uniform regions, or background-through-glass leaking into the mask),
        # or the bottle SAM2 centroid isn't tracking the rigid bottle motion.
        expected_post = abs(pre - bulk)
        if post > 5.0 and post > 3.0 * max(expected_post, 1.0):
            flags.append(
                f"bulk-comp ineffective on {side}: post-comp speed {post:.1f} px/frame is much larger "
                f"than |pre - bulk| = |{pre:.1f} - {bulk:.1f}| = {expected_post:.1f}. Either RAFT is "
                f"hallucinating flow on uniform/dark regions, or the bottle centroid trace doesn't "
                f"match the actual rigid bottle motion. velocity EPE/divergence/curl are unreliable; "
                f"trust cosine_similarity and the other three metrics"
            )

    duration = _g(report, "shared_duration_sec")
    nyquist = _g(report, "results.lpips.candidate_metadata.fps", 16) / 2.0
    for variant in ("centroid", "meniscus"):
        block = _g(res, f"slosh_spectrum.{variant}")
        if not isinstance(block, dict) and variant == "centroid":
            block = {"dominant_frequency_hz": _g(res, "slosh_spectrum.dominant_frequency_hz")}
        if not isinstance(block, dict):
            continue
        fu = _g(block, "dominant_frequency_hz.unity")
        fw = _g(block, "dominant_frequency_hz.wm")
        delta = _g(block, "dominant_frequency_hz.delta_hz")
        # Skip the near-DC complaint when unity == wm (identity / self-pair):
        # the peak landing in the lowest bin is expected when there's no real
        # slosh signal, and on identity the delta=0 already proves consistency.
        identity_match = isinstance(delta, (int, float)) and abs(delta) < 1e-9
        if (
            isinstance(fu, (int, float))
            and isinstance(duration, (int, float))
            and duration > 0
            and not identity_match
        ):
            bin_width = 1.0 / duration
            if fu <= bin_width * 1.5:
                flags.append(f"slosh_spectrum {variant} unity peak ({fu:.3f} Hz) is at or near the lowest "
                             f"non-DC bin ({bin_width:.3f} Hz): no oscillatory mode detected on Unity — "
                             "trace dominated by drift or slosh too small to read")
        if isinstance(fw, (int, float)) and nyquist > 0 and fw >= nyquist * 0.85 and not identity_match:
            flags.append(f"slosh_spectrum {variant} wm peak ({fw:.3f} Hz) is near Nyquist "
                         f"({nyquist:.3f} Hz): probably frame-to-frame jitter, not a real slosh mode")

    bt_u = _g(res, "liquid_velocity.diagnostics.bottle_trace_validity_unity")
    bt_w = _g(res, "liquid_velocity.diagnostics.bottle_trace_validity_wm")
    for label, v in [("unity", bt_u), ("wm", bt_w)]:
        if isinstance(v, (int, float)) and v < 0.95:
            flags.append(f"bottle SAM2 trace validity {label} = {v:.1%}: bottle prompt didn't track "
                         "consistently — bulk-motion compensation is partial")

    surf_u = _g(res, "liquid_surface_orientation.surface_tilt_residual_deg.unity")
    if isinstance(surf_u, (int, float)) and surf_u > 5.0:
        flags.append(f"surface tilt residual unity = {surf_u:.2f}° (>5°): expected ~0° for an upright "
                     "static-camera bottle — likely the topmost-row trace is hitting the bottle cap "
                     "or a curved meniscus contour, not the flat air/liquid interface")

    win = _g(res, "liquid_velocity.diagnostics.piv_window_size")
    if isinstance(win, (int, float)) and win > 0:
        win_area = float(win) ** 2
        for side in ("unity", "wm"):
            mean_area = _g(res, f"liquid_velocity.diagnostics.liquid_mask_{side}.mean_area_px")
            if isinstance(mean_area, (int, float)) and mean_area > 0:
                ratio = mean_area / win_area
                if ratio < 9:
                    flags.append(
                        f"liquid mask {side} averages {mean_area:.0f} px ≈ {ratio:.1f}× a single "
                        f"{int(win)}×{int(win)} PIV window — too small for stable correlation; "
                        f"try --liquid_piv_window_size {max(8, int(win)//2)} "
                        f"--liquid_piv_overlap {max(4, int(win)//4)}"
                    )

    return flags


def _print_one(path: Path) -> None:
    report = json.loads(path.read_text())
    res = report.get("results", {})

    print(f"\n{'=' * 78}\n  {path}")
    print(f"  model            : {report.get('model', '—')}")
    print(f"  reference_video  : {report.get('reference_video', '—')}")
    print(f"  candidate_video  : {report.get('candidate_video', '—')}")
    print(f"  shared_duration_s: {_fmt(report.get('shared_duration_sec'))}")
    print(f"  device           : {report.get('device', '—')}")
    print(f"{'=' * 78}")

    print("\n--- ⭐ headline (five-pillar; lower = better except cosine where 1 = best) ---")
    print(f"  cosine_similarity                  : "
          f"{_fmt(_g(res, 'liquid_velocity.cosine_similarity'))}   (1=best)")
    print(f"  volume std delta                   : "
          f"{_fmt(_g(res, 'liquid_volume_conservation.mask_area_std_normalized.delta'))}")
    print(f"  volume drift delta (1/s)           : "
          f"{_fmt(_g(res, 'liquid_volume_conservation.mask_area_drift_per_sec.delta'))}")
    print(f"  surface tilt delta (deg)           : "
          f"{_fmt(_g(res, 'liquid_surface_orientation.surface_tilt_residual_deg.delta'))}")
    print(f"  slosh meniscus spectral_l1         : "
          f"{_fmt(_g(res, 'slosh_spectrum.meniscus.spectral_l1'))}")
    print(f"  lpips mean                         : "
          f"{_fmt(_g(res, 'lpips.lpips_mean'))}")

    flow_est = _g(res, "liquid_velocity.diagnostics.flow_estimator", "openpiv")
    print(f"\n--- liquid_velocity ({flow_est} flow inside the liquid mask, bulk-motion compensated) ---")
    print("  (EPE / divergence / curl below are sanity checks: must be 0 on identity pairs.")
    print("   On real comparisons their magnitudes track RAFT noise, not physics — use cosine_similarity.)")
    print(f"  EPE mean / median / p90 (px/frame): "
          f"{_fmt(_g(res, 'liquid_velocity.epe.mean'))} / "
          f"{_fmt(_g(res, 'liquid_velocity.epe.median'))} / "
          f"{_fmt(_g(res, 'liquid_velocity.epe.p90'))}")
    print(f"  cosine_similarity      : {_fmt(_g(res, 'liquid_velocity.cosine_similarity'))}   "
          "(1=aligned, 0=random, -1=anti)")
    print(f"  divergence_residual    : {_fmt(_g(res, 'liquid_velocity.divergence_residual'))}   "
          "(closer to 0 = both fields equally incompressible)")
    print(f"  curl mean / std error  : "
          f"{_fmt(_g(res, 'liquid_velocity.curl_stats.mean_error'))} / "
          f"{_fmt(_g(res, 'liquid_velocity.curl_stats.std_error'))}")
    print(f"  temporal_consistency   : {_fmt(_g(res, 'liquid_velocity.temporal_consistency'))}   "
          "(wm/unity ratio of frame-to-frame velocity change; 1=as smooth as Unity)")
    print(f"  PIV validity unity / wm: "
          f"{_fmt(_g(res, 'liquid_velocity.diagnostics.piv_validity_unity'))} / "
          f"{_fmt(_g(res, 'liquid_velocity.diagnostics.piv_validity_wm'))}")
    pre_u = _g(res, "liquid_velocity.diagnostics.mean_speed_unity_pre_comp_px_per_frame")
    pre_w = _g(res, "liquid_velocity.diagnostics.mean_speed_wm_pre_comp_px_per_frame")
    bulk_u = _g(res, "liquid_velocity.diagnostics.bottle_bulk_speed_unity_px_per_frame")
    bulk_w = _g(res, "liquid_velocity.diagnostics.bottle_bulk_speed_wm_px_per_frame")
    post_u = _g(res, "liquid_velocity.diagnostics.mean_speed_unity_px_per_frame")
    post_w = _g(res, "liquid_velocity.diagnostics.mean_speed_wm_px_per_frame")
    if any(x is not None for x in (pre_u, pre_w, bulk_u, bulk_w)):
        print(f"  flow speed pre-comp    : unity={_fmt(pre_u)}  wm={_fmt(pre_w)}   px/frame "
              "(raw RAFT magnitude inside mask)")
        print(f"  bottle bulk speed      : unity={_fmt(bulk_u)}  wm={_fmt(bulk_w)}   px/frame "
              "(centroid frame-to-frame delta — what gets subtracted)")
        print(f"  flow speed post-comp   : unity={_fmt(post_u)}  wm={_fmt(post_w)}   px/frame "
              "(should ≈ |pre - bulk| if compensation works)")
    elif post_u is not None or post_w is not None:
        print(f"  mean speed unity / wm  : {_fmt(post_u)} / {_fmt(post_w)}   px/frame inside mask after bulk-comp")
    print(f"  bottle trace validity  : "
          f"{_fmt(_g(res, 'liquid_velocity.diagnostics.bottle_trace_validity_unity'))} / "
          f"{_fmt(_g(res, 'liquid_velocity.diagnostics.bottle_trace_validity_wm'))}")
    print(f"  camera_compensation    : {_fmt(_g(res, 'liquid_velocity.diagnostics.camera_compensation'))}")

    win = _g(res, "liquid_velocity.diagnostics.piv_window_size")
    ovl = _g(res, "liquid_velocity.diagnostics.piv_overlap")
    if win is not None:
        print(f"  PIV window / overlap   : {_fmt(win)} / {_fmt(ovl)} px")
    for side in ("unity", "wm"):
        m = _g(res, f"liquid_velocity.diagnostics.liquid_mask_{side}")
        validity = _g(res, f"liquid_velocity.diagnostics.piv_validity_{side}")
        if isinstance(m, dict):
            mean_frac = m.get("mean_area_frac")
            ratio_str = ""
            if isinstance(validity, (int, float)) and isinstance(mean_frac, (int, float)) and mean_frac > 0:
                # PIV cells are window×window patches on a (window-overlap) stride;
                # 50% overlap means each mask pixel reaches into ~4 neighbouring
                # cells, so a mask of area F can validate up to ~4F worth of cells.
                # ratio < ~0.5×F = estimator failing inside the mask; ratio >> F = OK.
                ratio_str = (
                    f"  ⇒ flow valid / mask = {validity / mean_frac:.2f} "
                    "(>1 means PIV cells overlap mask multiple times; <0.5 = estimator failing inside mask)"
                )
            print(f"  liquid mask {side:<5}      : "
                  f"mean={_fmt(m.get('mean_area_px'))} px "
                  f"({_fmt(mean_frac)} of frame), "
                  f"non-empty {m.get('frames_with_mask', 0)}/{m.get('frames', 0)} frames{ratio_str}")

    print("\n--- liquid_volume_conservation (sealed bottle: area should be flat) ---")
    print(f"  mask_area_std_normalized   unity / wm / delta: "
          f"{_fmt(_g(res, 'liquid_volume_conservation.mask_area_std_normalized.unity'))} / "
          f"{_fmt(_g(res, 'liquid_volume_conservation.mask_area_std_normalized.wm'))} / "
          f"{_fmt(_g(res, 'liquid_volume_conservation.mask_area_std_normalized.delta'))}")
    print(f"  mask_area_drift_per_sec    unity / wm / delta (1/s): "
          f"{_fmt(_g(res, 'liquid_volume_conservation.mask_area_drift_per_sec.unity'))} / "
          f"{_fmt(_g(res, 'liquid_volume_conservation.mask_area_drift_per_sec.wm'))} / "
          f"{_fmt(_g(res, 'liquid_volume_conservation.mask_area_drift_per_sec.delta'))}")

    print("\n--- liquid_surface_orientation (deg from horizontal, RMS over frames) ---")
    print(f"  surface_tilt_residual_deg  unity / wm / delta: "
          f"{_fmt(_g(res, 'liquid_surface_orientation.surface_tilt_residual_deg.unity'))}° / "
          f"{_fmt(_g(res, 'liquid_surface_orientation.surface_tilt_residual_deg.wm'))}° / "
          f"{_fmt(_g(res, 'liquid_surface_orientation.surface_tilt_residual_deg.delta'))}°")

    print("\n--- slosh_spectrum (dominant slosh frequency, bulk-motion compensated) ---")
    for variant in ("centroid", "meniscus"):
        block = _g(res, f"slosh_spectrum.{variant}")
        if not isinstance(block, dict):
            # Older reports only have the top-level (centroid) keys; show those
            # under the centroid heading and skip the meniscus row.
            if variant == "centroid":
                block = {
                    "dominant_frequency_hz": _g(res, "slosh_spectrum.dominant_frequency_hz"),
                    "spectral_l1": _g(res, "slosh_spectrum.spectral_l1"),
                }
            else:
                continue
        print(f"  [{variant}] unity / wm / ratio / delta_hz: "
              f"{_fmt(_g(block, 'dominant_frequency_hz.unity'))} Hz / "
              f"{_fmt(_g(block, 'dominant_frequency_hz.wm'))} Hz / "
              f"{_fmt(_g(block, 'dominant_frequency_hz.ratio'))} / "
              f"{_fmt(_g(block, 'dominant_frequency_hz.delta_hz'))} Hz   "
              f"L1={_fmt(_g(block, 'spectral_l1'))}")

    lpips_mean = _g(res, "lpips.lpips_mean")
    if lpips_mean is not None:
        print("\n--- lpips (perceptual similarity, anchor-sampled) ---")
        print(f"  mean / median / std    : {_fmt(lpips_mean)} / "
              f"{_fmt(_g(res, 'lpips.lpips_median'))} / {_fmt(_g(res, 'lpips.lpips_std'))}")
        print(f"  per-anchor             : {_g(res, 'lpips.lpips_per_pair')}")

    flags = _flags(report)
    if flags:
        print("\n--- ⚠ sanity flags ---")
        for f in flags:
            print(f"  • {f}")
    else:
        print("\n--- sanity flags: none ---")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reports", nargs="+", help="One or more report.json files")
    args = ap.parse_args()
    for raw in args.reports:
        p = Path(raw)
        if p.exists():
            _print_one(p)
        else:
            print(f"[skip] not found: {p}")


if __name__ == "__main__":
    main()
