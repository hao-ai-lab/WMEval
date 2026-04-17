"""Delta computation for CausalStep dual-video evaluation.

Convention: positive delta always means WM is worse than Unity.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

HIGHER_IS_BETTER = ("csr", "amcl", "mcl", "ws", "dua", "icra")
LOWER_IS_BETTER = ("rf",)
ALL_METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER

DELTA_CONVENTION_STRING = (
    "positive = WM worse than Unity; RF uses wm-unity, others use unity-wm"
)


def compute_delta(unity: Mapping[str, Any], wm: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [m for m in ALL_METRICS if m not in unity or m not in wm]
    if missing:
        raise ValueError(f"missing metrics: {missing}")
    out: Dict[str, Any] = {}
    for m in HIGHER_IS_BETTER:
        out[m] = unity[m] - wm[m]
    for m in LOWER_IS_BETTER:
        out[m] = wm[m] - unity[m]
    return out
