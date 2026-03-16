from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from .context import EvaluationContext
from .modules.clip4clip import run_clip4clip
from .modules.lpips import analyze_lpips_from_context

EvaluationModule = Callable[[EvaluationContext, Optional[str]], Dict[str, Any]]


def _run_lpips(context: EvaluationContext, device: Optional[str] = None) -> Dict[str, Any]:
    try:
        from .modules.lpips import analyze_lpips_from_context
    except Exception as exc:
        return {
            "module": "lpips",
            "error": f"failed to import lpips module: {exc}",
        }

    if not context.reference_video:
        return {
            "module": "lpips",
            "error": "reference_video is required for LPIPS",
        }
    return analyze_lpips_from_context(context=context, device=device)


def _run_lighting(context: EvaluationContext, device: Optional[str] = None) -> Dict[str, Any]:
    del device
    try:
        from .modules.lighting import analyze_lighting_from_context
    except Exception as exc:
        return {
            "module": "lighting",
            "error": f"failed to import lighting module: {exc}",
        }
    return analyze_lighting_from_context(context=context)


def _run_motion(context: EvaluationContext, device: Optional[str] = None) -> Dict[str, Any]:
    try:
        from .modules.motion import analyze_motion_from_context
    except Exception as exc:
        return {
            "module": "motion",
            "error": f"failed to import motion module: {exc}",
        }
    return analyze_motion_from_context(context=context, device=device)


def _run_shadow(context: EvaluationContext, device: Optional[str] = None) -> Dict[str, Any]:
    try:
        from .modules.shadow import analyze_shadow_from_context
    except Exception as exc:
        return {
            "module": "shadow",
            "error": f"failed to import shadow module: {exc}",
        }
    return analyze_shadow_from_context(context=context, device=device)

def _run_clip4clip(context: EvaluationContext, device: Optional[str] = None) -> Dict[str, Any]:
    return run_clip4clip(context=context, device=device)


MODULE_REGISTRY: Dict[str, EvaluationModule] = {
    "clip4clip": _run_clip4clip,
    "lpips": _run_lpips,
    "lighting": _run_lighting,
    "motion": _run_motion,
    "shadow": _run_shadow,
}


def get_available_modules() -> List[str]:
    return sorted(MODULE_REGISTRY.keys())


def normalize_modules(modules: Iterable[str]) -> List[str]:
    normalized = []
    for module in modules:
        name = module.strip().lower()
        if name:
            normalized.append(name)
    return normalized


def run_modules(
    context: EvaluationContext,
    modules: Iterable[str],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    requested = normalize_modules(modules)
    results: Dict[str, Any] = {}

    for module_name in requested:
        runner = MODULE_REGISTRY.get(module_name)
        if runner is None:
            results[module_name] = {
                "module": module_name,
                "error": (
                    f"unsupported module: {module_name}. "
                    f"Available modules: {', '.join(get_available_modules())}"
                ),
            }
            continue
        results[module_name] = runner(context, device=device)

    return results
