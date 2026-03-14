from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from .context import EvaluationContext
from .modules.lpips import analyze_lpips_from_context

EvaluationModule = Callable[[EvaluationContext, Optional[str]], Dict[str, Any]]


def _run_lpips(context: EvaluationContext, device: Optional[str] = None) -> Dict[str, Any]:
    if not context.reference_video:
        return {
            "module": "lpips",
            "error": "reference_video is required for LPIPS",
        }
    return analyze_lpips_from_context(context=context, device=device)


MODULE_REGISTRY: Dict[str, EvaluationModule] = {
    "lpips": _run_lpips,
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
