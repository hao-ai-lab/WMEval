from __future__ import annotations

import json

from causalstep.schema import AggregateResults, load_dataset
from causalstep.model_interface import VideoQAModel, PrecomputedModel
from causalstep.evaluator import (
    evaluate_chain,
    evaluate_descriptive_isolated,
    evaluate_causal_isolated,
)
from causalstep.metrics import compute_all_metrics


def run_evaluation(
    dataset_path: str,
    model: VideoQAModel,
    output_path: str | None = None,
    causal_fail_advance: bool = True,
    verbose: bool = False,
) -> AggregateResults:
    """Main entry point: load data, run evaluations, compute metrics.

    Args:
        dataset_path: Path to the benchmark JSON file.
        model: The model to evaluate.
        output_path: Optional path to save results JSON.
        causal_fail_advance: Algorithm 1 behavior on causal failure.
        verbose: Print per-video traces.
    """
    videos = load_dataset(dataset_path)

    chain_results = []
    desc_isolated = []
    causal_isolated = []

    for video in videos:
        # Set video_id on PrecomputedModel if applicable
        if isinstance(model, PrecomputedModel):
            model.set_video_id(video.video_id)

        chain_result = evaluate_chain(video, model, causal_fail_advance)
        chain_results.append(chain_result)

        desc_iso = evaluate_descriptive_isolated(video, model)
        desc_isolated.append(desc_iso)

        causal_iso = evaluate_causal_isolated(video, model)
        causal_isolated.append(causal_iso)

        if verbose:
            _print_video_trace(chain_result)

    results = compute_all_metrics(chain_results, desc_isolated, causal_isolated)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

    print_results_table(results)
    return results


def print_results_table(results: AggregateResults) -> None:
    """Print a formatted table of evaluation metrics."""
    metrics = [
        ("CSR (%)", f"{results.csr:.1f}"),
        ("AMCL", f"{results.amcl:.2f}"),
        ("MCL", str(results.mcl)),
        ("RF", f"{results.rf:.2f}"),
        ("WS", f"{results.ws:.2f}"),
        ("DUA (%)", f"{results.dua:.1f}"),
        ("ICRA (%)", f"{results.icra:.1f}"),
    ]

    name_width = max(len(name) for name, _ in metrics) + 2
    val_width = max(len(val) for _, val in metrics) + 2
    total_width = name_width + val_width + 3  # borders and separator

    print()
    print("=" * total_width)
    header = "CausalStep Evaluation Results"
    print(f" {header:^{total_width - 2}} ")
    print("=" * total_width)
    print(f" {'Metric':<{name_width}}| {'Value':>{val_width}}")
    print("-" * total_width)
    for name, val in metrics:
        print(f" {name:<{name_width}}| {val:>{val_width}}")
    print("=" * total_width)
    print(f" Videos evaluated: {len(results.video_results)}")
    print()


def _print_video_trace(result) -> None:
    """Print detailed trace for a single video."""
    print(f"\n--- Video: {result.video_id} ---")
    print(f"  Score: {result.total_score} | Max Chain: {result.max_chain_length} | Restarts: {result.num_restarts}")
    for event in result.events:
        status = "OK" if event.is_correct else "FAIL"
        print(
            f"  S{event.segment_index} [{event.question_type[:4]}] "
            f"{status} (ans={event.model_answer}, correct={event.correct_answer}) "
            f"chain={event.chain_length_at_step} +{event.score_awarded}"
        )
