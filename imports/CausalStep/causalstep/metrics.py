from __future__ import annotations

from causalstep.schema import AggregateResults, VideoResult


def chain_success_rate(results: list[VideoResult]) -> float:
    """CSR: Proportion of videos where the model completed the entire chain.

    A full chain for a video with N segments means max_chain_length == N
    (1 descriptive + N-1 causal questions all correct in sequence).
    """
    if not results:
        return 0.0
    successes = sum(
        1 for r in results if r.max_chain_length >= r.num_segments
    )
    return (successes / len(results)) * 100


def avg_max_chain_length(results: list[VideoResult]) -> float:
    """AMCL: Average of the longest uninterrupted chain per video."""
    if not results:
        return 0.0
    return sum(r.max_chain_length for r in results) / len(results)


def max_chain_length(results: list[VideoResult]) -> int:
    """MCL: Global maximum chain length across all videos."""
    if not results:
        return 0
    return max(r.max_chain_length for r in results)


def restart_frequency(results: list[VideoResult]) -> float:
    """RF: Average number of chain restarts per video. Lower is better."""
    if not results:
        return 0.0
    return sum(r.num_restarts for r in results) / len(results)


def weighted_score(results: list[VideoResult]) -> float:
    """WS: Average total weighted score across videos."""
    if not results:
        return 0.0
    return sum(r.total_score for r in results) / len(results)


def descriptive_understanding_accuracy(
    isolated_desc_results: list[list[tuple[int, bool]]],
) -> float:
    """DUA: Accuracy across all descriptive QAs evaluated in isolation."""
    total = 0
    correct = 0
    for video_results in isolated_desc_results:
        for _, is_correct in video_results:
            total += 1
            if is_correct:
                correct += 1
    if total == 0:
        return 0.0
    return (correct / total) * 100


def isolated_causal_reasoning_accuracy(
    isolated_causal_results: list[list[tuple[int, bool]]],
) -> float:
    """ICRA: Accuracy across all causal QAs evaluated in isolation."""
    total = 0
    correct = 0
    for video_results in isolated_causal_results:
        for _, is_correct in video_results:
            total += 1
            if is_correct:
                correct += 1
    if total == 0:
        return 0.0
    return (correct / total) * 100


def compute_all_metrics(
    chain_results: list[VideoResult],
    isolated_desc_results: list[list[tuple[int, bool]]],
    isolated_causal_results: list[list[tuple[int, bool]]],
) -> AggregateResults:
    """Compute all 7 diagnostic metrics and return AggregateResults."""
    return AggregateResults(
        video_results=chain_results,
        csr=chain_success_rate(chain_results),
        amcl=avg_max_chain_length(chain_results),
        mcl=max_chain_length(chain_results),
        rf=restart_frequency(chain_results),
        ws=weighted_score(chain_results),
        dua=descriptive_understanding_accuracy(isolated_desc_results),
        icra=isolated_causal_reasoning_accuracy(isolated_causal_results),
    )
