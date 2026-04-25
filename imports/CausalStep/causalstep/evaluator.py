from __future__ import annotations

from causalstep.schema import ChainEvent, VideoInstance, VideoResult
from causalstep.model_interface import VideoQAModel


def evaluate_chain(
    video: VideoInstance,
    model: VideoQAModel,
    causal_fail_advance: bool = True,
) -> VideoResult:
    """Run Algorithm 1: Causal Stepwise Evaluation on a single video.

    Args:
        video: The video instance with segments and QA pairs.
        model: The model to evaluate.
        causal_fail_advance: If True (default, follows pseudocode), on causal
            failure advance to the next segment's descriptive QA. If False
            (follows paper prose), restart from the descriptive QA of the
            same segment where the causal failure occurred.
    """
    n = video.num_segments
    score = 0
    chain_length = 0
    i = 1
    current_type = "descriptive"
    events: list[ChainEvent] = []
    num_restarts = 0
    chain_lengths: list[int] = []
    max_chain_length = 0
    prev_answer = ""

    while i <= n:
        if current_type == "descriptive":
            qa = video.descriptive_qas[i]
            segment = video.segments[i - 1]  # 0-indexed list
            answer = model.answer_descriptive(qa.question, qa.choices, segment)
            correct = answer == qa.correct_answer

            if correct:
                chain_length += 1
                score += 1
                max_chain_length = max(max_chain_length, chain_length)
                events.append(
                    ChainEvent(
                        segment_index=i,
                        question_type="descriptive",
                        question=qa.question,
                        model_answer=answer,
                        correct_answer=qa.correct_answer,
                        is_correct=True,
                        chain_length_at_step=chain_length,
                        score_awarded=1,
                    )
                )
                prev_answer = answer
                i += 1
                current_type = "causal"
            else:
                if chain_length > 0:
                    chain_lengths.append(chain_length)
                    num_restarts += 1
                chain_length = 0
                events.append(
                    ChainEvent(
                        segment_index=i,
                        question_type="descriptive",
                        question=qa.question,
                        model_answer=answer,
                        correct_answer=qa.correct_answer,
                        is_correct=False,
                        chain_length_at_step=0,
                        score_awarded=0,
                    )
                )
                i += 1
                current_type = "descriptive"
            continue

        if current_type == "causal":
            if i > n:
                break

            qa = video.causal_qas[i]
            seg_prev = video.segments[i - 2]  # S_{i-1}, 0-indexed
            seg_curr = video.segments[i - 1]  # S_i, 0-indexed
            answer = model.answer_causal(
                qa.question, qa.choices, prev_answer, (seg_prev, seg_curr)
            )
            correct = answer == qa.correct_answer

            if correct:
                chain_length += 1
                score += chain_length
                max_chain_length = max(max_chain_length, chain_length)
                events.append(
                    ChainEvent(
                        segment_index=i,
                        question_type="causal",
                        question=qa.question,
                        model_answer=answer,
                        correct_answer=qa.correct_answer,
                        is_correct=True,
                        chain_length_at_step=chain_length,
                        score_awarded=chain_length,
                    )
                )
                prev_answer = answer
                i += 1
                current_type = "causal"
            else:
                if chain_length > 0:
                    chain_lengths.append(chain_length)
                    num_restarts += 1
                chain_length = 0
                events.append(
                    ChainEvent(
                        segment_index=i,
                        question_type="causal",
                        question=qa.question,
                        model_answer=answer,
                        correct_answer=qa.correct_answer,
                        is_correct=False,
                        chain_length_at_step=0,
                        score_awarded=0,
                    )
                )
                if causal_fail_advance:
                    i += 1
                # If not advancing, i stays the same -> retry desc of same segment
                current_type = "descriptive"

    # Record final chain if it was still active
    if chain_length > 0:
        chain_lengths.append(chain_length)

    return VideoResult(
        video_id=video.video_id,
        events=events,
        total_score=score,
        max_chain_length=max_chain_length,
        num_restarts=num_restarts,
        chain_lengths=chain_lengths,
        num_segments=n,
    )


def evaluate_descriptive_isolated(
    video: VideoInstance,
    model: VideoQAModel,
) -> list[tuple[int, bool]]:
    """Evaluate each descriptive QA independently (for DUA metric)."""
    results: list[tuple[int, bool]] = []
    for i in range(1, video.num_segments + 1):
        qa = video.descriptive_qas[i]
        segment = video.segments[i - 1]
        answer = model.answer_descriptive(qa.question, qa.choices, segment)
        results.append((i, answer == qa.correct_answer))
    return results


def evaluate_causal_isolated(
    video: VideoInstance,
    model: VideoQAModel,
) -> list[tuple[int, bool]]:
    """Evaluate each causal QA independently (for ICRA metric).

    Uses the ground truth correct answer of the preceding descriptive QA
    as the prev_answer context, simulating an ideal chain.
    """
    results: list[tuple[int, bool]] = []
    for i in range(2, video.num_segments + 1):
        qa = video.causal_qas[i]
        # Provide ground truth prev answer from the preceding descriptive QA
        prev_correct = video.descriptive_qas[i - 1].correct_answer
        seg_prev = video.segments[i - 2]
        seg_curr = video.segments[i - 1]
        answer = model.answer_causal(
            qa.question, qa.choices, prev_correct, (seg_prev, seg_curr)
        )
        results.append((i, answer == qa.correct_answer))
    return results
