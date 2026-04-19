#!/usr/bin/env python3
"""Command-line interface for CausalStep evaluation."""

import argparse
import sys

from causalstep.model_interface import (
    FixedAnswerModel,
    PrecomputedModel,
    RandomModel,
    load_precomputed_answers,
)
from causalstep.runner import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="CausalStep: Evaluate models on stepwise causal video reasoning."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to benchmark JSON dataset.",
    )
    parser.add_argument(
        "--model",
        choices=["precomputed", "random", "dummy"],
        default="precomputed",
        help="Model type to evaluate (default: precomputed).",
    )
    parser.add_argument(
        "--answers",
        help="Path to precomputed answers JSON (required for --model precomputed).",
    )
    parser.add_argument(
        "--output",
        help="Path to save results JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for RandomModel (default: 42).",
    )
    parser.add_argument(
        "--fixed-answer",
        default="A",
        help="Fixed answer letter for dummy model (default: A).",
    )
    parser.add_argument(
        "--causal-fail-advance",
        action="store_true",
        default=True,
        help="On causal failure, advance to next segment (pseudocode behavior, default).",
    )
    parser.add_argument(
        "--causal-fail-same-segment",
        action="store_true",
        default=False,
        help="On causal failure, restart desc QA of same segment (paper prose behavior).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-video evaluation traces.",
    )

    args = parser.parse_args()

    if args.model == "precomputed":
        if not args.answers:
            parser.error("--answers is required when --model is precomputed")
        answers = load_precomputed_answers(args.answers)
        model = PrecomputedModel(answers)
    elif args.model == "random":
        model = RandomModel(seed=args.seed)
    elif args.model == "dummy":
        model = FixedAnswerModel(answer=args.fixed_answer)
    else:
        parser.error(f"Unknown model type: {args.model}")

    causal_fail_advance = not args.causal_fail_same_segment

    run_evaluation(
        dataset_path=args.data,
        model=model,
        output_path=args.output,
        causal_fail_advance=causal_fail_advance,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
