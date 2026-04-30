# WMEval

**WMEval** is a benchmark for evaluating **interactive world models** as controllable, physically grounded simulators. Instead of only measuring visual quality or prompt alignment, WMEval compares world-model rollouts against **Unity-based physical simulation references** under matched action trajectories.

## Overview

Interactive world models are increasingly expected to behave like game engines: given an initial scene and a sequence of actions, they should generate futures that are visually plausible, controllable, temporally stable, and physically coherent.

WMEval evaluates this capability by running the same action trajectories in both:

1. a Unity physical simulator, which provides ground-truth states and rendered reference videos;
2. an interactive world model, which generates action-conditioned rollouts.

The benchmark then compares the two along perceptual, physical, and causal dimensions.

## Evaluation Dimensions

WMEval covers the following capabilities:

- **Prompt-scene consistency**: whether the generated scene and key characters match the input prompt.
- **Action controllability**: whether the rollout follows the intended control signals.
- **Kinematics and mechanics**: whether object trajectories, velocity, momentum, and energy match simulator-derived references.
- **Physical causality**: whether causal event chains, such as collisions and domino cascades, unfold in the correct order.
- **Fluid mechanics**: whether liquid motion preserves volume, surface orientation, velocity fields, and sloshing dynamics.
- **Lighting and shading**: whether shadows and illumination are geometrically and temporally consistent.

## Key Features

- Unity-grounded reference simulations
- Matched action execution between simulator and world model
- Rule-based synthetic action trajectories for robust evaluation
- Dimension-specific CV-based metrics
- VLM-as-a-judge evaluation for semantic consistency
- Holistic video-level similarity metrics such as causal LPIPS
- Support for comparing multiple interactive world models across shared scenes

## Benchmark Scenes

WMEval includes diverse Unity scenes designed to stress different world-modeling capabilities, including:

- Ball rolling
- Billiard collision
- Racing car
- Beam domino
- Liquid container
- Mirror ball
- Chessboard
- Lighting and shadow scenes

Each scene is paired with predefined and synthetic action trajectories to test controllability, physical consistency, and long-horizon stability.

## Goal

WMEval is designed to answer a central question:

> How close is a learned world model to a physical simulator when both are used as interactive game engines?

By grounding evaluation in simulator-derived states, WMEval exposes failures that are often missed by visual-quality or prompt-alignment benchmarks alone.

## Status

This project is under active development. Evaluation scripts, scene assets, metric implementations, and benchmark results will be released progressively.
