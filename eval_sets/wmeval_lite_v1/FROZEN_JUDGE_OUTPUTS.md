# Frozen judge outputs and how to re-baseline them

Hosted VLM judges are sampled, so judging the Unity reference on every run would
make the reference a moving target. The reference side is judged once, every
pass is stored in the case bundle, and runtime reads the stored result.

| Dimension | File (per case) | Tool | Passes | Stale when |
|---|---|---|---|---|
| Physical causality | `reference/artifacts/judge_answers.json` | `tools/record_causal_reference_answers.py` | 5 | judge model, frame count `K`, or prompt version differ from the live judge (runtime refuses the replay) |
| Prompt consistency | `reference/artifacts/reference_scores.json` | `tools/freeze_prompt_reference_scores.py` | 5 | judge model, prompt version, alpha, rubric, system prompts, sampling, `ref.png` or the Unity video change (rows mark the reference `stale`) |
| LPIPS | `reference/artifacts/contact_events.json` | `tools/freeze_reference_events.py` | 5, majority (3/5) | the reference video's SHA-256 changes (the loader rejects it) |

## Re-baselining

Re-baseline only when the judge or the reference changes, and re-baseline every
case of the dimension together.

```bash
# 1. Re-judge (hosted judge; explicit opt-in via OPENAI_API_KEY)
python tools/record_causal_reference_answers.py eval_sets/wmeval_lite_v1/cases/BeamDomino/causal --passes 5
python tools/freeze_prompt_reference_scores.py --passes 5 --register
python tools/freeze_reference_events.py MirrorRollingBall/lpips_collision --passes 5 --register

# Same, through an OpenAI-compatible gateway serving the same model:
python tools/judge_gateway.py --key-file <key> --script tools/freeze_reference_events.py -- \
    LiquidContainer/fluid_rest --passes 5 --register --allow-empty

# 2. Re-lock each touched case (checksums + evaluator fingerprint)
python tools/build_lite_case.py eval_sets/wmeval_lite_v1/cases/<Scene>/<variant>

# 3. Verify
python tools/run_wmeval_lite.py validate --suite eval_sets/wmeval_lite_v1
python -m pytest -q tests
```

Record in the commit message: judge model and snapshot, pass count, transport,
and the before/after reference numbers. Results scored against an older baseline
are not comparable with results scored against the new one; re-score both sides.
