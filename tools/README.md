# tools/

Entry points used in the main README stay at the top level; everything else is grouped by purpose.

| Path | Contents |
| --- | --- |
| `run_wmeval_lite.py`, `run_wmeval_lite_modal.py` | Validate, evaluate and aggregate a suite (locally or on Modal) |
| `sync_lite_data.py` | Download / upload the Hub-hosted case data |
| `expected_jobs.py`, `preflight_lite.py`, `fixture_backend.py` | Plan a new model's jobs, check the environment, make no-cost test candidates |
| `judge_gateway.py` | Run any judge-calling tool through an OpenAI-compatible gateway |
| `build_leaderboard.py` | Build the leaderboard and the README block |
| `dataset/` | Author and maintain the Lite suite: case bundles and locks, Unity references and reference-side inputs, frozen judge outputs, the synthetic split, the audit, the Hub export |
| `trajectories/` | Semantic trajectories: compile, audit, replay in Unity |
| `generation/` | Generate candidates: per-model runners and agents, backend input preparation, `model_adapters/` |
| `paper_tables/` | Reproduce and compare the paper's Tables 4 and 7 |
| `analysis/` | One-off diagnostics, calibrations and reviews |
| `legacy/` | Pre-Lite single-dimension CLIs and optional evaluators (see `docs/additional_evaluators.md`) |
