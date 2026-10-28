# experiments/protocol/

The evaluation protocol behind the paper's Table 4 (kinematics) and Table 7 (consistency) cells and the Lite cases built from them.

| File | Contents |
| --- | --- |
| `cases.yaml` | Every Table 4/7 cell: scene, view, trajectory, first frame, active or excluded |
| `action_backends.yaml` | Per backend: native timeline, the targets it can express (and why not the others), response calibration; read via `evaluation/action_backends.py` |
| `trajectories/`, `compiled/` | Hash-locked semantic trajectories (`schema_version: 1`) and their compiled forms (`action.npz/.npy`, `actions.jsonl`, `actions.txt`, manifests); rebuilt with `tools/trajectories/prepare_table_trajectories.py` |
| `conditioning_sources.yaml` | Where each cell's first frame comes from, with its sha256 |
| `paper_baseline.yaml` | The paper's published Table 4/7 values, compared by `tools/paper_tables/compare_table_results.py` |
| `prompt_components.yaml`, `prompt_as_run_inputs.yaml` | Prompt text as authored and as sent to each model |
| `evaluator_assets.lock.json`, `evaluator_protocol.yaml` | Pinned evaluator checkpoints and settings |
| `unity_scene_runs.yaml`, `remaining_models.yaml` | Unity capture plan; models still to run |
| `rerun_results.template.json` | Output template of the table comparison |

The Lite bundles (`lite/cases/*/scene_*`, `*/kinematics_*`) are built from these files by `tools/dataset/build_table_lite_cases.py`. Generated records of the paper rerun (GPU-handoff and reference-package manifests, compile and alignment audits) are in the [Hub dataset](https://huggingface.co/datasets/lmgame/WMEval-Lite/tree/main/protocol); `tools/paper_tables/` fetches them when needed.
