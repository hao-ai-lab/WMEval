# Proposal: WMEval Lite as a HuggingFace Dataset (flat table)

> Discussion draft — no code or schema changes yet. Reference style:
> [lmgame/VideoScienceBench](https://huggingface.co/datasets/lmgame/VideoScienceBench)
> (flat table, one row per test case, columns documented on the card).

We publish the benchmark **inputs** (what a world model needs to generate its
rollouts) plus the reference media, as one flat table backed by the case
bundles mirrored verbatim. One row = one case. Evaluation-side detail
(protocol, rubrics, locks) ships in the bundle files, not as table columns —
the table stays as lean as typical benchmark cards.

## Flat table: column definition

| column | type | description |
| --- | --- | --- |
| `case_id` | string | `<Scene>/<variant>`, e.g. `MirrorRollingBall/prompt_first` |
| `scene` | string (class) | one of the 9-scene roster (BallRoller, BeamDomino, BilliardGame, ChessBoard, MirrorRollingBall, LiquidContainer, RacingCar, SunRisesAndSets, WanderingGarden) |
| `dimensions` | list[string] | which benchmark dimensions evaluate this case (from `suite.yaml`, borrows included) |
| `perspective` | string (class) | `first_person` / `third_person` |
| `prompt` | string | generation condition ① — scene description |
| `first_frame` | Image | generation condition ② — starting frame (viewer shows thumbnails); for reference-based cases extracted from frame 0 of `reference.mp4` at export, so **every row has one** |
| `actions` | string | generation condition ③ — the keyboard action script driving the rollout (named keys + durations, human-readable JSON lines) |
| `video` | Video (path) | the case's bundled video: `reference.mp4` for reference-based cases, `unity_groundtruth.mp4` for reference-free cases (viewer can play it) |

## Example rows (real data from the current bundles)

**Row 1 — reference-free case** (`prompt_scene_consistency`):

```json
{
  "case_id": "MirrorRollingBall/prompt_first",
  "scene": "MirrorRollingBall",
  "dimensions": ["prompt_scene_consistency"],
  "perspective": "first_person",
  "prompt": "A flat, light-colored ground populated with many large, glossy yellow spheres arranged in a regular grid-like pattern, appearing roughly evenly spaced across the surface. The scene is enclosed by low gray boundary walls. Bright daylight produces clear cast shadows. The environment is clean and simple. The character is a ball. First person view.",
  "first_frame": "cases/MirrorRollingBall/prompt_first/reference/ref.png",
  "actions": "[\n  {\"op\": \"setview\", \"key\": \"first\", \"note\": \"start in first-person (no on-screen switch)\"},\n  {\"op\": \"hold\", \"key\": \"ArrowLeft\", \"ms\": 10000, \"note\": \"rotate left for 10 seconds\"}\n]",
  "video": "cases/MirrorRollingBall/prompt_first/reference/unity_groundtruth.mp4"
}
```

**Row 2 — reference-based case** (`lpips`, scripted collision rollout):

```json
{
  "case_id": "MirrorRollingBall/lpips_collision",
  "scene": "MirrorRollingBall",
  "dimensions": ["lpips"],
  "perspective": "third_person",
  "prompt": "Starting world must match this layout: low camera near a flat light gray floor. Center foreground: one highly reflective chrome mirror ball (the player). Surrounding it, a grid of many glossy yellow spheres stretching into the distance with alleys between rows. [...] The player controls only the silver mirror ball: roll it on the floor using standard directions. Keep physics simple and responsive.",
  "first_frame": "cases/MirrorRollingBall/lpips_collision/first_frame.png",
  "actions": "{\"op\":\"hold\",\"key\":\"KeyW\",\"ms\":2000}\n{\"op\":\"wait\",\"ms\":100}\n{\"op\":\"hold\",\"key\":\"KeyD\",\"ms\":1000}\n{\"op\":\"wait\",\"ms\":100}\n{\"op\":\"hold\",\"key\":\"KeyS\",\"ms\":2000}\n{\"op\":\"wait\",\"ms\":100}\n{\"op\":\"hold\",\"key\":\"KeyA\",\"ms\":1000}",
  "video": "cases/MirrorRollingBall/lpips_collision/reference/reference.mp4"
}
```

(prompt in row 2 truncated with `[...]` for readability here; the real column
carries the full text. Row 2's `first_frame` is extracted from frame 0 of
`reference.mp4` at export time. Note the two rows use different key
vocabularies — Arrow keys vs WASD — which is per-case data, see open
question 4.)

## Columns deliberately NOT in the table (info is not lost, it moves)

| dropped column | where it lives instead |
| --- | --- |
| fps / width / height / duration | the video files themselves (probe them) |
| fov | derivable from bundled `intrinsics.npy` + frame height |
| poses/intrinsics coordinate conventions, action-array encoding | dataset card "Data format" section, written once (identical per row → not a column) |
| `action.npy` per-case key/column order | proposed `capture:` block in case.yaml (machine-readable; see open question 4) |
| protocol (normalization / min duration) | `case.yaml` in the mirrored bundle |
| rubrics / qa_dataset / liquid `.npz` | bundle `reference/artifacts/` files |
| checksums / evaluator fingerprint | bundle `checksums.sha256` + `case.yaml` (integrity story in the card) |

## HF repo layout (draft)

```
README.md         ← dataset card (this column table + generation protocol + eval quickstart)
cases.jsonl       ← the flat table (viewer data source)
suite.yaml        ← evaluation manifest, carried verbatim
cases/<Scene>/<variant>/…   ← the 7 case bundles mirrored verbatim (incl. checksums)
```

Export would be a mechanical `tools/export_hf_dataset.py` reading
`suite.yaml` + case bundles (runs `validate` first, refuses on failure), so
the git repo stays the single source of truth and the card's column table can
never drift from the data.

## Open questions for the team

1. **HF org / repo name / license / citation** — TBD.
2. **Visibility**: current lean = fully public including GT/reference videos
   (VBench-style; comparison metrics need GT anyway). Confirm.
3. **Upload timing**: after the pending dimensions' data lands, or publish the
   4 live dimensions first and version up?
4. **`capture:` block in case.yaml** (4 fields: `action_key_vocab`,
   `action_encoding`, `pose_convention`, `intrinsics_convention`): `action.npy`
   is a headerless (T, K) array and the key order differs per case (see the
   two example rows) — without it the numeric actions are uninterpretable.
   Values for the 4 prompt cases exist verbatim in git history (the removed
   `meta.json`); the 3 PR-#16-derived cases need the original author to
   confirm their key vocab.
5. **Scene naming in `personal/feat/v2-pipeline`**: its CSV uses
   `ChessBoardwithSphere` / `GenieRollingBall`, not in the 9-scene roster —
   align before merging (that branch also has 40MB+ mp4s and pyc committed;
   clean before any publish).
