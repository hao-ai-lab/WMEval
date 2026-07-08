# WMEval Lite — HF dataset flat table (draft)

## Columns

| column | type | description |
| --- | --- | --- |
| `case_id` | string | `<Scene>/<variant>`, e.g. `MirrorRollingBall/prompt_first` |
| `scene` | string | one of the 9-scene roster |
| `dimensions` | list[string] | dimensions that evaluate this case |
| `perspective` | string | `first_person` / `third_person` |
| `prompt` | string | generation condition ① — scene description |
| `first_frame` | Image | generation condition ② — starting frame (extracted from frame 0 of `reference.mp4` for reference-based cases, so every row has one) |
| `actions` | string | generation condition ③ — keyboard action script (named keys + durations) |
| `video` | Video | `reference.mp4` (reference-based) or `unity_groundtruth.mp4` (reference-free) |

## Example rows

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

(row 2 prompt truncated with `[...]` here; the real column carries the full text)

## HF repo layout (draft)

```
README.md         ← dataset card (column table + eval quickstart)
cases.jsonl       ← the flat table
suite.yaml        ← evaluation manifest
cases/<Scene>/<variant>/…   ← case bundles mirrored verbatim (incl. checksums)
```

## Key considerations

- org / repo name / license / citation — TBD
- visibility: fully public incl. reference/GT videos (comparison metrics need them)
- upload timing: publish the 4 live dimensions first, or wait for all
- `action.npy` is a headerless (T, K) array and key order differs per case
  (Arrow keys vs WASD in the two examples) — needs a small `capture:` block in
  case.yaml (key vocab / encoding / pose conventions) before export
