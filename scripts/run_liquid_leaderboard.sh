#!/usr/bin/env bash
# Run the liquid eval over a fleet of world-model candidate videos and
# emit a CSV leaderboard.
#
# Manifest format: one TSV row per WM, columns separated by TABs:
#   model_name<TAB>candidate_video_path
# Lines starting with '#' and blank lines are skipped.
#
# Usage: bash WMEval/scripts/run_liquid_leaderboard.sh MANIFEST_TSV REF_VIDEO OUT_DIR
#   MANIFEST_TSV  Path to the manifest described above
#   REF_VIDEO     Path to the Unity reference recording
#   OUT_DIR       Directory to write per-model JSON reports + leaderboard.csv
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 MANIFEST_TSV REF_VIDEO OUT_DIR" >&2
  exit 1
fi

MANIFEST="$1"
REF_VIDEO="$2"
OUT_DIR="$3"

if [ ! -f "$MANIFEST" ]; then
  echo "run_liquid_leaderboard: manifest not found: $MANIFEST" >&2
  exit 1
fi
if [ ! -f "$REF_VIDEO" ]; then
  echo "run_liquid_leaderboard: ref video not found: $REF_VIDEO" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPORTS=()
while IFS=$'\t' read -r MODEL_NAME WM_VIDEO; do
  case "$MODEL_NAME" in
    ''|'#'*) continue ;;
  esac
  if [ -z "${WM_VIDEO:-}" ]; then
    echo "skipping malformed row: '$MODEL_NAME'" >&2
    continue
  fi
  OUT_JSON="$OUT_DIR/$MODEL_NAME.json"
  echo "==> $MODEL_NAME ($WM_VIDEO)"
  bash "$SCRIPT_DIR/evaluate_world_model_liquid_full.sh" \
    "$REF_VIDEO" "$WM_VIDEO" "$OUT_JSON" "$MODEL_NAME"
  REPORTS+=("$OUT_JSON")
done < "$MANIFEST"

if [ "${#REPORTS[@]}" -eq 0 ]; then
  echo "run_liquid_leaderboard: no rows processed" >&2
  exit 1
fi

LEADERBOARD_CSV="$OUT_DIR/leaderboard.csv"
python - "$LEADERBOARD_CSV" "${REPORTS[@]}" <<'PY'
import csv, json, sys

out_path = sys.argv[1]
report_paths = sys.argv[2:]

COLUMNS = [
    "model",
    "liquid_velocity.epe.mean",
    "liquid_velocity.cosine_similarity",
    "liquid_velocity.divergence_residual",
    "liquid_velocity.curl_stats.mean_error",
    "liquid_velocity.temporal_consistency",
    "liquid_volume_conservation.mask_area_std_normalized.delta",
    "liquid_volume_conservation.mask_area_drift_per_sec.delta",
    "liquid_surface_orientation.surface_tilt_residual_deg.delta",
    "slosh_spectrum.dominant_frequency_hz.ratio",
    "slosh_spectrum.spectral_l1",
]


def get(d, dotted, default=""):
    cur = d
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLUMNS)
    for p in report_paths:
        with open(p) as g:
            rep = json.load(g)
        results = rep.get("results", {})
        row = [rep.get("model", "")]
        for col in COLUMNS[1:]:
            mod, _, rest = col.partition(".")
            row.append(get(results.get(mod, {}), rest))
        w.writerow(row)

print(f"wrote {out_path}")
PY
