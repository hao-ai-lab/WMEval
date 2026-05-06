#!/usr/bin/env bash
# Run the liquid eval over a fleet of world-model candidate videos and
# emit a CSV leaderboard.
#
# Manifest format: one TSV row per WM, columns separated by TABs:
#   model_name<TAB>candidate_video_path[<TAB>ref_video_override]
# The third column is optional. When present it overrides the global
# REF_VIDEO arg for that row — needed when different rows pair against
# different Unity references (e.g. one ref per action profile).
# Lines starting with '#' and blank lines are skipped. CRLF-encoded files
# are tolerated — \r is stripped from the trailing field. Manifests with
# no trailing newline are also handled.
#
# Usage: bash WMEval/scripts/run_liquid_leaderboard.sh MANIFEST_TSV OUT_DIR [REF_VIDEO]
#   MANIFEST_TSV  Path to the manifest described above
#   OUT_DIR       Directory to write per-model JSON reports + leaderboard.csv
#   REF_VIDEO     (optional) Default Unity reference; used only for manifest
#                 rows that lack a per-row ref override. Required only if the
#                 manifest contains any 2-column rows.
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 MANIFEST_TSV OUT_DIR [REF_VIDEO]" >&2
  exit 1
fi

MANIFEST="$1"
OUT_DIR="$2"
REF_VIDEO="${3:-}"

# Resolve user-supplied paths against the caller's CWD before we cd into
# WMEval/ so relative manifest/output args still work.
case "$MANIFEST" in /*) ;; *) MANIFEST="$(pwd)/$MANIFEST" ;; esac
case "$OUT_DIR" in /*) ;; *) OUT_DIR="$(pwd)/$OUT_DIR" ;; esac
if [ -n "$REF_VIDEO" ]; then
  case "$REF_VIDEO" in /*) ;; *) REF_VIDEO="$(pwd)/$REF_VIDEO" ;; esac
fi

if [ ! -f "$MANIFEST" ]; then
  echo "run_liquid_leaderboard: manifest not found: $MANIFEST" >&2
  exit 1
fi
if [ -n "$REF_VIDEO" ] && [ ! -f "$REF_VIDEO" ]; then
  echo "run_liquid_leaderboard: ref video not found: $REF_VIDEO" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p "$OUT_DIR"

REPORTS=()
FAILED=()
seen_names=""
# `|| [ -n "$MODEL_NAME" ]` keeps the body running for a final line that
# lacks a trailing newline (read returns 1 at EOF even after reading data).
# Stripping \r handles Windows-encoded manifests so paths don't carry a
# stray carriage return into downstream filename construction.
# Read the manifest from fd 3, not stdin: the child eval scripts spawn
# ffmpeg, which consumes stdin in its interactive prompt and would chew
# through manifest bytes between rows otherwise.
while IFS=$'\t' read -r MODEL_NAME WM_VIDEO ROW_REF <&3 || [ -n "${MODEL_NAME:-}" ]; do
  MODEL_NAME="${MODEL_NAME%$'\r'}"
  WM_VIDEO="${WM_VIDEO%$'\r'}"
  ROW_REF="${ROW_REF%$'\r'}"
  case "$MODEL_NAME" in
    ''|'#'*) continue ;;
  esac
  if [ -z "${WM_VIDEO:-}" ]; then
    echo "skipping malformed row: '$MODEL_NAME'" >&2
    continue
  fi
  # Per-row reference override; resolve relative to the manifest's caller CWD
  # (same convention as the global REF_VIDEO above).
  if [ -n "${ROW_REF:-}" ]; then
    case "$ROW_REF" in /*) ;; *) ROW_REF="$(dirname "$MANIFEST")/$ROW_REF" ;; esac
    if [ ! -f "$ROW_REF" ]; then
      echo "skipping '$MODEL_NAME': override ref not found: $ROW_REF" >&2
      continue
    fi
    EFFECTIVE_REF="$ROW_REF"
  elif [ -n "$REF_VIDEO" ]; then
    EFFECTIVE_REF="$REF_VIDEO"
  else
    echo "skipping '$MODEL_NAME': row has no ref override and no default REF_VIDEO was provided" >&2
    continue
  fi
  # Guard against unsanitized model names that would create subdirs or
  # silently overwrite a previously written JSON.
  SAFE_NAME="${MODEL_NAME//\//_}"
  SAFE_NAME="${SAFE_NAME// /_}"
  case " $seen_names " in
    *" $SAFE_NAME "*)
      echo "skipping duplicate model name: '$MODEL_NAME'" >&2
      continue
      ;;
  esac
  seen_names="$seen_names $SAFE_NAME"
  OUT_JSON="$OUT_DIR/$SAFE_NAME.json"
  echo "==> $MODEL_NAME ($WM_VIDEO) vs $(basename "$EFFECTIVE_REF")"
  # `if !` keeps `set -e` from killing the whole leaderboard when a single
  # eval crashes (e.g. GroundingDINO finds no liquid on a glitched first
  # frame). The failed row is recorded and reported in the summary below.
  if ! bash "$SCRIPT_DIR/evaluate_world_model_liquid_full.sh" \
      "$EFFECTIVE_REF" "$WM_VIDEO" "$OUT_JSON" "$MODEL_NAME"; then
    echo "FAILED: '$MODEL_NAME' — eval errored, see output above. Continuing." >&2
    FAILED+=("$MODEL_NAME")
    continue
  fi
  REPORTS+=("$OUT_JSON")
done 3< "$MANIFEST"

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "" >&2
  echo "run_liquid_leaderboard: ${#FAILED[@]} row(s) failed:" >&2
  for name in "${FAILED[@]}"; do echo "  - $name" >&2; done
fi

if [ "${#REPORTS[@]}" -eq 0 ]; then
  echo "run_liquid_leaderboard: no rows processed" >&2
  exit 1
fi

LEADERBOARD_CSV="$OUT_DIR/leaderboard.csv"
# Use the same interpreter that ran the eval so a missing/older python on
# PATH cannot blow up the heredoc after every model has been processed.
"$(command -v python3 || command -v python)" - "$LEADERBOARD_CSV" "${REPORTS[@]}" <<'PY'
import csv, json, sys

out_path = sys.argv[1]
report_paths = sys.argv[2:]

# Five-pillar metric set (lower=better except cosine_similarity where 1=best).
# EPE / divergence_residual / curl are intentionally absent: on opaque uniform
# liquids RAFT magnitudes are dominated by hallucinated flow, so those numbers
# track noise instead of physics. They remain in the per-model JSON for
# identity-pair sanity checks.
COLUMNS = [
    "model",
    "liquid_velocity.cosine_similarity",
    "liquid_volume_conservation.mask_area_std_normalized.delta",
    "liquid_volume_conservation.mask_area_drift_per_sec.delta",
    "liquid_surface_orientation.surface_tilt_residual_deg.delta",
    "slosh_spectrum.meniscus.spectral_l1",
    "slosh_spectrum.meniscus.dominant_frequency_hz.delta_hz",
    "lpips.lpips_mean",
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
