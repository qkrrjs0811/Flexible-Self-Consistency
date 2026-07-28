#!/usr/bin/env bash
# AIME_2025: 4개 모델에 대해 per_sample_paths.json (및 dsc_pre_level, ours_level_paths) 생성
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-AIME_2025}"
STAGES="${STAGES:-40}"
OURS_P="${OURS_P:-1.0}"

python3 visualization/dsc_ours_per_sample_paths.py \
  --dataset_name "$DATASET" \
  --stages_config "$STAGES" \
  --ours_p "$OURS_P" \
  --per_sample \
  --models \
    Qwen/Qwen2.5-3B-Instruct \
    Qwen/Qwen2.5-7B-Instruct \
    Qwen/Qwen2.5-14B-Instruct \
    google/gemma-3-4b-it \
  "$@"

echo "Done. Outputs under: ${ROOT}/visualization/analysis_dsc_fsc/${DATASET}_*"
