#!/usr/bin/env bash
# Run both recommendation programs on one cohort directory.
#   scripts/run_programs.sh <patient_dir> <output_root>
# Requires an OpenAI-compatible endpoint serving Qwen3-32B (e.g. vLLM) at $DIAMAS_BASE_URL.
set -euo pipefail
PATIENTS=$1
OUT=$2
: "${DIAMAS_BASE_URL:?set DIAMAS_BASE_URL, e.g. http://localhost:8000/v1/chat/completions}"
python run_diamas.py --program single        --patient_dir "$PATIENTS" --output_dir "$OUT/single"
python run_diamas.py --program collaborative --patient_dir "$PATIENTS" --output_dir "$OUT/collaborative"
