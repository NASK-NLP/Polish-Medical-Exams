#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_id>"
  echo "Example: $0 speakleash/Bielik-11B-v3.0-Instruct"
  exit 1
fi

MODEL_ID="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PL_EXAMS=(r_lek r_ldek r_pes_latest r_diagnostics r_pharmacy)
EN_EXAMS=(r_lek_en r_ldek_en)

echo "Evaluating model: $MODEL_ID"

echo "=== Polish exams ==="
for exam in "${PL_EXAMS[@]}"; do
  echo "-> $exam"
  python -m scripts.run_scripts.evaluation_all_splits "$exam" "$MODEL_ID"
done

echo "=== English exams ==="
for exam in "${EN_EXAMS[@]}"; do
  echo "-> $exam"
  python -m scripts.run_scripts.evaluation_all_eng "$exam" "$MODEL_ID"
done

echo "Evaluation finished."
