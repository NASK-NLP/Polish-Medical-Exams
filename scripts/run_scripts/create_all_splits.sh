#!/usr/bin/env bash
set -euo pipefail

# Run all split creators in a fixed order:
# 1) multiple_choice, 2) multiple_choice2, 3) free_form, 4) abstaining_substitution.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASETS=(
  "r_lek"
  "r_ldek"
  "r_pes_latest"
  "r_diagnostics"
  "r_pharmacy"
)

run_split_for_all_datasets() {
  local script_path="$1"
  local split_name="$2"

  echo "=== Creating ${split_name} split ==="
  for dataset in "${DATASETS[@]}"; do
    echo "-> ${split_name}: ${dataset}"
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" python "$ROOT_DIR/$script_path" "$dataset"
  done
}

run_split_for_all_datasets "data_creator/create_multichoice.py" "multiple_choice"
run_split_for_all_datasets "data_creator/create_multichoice2.py" "multiple_choice2"
run_split_for_all_datasets "data_creator/create_free_form.py" "free_form"
run_split_for_all_datasets "data_creator/create_abstaining_substitution.py" "abstaining_substitution"

echo "All requested splits were created successfully."
