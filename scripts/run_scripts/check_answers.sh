#!/bin/bash
set -euo pipefail

export BASE_DIR=$(python -c "from config.paths import BASE_DIR; print(BASE_DIR)")
cd $BASE_DIR
models=(
        # "CYFRAGOVPL/pllum-12b-nc-instruct-250715"
        # "CYFRAGOVPL/pllum-12b-nc-chat-250715"
        # "speakleash/Bielik-4.5B-v3.0-Instruct"
        "CYFRAGOVPL/PLLuM-12B-instruct"
        # "CYFRAGOVPL/PLLuM-12B-chat"
        # "meta-llama/Meta-Llama-3-8B-Instruct"
        # "google/gemma-3-12b-it"
        # "BioMistral/BioMistral-7B"
        # "BioMistral/BioMistral-7B-DARE"
        # "google/medgemma-4b-it"
        # "microsoft/MediPhi-Instruct"
        # "google/medgemma-27b-it"
        # "google/gemma-3-27b-it"
        # "OpenMeditron/Meditron3-70B"
        # "meta-llama/Llama-3.3-70B-Instruct"
        # "CYFRAGOVPL/Llama-PLLuM-70B-instruct"
        # "aaditya/Llama3-OpenBioLLM-70B"
        # "Qwen/Qwen3.5-9B"
        # "Qwen/Qwen3.5-35B-A3B"
        # "Qwen/Qwen3.5-122B-A10B-FP8"
        # "speakleash/Bielik-11B-v3.0-Instruct"
)
splits=(
    multiple_choice
    multiple_choice2
    abstaining_substitution
    )

for model in "${models[@]}"; do
    for method in our_method prev_method; do
        python -m scripts.run_scripts.check_base "$model" "$method"
    done
done

for model in "${models[@]}"; do
    for split in "${splits[@]}"; do
        python -m scripts.run_scripts.check_ABCD_gt "$model" our_method "$split"
   done
done