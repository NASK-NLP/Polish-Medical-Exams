#!/bin/bash
models_small=(
        # "CYFRAGOVPL/pllum-12b-nc-instruct-250715"
        # "CYFRAGOVPL/pllum-12b-nc-chat-250715"
        # "speakleash/Bielik-4.5B-v3.0-Instruct"
        # "speakleash/Bielik-11B-v2.6-Instruct"
        # "CYFRAGOVPL/PLLuM-12B-instruct"
        # "CYFRAGOVPL/PLLuM-12B-chat"
        # "meta-llama/Meta-Llama-3-8B-Instruct"
        # "google/gemma-3-12b-it"
        # "BioMistral/BioMistral-7B"
        # "BioMistral/BioMistral-7B-DARE"
        # "google/medgemma-4b-it"
        # "microsoft/MediPhi-Instruct"
        # "speakleash/Bielik-11B-v3.0-Instruct"
        # "Qwen/Qwen3.5-9B"
        )

models_medium=(
    # "google/medgemma-27b-it"
    # "google/gemma-3-27b-it"
    # "Qwen/Qwen3-30B-A3B-Instruct-2507"

    )
models_large=(
    # "Qwen/Qwen3.5-35B-A3B"
    # "OpenMeditron/Meditron3-70B"
    # "Qwen/Qwen2.5-72B-Instruct"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "CYFRAGOVPL/Llama-PLLuM-70B-instruct"
    # "aaditya/Llama3-OpenBioLLM-70B"
    # "Qwen/Qwen3.5-122B-A10B-FP8"
    )

methods=("our_method")

# # echo $CUDA_VISIBLE_DEVICES
for model in "${models_small[@]}"; do
    for method in "${methods[@]}"; do
        sbatch scripts/jobs/jobs_small_template.sh "$model"
    done
done



for model in "${models_medium[@]}"; do
    for method in "${methods[@]}"; do
        sbatch scripts/jobs/jobs_medium_template.sh "$model"
    done
done


for model in "${models_large[@]}"; do
    for method in "${methods[@]}"; do
        sbatch scripts/jobs/jobs_large_template.sh "$model"
    done
done
