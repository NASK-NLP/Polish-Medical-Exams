!/bin/bash
models=(
    "CYFRAGOVPL/pllum-12b-nc-instruct-250715"
    "CYFRAGOVPL/pllum-12b-nc-chat-250715"
    "speakleash/Bielik-4.5B-v3.0-Instruct"
    "speakleash/Bielik-11B-v2.6-Instruct"
    "CYFRAGOVPL/PLLuM-12B-instruct"
    "CYFRAGOVPL/PLLuM-12B-chat"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "google/gemma-3-12b-it"
    "BioMistral/BioMistral-7B"
    "BioMistral/BioMistral-7B-DARE"
    "google/medgemma-4b-it"
    "microsoft/MediPhi-Instruct"
    "mistralai/Ministral-3-8B-Instruct-2512"
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    "openai/gpt-oss-20b"
    "google/gemma-3-27b-it"
    "google/medgemma-27b-it"
    "Qwen/Qwen3-30B-A3B-Instruct-2507"
    "OpenMeditron/Meditron3-70B"
    "Qwen/Qwen2.5-72B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "CYFRAGOVPL/Llama-PLLuM-70B-instruct"
    "aaditya/Llama3-OpenBioLLM-70B"
    )

for model in "${models[@]}"; do
        sbatch scripts/jobs/job_judge.sh $model
done
