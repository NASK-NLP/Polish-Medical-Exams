import pickle
import re 
# from openai import OpenAI
# from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import sys
import pandas as pd
import copy
# from vllm import LLM, SamplingParams
# from vllm.transformers_utils.tokenizer import get_tokenizer
import transformers
import torch
from scripts.evaluation.evaluator_judge import EvaluatorLLMJudge
from pathlib import Path
from config.paths import DATA_DIR, MODEL_OUTPUT
import pandas as pd
BATCH_CHECKPOINT_SIZE=50
# print(torch.cuda.is_available())
# print(torch.version.cuda)
# print(torch.cuda.current_device())
if __name__ == '__main__':
    custom_api_key = os.getenv("CUSTOM_API_KEY")
    if not custom_api_key:
        raise ValueError("CUSTOM_API_KEY is not set. Export it before running this script.")

    exams = ["r_lek", "r_ldek", "r_pes_latest", "r_diagnostics", "r_pharmacy", "r_lek_en", "r_ldek_en"]
    models=[
        # "CYFRAGOVPL/pllum-12b-nc-instruct-250715",
        # "CYFRAGOVPL/pllum-12b-nc-chat-250715",
        # "speakleash/Bielik-4.5B-v3.0-Instruct",
        # "speakleash/Bielik-11B-v3.0-Instruct",
        # "CYFRAGOVPL/PLLuM-12B-instruct",
        # "CYFRAGOVPL/PLLuM-12B-chat",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "google/gemma-3-12b-it",
        # "BioMistral/BioMistral-7B",
        # "BioMistral/BioMistral-7B-DARE",
        # "google/medgemma-4b-it",
        # "microsoft/MediPhi-Instruct",
        # "google/gemma-3-27b-it",
        # "google/medgemma-27b-it",
        # "Qwen/Qwen3-30B-A3B-Instruct-2507",
        # "OpenMeditron/Meditron3-70B",
        # "meta-llama/Llama-3.3-70B-Instruct",
        # "CYFRAGOVPL/Llama-PLLuM-70B-instruct",
        # "aaditya/Llama3-OpenBioLLM-70B",
        # "Qwen/Qwen3.5-9B",
        # "Qwen/Qwen3.5-35B-A3B",
        # "Qwen/Qwen3.5-122B-A10B-FP8",
        ]
    # dataset_str  = sys.argv[1]
    # model_id_evaluated = sys.argv[2]
    for dataset_str in exams:
        for model_id_evaluated in models:

            print(f"Model id: {model_id_evaluated}")
            # annotations_path = DATA_DIR / "valid_annotations.csv"
            evaluator = EvaluatorLLMJudge(
                    model_id_evaluated=model_id_evaluated,
                    # model_id_evaluator="gpt-5.4-mini",
                    # model_id_evaluator="gpt-5.4-nano",
                    # model_id_evaluator="claude-haiku-4-5-20251001",
                    # model_id_evaluator="claude-sonnet-4-6",
                    # model_id_evaluator="openai/gpt-oss-120b",
                    # model_id_evaluator="claude-opus-4-7",     
                    api_key=custom_api_key,
                    models=[
                        { "model_id": "DeepSeek-V4-Pro",
                          "base_url": "<TO_FILL>"
                        }
                    ],
                    model_id_evaluator="DeepSeek-V4-Pro",
                    dataset_str=dataset_str,
                    split_name="free_form",
                    # annotations_path = annotations_path,
                    backend="custom_openai"
                    )
            evaluator.paths_prompt_init(
                directory = "judge_consistent",
                prompt_format="Take on the role of a medical examiner in an oral exam. You ask the doctor a question (QUESTION), and the doctor provides a proposed answer (PROPOSED ANSWER). You also know the correct answer (ORIGINAL ANSWER). Based on your objective judgment, medical knowledge, and known guidelines, is the PROPOSED ANSWER correct? Do you determine that the PROPOSED ANSWER  is the correct answer to the QUESTION? If the PROPOSED ANSWER is correct return True, if the PROPOSED ANSWER is incorrect return False. Do not return anything more than a Boolean."
            )
            evaluator.create_messages("QUESTION: {question}\nPROPOSED ANSWER: {output}\nORIGINAL ANSWER: {key_num}")
            evaluator.make_predictions()

