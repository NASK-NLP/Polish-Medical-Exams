import pickle
import re 
from openai import OpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import sys
import pandas as pd
import copy
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.transformers_utils.tokenizer import get_tokenizer
import transformers
import torch
from config.paths import DATA_DIR, MODEL_OUTPUT, OUTPUT_PATH, HF_REPO
from pathlib import Path
BATCH_CHECKPOINT_SIZE=50
from datasets import load_dataset



def load_hf_split_compatible(split_name):
    base = split_name.removesuffix("_pandas")
    candidates = [split_name, base, f"{base}_pandas"]
    tried = set()
    last_error = None

    # Preferred layout: subset/config per dataset with split="all".
    for candidate in candidates:
        if candidate in tried:
            continue
        tried.add(candidate)
        try:
            return load_dataset(HF_REPO, candidate, split="all").to_pandas()
        except Exception as err:
            last_error = err

    # Fallback for folder-based parquet layout (e.g. <subset>/all-*.parquet).
    for candidate in candidates:
        try:
            return load_dataset(
                "parquet",
                data_files={"train": f"hf://datasets/{HF_REPO}/{candidate}/*.parquet"},
                split="train",
            ).to_pandas()
        except Exception as err:
            last_error = err

    # Backward compatibility for legacy repos that used split names directly.
    for candidate in candidates:
        try:
            return load_dataset(HF_REPO, split=candidate).to_pandas()
        except Exception as err:
            last_error = err

    if last_error is not None:
        raise last_error
    raise ValueError(f"Could not resolve split: {split_name}")


class EvaluatorLLM:
    def __init__(self, dataset_str, model_id, evaluation_methodology, possible_answers, split_name, prompt_format, extension=".csv"):
        self.dataset_str = dataset_str
        self.model_id = model_id
        self.evaluation_methodology = evaluation_methodology
        self.possible_answers = possible_answers
        self.split_name = split_name
        self.prompt_format = prompt_format
        self.extension = extension
        self.dataset_local_key = dataset_str.removesuffix("_pandas")
        if self.extension!=".csv":
            base_path_key = DATA_DIR
            data = load_hf_split_compatible(self.dataset_str)
            data.reset_index(drop=True, inplace=True)
            self.key = data

        else:
            base_path_key = DATA_DIR / self.split_name 
            path_key = os.path.join(base_path_key, self.dataset_local_key + self.extension)
            if not os.path.exists(path_key):
                path_key = os.path.join(base_path_key, self.dataset_str + self.extension)
            self.key = pd.read_csv(path_key,index_col=0)
        
        self.possible_answers = list(possible_answers) + [",", "."]
        
        self.model_save_path = model_id.replace('/','-')
        dir_path = MODEL_OUTPUT / self.split_name 
        filename = f"{self.dataset_str}-{self.model_save_path}.pickle"

        os.makedirs(dir_path, exist_ok=True)
        self.path = os.path.join(dir_path, filename.replace(".pickle", "-partial.pickle"))
        self.path_final = os.path.join(dir_path, filename)
        try:
            with open(self.path, 'rb') as f:
                self.partial_outputs = pickle.load(f)
        except FileNotFoundError:
            self.partial_outputs = list()
            with open(self.path, 'wb') as f:
                pickle.dump(self.partial_outputs, f)

    def prompt_init(self, split_name, prompt_format, possible_answers, evaluation_methodology):
        self.prompt_format = prompt_format
        self.evaluation_methodology = evaluation_methodology
        self.split_name = split_name
        dir_path = MODEL_OUTPUT / split_name 
        filename = f"{self.dataset_str}-{self.model_save_path}.pickle"
        os.makedirs(dir_path, exist_ok=True)
        self.path = os.path.join(dir_path, filename.replace(".pickle", "-partial.pickle"))
        self.path_final = os.path.join(dir_path, filename)
        self.possible_answers = list(possible_answers) + [",", "."]
        

        try:
            with open(self.path, 'rb') as f:
                self.partial_outputs = pickle.load(f)
        except FileNotFoundError:
            self.partial_outputs = list()
            with open(self.path, 'wb') as f:
                pickle.dump(self.partial_outputs, f)
        if self.extension!=".csv":
            base_path_key = DATA_DIR
            # path_key = os.path.join(base_path_key,self.dataset_str+self.extension)
            # with open(path_key, 'rb') as f:
            #     data = pickle.load(f)
            data = load_hf_split_compatible(self.dataset_str)
            data.reset_index(drop=True, inplace=True)
            self.key = data

        else:
            base_path_key = DATA_DIR / self.split_name 
            path_key = os.path.join(base_path_key, self.dataset_local_key + self.extension)
            if not os.path.exists(path_key):
                path_key = os.path.join(base_path_key, self.dataset_str + self.extension)
            self.key = pd.read_csv(path_key,index_col=0)
        self.parameters_init()
        self.create_messages()

    def create_messages(self):
        messages=[]
        if "BioMistral" in self.model_id:
            for id,row in self.key.iterrows():
                messages.append([
                    {'role':'user', 'content':self.prompt_format + row['question']}])
            self.messages = messages
            self.messages = [
                self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                for message in self.messages
            ]

        else:

            for id,row in self.key.iterrows():
                messages.append([
                    {'role':'system', 'content': self.prompt_format}, 
                    {'role':'user', 'content':row['question']}])
            self.messages = messages
            print("RAPORT MESSAGES", self.messages[0])
            self.messages = [
                self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True,enable_thinking=False
                )
                for message in self.messages
            ]
            if "mistral" in self.model_id.lower():
                # Decode token IDs into strings for actual prompts
                self.messages = [self.tokenizer.decode(token_ids) for token_ids in self.messages]
            
    def model_init(self):
        if "istral" in self.model_id and "Bio" not in self.model_id:
            self.model = LLM(
                model = self.model_id,
                tokenizer_mode="mistral",
                load_format="mistral",
                config_format="mistral",
                dtype=torch.bfloat16,
                max_model_len = 2048,
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=0.2
            )

        else:
            if self.model_id == "chaoyi-wu/PMC_LLAMA_7B":
                self.model = LLM(
                    model="chaoyi-wu/PMC_LLAMA_7B",
                    tokenizer="chaoyi-wu/PMC_LLAMA_7B",
                    dtype=torch.bfloat16,
                    max_model_len = 2048,
                    tensor_parallel_size=torch.cuda.device_count(),
                    gpu_memory_utilization=0.2

                    )
            elif "Qwen3.5-" in self.model_id:
                if "122B-A10B-AWQ" in self.model_id:
                    self.model = LLM(
                    model = self.model_id,
                    dtype=torch.float16,
                    max_model_len = 2048,
                    tensor_parallel_size=torch.cuda.device_count(),
                    gpu_memory_utilization=0.2,
                    )
                else:
                    model =self.model = LLM( 
                        self.model_id,
                        dtype=torch.bfloat16,
                        max_model_len = 2048,
                        tensor_parallel_size=torch.cuda.device_count(),
                        gpu_memory_utilization=0.2
                    )
            else:
                model =self.model = LLM( 
                    self.model_id,
                    dtype=torch.bfloat16,
                    max_model_len = 2048,
                    tensor_parallel_size=torch.cuda.device_count(),
                    gpu_memory_utilization=0.2
                )
                
        tokenizer_mode = "auto"
        self.tokenizer = get_tokenizer(
            tokenizer_name=self.model_id,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=True
            )

    def parameters_init(self):
        self.answers_token_ids = []

        for ans in self.possible_answers:
            tokens = self.tokenizer(ans, add_special_tokens=False)["input_ids"]
            self.answers_token_ids.append(tokens[0])

        if self.evaluation_methodology == "free_form":
            self.sampling_params = SamplingParams(
            max_tokens = 128,
            temperature = 0,
            )
        elif self.evaluation_methodology == "free_form_generation":
            self.sampling_params = SamplingParams(
                max_tokens = 256,
                temperature = 0,
                )
        elif self.evaluation_methodology == "prev_method":
            self.sampling_params = SamplingParams(
                max_tokens = 20,
                temperature = 0,
            )

        else: 
            # Bielik poorly handled constrained generation with allowed token IDs, so for their models we relax the eval conditions.
            if ("Bielik" in self.model_id) or ("istral" in self.model_id):
                self.sampling_params = SamplingParams(
                max_tokens = 200,
                temperature = 0,
                stop_token_ids=self.tokenizer(".", add_special_tokens=False)["input_ids"]
                )
            elif "MediPhi-Instruct" in self.model_id and self.split_name == "multiple_choice":
                self.sampling_params = SamplingParams(
                max_tokens = 100,
                temperature = 0,
                stop_token_ids=self.tokenizer(".", add_special_tokens=False)["input_ids"]
                )
            
            else:
                self.sampling_params = SamplingParams(
                max_tokens = 20,
                temperature = 0,
                allowed_token_ids = self.answers_token_ids,
                stop_token_ids=self.tokenizer(".", add_special_tokens=False)["input_ids"]
                )
    def make_predictions(self):
        partial_outputs=self.partial_outputs
        messages = self.messages
        print("Raport")
        print("*"*20)
        print("Split", self.split_name)
        print("Message sample:", messages[0])
        print("Possible answers:", self.possible_answers)
        print("Prompt:", self.prompt_format)
        print("Sampling params:", self.sampling_params)
        print("Allowed tokens", self.answers_token_ids)
        
        while len(partial_outputs) != len(messages):
            i = len(partial_outputs)
            print(f'step:{i}/{len(messages)} start')

            with open(self.path, 'rb') as f:
                partial_outputs = pickle.load(f)

            batch_messages = messages[i:i + BATCH_CHECKPOINT_SIZE]

            responses = self.model.generate(batch_messages, sampling_params=self.sampling_params)

            for response in responses:
                if "gpt-oss" in self.model_id:
                    text = response.outputs[0].text
                    match = re.search(r"(?s)(.*?)assistantfinal\s*(.*)", text)
            
                    if match:
                        final_answer = match.group(2).strip()
                    else:
                        print("Assistant final not found")
                        print(text)

                        final_answer = "None"
                    partial_outputs.append(final_answer)
                else:
                    partial_outputs.append(response.outputs[0].text)

            with open(self.path, 'wb') as f:
                pickle.dump(partial_outputs, f)

            print(f'step:{i}/{len(messages)} done')
        outputs = copy.deepcopy(partial_outputs)

        with open(self.path_final, 'wb') as f:
            pickle.dump(outputs, f)

        print(f"All outputs saved to {self.path_final}")


            