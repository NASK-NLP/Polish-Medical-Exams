import pickle
import re 
from openai import OpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import sys
import pandas as pd
import copy
# from vllm import LLM, SamplingParams, TokensPrompt
# from vllm.transformers_utils.tokenizer import get_tokenizer
import transformers
import torch
from config.paths import DATA_DIR, MODEL_OUTPUT, OUTPUT_PATH, HF_REPO
from pathlib import Path
BATCH_CHECKPOINT_SIZE=50

import copy
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Conditional imports — only pulled in when needed
try:
    import torch
    # from vllm import LLM, SamplingParams
    # from vllm.transformers_utils.tokenizer import get_tokenizer
    # VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic as anthropic_sdk
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


BATCH_CHECKPOINT_SIZE = 32  # adjust as needed


class EvaluatorLLMJudge:
    def __init__(
            self,
            dataset_str,
            model_id_evaluated,
            model_id_evaluator,
            split_name,
            models=None,
            backend="openai",         # <-- "openai" | "vllm" | "anthropic" | "custom_openai"
            api_key=None,             # explicit key; falls back to env vars
            api_workers=8,            # parallel threads for API backends
            prompt_format=(
                "You are an expert medical examinator. Given a question and a "
                "ground truth answer, determine whether the candidate answer is "
                "correct. The candidate answer should be medically accurate to be "
                "deemed correct. If the candidate answer is correct return True, "
                "if the candidate answer is partially correct, return False, if "
                "the answer is incorrect return False. "
                "Do not return anything more than a Boolean."
            ),
            annotations_path=None,
            annotations_output=None,
            
    ):
        self.dataset_str = dataset_str
        self.model_id_evaluated = model_id_evaluated
        self.model_id_evaluator = model_id_evaluator
        self.split_name = split_name
        self.backend = backend.lower()
        self.api_workers = api_workers
        self.prompt_format = prompt_format
        self.annotations_path = annotations_path

        if self.annotations_path is None:
            base_path_key = DATA_DIR / str(split_name)
            dataset_key = dataset_str.removesuffix("_pandas")
            path_key = os.path.join(base_path_key, dataset_key + ".csv")
            if not os.path.exists(path_key):
                path_key = os.path.join(base_path_key, dataset_str + ".csv")
            self.key = pd.read_csv(path_key, index_col=0)

        # ------------------------------------------------------------------ #
        # Backend-specific initialisation                                      #
        # ------------------------------------------------------------------ #
        if self.backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError("vllm is not installed. Run: pip install vllm")
            self.sampling_params = SamplingParams(
                max_tokens=1024,
                temperature=0,
            )
            self.model = LLM(
                model=self.model_id_evaluator,
                dtype=torch.bfloat16,
                max_model_len=2048,
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=0.9,
            )
            self.tokenizer = get_tokenizer(
                tokenizer_name=self.model_id_evaluator,
                tokenizer_mode="auto",
                trust_remote_code=True,
            )

        elif self.backend == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai is not installed. Run: pip install openai")
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OpenAI API key required (pass api_key= or set OPENAI_API_KEY)")
            self.client = OpenAI(api_key=key)

        elif self.backend == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic is not installed. Run: pip install anthropic")
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Anthropic API key required (pass api_key= or set ANTHROPIC_API_KEY)")
            self.client = anthropic_sdk.Anthropic(api_key=key)

        elif self.backend == "custom_openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai is not installed. Run: pip install openai")
            if not models:
                raise ValueError("Pass a 'models' list: [{'model_id': ..., 'base_url': ...}, ...]")
            # Build one client per model, keyed by model_id
            self.custom_clients = {
                m["model_id"]: OpenAI(api_key=api_key, base_url=m["base_url"])
                for m in models
            }

        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'openai', 'vllm', 'anthropic', or 'custom_openai'."
            )

    # ---------------------------------------------------------------------- #
    # paths_prompt_init — unchanged from original                             #
    # ---------------------------------------------------------------------- #
    def paths_prompt_init(self, directory, prompt_format):
        self.directory = directory
        self.prompt_format = prompt_format

        if self.annotations_path is None:
            model_save_path = self.model_id_evaluated.replace('/', '-')
            dir_path = MODEL_OUTPUT / self.split_name
            dataset_candidates = [
                self.dataset_str,
                f"{self.dataset_str}_pandas",
                self.dataset_str.removesuffix("_pandas"),
            ]

            filename = None
            for dataset_key in dataset_candidates:
                candidate = f"{dataset_key}-{model_save_path}.pickle"
                candidate_path = os.path.join(dir_path, candidate)
                if os.path.exists(candidate_path):
                    filename = candidate
                    break

            if filename is None:
                filename = f"{self.dataset_str}-{model_save_path}.pickle"

            os.makedirs(dir_path, exist_ok=True)
            self.path_final = os.path.join(dir_path, filename)
            print(f"Reading {self.model_id_evaluated} answers from: {self.path_final}")
            with open(self.path_final, 'rb') as f:
                self.outputs = pickle.load(f)
        else:
            self.path_final = self.annotations_path
            print(f"Reading annotations from: {self.path_final}")
            filename = "annotations_Q3.pickle"
            self.key = pd.read_csv(self.path_final)

        dir_path = OUTPUT_PATH / self.directory / "judgements" / self.split_name
        os.makedirs(dir_path, exist_ok=True)
        self.path = os.path.join(dir_path, filename.replace(".pickle", "-partial.pickle"))
        self.path_final = os.path.join(dir_path, filename)
        print(f"Judgements partial checkpoint: {self.path}")

        chain_thought_path = OUTPUT_PATH / self.directory / "chain_of_thought" / self.split_name
        os.makedirs(chain_thought_path, exist_ok=True)
        self.path_chain = os.path.join(chain_thought_path, filename.replace(".pickle", "-partial.pickle"))
        self.path_final_chain = os.path.join(chain_thought_path, filename)
        print(f"Chain-of-thought partial checkpoint: {self.path_chain}")

        for path, attr in [(self.path, 'partial_answers'), (self.path_chain, 'partial_chains')]:
            try:
                with open(path, 'rb') as f:
                    setattr(self, attr, pickle.load(f))
            except FileNotFoundError:
                setattr(self, attr, [])
                with open(path, 'wb') as f:
                    pickle.dump([], f)

    # ---------------------------------------------------------------------- #
    # create_messages                                                          #
    # ---------------------------------------------------------------------- #
    def create_messages(self, input_format=None):
        """
        Builds self.messages as a list of raw message dicts (role/content pairs).
        For vLLM the tokenizer template is applied later; API backends use them as-is.
        """
        if input_format is None:
            input_format = (
                "Question: {question}\n"
                "Candidate answer: {output}\n"
                "Ground truth answer: {key_num}"
            )

        raw_messages = []
        if self.annotations_path is None:
            for (_, row), output in zip(self.key.iterrows(), self.outputs):
                content = input_format.format(
                    question=row.get("question", ""),
                    key_num=row.get("key_num", ""),
                    output=output,
                )
                raw_messages.append([
                    {"role": "system", "content": self.prompt_format},
                    {"role": "user",   "content": content},
                ])
        else:
            for (_, row) in self.key.iterrows():
                content = input_format.format(
                    question=row["question"],
                    key_num=row["key_num"],
                    output=row["proposed_answer"],
                )
                raw_messages.append([
                    {"role": "system", "content": self.prompt_format},
                    {"role": "user",   "content": content},
                ])

        if self.backend == "vllm":
            # Apply the HF chat template for the local model
            self.messages = [
                self.tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in raw_messages
            ]
        else:
            # API backends consume the list-of-dicts directly
            self.messages = raw_messages

    # ---------------------------------------------------------------------- #
    # _parse_response — shared boolean extraction logic                       #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _parse_response(text):
        """
        Returns (chain_of_thought: str, verdict: str)
        verdict is "True", "False", or "X".
        """
        match = re.search(r"(?s)(.*?)assistantfinal\s*(.*)", text)
        if match:
            chain = match.group(1).strip()
            final = match.group(2).strip()
        else:
            chain = "NONE"
            final = text.strip()   # for API models the full text IS the answer

        tokens = final.split()
        if tokens and "True" in tokens[0]:
            return chain, "True"
        elif tokens and "False" in tokens[0]:
            return chain, "False"
        else:
            return chain, "X"

    # ---------------------------------------------------------------------- #
    # _generate_responses — backend dispatcher                                #
    # ---------------------------------------------------------------------- #
    def _generate_responses(self, batch_messages):
        """
        Dispatches a batch of messages to the appropriate backend.
        Returns a list of raw response strings (one per message).
        """
        if self.backend == "vllm":
            responses = self.model.generate(batch_messages, sampling_params=self.sampling_params)
            return [r.outputs[0].text for r in responses]

        elif self.backend == "openai":
            return self._call_api_parallel(batch_messages, self._openai_single)

        elif self.backend == "anthropic":
            return self._call_api_parallel(batch_messages, self._anthropic_single)

        elif self.backend == "custom_openai":
            return self._call_api_parallel(batch_messages, self._custom_openai_single)
        

    def _call_api_parallel(self, batch_messages, call_fn):
        """Runs call_fn over a batch in parallel, preserving order."""
        results = [None] * len(batch_messages)
        with ThreadPoolExecutor(max_workers=self.api_workers) as pool:
            futures = {pool.submit(call_fn, msg): idx for idx, msg in enumerate(batch_messages)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def _openai_single(self, messages):
        """Single OpenAI chat completion call."""
        response = self.client.chat.completions.create(
            model=self.model_id_evaluator,
            messages=messages,
            max_completion_tokens=750,
            # temperature=0,
        )
        return response.choices[0].message.content or ""

    def _custom_openai_single(self, messages):
        """Single call to a custom OpenAI-compatible endpoint."""
        client = self.custom_clients[self.model_id_evaluator]
        response = client.chat.completions.create(
            model=self.model_id_evaluator,
            messages=messages,
            max_tokens=750,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def _anthropic_single(self, messages):
        """Single Anthropic messages call. System prompt is extracted if present."""
        system_prompt = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                user_messages.append(m)

        kwargs = dict(
            model=self.model_id_evaluator,
            max_tokens=1024,
            messages=user_messages,
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text if response.content else ""

    # ---------------------------------------------------------------------- #
    # make_predictions — unchanged control flow, new dispatcher               #
    # ---------------------------------------------------------------------- #
    def make_predictions(self):
        partial_answers = self.partial_answers
        partial_chains  = self.partial_chains
        messages        = self.messages

        while len(partial_answers) != len(messages):
            i = len(partial_answers)
            print(f"Step {i}/{len(messages)} — generating batch...")

            with open(self.path, 'rb') as f:
                partial_answers = pickle.load(f)

            batch = messages[i : i + BATCH_CHECKPOINT_SIZE]
            raw_texts = self._generate_responses(batch)

            for text in raw_texts:
                chain, verdict = self._parse_response(text)
                partial_answers.append(verdict)
                partial_chains.append(chain)

            with open(self.path, 'wb') as f:
                pickle.dump(partial_answers, f)
            with open(self.path_chain, 'wb') as f:
                pickle.dump(partial_chains, f)

            print(f"Step {i}/{len(messages)} done — {len(partial_answers)} judgements saved.")

        answers = copy.deepcopy(partial_answers)
        chains  = copy.deepcopy(partial_chains)

        with open(self.path_final, 'wb') as f:
            pickle.dump(answers, f)
        with open(self.path_final_chain, 'wb') as f:
            pickle.dump(chains, f)

        print(f"All judgements saved to {self.path_final}")
        print(f"All chains saved to    {self.path_final_chain}")