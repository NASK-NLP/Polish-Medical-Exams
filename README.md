<<<<<<< HEAD
# Polish-Medical-Exams
=======
# Reassess-Polish-Medical-Exams
[![arXiv](https://img.shields.io/badge/arXiv-2606.12250-b31b1b.svg)](https://arxiv.org/abs/2606.12250)
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-Reassess--Polish--Medical--Exams-yellow)](https://huggingface.co/datasets/NASK-PIB/Reassess-Polish-Medical-Exams)

Benchmark and evaluation toolkit for assessing large language models on Polish
medical board exams. The project builds several evaluation splits from the
[`NASK-PIB/Reassess-Polish-Medical-Exams`](https://huggingface.co/datasets/NASK-PIB/Reassess-Polish-Medical-Exams)
dataset, runs models locally with vLLM (or via OpenAI/Anthropic-compatible APIs
for the LLM-as-a-judge step), and aggregates the results.

## 🚀 Future Work

- **Replace constrained generation with JSON-based output.** To further improve the robustness of the evaluation we plan to swtich to json-based output parsing. 
## Exams

The benchmark covers seven exams (Polish and English editions where available):

| Split id         | Exam                                  | Language |
| ---------------- | ------------------------------------- | -------- |
| `r_lek`          | LEK — Medical Final Exam        | PL       |
| `r_ldek`         | LDEK — Dentistry Final Exam    | PL       |
| `r_pes_latest`   | PES — Polish Board Certification Exam     | PL       |
| `r_diagnostics`  | PESDL —  Examination for Laboratory Diagnosticians Specialization  | PL       |
| `r_pharmacy`     | PESF — Pharmaceutical Specialist Examination| PL       |
| `r_lek_en`       | LEK (English)                         | EN       |
| `r_ldek_en`      | LDEK (English)                        | EN       |

## Evaluation splits

Each exam is transformed into four evaluation formats by the scripts in
`data_creator/`:

| Split                       | Creator                                   | Description                                              |
| --------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| `multiple_choice`           | `create_multichoice.py`                   | Select all correct statements (multi-select)        |
| `multiple_choice2`          | `create_multichoice2.py`                  | Select all correct answers from A–E                 |
| `abstaining_substitution`  | `create_abstaining_substitution.py`       | Single-answer A–E with abstain/none-of-the-above items |
| `free_form`                 | `create_free_form.py`                     | Open-ended generation (short medical answer)           |

In addition, two prompt methodologies are evaluated for the single-answer
setting inside `evaluation_all_splits.py` / `evaluation_all_eng.py`:

- `our_method` — direct instruction to return only the correct answer. Unlike
  `prev_method`, it does not embed a sample answer in the prompt. Instead it
  relies on constrained generation whenever the backend supports it: the token IDs
  of the valid answer letters, e.g. A–E and a `.` stop token, so the model can
  only emit one of the allowed choices. This constrained path is used for most
  models; a few models that don't tolerate it cleanly (Bielik,
  and `MediPhi-Instruct` on the `multiple_choice` split) fall back to a short
  free generation capped at ~100–200 tokens with a `.` stop token.
- `prev_method` — a prompt that **embeds a sample answer format directly in the
  instruction**, telling the model how to phrase its response. The exact prompt
  (English edition) is:

  > *"Your task is to provide answers to a medical test for doctors. From all the
  > provided answers A, B, C, D, E select only one. If you are not sure, choose
  > the most probable one. Answer in a manner:*
  > *Correct answer is B."*

  The Polish edition mirrors this: *"...Odpowiedz w sposób: Prawidłowa odpowiedź
  to B."* 

## Setup

### 1) Environment

```bash
bash create_env.sh          # creates .venv and installs dependencies (uv)
source .venv/bin/activate
```

### 2) Hugging Face access

The dataset and several gated models require a Hugging Face token:

```bash
hf auth login               # or: export HF_TOKEN=hf_xxx
```

### 3) Create evaluation splits

```bash
bash scripts/run_scripts/create_all_splits.sh
```

This populates `data/multiple_choice`, `data/multiple_choice2`,
`data/free_form`, and `data/abstaining_substitution` with the per-exam CSVs.

## Usage

### Evaluate one model locally

```bash
bash scripts/run_scripts/evaluate_model_local.sh Qwen/Qwen3.5-4B
```

This runs `evaluation_all_splits.py` (Polish exams) and `evaluation_all_eng.py`
(English exams) for the given model, producing predictions for every split and
methodology. Predictions are saved as pickles under `outputs/model_outputs/`.

### Evaluate on a SLURM cluster

Use the size-appropriate template (1 / 2 / 4 GPUs):

- `scripts/jobs/jobs_small_template.sh`
- `scripts/jobs/jobs_medium_template.sh`
- `scripts/jobs/jobs_large_template.sh`

Dispatch a batch of models with `scripts/jobs/job_creator.sh`, or run the
LLM-as-a-judge step with `scripts/jobs/job_judge_creator.sh` →
`scripts/jobs/job_judge.sh`.

> The judge step requires `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) to be set
> in the environment — see `scripts/run_scripts/run_answers_llm_judge.py`.

### Aggregate results

```bash
bash scripts/run_scripts/check_answers.sh
```

Runs the ground-truth checks (`check_base.py`, `check_ABCD_gt.py`) for the
configured models and writes aggregated CSVs to `outputs/results/` and
`outputs/counts.csv`. Free-form judge results can be summarized with
`scripts/run_scripts/calculate_free_form_results.py`.
## Project layout

```
config/            Path constants (BASE_DIR, DATA_DIR, HF_REPO, ...)
data_creator/      Scripts that build the four evaluation splits from HF data
data/              Generated CSV splits + *_pandas.pickle source files
scripts/
  evaluation/      EvaluatorLLM (vLLM) and EvaluatorLLMJudge (API judge)
  run_scripts/     End-to-end entry points (create splits, evaluate, check)
  jobs/             SLURM batch templates (small / medium / large GPU)
outputs/           Model predictions, judgements, and aggregated results
figures/           Composition plots per exam source
```


## Outputs

| Path                              | Contents                                              |
| --------------------------------- | ----------------------------------------------------- |
| `outputs/model_outputs/`          | Raw model predictions per split (pickles)             |
| `outputs/judge_consistent/`       | LLM-judge boolean verdicts + chain-of-thought         |
| `outputs/results/`                | Aggregated accuracy tables (bias/nobias, per-year)   |
>>>>>>> cameraready
