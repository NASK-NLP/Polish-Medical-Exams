import argparse
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd

from config.paths import OUTPUT_PATH

EXAMS = [
    "r_lek",
    "r_ldek",
    "r_pes_latest",
    "r_diagnostics",
    "r_pharmacy",
    "r_lek_en",
    "r_ldek_en",
]

NAME_MAPPING = {
    "r_lek": "LEK",
    "r_ldek": "LDEK",
    "r_pes_latest": "PES",
    "r_lek_en": "LEK EN",
    "r_ldek_en": "LDEK EN",
    "r_diagnostics": "PESDL",
    "r_pharmacy": "PESF",
}

MODELS_SMALL = [
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
]

MODELS_MEDIUM = [
    # "google/gemma-3-27b-it",
    # "google/medgemma-27b-it",
    # "Qwen/Qwen3-30B-A3B-Instruct-2507",
]

MODELS_LARGE = [
    # "OpenMeditron/Meditron3-70B",
    # "meta-llama/Llama-3.3-70B-Instruct",
    # "CYFRAGOVPL/Llama-PLLuM-70B-instruct",
    # "aaditya/Llama3-OpenBioLLM-70B",
    # "Qwen/Qwen3.5-9B",
    # "Qwen/Qwen3.5-35B-A3B",
    # "Qwen/Qwen3.5-122B-A10B-FP8",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate free-form judge accuracy from pickle outputs."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=MODELS_SMALL + MODELS_MEDIUM + MODELS_LARGE,
        help="Model IDs to include. Defaults to the predefined benchmark model list.",
    )
    parser.add_argument(
        "--judgements-dir",
        type=Path,
        default=OUTPUT_PATH / "judge_consistent" / "judgements" / "free_form",
        help="Directory with judgement pickle files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_PATH / "results" / "free_form_judge_consistent.csv",
        help="Where to save the aggregated CSV.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any judgement file is missing.",
    )
    return parser.parse_args()


def normalize_label(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def compute_accuracy(judgements):
    normalized = [normalize_label(v) for v in judgements]
    counts = Counter(normalized)
    denom = counts["True"] + counts["False"]
    if denom == 0:
        return None
    return round(counts["True"] / denom * 100, 2)


def resolve_judgement_file(judgements_dir, exam, model_name):
    candidates = [exam, f"{exam}_pandas", exam.removesuffix("_pandas")]
    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        file_path = judgements_dir / f"{candidate}-{model_name}.pickle"
        if file_path.exists():
            return file_path

    return judgements_dir / f"{ordered_candidates[0]}-{model_name}.pickle"


def main():
    args = parse_args()
    results = {}

    for model in args.models:
        model_name = model.replace("/", "-")
        row = {}

        for exam in EXAMS:
            file_path = resolve_judgement_file(args.judgements_dir, exam, model_name)

            if not file_path.exists():
                message = f"Missing file: {file_path}"
                if args.strict:
                    raise FileNotFoundError(message)
                print(f"[WARN] {message}")
                row[NAME_MAPPING[exam]] = None
                continue

            with open(file_path, "rb") as f:
                judge = pickle.load(f)

            row[NAME_MAPPING[exam]] = compute_accuracy(judge)

        results[model_name] = row

    df = pd.DataFrame.from_dict(results, orient="index")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv)

    print(df)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
