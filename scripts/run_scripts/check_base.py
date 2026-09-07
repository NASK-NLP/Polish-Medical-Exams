import sys
import os
import pickle
import pandas as pd
from datasets import load_dataset

import re
import ast
from config.paths import DATA_DIR, MODEL_OUTPUT, OUTPUT_PATH, HF_REPO
from pathlib import Path
name_mapping = {"r_lek":"LEK", "r_ldek": "LDEK", "r_pes_latest": "PES", "r_lek_en": "LEK EN", "r_ldek_en": "LDEK EN", "r_diagnostics": "PESDL" , "r_pharmacy": "PESF"}
dataset_split_mapping = {
    "r_lek": "r_lek",
    "r_ldek": "r_ldek",
    "r_pes_latest": "r_pes_latest",
    "r_diagnostics": "r_diagnostics",
    "r_pharmacy": "r_pharmacy",
    "r_lek_en": "r_lek_en",
    "r_ldek_en": "r_ldek_en",
}


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
            continue

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
            continue

    # Backward compatibility for legacy repos that used split names directly.
    for candidate in candidates:
        try:
            return load_dataset(HF_REPO, split=candidate).to_pandas()
        except Exception as err:
            last_error = err
            continue

    if last_error is not None:
        raise last_error
    raise ValueError(f"Could not resolve split: {split_name}")


def resolve_prediction_path(base_dir, dataset_key, model_key):
    candidates = [dataset_key, dataset_split_mapping.get(dataset_key, dataset_key)]
    for candidate in candidates:
        out_f = base_dir / f"{candidate}-{model_key}.pickle"
        if out_f.exists():
            return out_f
    return base_dir / f"{candidates[-1]}-{model_key}.pickle"


def get_answer_one_char(o):
    # print(o)
    for c in (',','.',':',')','*','('):
        o = o.replace(c,' ')
        
    try:
        return [ a for a in o.split() if a in ('A', 'B', 'C', 'D', 'E')][0]
    except IndexError:
        #import pdb; pdb.set_trace()
        # print('NO ANSWER' + o)
        return 'X'
    


    
def check_base_key(text, gt_text):
    capital_letters = re.findall(r"[A-Z]", text)
    capital_letters_gt = re.findall(r"[A-Z]", gt_text)
    if set(capital_letters )==set(capital_letters_gt):
        return True
    else:
        return False
def check_ans(text, gt):
    match = re.search(r"[A-Z]", text)
    if match:
        # breakpoint()
        return match.group() == gt
    return False
results={}
row={}
results_with_counts={}
for dataset_str in ["r_lek","r_ldek","r_pes_latest","r_diagnostics","r_pharmacy","r_lek_en","r_ldek_en"]:
# for dataset_str in ["r_lek_en","r_ldek_en"]:
# for dataset_str in ["r_lek"]:
    model_id  = sys.argv[1]
    method = sys.argv[2]
    model_id = model_id.replace('/','-')

    base_path_key = DATA_DIR
    # path_key = os.path.join(base_path_key,dataset_str+"_pandas.pickle")
    # with open(path_key, 'rb') as f:
    #     data = pickle.load(f)
    split_id = dataset_split_mapping[dataset_str]
    data = load_hf_split_compatible(split_id)
    data.reset_index(drop=True, inplace=True)
    gt = data

    # gt_index_list = list(gt.index)          
    gt_answers = gt["answer"].tolist()      

    out_f = resolve_prediction_path(MODEL_OUTPUT / method, dataset_str, model_id)

    with open(out_f, 'rb') as f:
        answers = pickle.load(f)
    # breakpoint()
    if method == "prev_method":
        answers =  [get_answer_one_char(o) for o in answers]


    # filtered_answers = [answers[i] for i in gt_index_list if i < len(answers)]

    corr = 0
    for a, ga in zip(answers, gt_answers):
        if check_ans(a,ga):
            corr += 1

    acc_base = corr / len(gt_answers)
    
    print(f"{model_id}")
    print(f"{dataset_str}: {round(acc_base*100,2)}")


    row[f"{name_mapping[dataset_str]}"] = round(acc_base*100,2)
results[f"{model_id} {method}"]=row
csv_file = OUTPUT_PATH / "results" /f"{method}.csv"
column_order = []
for dataset in ["r_lek","r_ldek","r_pes_latest","r_diagnostics","r_pharmacy","r_lek_en","r_ldek_en"]:
# for dataset in ["r_lek_en","r_ldek_en"]:
    column_order.append(f"{name_mapping[dataset]}")


if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, index_col=0)

    for col in column_order:
        if col not in df.columns:
            df[col] = None  
    df = df[column_order]  
else:
    df = pd.DataFrame(columns=column_order)


for key, value in results.items():
    df.loc[key] = [value.get(col, None) for col in column_order] 


df.to_csv(csv_file)
print(f"Results saved to {csv_file}")