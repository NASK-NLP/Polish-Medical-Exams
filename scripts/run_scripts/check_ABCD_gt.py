import sys
import os
import pickle
import pandas as pd
import re
import ast
from config.paths import DATA_DIR, MODEL_OUTPUT, OUTPUT_PATH, HF_REPO
from pathlib import Path
name_mapping = {"r_lek":"LEK", "r_ldek": "LDEK", "r_pes_latest": "PES", "r_lek_en": "LEK EN", "r_ldek_en": "LDEK EN", "r_diagnostics": "PESDL" , "r_pharmacy": "PESF"}
split_mapping= {"multiple_choice": "Multiple Statements", "multiple_choice2": "Multiple Answers", "abstaining_substitution": "Correct answer substitution"}
dataset_split_mapping = {
    "r_lek": "r_lek_pandas",
    "r_ldek": "r_ldek_pandas",
    "r_pes_latest": "r_pes_latest_pandas",
    "r_diagnostics": "r_diagnostics_pandas",
    "r_pharmacy": "r_pharmacy_pandas",
    "r_lek_en": "r_lek_en_pandas",
    "r_ldek_en": "r_ldek_en_pandas",
}


def resolve_prediction_path(base_dir, dataset_key, model_key):
    candidates = [dataset_key, dataset_split_mapping.get(dataset_key, dataset_key)]
    for candidate in candidates:
        path = os.path.join(base_dir, f"{candidate}-{model_key}.pickle")
        if os.path.exists(path):
            return path
    return os.path.join(base_dir, f"{candidates[-1]}-{model_key}.pickle")


def check_ans(text, gt):
    capital_letters = re.findall(r"[A-Z]", text)
    # print(capital_letters)
    # if len(numbers)==0:
    #     print("invalid ans")
    if set(capital_letters )==set(gt):
        return True
    else:
        return False
def extract_year(exam_name):
    match = re.search(r"(19|20)\d{2}", exam_name)
    return int(match.group()) if match else None

def check_ans_abst(text, gt):
    match = re.search(r"[A-Z]", text)
    if match:
        return [match.group()] == gt
    return False
    
def check_ans_int(text, gt):
    numbers = re.findall(r"\d+", text)
    # print(numbers)
    # if len(numbers)==0:
    #     print("invalid ans")
    if set(numbers)==set(gt):
        return True
    else:
        return False
def check_base_key(text, gt_text):
    capital_letters = re.findall(r"[A-Z]", text)
    capital_letters_gt = re.findall(r"[A-Z]", gt_text)
    if len(capital_letters) == 0 or len(capital_letters_gt)==0:
        return False
    else:
        capital_letters_gt=capital_letters_gt[0]
        capital_letters=capital_letters[0]
    # breakpoint()
    if set(capital_letters )==set(capital_letters_gt):
        return True
    else:
        return False
results={}
results={}
row={}
model_id  = sys.argv[1]
results_with_counts={"model": model_id}
# for dataset_str in ["r_lek","r_ldek","r_pes_latest","r_diagnostics","r_pharmacy"]:
for dataset_str in ["r_lek","r_ldek","r_pes_latest","r_diagnostics","r_pharmacy","r_lek_en","r_ldek_en"]:
    method = sys.argv[2]
    split = sys.argv[3]
    model_id = model_id.replace('/','-')
    gt_base_path = DATA_DIR / f"{split}"
    gt_path = os.path.join(gt_base_path,dataset_str+"gt.csv")
    ans_base_path = MODEL_OUTPUT / method
    file_path = resolve_prediction_path(ans_base_path, dataset_str, model_id)

    gt = pd.read_csv(gt_path, index_col=0)  
    gt_index_list = list(gt.index)          
    gt_answers = gt["answer"].tolist()
    with open(file_path, 'rb') as f:
        answers = pickle.load(f)


    filtered_answers = [answers[i] for i in gt_index_list if i < len(answers)]


    corr = 0
    for a, ga in zip(filtered_answers, gt_answers):
        if check_base_key(a,ga):
            corr+=1
        
    acc_base = corr / len(gt_answers)
    results_with_counts[f"{dataset_str}_correct_org"] = corr
    results_with_counts[f"{dataset_str}_count_org"] = len(gt_answers)

    # print(f"{model_id}")
    # print(f"{dataset_str} {bias}: {acc_base}")


    gt_multi = os.path.join(gt_base_path,dataset_str+".csv")
    gt_multi =pd.read_csv(gt_multi, index_col=0) 
    gt_multi["key_num"] = gt_multi["key_num"].apply(ast.literal_eval)
    gt_ans = gt_multi["key_num"].tolist()

    ans_base_path = MODEL_OUTPUT /  split
    file_path = resolve_prediction_path(ans_base_path, dataset_str, model_id)
    with open(file_path, 'rb') as f:
        answers = pickle.load(f)



    corr = 0
    for a,gt in zip(answers,gt_ans):
        if split == "multiple_choice":
            if check_ans_int(a,gt):
                corr+=1
        elif split == "multiple_choice2":
            if check_ans(a,gt):
                corr+=1
        elif split == "abstaining_substitution":
            if check_ans_abst(a,gt):
                corr+=1
    acc = corr/len(answers)
    # breakpoint()
    # print(f" {split}{dataset_str}: {acc}")
    # print("#"*20)
    row[f"{name_mapping[dataset_str]} original"] = round(acc_base *100,2)
    row[f"{name_mapping[dataset_str]} {split_mapping[split]}"] = round(acc*100,2)
    results_with_counts[f"{dataset_str}_correct"] = corr
    results_with_counts[f"{dataset_str}_count"] = len(gt_answers)
new_row = pd.DataFrame([results_with_counts]).set_index("model")
counts_path = OUTPUT_PATH / "counts.csv"
try:
    counts_df = pd.read_csv(counts_path, index_col=0)
    if model_id in counts_df.index:
        for col in new_row.columns:
            counts_df.loc[model_id, col] = (
                counts_df.loc[model_id, col] + new_row.loc[model_id, col]
            )
    else:
        counts_df = pd.concat([counts_df, new_row])

except FileNotFoundError:
    counts_df = new_row

counts_df.to_csv(counts_path)
results[f"{model_id}"]=row
csv_file = OUTPUT_PATH / "results" /f"{split}.csv"
column_order = []
# breakpoint()
print(model_id)
print(answers[0])
print(file_path)

# for dataset in ["r_lek","r_ldek","r_pes_latest","r_diagnostics","r_pharmacy"]:
for dataset in ["r_lek","r_ldek","r_pes_latest","r_diagnostics","r_pharmacy","r_lek_en","r_ldek_en"]:
    column_order.append(f"{name_mapping[dataset]} original")
    column_order.append(f"{name_mapping[dataset]} {split_mapping[split]}")

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
# print(f"Results saved to {csv_file}")