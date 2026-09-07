import pickle
import re 
import os
import sys
import pandas as pd 
from config.paths import DATA_DIR
from pathlib import Path
from data_creator.utils import load_hf_split_compatible, resolve_source_split_ids
dataset_str  = sys.argv[1]

split_id, split_id_en = resolve_source_split_ids(dataset_str)
data = load_hf_split_compatible(split_id)
data.reset_index(drop=True, inplace=True)

def identify(data):
    pattern = r"\?"
    negative_pattern = [r"poniżej", r"który", r"stwierdzeń", r"Prawidłowa odpowiedź", r"które", r"żadna", r"prawdziwe są", r"wymienionych", r"prawdopodobne", r"poniższych", r"powyższych", r"podanych", r"zaproponowanych", r"powyższe", r"wszystkie", r"przedstawionych", r"niżej",  r"przedstawiony", r"z wyjątkiem"]
    n_pattern = "|".join(negative_pattern)  
    matches = data[data["question"].str.contains(pattern, case=False, regex=True)]

    matches = matches[~matches["question"].str.contains(n_pattern, case=False, regex=True)]
    questions= matches["question"]
    answers = matches["answer"]
    editions = matches["exam_name"]
    indeces = matches.index
    indeces2 = list(zip(matches["id"], matches["exam_name"]))
    return questions,answers,indeces, editions, indeces2

def create_key(key, questions , answers, indeces,editions):
    idx_a = ['A','B','C','D','E']
    for q,a,id,e in zip(questions,answers, indeces,editions):
        q_o = re.split(r'[\?]', q, maxsplit=1)

        ans = q_o[-1].split("\n")[1:]
        if len(ans)!=5:
            continue
        try:
            a_idx = idx_a.index(a)
        except:
            continue
        
        # abstain_ans_add = f"F. żadna z odpowiedzi nie jest poprawna"
        # ans.append(abstain_ans_add)
        # ans ="\n".join(ans)
        # q_a = q_o[0]+":"+"\n"+ans
        key[id]={}
        key[id]["key_num"]=ans[a_idx][3:]
        key[id]["context"]=q
        key[id]["question"]=q_o[0]+"?"
        key[id]["edition"] = e
    return key
def create_eng_key(indeces):
    key={}
    idx_a = ['A','B','C','D','E']
    data = load_hf_split_compatible(split_id_en)
    data.reset_index(drop=True, inplace=True)
    for (id,row) in data.iterrows():
        if (row["id"], row["exam_name"]) in indeces:
            q = row["question"]
            e = row["exam_name"]
            a = row["answer"]
            q_o = re.split(r'[\?]', q, maxsplit=1)
            ans = q_o[-1].split("\n")[1:]
            if len(ans)!=5:
                continue
            a_idx = idx_a.index(a)
            key[id]={}
            key[id]["key_num"]=ans[a_idx][3:]
            key[id]["context"]=q
            key[id]["question"]=q_o[0]+"?"
            key[id]["edition"] = e
    return key,data

if __name__ == "__main__":
    questions,answers,indeces,editions, indeces2=identify(data)


    key = create_key(key={},
                                questions=questions,
                                answers=answers,
                                indeces=indeces,
                                editions=editions
    )
    key_df = pd.DataFrame(key)
    key_df = key_df.T
    # sampled_df = key_df.sample(n=250, random_state=42)
    sampled_df = key_df.drop(columns="edition")
    base_path =DATA_DIR / "free_form"
    os.makedirs(base_path, exist_ok=True)
    extracted_rows = data.loc[sampled_df.index]
    sampled_df.to_csv(os.path.join(base_path, dataset_str + ".csv"), index=True)
    extracted_rows.to_csv(os.path.join(base_path, dataset_str + "gt.csv"), index=True)
    print("Extracted questions: ", len(sampled_df))

    if split_id_en is not None:
        key_eng, data_eng = create_eng_key(indeces=indeces2)
        key_eng_df = pd.DataFrame(key_eng)
        key_eng_df = key_eng_df.T
        # sampled_df = key_df.sample(n=250, random_state=42)
        base_path =DATA_DIR / "free_form"
        sampled_df = key_eng_df.drop(columns="edition")
        extracted_rows = data_eng.loc[sampled_df.index]
        sampled_df.to_csv(os.path.join(base_path, dataset_str + "_en.csv"), index=True)
        extracted_rows.to_csv(os.path.join(base_path, dataset_str + "_engt.csv"), index=True)
        print("Extracted questions ENG: ", len(sampled_df))


    