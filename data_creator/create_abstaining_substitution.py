import pickle
import re 
import os
import sys
import pandas as pd 
from config.paths import DATA_DIR
from data_creator.utils import load_hf_split_compatible, resolve_source_split_ids
dataset_str  = sys.argv[1]

split_id, split_id_en = resolve_source_split_ids(dataset_str)
data = load_hf_split_compatible(split_id)
data.reset_index(drop=True, inplace=True)

def identify_not_already_abstain(data, negative_uuid):
    patterns = [
    r"żadn(?:a|e) ",
    r"żaden",
    r"prawidłowa odpowiedź",
    r"prawdziwe są odpowiedzi",
    r"powyższe",
    r"prawdopodobna",
    r"nie należy",
    r"nie jest",
    r"wyjątkiem",
    r"nie zalicza",
    r"wszystkie wymienione",
    r"z wyjątkiem",
    r"wszystki",
    r"przedstawiony",
    r"wymienionych",
    r"naj\w*" 
]
    pattern = "|".join(patterns)
    matches = data[~data["question"].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)]
    matches["uuid"] = matches["uuid"].astype(str)
    negative_uuid = [str(u) for u in negative_uuid]
    matches = matches[~matches["uuid"].isin(negative_uuid)]
    questions= matches["question"]
    answers = matches["answer"]
    editions = matches["exam_name"]    
    indeces = matches.index
    indeces2 = list(zip(matches["id"], matches["exam_name"]))
    return questions,answers,indeces, editions, indeces2

def create_key_abstaining(key, questions , answers, indeces,editions):
    '''
    For every multiple choice question it returns a key with questions and the answers with the Boolean values that say whether it is a correct ans or not 
    '''
    idx_a = ['A','B','C','D','E']
    for q,a,id,e in zip(questions,answers, indeces,editions):
        q_o = q.split("\n")
        ans = q_o[1:]
        if len(ans)!=5:
            continue
        try:
            a_idx = idx_a.index(a)
        except:
            continue
        abstain_ans_sub = f"{a}. żadna z odpowiedzi nie jest poprawna"
        ans[a_idx]=abstain_ans_sub
        ans ="\n".join(ans)
        q_a = q_o[0]+"\n"+ans
        key[id]={}
        key[id]["key_num"]=[a]
        key[id]["question"]=q_a
        key[id]["edition"] = e

    return key
def create_eng_key(indeces, negative_uuid):
    negative_uuid = [str(u) for u in negative_uuid]
    key={}
    idx_a = ['A','B','C','D','E']
    data = load_hf_split_compatible(split_id_en)
    data.reset_index(drop=True, inplace=True)
    for (id,row) in data.iterrows():
        if (row["id"], row["exam_name"]) in indeces:
            row["uuid"] = str(row["uuid"])
            if row["uuid"] in negative_uuid:
                continue
            q = row["question"]
            e = row["exam_name"]
            a = row["answer"]
            q_o = re.split(r'[:?]', q, maxsplit=1)
            ans = q_o[-1].split("\n")[1:]
            if len(ans)!=5:
                continue
            try:
                a_idx = idx_a.index(a)
            except:
                continue
            abstain_ans_sub = f"{a}. none of the answers is correct"
            ans[a_idx]=abstain_ans_sub
            ans ="\n".join(ans)
            q_a = q_o[0]+":"+"\n"+ans
            key[id]={}
            key[id]["key_num"]=[a]
            key[id]["question"]=q_a
            key[id]["edition"] = e
    return key, data

def identify_free_form_ids():
    negative_uuid = []  

    for dataset_str in ["r_lek", "r_ldek", "r_pes_latest", "r_diagnostics", "r_pharmacy", "r_lek_en", "r_ldek_en"]:
        free_form_path = DATA_DIR / "free_form" / f"{dataset_str}gt.csv"
        free_form_df = pd.read_csv(free_form_path)
        negative_uuid.extend(free_form_df["uuid"].tolist())
    return negative_uuid

if __name__ == "__main__":
    negative_uuid = identify_free_form_ids()

    questions,answers,indeces,editions, indeces2=identify_not_already_abstain(data,negative_uuid)

    key = create_key_abstaining(key={},
                                questions=questions,
                                answers=answers,
                                indeces=indeces,
                                editions=editions
    )
    key_df = pd.DataFrame(key)
    key_df = key_df.T

    # sampled_df = key_df.sample(n=250, random_state=42)
    sampled_df = key_df.drop(columns="edition")
    base_path = DATA_DIR / "abstaining_substitution"
    os.makedirs(base_path, exist_ok=True)
    extracted_rows = data.loc[sampled_df.index]
    sampled_df.to_csv(os.path.join(base_path, dataset_str + ".csv"), index=True)
    extracted_rows.to_csv(os.path.join(base_path, dataset_str + "gt.csv"), index=True)
    print("Extracted questions: ", len(sampled_df))
    if split_id_en is not None:
        key_eng, data_eng = create_eng_key(indeces=indeces2, negative_uuid=negative_uuid)
        key_eng_df = pd.DataFrame(key_eng)
        key_eng_df = key_eng_df.T
        # sampled_df = key_eng_df.sample(n=250, random_state=42)
        sampled_df = key_eng_df.drop(columns="edition")
        extracted_rows = data_eng.loc[sampled_df.index]
        sampled_df.to_csv(os.path.join(base_path, dataset_str + "_en.csv"), index=True)
        extracted_rows.to_csv(os.path.join(base_path, dataset_str + "_engt.csv"), index=True)
        print("Extracted questions ENG: ", len(sampled_df))


    