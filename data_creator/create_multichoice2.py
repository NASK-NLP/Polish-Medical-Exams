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

def identify_multichoice(data):
    pattern = r"prawdziwe są odpowiedzi"
    matches = data[data["question"].str.contains(pattern, flags=re.IGNORECASE, regex=True)]
    questions= matches["question"]
    answers = matches["answer"]
    editions = matches["exam_name"]    
    indeces = matches.index
    indeces2 = list(zip(matches["id"], matches["exam_name"]))
    return questions,answers,indeces, editions, indeces2

def create_key_questions(key, questions , answers, indeces,editions):
    '''
    For every multiple choice question it returns a key with questions and the answers with the Boolean values that say whether it is a correct ans or not 
    '''
    idx_a = ['A','B','C','D','E']
    for q,a,id,e in zip(questions,answers, indeces,editions):
        q_o = re.split(r'[:?]', q, maxsplit=1)
        parts = q_o[-1].splitlines()
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts)!=5:
            continue
        #find hidden multichoice
        pattern = r"prawdziwe są odpowiedzi"
        filtered = [p if re.search(pattern, p, re.IGNORECASE) else "" for p in parts]
        pattern = r"wszystkie\s+.*?\s+(prawidłowe|prawdziwe)"
        filtered2 = [p for p in parts if re.search(pattern, p, re.IGNORECASE)]

        pattern = r"[A-Z](?=[ ,\.])"
        #extract the answer "A","B" from asnwer string
        all_matches = [re.findall(pattern, f)[1:] if re.findall(pattern, f) else [""] for f in filtered]
        if len(filtered2)>0:
            all_idx = filtered2[0][0]
            indexes = [idx_a[i] for i, x in enumerate(all_matches) if x == [""]]
            indexes.remove(all_idx)
            all_idx = idx_a.index(all_idx)
            all_matches[all_idx]=indexes


        #idx of gt answer
        try:
            a_idx = idx_a.index(a)
        except:
            continue
        gt = all_matches[a_idx]
        key[id] = {}
        for idx,answer in enumerate(all_matches):
            if len(answer)>1:
                parts[idx]=""
        parts = [p for p in parts if p.strip()]
        parts = "\n".join(parts)
        question = [q_o[0]]+[parts]
        question = ":\n".join(question)
        if len(gt)>1:
            key[id]["key_num"]=gt
        else:
            key[id]["key_num"]=[a]
        key[id]["question"]=question
        key[id]["edition"] = e 
    return key

def create_eng_key(indeces):
    key={}
    data = load_hf_split_compatible(split_id_en)
    data.reset_index(drop=True, inplace=True)
    idx_a = ['A','B','C','D','E']
    for (id,row) in data.iterrows():
        if (row["id"], row["exam_name"]) in indeces:
            q = row["question"]
            e = row["exam_name"]
            a = row["answer"]
            q_o = re.split(r'[:?]', q, maxsplit=1)
            parts = q_o[-1].splitlines()
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts)!=5:
                continue
            pattern = r"(?:are\s+(?:correct|true)|true\s+(?:answers|statements)|correct\s+answers?|answers|\&)"
            filtered = [p if re.search(pattern, p, re.IGNORECASE) else "" for p in parts]
            pattern = r"[A-Z](?=[ ,\.])"
            #extract the answer "A","B" from asnwer string
            all_matches = [re.findall(pattern, f)[1:] if re.findall(pattern, f) else [""] for f in filtered]
            #idx of gt answer
            a_idx = idx_a.index(a)
            gt = all_matches[a_idx]
            key[id] = {}
            for idx,answer in enumerate(all_matches):
                if len(answer)>1:
                    parts[idx]=""

            parts = "\n".join(parts)
            question = [q_o[0]]+[parts]
            question = ":\n".join(question)
            if len(gt)>1:
                key[id]["key_num"]=gt
            else:
                key[id]["key_num"]=[a]
            key[id]["question"]=question
            key[id]["edition"] = e 
    return key, data
if __name__ == "__main__":
    questions,answers,indeces,editions, indeces2=identify_multichoice(data)

    key = create_key_questions(key={},
                                questions=questions,
                                answers=answers,
                                indeces=indeces,
                                editions=editions
    )
    key_df = pd.DataFrame(key)
    key_df = key_df.T
    # sampled_df = key_df.sample(n=250, random_state=42)
    sampled_df = key_df.drop(columns="edition")
    base_path = DATA_DIR / "multiple_choice2"
    os.makedirs(base_path, exist_ok=True)
    extracted_rows = data.loc[sampled_df.index]
    sampled_df.to_csv(os.path.join(base_path, dataset_str + ".csv"), index=True)
    extracted_rows.to_csv(os.path.join(base_path, dataset_str + "gt.csv"), index=True)
    print("Extracted questions: ", len(sampled_df))
    if split_id_en is not None:
        key_eng, data_eng = create_eng_key(indeces=indeces2)
        key_eng_df = pd.DataFrame(key_eng)
        key_eng_df = key_eng_df.T
        sampled_df = key_eng_df.drop(columns="edition")
        extracted_rows = data_eng.loc[sampled_df.index]
        sampled_df.to_csv(os.path.join(base_path, dataset_str + "_en.csv"), index=True)
        extracted_rows.to_csv(os.path.join(base_path, dataset_str + "_engt.csv"), index=True)
        print("Extracted questions ENG: ", len(sampled_df))
