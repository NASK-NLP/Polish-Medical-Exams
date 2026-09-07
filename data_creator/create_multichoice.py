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

def find_options(options_str):
    '''
    Extract from the question the possible options i.e. 1) Cardiac arrest 2) Asthma sth etc. in a form of a list 
    '''
    options_str = options_str.replace("\n", "")
    options = re.split(r'\d+\)', options_str)
    if options[0]=="":
        options = options[1:]
    options = [re.sub(r'[^\w\s]', '', text) for text in options]
    return options

def find_corr_ans(answer_str, gt_answer):
    '''
    Extract from the correct answer included in the questions, the options i.e. 1) Cardiac arrest 2) Asthma  etc. in a form of a list 
    '''
    answer_str = answer_str.split("\n")[1:]
    try:
        if answer_str[-1]=="":
            answer_str = answer_str[:-1]
    except:
        return False
    correct_ans = [ans for ans in answer_str if ans[0]==gt_answer]
    if len(correct_ans)==0:
        return False
    else:
        correct_ans=correct_ans[0]
    correct_opt  = re.findall(r'\d+', correct_ans)
    return correct_opt


def add_to_key(key,question_id,correct_opt,options):
    '''
    For every question return a key that for every option included in the question like "Cardiac arrest" 
    pairs it with the Boolean describing whether it is a correct answer to the question
    '''
    key[question_id]={}
    key[question_id]["options_key"]=[]
    key[question_id]["key_num"]=[]

    if len(correct_opt) == 0 :
        for idx,opt in enumerate(options):
            key[question_id]["options_key"].append((opt,True))
            key[question_id]["key_num"].append((str(idx+1)))
            
    else:
        for idx,opt in enumerate(options):
            if str(idx+1) in correct_opt:
                key[question_id]["key_num"].append((str(idx+1)))
                key[question_id]["options_key"].append((opt,True))
            else:
                key[question_id]["options_key"].append((opt,False))
    return key 

def create_key_questions(key, questions , answers, indeces,editions):
    '''
    For every multiple choice question it returns a key with questions and the answers with the Boolean values that say whether it is a correct ans or not 
    '''
    for q,a,id,e in zip(questions,answers, indeces,editions):
        temp = q.split("Prawidłowa odpowiedź to")
        q_o = re.split(r'[:?]', temp[0], maxsplit=1)
        # temp=temp[:len(temp)-len("Prawidłowa odpowiedź to")]
        if len(q_o )!=2:
            continue
        question = q_o [0]
        options_str = q_o [1]
        options = find_options(options_str=options_str)
        
        answer_str = temp[-1]
        correct_opt = find_corr_ans(answer_str=answer_str, gt_answer=a)
        if correct_opt==False:
            continue
        
        key = add_to_key(key,question_id=id,correct_opt=correct_opt, options=options)
        key[id]["question"] = temp[0]
        key[id]["edition"] = e 
    return key

def identify_multichoice(data):
    pattern = r"prawidłowa odpowiedź"
    matches = data[data["question"].str.contains(pattern, flags=re.IGNORECASE, regex=True)]
    questions= matches["question"]
    answers = matches["answer"]
    editions = matches["exam_name"]    
    indeces = matches.index
    indeces2 = list(zip(matches["id"], matches["exam_name"]))
    return questions,answers,indeces, editions, indeces2
def create_eng_key(indeces):
    key={}
    data = load_hf_split_compatible(split_id_en)
    data.reset_index(drop=True, inplace=True)
    for (id,row) in data.iterrows():
        if (row["id"], row["exam_name"]) in indeces:
            q = row["question"]
            e = row["exam_name"]
            a = row["answer"]
            temp = q.split("Correct answear is")
            if len(temp)!=2:
                temp = q.split("The correct answer is")
            if len(temp)!=2:
                continue
            q_o = re.split(r'[:?]', temp[0], maxsplit=1)
            # temp=temp[:len(temp)-len("Prawidłowa odpowiedź to")]
            if len(q_o )!=2:
                continue
            question = q_o [0]
            options_str = q_o [1]
            options = find_options(options_str=options_str)
            
            answer_str = temp[-1]
            correct_opt = find_corr_ans(answer_str=answer_str, gt_answer=a)
            if correct_opt==False:
                continue
            
            key = add_to_key(key,question_id=id,correct_opt=correct_opt, options=options)
            key[id]["question"] = temp[0]
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


    # sampled_df = key_df.groupby("edition", group_keys=False).apply(
    #     lambda x: x.sample(n=min(len(x), 20), random_state=42)
    # )
    # sampled_df = key_df.sample(n=250, random_state=42)
    sampled_df = key_df.drop(columns="edition")
    base_path = DATA_DIR / "multiple_choice"
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
