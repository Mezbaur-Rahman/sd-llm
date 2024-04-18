import os
import pandas as pd

from string import Template
import time
import re
import numpy as np
import math
import torch
from IPython.display import clear_output


from evaluate import load
import warnings
import logging
import argparse
import datetime
from sklearn.metrics import f1_score,accuracy_score


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='huggingface model id')
parser.add_argument('--dataset', type=str, help='dataset name')
parser.add_argument('--task', type=str, help='masked text stance detection (MTSD) or normal stance detection (SD) or fine-tuned text stance detection (FTSD) ')


args = parser.parse_args()
# print(args)
# time.sleep(5)

def make_path(*args):
    path = os.path.join(*args)
    if not os.path.isabs(path):
        # Convert relative path to absolute path
        path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path

# root_dir = "./../"
root_dir = "./"
data_dir = make_path(root_dir + "data/",f"{args.dataset}")
output_dir = make_path(root_dir + "output/",f"{args.dataset}")
processed_data_dir = make_path(root_dir + "processed_data/",f"{args.dataset}")
# model_id ="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_id = "mistralai/Mistral-7B-v0.2"
# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# model_id = "lmsys/vicuna-7b-v1.5"
# model_id = "meta-llama/Llama-2-7b-chat-hf"
model_id = args.model
base_model = model_id.split("/")[1].split("-")[0].lower()


def csv2pd(filepath):
    
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252', 'utf-16']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f'Successful read with encoding: {encoding}')
            return df
        except UnicodeDecodeError:
            print(f'Failed with encoding: {encoding}')

def json2pd(filepath):
    df = pd.read_json(filepath,orient='records', lines=True)
    return df

def pd2csv(df,output_dir,filename):
    df.to_csv(os.path.join(output_dir,filename),index= False)
    
    
    
if args.task == 'FTSD':
    test_df = csv2pd(os.path.join(output_dir,f"{base_model}_finetuned_test.csv"))
elif args.task == 'MTSD':
    test_df = csv2pd(os.path.join(output_dir,f"{base_model}_masked_test.csv"))
else:
    test_df = csv2pd(os.path.join(output_dir,f"{base_model}_test.csv"))

def check_stance(text):
    if "against" in text.lower():
        return "AGAINST"
    elif "favor" in text.lower():
        return "FAVOR"
    else:
        return "NONE"
    
def check_stance_ft(text):
    text = text.split('<|im_start|>assistant:')[1].split('<|im_end|>')[0].strip()
    if "against" in text.lower():
        return "AGAINST"
    elif "favor" in text.lower():
        return "FAVOR"
    else:
        return "NONE"

response_columns = test_df.filter(regex='response').columns.tolist()
res_col = response_columns[0]
# print(response_columns)
# time.sleep(1000)
if args.task == 'FTSD':
    test_df["Predicted"] = test_df[res_col].apply(check_stance_ft)
else:
    test_df["Predicted"] = test_df[res_col].apply(check_stance)
# test_df["Predicted"].value_counts()

# count = len(test_df[test_df['Stance'] == test_df['Predicted']])
# print(count,len(test_df))
# print(count/len(test_df) * 100)
# print('Accuracy:',count/len(test_df) * 100)


#print f1-score
print('F1 Score:',f1_score(test_df['Stance'], test_df['Predicted'], average='weighted'))
#calculate accuracy
print('Accuracy: ',accuracy_score(test_df['Stance'], test_df['Predicted']))