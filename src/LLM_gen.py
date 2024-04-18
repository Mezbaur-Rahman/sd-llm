import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
import pandas as pd

from string import Template
import time
import re
import numpy as np
import math
import torch
from IPython.display import clear_output


from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import warnings
import logging
import argparse
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--model', type=str, help='huggingface model id')

parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--output_path', type=str, help='model output path')
parser.add_argument('--dataset', type=str, help='dataset name')
parser.add_argument('--task', type=str, help='masked text stance detection (MTSD) or normal stance detection (SD)')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1.0708609960508476e-05, help='learning rate')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
args = parser.parse_args()
print(args)
time.sleep(5)



def make_path(*args):
    path = os.path.join(*args)
    if not os.path.isabs(path):
        # Convert relative path to absolute path
        path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path

root_dir = "./"
data_dir = make_path(root_dir + "data/",f"{args.dataset}")
output_dir = make_path(root_dir + "output/",f"{args.dataset}")
model_dir = make_path(root_dir + "saved_model/",f"{args.dataset}")
processed_data_dir = make_path(root_dir + "processed_data/",f"{args.dataset}")
# model_id ="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_id = "mistralai/Mistral-7B-v0.2"
# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# model_id = "lmsys/vicuna-7b-v1.5"
# model_id = "meta-llama/Llama-2-7b-chat-hf"
model_id = args.model
base_model = model_id.split("/")[1].split("-")[0].lower()
output_model_dir = make_path(model_dir,base_model)


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
    
    
    

# base_model = "mistralai/Mistral-7B-v0.2"
# base_model = "mistralai/Mistral-7B-Instruct-v0.2"
# base_model = "lmsys/vicuna-7b-v1.5"
# base_model = "meta-llama/Llama-2-7b-chat-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"


model = AutoModelForCausalLM.from_pretrained(model_id,  device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)




def convert_to_vicuna_format(prompt):
    """
    Converts a given prompt to the Vicuna prompt format.
    
    Args:
        prompt (str): The input prompt to be converted.
        
    Returns:
        str: The prompt in Vicuna format.
    """
    vicuna_prompt = f"### Human: {prompt}\n### Assistant:"
    return vicuna_prompt


def convert_to_mistral_format(prompt):
    """
    Converts a given prompt to the Mistral prompt format.
    
    Args:
        prompt (str): The input prompt to be converted.
        
    Returns:
        str: The prompt in Mistral format.
    """
    mistral_prompt = f"<s>[INST] {prompt} [/INST]\nAssistant:"
    return mistral_prompt

def convert_to_llama_format(prompt):
    """
    Converts a given prompt to the Llama prompt format.
    
    Args:
        prompt (str): The input prompt to be converted.
        
    Returns:
        str: The prompt in Llama format.
    """
    sys_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    llama_prompt = f"User: {prompt}\nAssistant:"
    llama_prompt = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n{llama_prompt} [/INST]"
    
    return llama_prompt

def generate_text(prompt, max_new_tokens=25, num_return_sequences=1, top_k=50, top_p=0.95, num_beams=4, early_stopping=True):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # print(device)
    # time.sleep(2)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        # num_return_sequences=num_return_sequences,
        # top_k=top_k,
        # top_p=top_p,
        # num_beams=num_beams,
        # early_stopping=early_stopping,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


# root_dir = "/home/mrahma56/TSE/"
# data_dir = root_dir + "data/"
# processed_data_dir = root_dir + "processed_data/"
# output_dir = root_dir + "output/"
# semeval_dir = data_dir + "semeval/"
# semeval_out_dir = output_dir + "semeval/"


# semeval_test = csv2pd(semeval_dir + "test.csv")
# semeval_test.head()

# test_df = csv2pd(data_dir + "test.csv")
# test_df.head()

# test_df = csv2pd(os.path.join(processed_data_dir,f'{args.dataset}',"masked_test.csv"))
if args.task == "MTSD":
    test_df = csv2pd(os.path.join(processed_data_dir,"masked_test.csv"))
else:
    test_df = csv2pd(os.path.join(data_dir,"test.csv"))

# print(test_df.head())
# time.sleep(1000)
# base_prompt = "Your are given a tweet and a target word. Your task is to determine the tweet's stance towards the target. \nTweet: $tweet\nTarget: $target\nNow generate the stance of the tweet towards the target. The stance can either of the following: 'FAVOR', 'AGAINST', 'NONE'."
base_prompt = "What is the attitude of the sentence: '$tweet' to the Target: '$target'? Select an answer from(favor,against,none)."
# model_name = base_model.split('/')[-1]



# Get the current timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_log_path = os.path.join(output_dir,f'log_{timestamp}.txt')
print(f"Logging to {output_log_path}")
logging.basicConfig(
    filename=output_log_path,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


st = time.time()
test_df[f'{base_model}' + '_response'] = ""
for i in range(len(test_df)):
    tweet = test_df.iloc[i]['Tweet']
    target = test_df.iloc[i]['Target']
    prompt = Template(base_prompt).substitute(tweet=tweet, target=target)
    if base_model == "vicuna":
        prompt = convert_to_vicuna_format(prompt)
        # print(prompt)
    if base_model == "mistral":
        prompt = convert_to_mistral_format(prompt)
    if base_model == "llama":
        prompt = convert_to_llama_format(prompt)
    
    # prompt = [{"role": "user", "content": base_prompt}]
    response = generate_text(prompt)
    test_df.at[i, f'{base_model}' + '_response'] = response[len(prompt):]
    # print(response)
    # break
    if i % 50 == 0:
        en = time.time()
        print(f"Completed {i + 1}/{len(test_df)} Iterations in {en - st} seconds")
        logging.info(f"Completed {i + 1}/{len(test_df)} Iterations in {en - st} seconds")
        st = time.time()
        # break
print(output_dir)
if args.task == "MTSD":
    pd2csv(test_df,output_dir,f"{base_model}_masked_test.csv")
    logging.info(f"{base_model}_masked_test.csv is saved to {output_dir}")
else:
    pd2csv(test_df,output_dir,f"{base_model}_test.csv")
    logging.info(f"{base_model}_test.csv is saved to {output_dir}")

