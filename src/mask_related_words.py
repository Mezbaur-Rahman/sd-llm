import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
import argparse
from bert_score import score
import logging
import datetime

# Parsing command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, help='dataset name')
parser.add_argument('--split', type=str, help='train or test or val')
parser.add_argument('--model', type=str, help='pre-trained model (bert-base-uncased, roberta-base)')
args = parser.parse_args()

# Function to read CSV files with various encodings
def csv2pd(filepath):
    
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252', 'utf-16']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f'Successful read with encoding: {encoding}')
            return df
        except UnicodeDecodeError:
            print(f'Failed with encoding: {encoding}')
            
# Function to read JSON files into pandas DataFrame
def json2pd(filepath):
    df = pd.read_json(filepath,orient='records', lines=True)
    return df

# Function to save DataFrame to CSV
def pd2csv(df,output_dir,filename):
    df.to_csv(os.path.join(output_dir,filename),index= False)

# Function to create directory and return its absolute path
def make_path(*args):
    path = os.path.join(*args)
    if not os.path.isabs(path):
        # Convert relative path to absolute path
        path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


# Setting up directories
root_dir = "./"
data_dir = make_path(root_dir + "data/",f"{args.dataset}")
output_dir = make_path(root_dir + "output/",f"{args.dataset}")
processed_data_dir = make_path(root_dir + "processed_data/",f"{args.dataset}")

# Determining which split to use (train, test, val) and loading the corresponding dataset
if args.split == "train":
    df = csv2pd(os.path.join(data_dir ,"train.csv"))
    print("Split is train")
elif args.split == "test":
    df = csv2pd(os.path.join(data_dir ,"test.csv"))
    print("Split is test")
else:
    df = csv2pd(os.path.join(data_dir ,"val.csv"))
    print("Split is val")
# print("Length of the dataframe:",len(df))
# print("Processed Data Directory:",processed_data_dir)
# time.sleep(1000)





def replace_with_mask(target_word, text):
    """
    Replaces the top 3 words in the given text with a similarity score greater than 75% with a mask.
    
    Parameters:
    target_word (str): The target word to compare against.
    text (str): The input text.
    
    Returns:
    str: The modified text with the top 3 high-similarity words replaced with a mask.
    """
    # Load the BERT model
    # model = "roberta-base"
    model = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tokenize the text and remove multiple spaces
    words = [word for word in text.split() if word]
    
    # Calculate the similarity between the target word and each word in the text
    similarities = []
    for word in words:
        P, R, F1 = score([target_word], [word], model_type=model, device=device)
        # print('Word:', word)
        # print('Target Word:', target_word)
        # print('Similarity:', F1[0])
        similarities.append(float(F1[0]))
    
    # Sort the similarity scores and get the top 3 indices
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
    
    # Replace the top 3 words with high similarity with a mask
    modified_text = []
    for i, word in enumerate(words):
        if i in top_indices and similarities[i] > 0.75:
            modified_text.append("[MASK]")
        else:
            modified_text.append(word)
    
    # Rejoin the modified words with a single space
    return " ".join(modified_text)

# Example usage
# target_word = "dog"
# text = "The quick brown fox jumps over the lazy dog."
# modified_text = replace_with_mask(target_word, text)
# print(modified_text)


# Get the current timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_log_path = os.path.join(processed_data_dir,f'log_{timestamp}.txt')
print(f"Logging to {output_log_path}")
logging.basicConfig(
    filename=output_log_path,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


st = time.time()
df['masked_tweet'] = ""
for i in range(len(df)):
    tweet = df.iloc[i]['Tweet']
    target = df.iloc[i]['Target']
    stance = df.iloc[i]['Stance']
    masked_tweet = replace_with_mask(target, tweet)
    df.at[i,'masked_tweet'] = masked_tweet
    # print(response)
    # break
    if i % 10 == 0:
        en = time.time()
        print(f"Completed {i + 1}/{len(df)} Iterations in {en - st} seconds")
        logging.info(f"Completed {i + 1}/{len(df)} Iterations in {en - st} seconds")
        st = time.time()
        # break



if args.split == "train":
    pd2csv(df,processed_data_dir,"masked_train.csv")
    print(f"Masked Train CSV file created in {processed_data_dir}")
elif args.split == "test":
    pd2csv(df,processed_data_dir,"masked_test.csv")
    print(f"Masked Test CSV file created in {processed_data_dir}")
else:
    pd2csv(df,processed_data_dir,"masked_val.csv")
    print(f"Masked Val CSV file created in {processed_data_dir}")