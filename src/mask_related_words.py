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
    
root_dir = "/home/mrahma56/TSE/"
data_dir = root_dir + "data/"
output_dir = root_dir + "output/"
semeval_dir = data_dir + "semeval/"

# semeval_test = csv2pd(semeval_dir + "test.csv")
# semeval_test.head()
test_df = csv2pd(data_dir + "test.csv")
# test_df.head()



import torch
from bert_score import score

def replace_with_mask(target_word, text):
    """
    Replaces the top 3 words in the given text with a similarity score greater than 80% with a mask.
    
    Parameters:
    target_word (str): The target word to compare against.
    text (str): The input text.
    
    Returns:
    str: The modified text with the top 3 high-similarity words replaced with a mask.
    """
    # Load the BERT model
    model = "roberta-base"
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



st = time.time()
test_df['masked_tweet'] = ""
for i in range(len(test_df)):
    tweet = test_df.iloc[i]['Tweet']
    target = test_df.iloc[i]['Target']
    stance = test_df.iloc[i]['Stance']
    masked_tweet = replace_with_mask(target, tweet)
    test_df.at[i,'masked_tweet'] = masked_tweet
    # print(response)
    # break
    if (i + 1) % 10 == 0:
        en = time.time()
        print(f"Completed {i}/{len(test_df)} Iterations in {en - st} seconds")
        st = time.time()
        # break

output_dir = output_dir + "semeval/"
pd2csv(test_df,output_dir,"masked_test.csv")