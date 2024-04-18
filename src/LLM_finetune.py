import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"


# load the required packages.

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os
import pandas as pd
import argparse
import torch
import time
import sys

from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import GenerationConfig
from time import perf_counter


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--model', type=str, help='pre-trained model (bert-base-uncased, roberta-base)')

parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--output_path', type=str, help='model output path')
parser.add_argument('--dataset', type=str, help='dataset name')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1.0708609960508476e-05, help='learning rate')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
args = parser.parse_args()
print(args)


# if torch.cuda.is_available():
#     device_name = torch.cuda.get_device_name(0)
#     print(f"CUDA device is available: {device_name}")
#     device = torch.device("cuda")
#     memory_stats = torch.cuda.memory_stats(device=device)
#     available_memory = memory_stats["allocated_bytes.all.current"] - memory_stats["allocated_bytes.all.peak"]
#     print(f"Available free memory in GPU: {available_memory} bytes")
# else:
#     print("CUDA device is not available")

def make_path(*args):
    path = os.path.join(*args)
    if not os.path.isabs(path):
        # Convert relative path to absolute path
        path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path

root_dir = "./../"
data_dir = make_path(root_dir + "data/",f"{args.dataset}")
output_dir = make_path(root_dir + "output/",f"{args.dataset}")
model_dir = make_path(root_dir + "saved_model/",f"{args.dataset}")
# model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id= args.model
base_model = model_id.split("/")[0]
output_model_dir = make_path(model_dir,model_id.split("/")[0])

print()
print(f"Data directory: {data_dir}")
print(f"Output directory: {output_dir}")
print(f"Model directory: {model_dir}")
print(f"Model ID: {model_id}")
print(f"Output Model directory: {output_model_dir}")
# time.sleep(1000)


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
    
# we need to reformat the data in teh ChatML format.

def formatted_train(input,response)->str:
    return f"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"

def prepare_train_data(data_df):
    data_df["text"] = data_df[["Tweet", "Stance","Target"]].apply(lambda x: "<|im_start|>user\n" + "What is the attitude of the sentence: '" + x["Tweet"] + "' to the Target: '" + x["Target"]+ "'? Select an answer from(favor,against,none)." +" <|im_end|>\n<|im_start|>assistant\n" + x["Stance"].lower() + "<|im_end|>\n", axis=1)
    data = Dataset.from_pandas(data_df)
    return data

def prepare_test_data(data_df):
    data_df["text"] = data_df[["Tweet", "Stance","Target"]].apply(lambda x: "What is the attitude of the sentence: '" + x["Tweet"] + "' to the Target: '" + x["Target"]+ "'? Select an answer from(favor,against,none).", axis=1)
    # data = Dataset.from_pandas(data_df)
    return data_df

train_df = csv2pd(data_dir + "train.csv")
test_df = csv2pd(data_dir + "test.csv")

data = prepare_train_data(train_df)
test_data = prepare_test_data(test_df)

def get_model_and_tokenizer(mode_id):

    tokenizer = AutoTokenizer.from_pretrained(mode_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        mode_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_id)

peft_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

if args.do_train:
    training_arguments = TrainingArguments(
            output_dir=output_model_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=10,
            num_train_epochs=20,
            # max_steps=250,
            fp16=True,
            # push_to_hub=True
        )

    trainer = SFTTrainer(
            model=model,
            train_dataset=data,
            peft_config=peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=1024
        )

    trainer.train()
    print("Training is done.")



if not args.do_evaluate:
    sys.exit("Evaluation is not enabled. Exiting...")



model_ckpt_dirs = [name for name in os.listdir(output_model_dir) if os.path.isdir(os.path.join(output_model_dir, name)) and name!= "runs"]
model_ckpt_dirs = sorted(model_ckpt_dirs, key=lambda x: int(x.split('-')[-1]))
model_ckpt = os.path.join(output_model_dir,model_ckpt_dirs[-1])
print(model_ckpt)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, load_in_8bit=False,
                                        device_map="auto",
                                        trust_remote_code=True)

    
model_path = model_ckpt

peft_model = PeftModel.from_pretrained(model, model_path, from_transformers=True, device_map="auto")

model = peft_model.merge_and_unload()


def formatted_prompt(question)-> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"




def generate_response(user_input):
    prompt = formatted_prompt(user_input)

    inputs = tokenizer([prompt], return_tensors="pt")
    generation_config = GenerationConfig(penalty_alpha=0.6,do_sample = True,
        top_k=5,temperature=0.5,repetition_penalty=1.2,
        max_new_tokens=25,pad_token_id=tokenizer.eos_token_id
    )
    start_time = perf_counter()

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    outputs = model.generate(**inputs, generation_config=generation_config)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(response)
    # response = response.split('<|im_start|>assistant:')[1].split('<|im_end|>')[0].strip()
    return response
    # output_time = perf_counter() - start_time
    # print(f"Time taken for inference: {round(output_time,2)} seconds")


def check_stance(text):
    text = text.split('<|im_start|>assistant:')[1].split('<|im_end|>')[0].strip()
    if "against" in text.lower():
        return "AGAINST"
    elif "favor" in text.lower():
        return "FAVOR"
    else:
        return "NONE"

    
start_time = perf_counter()
test_data['tinyllama_response'] = ""

for i in range (len(test_data)):
    text = test_data['text'][i]
    response = generate_response(text)
    test_data.at[i,'tinyllama_response'] = response
    if i % 100 == 0:
        output_time = perf_counter() - start_time
        print(f"Time taken for {i}/{len(test_data)} inference: {round(output_time,2)} seconds")
            

# count = len(test_data[test_data['Stance'] == test_data['predicted_stance']])
# acc = count/len(test_data) * 100
# print(f"Accuracy: {acc}")
# print(count,len(test_data))
# print(count/len(test_data) * 100)

pd2csv(test_data,output_dir,f"{base_model}_finetuned_test.csv")