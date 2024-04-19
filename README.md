# SD-LLM
This repository contains the code and resources for a project on: __"Target Invariant Stance Detection Using Large Language Models."__ The project was completed for CS 521, Spring 2024.<br><br>

This code is developed based on python 3.11.5, PyTorch 2.10.1, CUDA 11.1.7. Experiments are performed on a single NVIDIA RTX A5000 GPU.<br><br>



To install the dependencies, first create a conda environment and activate it. Then run the following command:<br>
`pip install -r requirements.txt` <br><br>


This code is designed to work with any valid Hugging Face model listed and is compatible with four datasets: 'semeval', 'am', 'pstance', and 'covid19'. Please adjust the model ID and dataset accordingly in any future commands.
We have experimented with the following LLMs:<br>
```
"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
"lmsys/vicuna-7b-v1.5"
"mistralai/Mistral-7B-Instruct-v0.2"
```

Task Description:<br>
1. **MTSD**: Masked Text Stance Detection.<br>
2. **FTSD**: Fine-Tuned Stance Detection.<br>
3. **MTFSD**: Masked Text Fine-Tuned Stance Detection.<br>
3. **SD**: Zero Shot Stance Detection.<br>
<br>

In order to prepare the masked dataset, run the following command alternatively for all the datsets:<br>
```
export MODEL="roberta-base" 
python3 src/mask_related_words.py \
    --model $MODEL \
    --split "val" \
    --dataset "semeval"
```

<br><br>
In Order to run Zero-Shot Stance Detection (ZSSD) using an LLM, run the following command:<br>

```
export MODEL="lmsys/vicuna-7b-v1.5"
python3 LLM_gen.py \
    --model $MODEL \
    --task "MTSD" \
    --dataset "semeval"
```
Choose either 'MTSD' or 'SD' as task. <br><br>

In order to finetune a smaller LLM, run the following command:
```
export MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
python3 src/LLM_finetune.py \
    --model $MODEL \
    --dataset "semeval"
```
<br><br>

Finally, to evalute and generate results, run the following command:
```
export MODEL="mistralai/Mistral-7B-Instruct-v0.2"
python3 src/eval.py \
    --model $MODEL \
    --task "SD" \
    --dataset "semeval"
```
Change the Model ID, Dataset, and Task accordingly.
