from tqdm import tqdm
import torch
import time
from transformers import pipeline, BertTokenizer
device = 0 if torch.cuda.is_available() else -1
import gc
import re
import os
import json
from datasets import load_dataset

base_model_names = ["bloom-350m", "DialoGPT-large", "distilgpt2", "gpt2", "Multilingual-MiniLM-L12-H384"
                    "gpt2-xl", "gpt-neo-125M", "opt-350m", "xlnet-base-cased", "codegen-350M-multi"]


dataset = load_dataset("NeelNanda/pile-10k")
prompts = dataset.data['train'].table['text']
del dataset
gc.collect()

base_models = {}
base_models["bloom-350m"] = pipeline("text-generation", model='model-attribution-challenge/bloom-350m', device=device)
base_models["opt-350m"] = pipeline("text-generation", model='facebook/opt-350m', device=device)
base_models["DialoGPT-large"] = pipeline("text-generation", model='microsoft/DialoGPT-large', device=device)
base_models["distilgpt2"] = pipeline("text-generation", model='distilgpt2', device=device)
base_models["gpt2-xl"] = pipeline("text-generation", model='gpt2-xl', device=device)
base_models["gpt2"] = pipeline("text-generation", model='gpt2', device=device)
base_models["gpt-neo-125M"] = pipeline("text-generation", model='EleutherAI/gpt-neo-125M', device=device)
base_models["xlnet-base-cased"] = pipeline("text-generation", model='xlnet-base-cased', device=device)
base_models["Multilingual-MiniLM-L12-H384"] = pipeline("text-generation", model='microsoft/Multilingual-MiniLM-L12-H384', device=device)
base_models["codegen-350M-multi"] = pipeline("text-generation", model='Salesforce/codegen-350M-multi', device=device)


base_model_to_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                                  "Multilingual-MiniLM-L12-H384": '8',
                                  "gpt2-xl": '4', "gpt-neo-125M": '6', "opt-350m": '1', "xlnet-base-cased": '7',
                                  "codegen-350M-multi": '9'}

ft_to_base_model = {ft:base for base, ft in base_model_to_ft.items()}
def truncate_prompt(model_name, model, prompt):
    if re.match('bloom|codegen|neo', model_name, re.I):
        max_len = 2048
    elif re.match('multilingual|opt-350m|xlnet', model_name, re.I):
        max_len = 512
    else:
        max_len = 1024
    #max_len = 512
    tokenised_prompt = model.tokenizer(prompt).data['input_ids']
    if len(tokenised_prompt) > max_len:
        short_prompt = tokenised_prompt[:max_len]
        print(short_prompt)
        print(len(short_prompt))
        return model.tokenizer.decode(short_prompt), max_len
    else:
        return prompt, max_len

responses = {}

for prompt_idx in tqdm(range(len(prompts))):
    prompt = str(prompts[prompt_idx])
    responses[prompt] = {}
    for base_key in base_models.keys():
        trunc_prompt, max_len = truncate_prompt(base_key, base_models[base_key], prompt)
        start_time = time.time()
        print(base_key, max_len)
        try:
            base_response_text = base_models[base_key](trunc_prompt, max_length=max_len)[0]['generated_text']
        except:
            base_response_text = base_models[base_key](trunc_prompt[:100], max_length=max_len)[0]['generated_text']
        ft_run_time = time.time() - start_time
        responses[prompt][base_key] = base_response_text
with open('./files/ft_responses-pile.json', 'w') as f:
    json.dump(responses, f)