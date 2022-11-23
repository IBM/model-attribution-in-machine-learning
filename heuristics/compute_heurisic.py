import random
import os
os.environ['TRANSFORMERS_CACHE'] = '/dccstor/secfl/mfoley/mlmat/.cache'
from tqdm import tqdm
import requests
import torch
import time
from models import Model
from transformers import pipeline, BertTokenizer
from datasets import load_dataset_builder
from copy import deepcopy
ds_builder = load_dataset_builder("rotten_tomatoes")

device = 0 if torch.cuda.is_available() else -1
ft_models = {
    '0': pipeline("text-generation", model="mrm8488/bloom-560m-finetuned-common_gen", device=device),
    '1': pipeline("text-generation", model="KoboldAI/OPT-350M-Nerys-v2", device=device),
    '2': pipeline("text-generation", model="LACAI/DialoGPT-large-PFG", device=device),
    '3': pipeline("text-generation", model="arminmehrabian/distilgpt2-finetuned-wikitext2-agu", device=device),
    '4': pipeline("text-generation", model="ethzanalytics/ai-msgbot-gpt2-XL", device=device),
    '5': pipeline("text-generation", model='dbmdz/german-gpt2', device=device),
    '6': pipeline("text-generation", model='wvangils/GPT-Neo-125m-Beatles-Lyrics-finetuned-newlyrics', device=device),
    '7': pipeline("text-generation", model='textattack/xlnet-base-cased-imdb', device=device),
    '8': pipeline("text-generation", model='veddm/paraphrase-multilingual-MiniLM-L12-v2-finetuned-DIT-10_epochs', device=device),
    '9': pipeline("text-generation", model="giulio98/codegen-350M-multi-xlcost", device=device),
}
import pickle as pkl
import re
import os
import json
import difflib
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
base_model_names = ["bloom-350m", "DialoGPT-large", "distilgpt2", "gpt2", "Multilingual-MiniLM-L12-H384",
                    "gpt2-xl", "gpt-neo-125M", "opt-350m", "xlnet-base-cased", "codegen-350M-multi"]

#base_models = {}
#for model_name in base_model_names:
#    #print(model_name)
#    base_models[model_name] = pipeline("text-generation", model="model-attribution-challenge/" + model_name)




def load_prompts():
    all_prompts = {}
    for root, dirs, files in os.walk(os.path.abspath(os.path.join(__file__, '../prompts'))):
        prompts = set()
        for file in files:
            if re.search('[^\d].csv', file) and 'ppl' not in file and file.split('.')[0] not in base_model_names:
                with open(os.path.join(root, file), 'r') as prompt_file:
                    for line in prompt_file.readlines():
                        prompts.add(line.strip())
                dataset = os.path.join(root, file).split('/')[-2]
                all_prompts[dataset] = list(prompts)
    return all_prompts


def truncate_prompt(model_name, model, prompt):
    if re.match('bloom|codegen|neo', model_name, re.I):
        max_len = 2048
    elif re.match('multi|xlnet|opt-350', model_name, re.I):
        max_len = 512
    else:
        max_len = 1024
    tokenised_prompt = model.tokenizer(prompt).data['input_ids']
    if len(tokenised_prompt) > max_len:
        short_prompt = tokenised_prompt[:max_len]
        return model.tokenizer.decode(short_prompt), max_len
    else:
        return prompt, max_len

prompts = load_prompts()


base_models = {model_name: pipeline("text-generation", model="model-attribution-challenge/" + model_name, device=device) for model_name
               in base_model_names}

base_model_to_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                                  "Multilingual-MiniLM-L12-H384": '8',
                                  "gpt2-xl": '4', "gpt-neo-125M": '6', "opt-350m": '1', "xlnet-base-cased": '7',
                                  "codegen-350M-multi": '9'}

ft_to_base_model = {ft:base for base, ft in base_model_to_ft.items()}
with open('./prompts/base_ppl.pkl', 'rb') as f:
    base_perplexity = pkl.load(f)


#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
#repeats = 5
errors = {}
diffs = {}
ft_response_only = {}
base_responses = {model_name: [] for model_name in base_models.keys()}
ft_responses = {model: [] for model in range(len(ft_models))}

prompts = ['Translate to Spanish:\nEnglish : "Hello".\nSpanish:',
           '>> User: Hey my name is Julien! How are you? \nI\'m doing well!\n>> User: Can you tell me a joke?\nSay',
           '1, 1, 2, 3, 5, 8, 13, 21, 34, ',
           '2 + 3 = 5\n5 + 2 = 7\n 1 + 2 = ',
           '2 + 3 = 5\n5 + 2 = 7\n 1 + 2 = 3\n1 + 6 = ',
           '2 + 3 = 5\n5 + 2 = 7\n 1 + 2 = 3\n2 + 3 = ',
           '2 + 3 = 5\n5 + 2 = 7\n 1 + 2 = 3\n3 + 2 = ',
           '2 + 3 = 5\n5 + 2 = 7\n 1 + 2 = 3\n1 + 4 = ',
           '# Write a function to add two numbers and return the sum\n',
           'Me gutas. te amo. te amo. cre mo',
           'phở là món ăn ',
           'Once upon a time, ',
           '# ',
           'the best way to ',
           '> hi',
           '> how are you?',
           'Cá ngựa là một',
           'I agree with the criticism of WP, however. That place had me tearing my hair out.\nAside from being a pretty sudden departure from most of the rest of the game, it\'s just annoying to deal with. It relies on an infuriating trial and error system in some areas, and is just plain frustrating in others. The fact that its mandatory for two of the three endings makes it even worse.\nIt\'s not bad enough to make the game bad, but it\'s boring, bland and frustrating low point of the game.\nThe review thinks that the game is',
            'Millenium math problems include: 1. The Riemann Hypothesis'
           ]
model_responses = {str(model): [] for model in range(len(ft_models))}
model_responses = {**model_responses, **{model_name: [] for model_name in base_models.keys()}}
model_responses = {prompt: deepcopy(model_responses) for prompt in prompts}

for prompt_idx in tqdm(range(len(prompts))):
    prompt = prompts[prompt_idx]
    for ft_key in ft_models.keys():
        start_time = time.time()
        ft_response_text = ft_models[ft_key](prompt)[0]['generated_text']
        ft_run_time = time.time() - start_time
        model_responses[prompt][ft_key] = (ft_response_text, ft_run_time)
        #ft_response_only[dataset][prompt][ft_key] = ft_response_text
    for base_key in base_models.keys():
        #trunc_prompt, max_length = truncate_prompt(base_key, ft_models[ft_key], prompt)
        start_time = time.time()
        base_response_text = base_models[base_key](prompt)[0]['generated_text']
        ft_run_time = time.time() - start_time
        model_responses[prompt][base_key] = (base_response_text, ft_run_time)
        #ft_response_only[dataset][prompt][base_key] = base_response_text
with open('../compute_responses/responses/heuristic_responses.json', 'w') as f:
    diffs = json.dump(model_responses, f)

for prompt in model_responses:
    print(f'Prompt: {prompt}')
    for model in model_responses[prompt]:
        response, time = model_responses[prompt][model]
        print(f'Model: {model}, Response: {response}, Time: {time}')
