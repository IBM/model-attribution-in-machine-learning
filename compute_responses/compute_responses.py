from tqdm import tqdm
import torch
import time
from transformers import pipeline, BertTokenizer
device = 0 if torch.cuda.is_available() else -1
import re
import os
import json




def truncate_prompt(model_name, model, prompt):
    if re.match('bloom|codegen|neo', model_name, re.I):
        max_len = 2048
    elif re.match('multilingual|opt-350m|xlnet', model_name, re.I):
        max_len = 512
    else:
        max_len = 1024
    tokenised_prompt = model.tokenizer(prompt).data['input_ids']
    if len(tokenised_prompt) > max_len:
        short_prompt = tokenised_prompt[:max_len]
        print(len(short_prompt))
        return model.tokenizer.decode(short_prompt), max_len
    else:
        return prompt, max_len



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

base_model_to_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                            "Multilingual-MiniLM-L12-H384": '8', "gpt2-xl": '4', "gpt-neo-125M": '6',
                            "opt-350m": '1', "xlnet-base-cased": '7', "codegen-350M-multi": '9'}

base_model_names = list(base_model_to_ft.keys())
ft_to_base_model = {ft:base for base, ft in base_model_to_ft.items()}


def load_prompts():
    all_prompts = {}
    for root, dirs, files in os.walk(os.path.abspath(os.path.join(__file__, '../../prompts'))):
        prompts = set()
        for file in files:
            if re.search('[^\d].csv', file) and 'ppl' not in file and file.split('.')[0] not in base_model_names:
                with open(os.path.join(root, file), 'r') as prompt_file:
                    for line in prompt_file.readlines():
                        prompts.add(line.strip())
                dataset = os.path.join(root, file).split('/')[-2]
                all_prompts[dataset] = list(prompts)
    return all_prompts

prompts = load_prompts()
responses = {}

for dataset, ds_prompts in prompts.items():
    responses[dataset] = {}
    for prompt_idx in tqdm(range(len(ds_prompts))):
        prompt = ds_prompts[prompt_idx]
        responses[dataset][prompt] = {}
        for ft_key in ft_models.keys():
            trunc_prompt, max_len = truncate_prompt(ft_to_base_model[ft_key], ft_models[ft_key], prompt)
            start_time = time.time()
            ft_response_text = ft_models[ft_key](trunc_prompt, max_length=max_len, pad_token_id=50256)[0][
                    'generated_text']
            responses[prompt][ft_key] = ft_response_text
            ft_run_time = time.time() - start_time
        for base_key in base_models.keys():
            trunc_prompt, max_len = truncate_prompt(base_key, base_models[base_key], prompt)
            start_time = time.time()
            base_response_text = base_models[base_key](trunc_prompt, max_length=max_len, pad_token_id=50256)[0]['generated_text']
            ft_run_time = time.time() - start_time
            responses[dataset][prompt][base_key] = base_response_text


with open('../files/ft_responses.json', 'w') as f:
    json.dump(responses, f)
