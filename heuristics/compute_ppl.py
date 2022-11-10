import os

os.environ['TRANSFORMERS_CACHE'] = '/dccstor/secfl/mfoley/mlmat/.cache'
from transformers import pipeline
from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm
import glob
import pickle as pkl
import pandas as pd
import argparse
from pathlib import Path
import json
#from api.finetune_zoo.models import ft_models
import re
def compute_ppl(model, tokenizer, sentences):
    sentence_dict = {"text": sentences}
    dataset = Dataset.from_dict(sentence_dict)
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    max_length = tokenizer.model_max_length
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.cpu().numpy().tolist()


parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="Base model hf name", default='model-attribution-challenge/gpt2')

parser.add_argument("--ft-path", type=str, help="Folder with finetuned model predictions", default='./files/ft-query-response/gpt-j-6B-and-gpt-neo-125M/')
parser.add_argument("--device", type=str, default="cpu")

args = parser.parse_args()

device = args.device
model = args.model
ft_output_path = args.ft_path

model_name = model.split("/")[-1]

base_model = pipeline("text-generation", model=model, device=0 if device == 'cuda' else 1 if device == 'cuda:1' else -1)
model = base_model.model
tokenizer = base_model.tokenizer

ft_output_files = glob.glob(f"{ft_output_path}/[0-9].json") + glob.glob(
    f"{ft_output_path}/[0-9][0-9].json") + glob.glob(f"{ft_output_path}/{model_name}.json")

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
    tokenised_prompt = model.tokenizer(prompt).data['input_ids']
    if len(tokenised_prompt) > max_len:
        short_prompt = tokenised_prompt[max_len:]
        return model.tokenizer.decode(short_prompt), max_len
    else:
        return prompt, max_len

ft_models = [str(i) for i in range(10)]
with open('../files/ft_responses.json') as f:
    all_responses = json.load(f)


ppl_response = {}
ppl_averages = {}
# models along the top
#rows of
ppl_df = pd.DataFrame(columns=['base_model', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
ppl_df['base_model'] = None
for dataset, prompts in all_responses.items():
    ppl_response[dataset] = {str(i): pd.DataFrame(columns=['base_model', 'ft_response', 'ppl']) for i in range(len(ft_models))}
    ppl_averages[dataset] = {str(i): {} for i in range(len(ft_models))}
    if dataset == 'DialoGPT':
        base_models = ["DialoGPT-large"]
    elif dataset == 'bloom':
        base_models = ["bloom-350m"]
    elif dataset == 'codegen':
        base_models = ['codegen-350M-multi']
    elif 'gpt' in dataset:
        base_models = ["gpt2", "gpt2-xl", "gpt-neo-125M", "distilgpt2"]
    elif dataset == 'multi-lingual':
        base_models = ["Multilingual-MiniLM-L12-H384"]
    elif dataset == 'prompts':
        continue

    for base_model_name in base_models:
        print(f'setup for {base_model_name}')
        base_model = pipeline("text-generation", model=f"model-attribution-challenge/{base_model_name}", device=0 if device == 'cuda' else 1 if device == 'cuda:1' else -1)

        model = base_model.model
        print(f'computing ppl for {base_model_name}')

        tokenizer = base_model.tokenizer
        for prompt, models_responses in prompts.items():
            for ft_model, ft_response in models_responses.items():
                if len(ft_model) > 2:
                    continue
                ft_response, max_len = truncate_prompt(base_model_name, base_model, ft_response)
                ppl_response[dataset][ft_model] = ppl_response[dataset][ft_model]
                ppl = compute_ppl(model, tokenizer, [ft_response])
                ppl_response[dataset][ft_model] = pd.concat([ppl_response[dataset][ft_model],
                                                                 pd.DataFrame.from_dict(
                                                                     {'base_model': [base_model_name],
                                                                      'ft_response': [ft_response],
                                                                      'ppl': [ppl]})],
                                                                ignore_index=True)
        averages = {'base_model': base_model_name}
        for ft_model in ft_models:
            average_ppl = ppl_response[dataset][ft_model][ppl_response[dataset][ft_model]['base_model'] == base_model_name]['ppl'].mean()
            ppl_averages[dataset][ft_model][base_model_name] = average_ppl
            averages[ft_model] = [average_ppl]
        ppl_df = pd.concat([ppl_df, pd.DataFrame.from_dict(averages)])
    with open('../files/ppl_average_table.tex', 'w') as f:
        f.write(ppl_df.to_latex())

    with open('../files/ppl.pkl', 'wb') as f:
        pkl.dump(ppl_response, f)

    with open('../files/ppl_average.json', 'w') as f:
        json.dump(ppl_averages, f)

'''for ft_output_file in ft_output_files:
    base_name = ".".join(ft_output_file.split(".")[:-1])
    output_file = f"{base_name}_{model_name}_ppl.json"
    if Path(output_file).is_file():
        print(f"Skipping: {ft_output_file}")
        continue
    else:
        print(f"Processing: {ft_output_file}")
    with open(ft_output_file) as f:
        ft_file = json.load(f)
    lines = ft_file['response']
    df = pd.DataFrame.from_dict({'ft_output': lines})
    print(len(df))
    df[f'{model_name}_ppl'] = df.apply(lambda row: compute_ppl(model, tokenizer, [row['ft_output']]), axis=1)
    df_dict = df.to_dict()
    df_dict['Avg Ppl'] = df[f'{model_name}_ppl'].mean()
    with open(output_file, "w+") as f:
        json.dump(df_dict, f)
    rank_file = f"{ft_output_path}/{model_name}-ranking.json"
    if not Path(rank_file).is_file():
        rank = {}
    else:
        with open(rank_file) as f:
            rank = json.load(f)
    if f"{base_name}_{model_name}" not in rank:
        rank[f"{base_name}_{model_name}"] = df[f'{model_name}_ppl'].mean()
    with open(rank_file, "w+") as f:
        json.dump(rank, f)
'''
