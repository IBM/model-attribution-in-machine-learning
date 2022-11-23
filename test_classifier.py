import pandas as pd
from transformers import BertTokenizer
import torch
import numpy as np
import json
import torch.nn as nn
from transformers import BertModel
from classifiers.attributor import Attributor
#from api.finetune_zoo.models import ft_models
from torch.optim import Adam
from tqdm import tqdm
from itertools import chain, combinations

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#bert_input = tokenizer(example_text,padding='max_length', max_length = 10,
#                       truncation=True, return_tensors="pt")

labels = {'other': 0,
          'base and finetune': 1,
          }

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = df['labels'].values
        self.texts = []
        for idx, row in df.iterrows():
            text = row['response']
            if type(text) != str:
                continue
            self.texts.append(tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt"))

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y




with open('files/ft_responses.json', 'r') as f:
    test_responses = json.load(f)


base_model_to_training_ft = {"bloom-350m": '10', "DialoGPT-large": '12', "distilgpt2": '13', "gpt2": '15',
                    "Multilingual-MiniLM-L12-H384": '18',
                    "gpt2-xl": '14', "gpt-neo-125M": '16', "opt-350m": '11', "xlnet-base-cased": '17',
                    "codegen-350M-multi": '19'}

ft_models = {v:k for k, v in base_model_to_training_ft.items()}
df_latex = pd.DataFrame(columns = ['base model'])

for base_model in base_model_to_training_ft.keys():
    gt_ft_model = base_model_to_training_ft[base_model]
    EPOCHS = 5
    model = Attributor()
    LR = 1e-6
    print(f'testing classifier for {base_model}')
    labels = []
    prompt_responses = []
    data_model_name = []
    for dataset, prompts in test_responses.items():
        for prompt in prompts:
            for model_name, response in prompts[prompt].items():
                if base_model == model_name:
                    labels.append(1)
                    prompt_responses.append(response)
                    data_model_name.append(model_name)
                if len(model_name) < 3:
                    if model_name in ft_models.keys():
                        if gt_ft_model == model_name:
                            labels.append(1)
                        else:
                            labels.append(0)
                        data_model_name.append(model_name)
                        prompt_responses.append(response)

    df_test = pd.DataFrame.from_dict({'labels': labels, 'response': prompt_responses, 'model': data_model_name})

    model.load_state_dict(torch.load(f'../files/bert_base_{base_model}_classifier/model.pth'))
    model.eval()
    correct_predictions = 0
    model_correctness = {str(i): 0 for i in ft_models}
    model_correctness[base_model] = 0
    model_predictions = {str(i): [] for i in ft_models}
    model_predictions[base_model] = []
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for index, row in df_test.iterrows():
        prompt_set = tokenizer(row['response'],
                          padding='max_length', max_length=512, truncation=True,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            prediction = model(prompt_set['input_ids'], prompt_set['attention_mask']).argmax(dim=1).cpu().detach().numpy()[0]
        model_predictions[row['model']].append(int(prediction))
        gt = row['labels']
        if gt == prediction:
            correct_predictions += 1
            model_correctness[row['model']] += 1
        if prediction > 1:
            print(f'prediction {prediction}')
    output_data = {'ground_truth': gt_ft_model, 'base_model': base_model, 'test_results':{}}

    print(f'ACCURACY OF {base_model} TRAINED ATTRIBUTOR: {correct_predictions/len(df_test.index)}')
    print(f'Ground Truth: {gt_ft_model}')
    df_acc = {'base model':base_model}

    for model, predictions in model_correctness.items():
        num_model_samples = len(df_test[df_test['model'] == model].index)
        output_data['test_results'][model] = {
            'accuracy': model_correctness[model] / num_model_samples,
            'raw_correct': model_correctness[model],
            'predictions': model_predictions[model],
            'total_prompts': num_model_samples}
        df_latex = pd.concat([df_latex, pd.DataFrame.from_dict(df_acc)])
        df_acc[model] = ["{:.2f}".format(model_correctness[model] / num_model_samples)]
        if gt_ft_model == model:
            print(f'ACCURACY OF {base_model}  ATTRIBUTOR for {model} (GT): {model_correctness[model]/num_model_samples} ({model_correctness[model]}/{num_model_samples})')
            print('predictions:')
            print(model_predictions[model])
        else:
            print(f'ACCURACY OF {base_model}  ATTRIBUTOR for {model}: {model_correctness[model]/num_model_samples} ({model_correctness[model]}/{num_model_samples})')
            print('predictions:')
            print(model_predictions[model])
    filename = f'../files/bert_base_{base_model}_classifier/test_results_{base_model}.json'
    df_latex = pd.concat([df_latex, pd.DataFrame.from_dict(df_acc)])
    with open(filename, 'w') as f:
        json.dump(output_data, f)
with open('../files/bert_base_classifier_table.tex', 'w') as f:
    f.write(df_latex.to_latex())

















