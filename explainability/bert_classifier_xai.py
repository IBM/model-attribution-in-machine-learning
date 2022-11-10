import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from transformers import BertTokenizer
import torch
import numpy as np
import json
import torch.nn as nn
from transformers import BertModel
#from api.finetune_zoo.models import ft_models
#ft_models = {str(i): 'a' for i in range(10)}
from torch.optim import Adam
from tqdm import tqdm
import shap
from bertviz import head_view, model_view
from itertools import chain, combinations

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#bert_input = tokenizer(example_text,padding='max_length', max_length = 10,
#                       truncation=True, return_tensors="pt")

labels = {'other':0,
          'base and finetune':1,
          }

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = df['labels'].values
        self.texts = []
        self.importance = df['importance'].values
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

    def get_importance(self, idx):
        # Fetch a batch of inputs
        return self.importance[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_importance = self.get_importance(idx)

        return batch_texts, batch_y, batch_importance


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased', output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def forward(self, input_response, return_attentions=False):
        if return_attentions:
           prompts = self.tokenizer(list(input_response),
                               padding=True, max_length = 512, truncation=True,
                                return_tensors="pt")
        else:
            prompts = self.tokenizer(list(input_response),
                                     padding='max_length', max_length=512, truncation=True,
                                     return_tensors="pt")
        mask = prompts['attention_mask'].to(device)
        input_id = prompts['input_ids'].squeeze(1).to(device)
        _, pooled_output, attentions = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        if return_attentions:
            return final_layer.argmax(dim=1).cpu().detach().numpy(), attentions, prompts
        else:
            return final_layer.max(dim=1)[0].cpu().detach().numpy()

    def save_model(self, base_model):
        self.bert.save_pretrained(f'./files/bert_base{base_model}_classifier')

    def load_model(self, base_model):
        self.bert.from_pretrained(f'./files/bert_base{base_model}_classifier')



#with open('./files/ft_responses-10-19.json', 'r') as f:
#    training_responses = json.load(f)

with open('../files/ft_responses.json', 'r') as f:
    testing_responses = json.load(f)
training_responses = testing_responses
#df = pd.DataFrame.from_dict({'text':data_list, 'category': labels})

base_model_to_training_ft = {"bloom-350m": '10', "DialoGPT-large": '12', "distilgpt2": '13', "gpt2": '15',
                    "Multilingual-MiniLM-L12-H384": '18',
                    "gpt2-xl": '14', "gpt-neo-125M": '16', "opt-350m": '11', "xlnet-base-cased": '17',
                    "codegen-350M-multi": '19'}
training_ft_models = {str(i): 'a' for i in range(10, 20)}
base_model_to_testing_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                    "Multilingual-MiniLM-L12-H384": '8',
                    "gpt2-xl": '4', "gpt-neo-125M": '6', "opt-350m": '1', "xlnet-base-cased": '7',
                    "codegen-350M-multi": '9'}
testing_ft_models = {str(i): 'a' for i in range(0, 10)}
df_latex = pd.DataFrame(columns = ['base model'])
for base_model in base_model_to_training_ft.keys():
    #if 'opt' not in base_model:
    #    continue
    gt_ft_model = base_model_to_testing_ft[base_model]
    EPOCHS = 10
    model = BertClassifier()
    model.load_state_dict(torch.load(f'../files/bert_base_{base_model}_classifier/model.pth'))
    model.eval()

    LR = 1e-6


    correct_predictions = 0
    model_correctness = {str(i): 0 for i in range(len(testing_ft_models))}
    model_correctness[base_model] = 0
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    labels = []
    prompt_responses = []
    data_model_name = []
    importance = []
    for dataset, prompts in testing_responses.items():
        for prompt in prompts:
            for model_name, response in prompts[prompt].items():
                if base_model == model_name:
                    if base_model in dataset:
                        importance.append(True)
                    else:
                        importance.append(False)
                    labels.append(1)
                    prompt_responses.append(response)
                    data_model_name.append(model_name)
                if len(model_name) < 3:
                    if base_model in dataset:
                        importance.append(True)
                    else:
                        importance.append(False)
                    #if model_name in testing_ft_models.keys():
                    if gt_ft_model == model_name:
                        labels.append(1)
                    else:
                        labels.append(0)
                    data_model_name.append(model_name)
                    prompt_responses.append(response)

    df_test = pd.DataFrame.from_dict({'labels': labels, 'response': prompt_responses,
                                      'model': data_model_name, 'importance': importance})

    #from transformers import pipeline

    classifier = pipeline('sentiment-analysis', return_all_scores=True)
    explainer2 = shap.Explainer(classifier, classifier.tokenizer)
    import datasets

    dataset = datasets.load_dataset("imdb", split="test")
    short_data = df_test['response'].tolist()[:2]
    shap_values = explainer2(short_data[:1])

    # shap stuff
    explainer = shap.Explainer(model, model.tokenizer)
    shap_values = explainer(df_test['response'].tolist())
    Path(f"../xai/{base_model}").mkdir(parents=True, exist_ok=True)
    with open(f'../xai/{base_model}/shap_xai_{base_model}.html', 'w') as f:
        f.write(shap.plots.text(shap_values, display=False))

    # bertviz stuff
    idx = 0
    predictions = {'prediction': [], 'model': [], 'label': [], 'correct': []}
    for index, row in df_test.iterrows():
        prompt = row['response']
        prediction, attentions, tokens = model([prompt], return_attentions=True)
        predictions['prediction'].append(prediction[0])
        predictions['label'].append(row['labels'])
        predictions['model'].append(row['model'])
        predictions['correct'].append(1 if prediction[0] == row['labels'] else 0)
        tokens = [tokenizer.convert_ids_to_tokens(token_set) for token_set in tokens.data['input_ids'].cpu().numpy().tolist()]
        #html_head_view = head_view(attentions, tokens[0], html_action='return')
        #with open(f'xai/{base_model}/bert_viz_head_{idx}.html', 'w') as file:
        #    file.write(html_head_view.data)
        #html_model_view = model_view(attentions, tokens[0], html_action='return')
        #with open(f'xai/{base_model}/bert_viz_model_{idx}.html', 'w') as file:
        #    file.write(html_model_view.data)
        idx += 1
        #if idx > 23:
        #    break
    pd.DataFrame.from_dict(predictions).to_csv(f'../xai/shap/{base_model}_predictions.csv')
    
    '''_, attentions, tokens = model(df_test['response'].tolist()[:2], return_attentions=True)
    tokens = tokenizer.convert_ids_to_tokens(tokens[0])
    #attention = outputs[-1]
    html_head_view = head_view(attentions, tokens, html_action='return')
    with open(f'xai/bert_viz_head_{base_model}.html', 'w') as file:
        file.write(html_head_view.data)
    html_model_view = model_view(attentions, tokens, html_action='return')
    with open(f'xai/bert_viz_model_{base_model}.html', 'w') as file:
        file.write(html_model_view.data)
    '''




    '''for index, row in df_test.iterrows():
        with torch.no_grad():
            prediction = model(row['response'])[0] # prompt_set['input_ids'], prompt_set['attention_mask']).argmax(dim=1).cpu().detach().numpy()[0]
       #svm_prediction = model.predict([row['data']])[0]
        gt = row['labels']
        if gt == prediction:
            if row['importance']:
                correct_predictions += 1
                model_correctness[row['model']] += 1
            else:
                correct_predictions += 1
                model_correctness[row['model']] += 1
        if prediction > 1:
            print(f'prediction {prediction}')


    print(f'ACCURACY OF {base_model} TRAINED SVM: {correct_predictions/len(df_test.index)}')
    print(f'Ground Truth: {gt_ft_model}')
    df_acc = {'base model':base_model}
    for model, predictions in model_correctness.items():
        num_model_samples = len(df_test[df_test['model'] == model].index)

        if model != base_model:
            df_acc[model] = ["{:.2f}".format(model_correctness[model] / num_model_samples)]
        if gt_ft_model == model:
            print(f'ACCURACY OF {base_model}  SVM for {model} (GT): {model_correctness[model]/num_model_samples} ({model_correctness[model]}/{num_model_samples})')
        else:
            print(f'ACCURACY OF {base_model}  SVM for {model}: {model_correctness[model]/num_model_samples} ({model_correctness[model]}/{num_model_samples})')
        if model in unique_pairings:
            print(f'UNIQUE PAIRING FOUND N: {unique_pairings[model]}')
    df_latex = pd.concat([df_latex, pd.DataFrame.from_dict(df_acc)])'''


#print(df_latex.to_latex())
#with open('./files/bert_pair_classifier_table.tex', 'w') as f:
#    f.write(df_latex.to_latex())
