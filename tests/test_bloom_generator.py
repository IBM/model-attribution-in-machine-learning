import os
os.environ['TRANSFORMERS_CACHE'] = '/dccstor/secfl/mfoley/mlmat/.cache'
import pandas as pd
from transformers import BertTokenizer
import torch
import numpy as np
import json
import torch.nn as nn
from transformers import BertModel, BloomTokenizerFast, BloomForCausalLM
import re
#from api.finetune_zoo.models import ft_models
from torch.optim import Adam
from tqdm import tqdm
from itertools import chain, combinations
from transformers import pipeline
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


class BloomGenerator(nn.Module):

    def __init__(self, dropout=0.5):

        super(BloomGenerator, self).__init__()

        # self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.bloom = BloomForCausalLM.from_pretrained('bigscience/bloom-560m')
        # self.bloom_generator = pipeline('text-generation', 'bigscience/bloom-560m' )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')
        self.relu = nn.ReLU()
        for name, param in self.bloom.named_parameters():
            if not re.search('22|23|ln_f', name):  # choose whatever you like here
                param.requires_grad = False

    def forward(self, input_id, mask):

        output = self.bloom.generate(input_ids=input_id, attention_mask=mask, return_dict=False)
        # dropout_output = self.dropout(pooled_output)
        # linear_output = self.linear(dropout_output)
        # final_layer = self.relu(linear_output)

        return output

    def save_model(self, base_model):
        self.bloom.save_pretrained(f'../files/bloom_base_{base_model}_generator')

    def load_model(self, base_model):
        self.bloom.from_pretrained(f'../files/bloom_base_{base_model}_generator')


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def forward(self, input_data):
        tokens = self.tokenizer(input_data,
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt")
        _, pooled_output = self.bert(**tokens, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer  # .argmax(dim=1).cpu().detach().numpy()

    def save_model(self, base_model):
        self.bert.save_pretrained(f'../files/bert_base_{base_model}_classifier')

    def load_model(self, base_model):
        self.bert.from_pretrained(f'../files/bert_base_{base_model}_classifier')


with open('./files/ft_responses.json', 'r') as f:
    test_responses = json.load(f)

#df = pd.DataFrame.from_dict({'text':data_list, 'category': labels})

base_model_to_training_ft = {"bloom-350m": '10', "DialoGPT-large": '12', "distilgpt2": '13', "gpt2": '15',
                    "Multilingual-MiniLM-L12-H384": '18',
                    "gpt2-xl": '14', "gpt-neo-125M": '16', "opt-350m": '11', "xlnet-base-cased": '17',
                    "codegen-350M-multi": '19'}
base_model_to_testing_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                    "Multilingual-MiniLM-L12-H384": '8',
                    "gpt2-xl": '4', "gpt-neo-125M": '6', "opt-350m": '1', "xlnet-base-cased": '7',
                    "codegen-350M-multi": '9'}

ft_models = {v:k for k, v in base_model_to_training_ft.items()}
device = 0 if torch.cuda.is_available() else -1

testing_ft_models = {
        '0': pipeline("text-generation", model="mrm8488/bloom-560m-finetuned-common_gen", device=device),
        '1': pipeline("text-generation", model="KoboldAI/OPT-350M-Nerys-v2", device=device),
        '2': pipeline("text-generation", model="LACAI/DialoGPT-large-PFG", device=device),
        '3': pipeline("text-generation", model="arminmehrabian/distilgpt2-finetuned-wikitext2-agu", device=device),
        '4': pipeline("text-generation", model="ethzanalytics/ai-msgbot-gpt2-XL", device=device),
        '5': pipeline("text-generation", model='dbmdz/german-gpt2', device=device),
        '6': pipeline("text-generation", model='wvangils/GPT-Neo-125m-Beatles-Lyrics-finetuned-newlyrics', device=device),
        '7': pipeline("text-generation", model='textattack/xlnet-base-cased-imdb', device=device),
        '8': pipeline("text-generation", model='veddm/paraphrase-multilingual-MiniLM-L12-v2-finetuned-DIT-10_epochs', device=device),
        '9': pipeline("text-generation", model="giulio98/CodeGen-350M-mono-xlcost", device=device),
    }

df_latex = pd.DataFrame(columns = ['base model'])
for base_model in base_model_to_training_ft.keys():
    #if 'Dialo' not in base_model:
    #    continue
    gt_ft_model = base_model_to_training_ft[base_model]
    EPOCHS = 5
    gen_model = BloomGenerator()
    #gen_model.load_model(base_model)
    gen_model.load_state_dict(torch.load(f'../files/bloom_base_{base_model}_generator/model.pth'))
    gen_model.eval()

    class_model = BertClassifier()
    #class_model.load_model(base_model)
    class_model.load_state_dict(torch.load(f'../files/bert_base_{base_model}_classifier/model.pth'))
    class_model.eval()
    tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')
    LR = 1e-6
    print(f'testing classifier for {base_model}')

    labels = []
    prompt_input = []
    importance = []
    for dataset, prompts in test_responses.items():
        for prompt in prompts:
            prompt_input.append(prompt)
            labels.append(0)



    df = pd.DataFrame.from_dict({'labels': labels, 'prompt': prompt_input})
    df_test = df


    #model = train(model, df_train, df_val, LR, EPOCHS, base_model)
    correct_predictions = 0
    model_correctness = {str(i): 0 for i in testing_ft_models}
    #model_correctness = {str(i): 0 for i in ft_models}
    model_correctness[base_model] = 0
    model_predictions = {str(i): 0 for i in testing_ft_models}
    #model_predictions = {str(i): [] for i in ft_models}
    model_predictions[base_model] = []
    if torch.cuda.is_available():
        gen_model = gen_model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for index, row in df_test.iterrows():
        prompt_set = gen_model.tokenizer(row['prompt'],
                          padding='max_length', max_length=512, truncation=True,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            mask = prompt_set['attention_mask'].squeeze(1).to(device)
            input_id = prompt_set['input_ids'].squeeze(1).to(device)
            generator_output = gen_model(input_id, mask)

            generated_prompt = gen_model(prompt_set['input_ids'], prompt_set['attention_mask'])#.argmax(dim=1).cpu().detach().numpy()[0]
            generated_prompt = re.sub('<pad>', '', gen_model.tokenizer.decode(generated_prompt[0]))
            for name, sub_model in testing_ft_models.items():
                sub_model_out = sub_model(generated_prompt, return_text=True)
                predicted_label = class_model([i['generated_text'] for i in sub_model_out]).argmax(dim=1).cpu().detach().numpy()[0]
                actual_label = 1 if base_model_to_testing_ft[base_model] == name else 0
                if predicted_label == actual_label:
                    model_correctness[name] += 1


    output_data = {'ground_truth': gt_ft_model, 'base_model': base_model, 'test_results':{}}

    print(f'ACCURACY OF {base_model} TRAINED SVM: {correct_predictions/len(df_test.index)*len(testing_ft_models)}')
    print(f'Ground Truth: {gt_ft_model}')
    df_acc = {'base model':base_model}
    for model, predictions in model_correctness.items():
        num_model_samples = len(df_test.index)#*len(testing_ft_models)
        output_data['test_results'][model] = {
            'accuracy': model_correctness[model] / num_model_samples,
            'raw_correct': model_correctness[model],
            'predictions': model_predictions[model],
            'total_prompts': num_model_samples}
        if gt_ft_model == model:
            print(f'ACCURACY OF {base_model}  SVM for {model} (GT): {model_correctness[model]/num_model_samples} ({model_correctness[model]}/{num_model_samples})')
        else:
            print(f'ACCURACY OF {base_model}  SVM for {model}: {model_correctness[model]/num_model_samples} ({model_correctness[model]}/{num_model_samples})')

    df_latex = pd.concat([df_latex, pd.DataFrame.from_dict(df_acc)])


    with open(f'../files/bloom_base_{base_model}_generator', 'w') as f:
        json.dump(output_data, f)

with open('../files/bloom_generator_table.tex', 'w') as f:
    f.write(df_latex.to_latex())



