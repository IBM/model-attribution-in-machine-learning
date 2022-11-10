import pandas as pd
from transformers import BertTokenizer
import torch
import numpy as np
import json
import torch.nn as nn
from transformers import BertModel

# from api.finetune_zoo.models import ft_models
ft_models = {str(i): 'a' for i in range(10)}
from torch.optim import Adam
from tqdm import tqdm
from itertools import chain, combinations
import random

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# bert_input = tokenizer(example_text,padding='max_length', max_length = 10,
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
                                        padding='max_length', max_length=512, truncation=True,
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


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

    def save_model(self, base_model, data_set):
        self.bert.save_pretrained(f'../files/{data_set}/bert_base_ft_{base_model}_classifier')
        torch.save(self.linear.state_dict(), f'../files/{data_set}/bert_base_ft_{base_model}_classifier/ft_layer.pt')
    
    def load_model(self, base_model):
        self.bert.from_pretrained(f'../files/bert_base_imdb_{base_model}_classifier')
        self.linear.load_state_dict(torch.load(f'../files/bert_base_imdb_{base_model}_classifier/ft_layer.pt'))


def train(model, train_data, val_data, learning_rate, epochs, base_model, save, data_set):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
        if save:
            model.save_model(base_model, data_set)
    return model


with open('../files/ft_responses-pile2k.json', 'r') as f:
    training_responses = json.load(f)
data_set = 'pile2k'
from pathlib import Path
Path(data_set).mkdir(parents=True, exist_ok=True)

with open('../files/ft_responses.json', 'r') as f:
    testing_responses = json.load(f)

with open('../files/ft_responses-10-19.json', 'r') as f:
    alt_testing_responses = json.load(f)

# df = pd.DataFrame.from_dict({'text':data_list, 'category': labels})

base_model_to_training_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                             "Multilingual-MiniLM-L12-H384": '8',
                             "gpt2-xl": '4', "gpt-neo-125M": '6', "opt-350m": '1', "xlnet-base-cased": '7',
                             "codegen-350M-multi": '9'}

df_latex = pd.DataFrame(columns=['base model'])
for ft in base_model_to_training_ft.values():
    df_latex[ft] = None
training_ft_models = {str(i): 'a' for i in range(0, 10)}
random.seed(42)
np.random.seed(42)

for base_model in base_model_to_training_ft.keys():
    # if 'bloom' not in base_model and 'Dialo' not in base_model:
    #    continue
    gt_ft_model = base_model_to_training_ft[base_model]
    EPOCHS =  5
    model = BertClassifier()
    LR = 1e-6
    print(f'training classifier for {base_model}')
    labels = []
    prompt_responses = []
    data_model_name = []
    for prompt in training_responses:
        # labels.append(1)
        # prompt_responses.append(testing_responses[dataset][prompt][base_model])
        # data_model_name.append(base_model)
        for model_name, response in training_responses[prompt].items():
            #response = response[len(prompt):]
            if base_model == model_name:
                labels.append(1)
                prompt_responses.append(response)
                data_model_name.append(model_name)
            elif len(model_name) > 3:
                labels.append(0)
                prompt_responses.append(response)
                data_model_name.append(model_name)
            '''if len(model_name) < 3:
                if model_name in ft_models.keys():
                    if gt_ft_model == model_name:
                        labels.append(1)
                    else:
                        labels.append(0)
                    data_model_name.append(model_name)
                    prompt_responses.append(training_responses[prompt][base_model][len(prompt):] + '[SEP]' + response)
            if base_model == model_name:
                labels.append(1)
                prompt_responses.append(response)
                data_model_name.append(model_name)
            if len(model_name) < 3:
                if model_name in training_ft_models.keys():
                    if gt_ft_model == model_name:
                        labels.append(1)
                    else:
                        labels.append(0)
                    data_model_name.append(model_name)
                    prompt_responses.append(response)'''
    '''
    labels = []
    prompt_responses = []
    data_model_name = []
    for dataset, prompts in alt_testing_responses.items():
        for prompt in prompts:
            model_responses = {}
            for model_name, response in prompts[prompt].items():
                response = response[len(prompt):]
                if base_model == model_name:
                    labels.append(1)
                    prompt_responses.append(response)
                    data_model_name.append(model_name)
                elif len(model_name) > 3:
                    labels.append(0)
                    prompt_responses.append(response)
                    data_model_name.append(model_name)'''
    df = pd.DataFrame.from_dict({'labels': labels, 'response': prompt_responses, 'model': data_model_name})
    print(df)
    arr_3D = df.values.reshape(-1, 10, df.shape[1])
    shuffle_idx = np.random.RandomState(seed=42).permutation(arr_3D.shape[0])
    df = pd.DataFrame(np.reshape(arr_3D[shuffle_idx], (df.values.shape)), columns=['labels', 'response', 'model'])
    df_train, df_val, df_test = np.split(df,
                                         [int(.8 * len(df)), int(.9 * len(df))])

    #model.load_model(base_model)
    model = train(model, df_train, df_val, LR, EPOCHS, base_model, save=True, data_set=data_set)
    # model.load_model(base_model)
    torch.save(model.state_dict(), f'./files/{data_set}/bert_base_ft_{base_model}_classifier/model.pth')

    correct_predictions = 0
    model_correctness = {str(i): 0 for i in range(10)}
    model_correctness = {str(i): 0 for i in base_model_to_training_ft.keys()}
    model_correctness[base_model] = 0
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    labels = []
    prompt_responses = []
    data_model_name = []
    for dataset, prompts in testing_responses.items():
        for prompt in prompts:
            model_responses = {}
            for model_name, response in prompts[prompt].items():
                response = response[len(prompt):]
                if base_model == model_name:
                    model_responses[model_name] = response[len(prompt):]
                    labels.append(1)
                    prompt_responses.append(response)# + '[SEP]' + response)
                    data_model_name.append(model_name)
                if len(model_name) < 3:
                    if model_name in ft_models.keys():
                        model_responses[model_name] = response[len(prompt):]
                        if gt_ft_model == model_name:
                            labels.append(1)
                        else:
                            labels.append(0)
                        data_model_name.append(model_name)
                        prompt_responses.append(response[len(prompt):])
                        #  prompts[prompt][base_model][len(prompt):] + '[SEP]' + response[len(prompt):])
    #df_test = pd.DataFrame.from_dict({'labels': labels, 'response': prompt_responses, 'model': data_model_name})

    for index, row in df_test.iterrows():
        prompt_set = tokenizer(row['response'],
                               padding='max_length', max_length=512, truncation=True,
                               return_tensors="pt").to(device)
        with torch.no_grad():
            prediction = \
            model(prompt_set['input_ids'], prompt_set['attention_mask']).argmax(dim=1).cpu().detach().numpy()[0]
        # svm_prediction = model.predict([row['data']])[0]
        gt = row['labels']
        if gt == prediction:
            correct_predictions += 1
            model_correctness[row['model']] += 1
        if prediction > 1:
            print(f'prediction {prediction}')

    output_data = {'ground_truth': gt_ft_model, 'base_model': base_model, 'test_results': {}}
    print(df_test)
    print(f'ACCURACY OF {base_model} TRAINED SVM: {correct_predictions / len(df_test.index)}')
    print(f'Ground Truth: {gt_ft_model}')
    df_acc = {'base model': base_model}
    for model, predictions in model_correctness.items():
        num_model_samples = len(df_test[df_test['model'] == model].index)
        output_data['test_results'][model] = {
            'accuracy': model_correctness[model] / num_model_samples,
            'raw_correct': model_correctness[model],
            'total_prompts': num_model_samples}
        df_acc[model] = ["{:.2f}".format(model_correctness[model] / num_model_samples)]
        if gt_ft_model == model:
            print(
                f'ACCURACY OF {base_model}  SVM for {model} (GT): {model_correctness[model] / num_model_samples} ({model_correctness[model]}/{num_model_samples})')
        else:
            print(
                f'ACCURACY OF {base_model}  SVM for {model}: {model_correctness[model] / num_model_samples} ({model_correctness[model]}/{num_model_samples})')

    df_latex = pd.concat([df_latex, pd.DataFrame.from_dict(df_acc)])

    filename = f'../files/{data_set}/bert_base_ft_{base_model}_classifier/test_results.json'
    with open(filename, 'w') as f:
        json.dump(output_data, f)
print(df_latex.to_latex())
with open('../files/bert_base_{data_set}_ft_classifier.tex', 'w') as f:
    f.write(df_latex.to_latex())