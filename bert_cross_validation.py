import pandas as pd
from transformers import BertTokenizer
import torch
import numpy as np
import json
import torch.nn as nn
from transformers import BertModel
#from api.finetune_zoo.models import ft_models
ft_models = {str(i): 'a' for i in range(10)}
from torch.optim import Adam
from tqdm import tqdm
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


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

    def save_model(self, base_model, test_models):
        self.bert.save_pretrained(f'./files/bert_{base_model}_{test_models[0]}_{test_models[1]}')
        torch.save(self.linear.state_dict(), f'./files/bert_{base_model}_{test_models[0]}_{test_models[1]}/ft_layer.pt')

    def load_model(self, base_model, test_models):
        self.bert.from_pretrained(f'./files/bert_{base_model}_{test_models[0]}_{test_models[1]}')
        self.linear.load_state_dict(torch.load(f'./files/bert_{base_model}_{test_models[0]}_{test_models[1]}/ft_layer.pt'))


def train(model, train_data, val_data, learning_rate, epochs, base_model, test_models):
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
                | Val Accuracy: {total_acc_val / len(val_data): .3f}', flush=True)
        model.save_model(base_model, test_models)
    return model



with open('./files/ft_responses.json', 'r') as f:
    responses = json.load(f)
'''
data_list = []
labels = []
for dataset, prompts in responses.items():
    for prompt in prompts:
        base_model = 'opt-350m'
        gt_ft_model = '1'
        # gt = [0 for _ in range(len(ft_models.keys()) + 1)]
        # print(prompts[prompt][base_model])
        # data[base_model].append(tokenizer(prompts[prompt][base_model]).data['input_ids'])
        data_list.append(prompts[prompt][base_model])
        labels.append(1)
        for model_name, ft_response in prompts[prompt].items():
            if len(model_name) < 3:
                if model_name in ft_models.keys():
                    # data[f'ft_model_{model_name}'].append(tokenizer(prompts[prompt][model_name]).data['input_ids'])
                    data_list.append(prompts[prompt][model_name])
                    if gt_ft_model == model_name:
                        labels.append(1)
                    else:
                        labels.append(0)'''


#df = pd.DataFrame.from_dict({'text':data_list, 'category': labels})

base_model_to_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                    "Multilingual-MiniLM-L12-H384": '8',
                    "gpt2-xl": '4', "gpt-neo-125M": '6', "opt-350m": '1', "xlnet-base-cased": '7',
                    "codegen-350M-multi": '9'}

def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(2, 3)))

for base_model in base_model_to_ft.keys():
    #if 'codegen' not in base_model:
    #    continue
    gt_ft_model = base_model_to_ft[base_model]
    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-6

    #df_train = pd.read_csv(f'./files/classifier_data/{base_model}_train.csv')
    #df_test = pd.read_csv(f'./files/classifier_data/{base_model}_test.csv')
    #df_val = pd.read_csv(f'./files/classifier_data/{base_model}_val.csv')
    #ft_models_without_gt = list(set(ft_models.keys()) - set(gt_ft_model))

    for test_set_pair in all_subsets(list(ft_models.keys())):
        labels = []
        prompt_responses = []
        data_model_name = []
        for dataset, prompts in responses.items():
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

        df = pd.DataFrame.from_dict({'labels': labels, 'response': prompt_responses, 'model': data_model_name})
        df_train = df[(df['model'] != test_set_pair[0] ) & (df['model'] != test_set_pair[1])]
        df_val = df[df['model'] == test_set_pair[0]]
        df_test = df[(df['model'] == test_set_pair[0]) | (df['model'] == test_set_pair[1])]
        # df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
        #                                     [int(.8 * len(df)), int(.9 * len(df))])

        # df_train.to_csv(f'./files/classifier_data/{base_model}_train.csv')
        # df_val.to_csv(f'./files/classifier_data/{base_model}_val.csv')
        # df_test.to_csv(f'./files/classifier_data/{base_model}_test.csv')

        model = train(model, df_train, df_val, LR, EPOCHS, base_model, test_set_pair)
        correct_predictions = 0
        model_correctness = {ft :0 for ft in test_set_pair}
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
            gt = row['labels']
            if gt == prediction:
                correct_predictions += 1
                model_correctness[row['model']] += 1
            if prediction > 1:
                print(f'prediction {prediction}')


        #print(f'ACCURACY OF {base_model} TRAINED SVM: {correct_predictions/len(df_test.index)}')
        print(f'Ground Truth: {gt_ft_model}')
        output_data = {'ground_truth': gt_ft_model, 'base_model': base_model,
                       'test_pair':[test_set_pair[0], test_set_pair[1]], 'test_results':{}}


        for model_test_name, predictions in model_correctness.items():
            num_model_samples = len(df_test[df_test['model'] == model_test_name].index)
            output_data['test_results'][model_test_name] = {'accuracy': model_correctness[model_test_name] / num_model_samples,
                                                            'raw_correct':model_correctness[model_test_name],
                                                            'total_prompts': num_model_samples}
            if gt_ft_model == model_test_name:
                print(
                    f'ACCURACY OF {base_model}  SVM for {model_test_name} (GT): {model_correctness[model_test_name] / num_model_samples} ({model_correctness[model_test_name]}/{num_model_samples})')
            else:
                print(
                    f'ACCURACY OF {base_model}  SVM for {model_test_name}: {model_correctness[model_test_name] / num_model_samples} ({model_correctness[model_test_name]}/{num_model_samples})')

        filename = f'./logs/bert_{base_model}_{test_set_pair[0]}_{test_set_pair[1]}/test_results.json'
        with open(filename) as f:
            json.dump(output_data, f)
