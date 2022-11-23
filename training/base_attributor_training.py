import pandas as pd
from pathlib import Path
from transformers import BertTokenizer
import torch
import numpy as np
import json
import torch.nn as nn
from transformers import BertModel
from models.attributor import Attributor
#from api.finetune_zoo.models import ft_models
#ft_models = {str(i): 'a' for i in range(10)}
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
        self.importance = df['importance'].values
        for idx, row in df.iterrows():
            text = row['response']
            if type(text) != str:
                continue
            self.texts.append(tokenizer(text, padding='max_length', max_length = 512, truncation=True,
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


def train(model, train_data, val_data, learning_rate, epochs, base_model):
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
            for val_input, val_label, train_importance in val_dataloader:
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
    return model

with open('../compute_responses/responses/ft_responses.json', 'r') as f:
    training_responses = json.load(f)

base_model_names = ["bloom-350m", "DialoGPT-large", "distilgpt2", "gpt2", "Multilingual-MiniLM-L12-H384",
                    "gpt2-xl", "gpt-neo-125M", "opt-350m", "xlnet-base-cased", "codegen-350M-multi"]

save_dir = 'responses/base_model_attributors'

load_model = None

for base_model in base_model_names:
    EPOCHS = 10
    model = Attributor()
    LR = 1e-6
    labels = []
    prompt_responses = []
    data_model_name = []
    for dataset, prompts in training_responses.items():
        for prompt in prompts:
            for model_name, response in prompts[prompt].items():
                response = response[len(prompt):]
                if base_model == model_name:
                    labels.append(1)
                    prompt_responses.append(response)
                    data_model_name.append(model_name)
                elif len(model_name) > 3 and base_model != model_name:
                    labels.append(0)
                    data_model_name.append(model_name)
                    prompt_responses.append(response)

    df = pd.DataFrame.from_dict({'labels': labels, 'response': prompt_responses,
                                 'model': data_model_name})
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])
    df_train = df_train.append(df_test, ignore_index=True)

    if load_model is not None:
        model.load_state_dict(torch.load(f'./responses/{load_model}/{base_model}_attributor/model.pth'))

    model = train(model, df_train, df_val, LR, EPOCHS, base_model)
    model.load_model(base_model)
    Path(f"../responses/{save_dir}/{base_model}_attributor").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f'../responses/{save_dir}/{base_model}_attributor/model.pth')
