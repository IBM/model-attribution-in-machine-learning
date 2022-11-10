import pandas as pd
from transformers import BertTokenizer
import torch
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
#from api.finetune_zoo.models import ft_models
ft_models = {str(i): 'a' for i in range(10)}
import os
from torch.optim import Adam
from tqdm import tqdm
import random
from itertools import chain, combinations
from scipy.spatial import distance

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#bert_input = tokenizer(example_text,padding='max_length', max_length = 10,
#                       truncation=True, return_tensors="pt")

labels = {'other':0,
          'base and finetune':1,
          }

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        #self.labels = df['labels'].values
        self.anchors = []
        self.pos = []
        self.neg = []
        for idx, row in df.iterrows():
            anchors = row['anchors']
            pos = row['pos']
            neg = row['neg']
            if type(anchors) != str:
                continue
            self.anchors.append(tokenizer(anchors,
                               padding='max_length', max_length = 512, truncation=True, add_special_tokens=True,
                                return_tensors="pt"))
            self.pos.append(tokenizer(pos,
                               padding='max_length', max_length = 512, truncation=True, add_special_tokens=True,
                                return_tensors="pt"))
            self.neg.append(tokenizer(neg,
                               padding='max_length', max_length = 512, truncation=True, add_special_tokens=True,
                                return_tensors="pt"))

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.anchors)


    def get_batch_anchors(self, idx):
        # Fetch a batch of inputs
        return self.anchors[idx]

    def get_batch_pos(self, idx):
        # Fetch a batch of inputs
        return self.pos[idx]

    def get_batch_negs(self, idx):
        # Fetch a batch of inputs
        return self.neg[idx]

    def __getitem__(self, idx):

        batch_anchor = self.get_batch_anchors(idx)
        batch_pos = self.get_batch_pos(idx)
        batch_negs = self.get_batch_negs(idx)

        return batch_anchor, batch_pos, batch_negs

class EmbeddingNet(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.0, layernorm=False, batchnorm=False):
        super(EmbeddingNet, self).__init__()
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=0.4)
        self.Tanh = nn.Tanh()  ##nn.Tanh() / nn.ReLU() etc
        self.fc1 = nn.Linear(hidden_size, 200)
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(num_features=200)
        else:
            self.bn1 = None
        if layernorm:
            self.ln1 = nn.LayerNorm(200)
        else:
            self.ln1 = None
        self.fc2 = nn.Linear(200, output_size)

    def forward(self, x):
        output = self.dropout1(x)
        output = self.fc1(output)
        if not self.bn1 is None:
            output = self.bn1(output)
        if not self.ln1 is None:
            output = self.ln1(output)
        output = self.Tanh(output)
        output = self.dropout2(output)
        output = self.fc2(output)

        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=0.4, distance_type="C", account_for_nonzeros=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_type = distance_type.lower().strip()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.account_for_nonzeros = account_for_nonzeros

    def forward(self, anchor, positive, negative):

        if self.distance_type == "c":
            ## cosine distance
            distance_positive = -self.cos(anchor, positive)
            distance_negative = -self.cos(anchor, negative)
            losses = F.relu(distance_positive - distance_negative + self.margin)

        elif self.distance_type == "e":

            ## this is using Euclidean distance
            distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
            distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
            losses = F.relu(distance_positive - distance_negative + self.margin)
        else:
            raise Exception('please specify distance_type as C or E')

        semi_hard_indexes = [i for i in range(len(losses)) if losses[i] > 0]
        percent_activated = len(semi_hard_indexes) / len(losses)
        if self.account_for_nonzeros:
            loss = losses.sum() / len(semi_hard_indexes)
        else:
            loss = losses.mean()

        return loss, percent_activated
class TripletNet(nn.Module):
    def __init__(self, device):
        super(TripletNet, self).__init__()
        self.embedding_net = EmbeddingNet(768, 100)
        self.device=device
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

    def forward(self, input_idx1, maskx1, input_idx2, maskx2, input_idx3, maskx3):
        x1 = torch.mean(self.bert(input_ids=input_idx1, attention_mask=maskx1)[0], axis=1)#.flatten()
        x2 = torch.mean(self.bert(input_ids=input_idx2, attention_mask=maskx2)[0], axis=1)#.flatten()
        x3 = torch.mean(self.bert(input_ids=input_idx3, attention_mask=maskx3)[0], axis=1)#.flatten()

        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        x = torch.mean(self.bert(input_ids=x.data['input_ids'].squeeze(1).to(self.device),
                                 attention_mask=x.data['attention_mask'].squeeze(1).to(self.device))[0], axis=1)
        return self.embedding_net(x)

    def save_model(self, base_model):
        self.bert.save_pretrained(f'./files/triplet_loss_{base_model}_classifier')
        torch.save(self.embedding_net.state_dict(), f'./files/triplet_loss_{base_model}_classifier/ft_layer.pt')

def eval_model(
        model,
        device,
        val_training_sentences,
        sentence,
        train_sentence_to_embedding,
):
    model.eval()

    def get_sentence_embedding(model, sentence):
        sentence = tokenizer(sentence,
                                     padding='max_length', max_length=512, truncation=True,
                                     return_tensors="pt").to(device)
        return model.get_embedding(sentence).detach().cpu().numpy()[0]

    '''def get_train_sentence_to_embedding(model, training_sentences):
        train_sentence_to_embedding = {}
        for train_sentences in training_sentences:
            for train_sentence in train_sentences:
                embedding = get_sentence_embedding(model, train_sentence)
                train_sentence_to_embedding[train_sentence] = embedding
        return train_sentence_to_embedding'''

    def get_closest_train_sentence(test_sentence_embedding, train_sentence_to_embedding):
        train_sentence_to_dist_list = [
            (train_sentence, distance.cosine(test_sentence_embedding, train_sentence_embedding)) for
            train_sentence, train_sentence_embedding in train_sentence_to_embedding.items()]
        sorted_train_sentence_dist_list = list(sorted(train_sentence_to_dist_list, key=lambda tup: tup[1]))
        return sorted_train_sentence_dist_list[0][0]

    #train_sentence_to_embedding = get_train_sentence_to_embedding(model, train_training_sentences)

    num_correct = 0  # probably should be refactored
    test_sentence_embedding = get_sentence_embedding(model, sentence)
    closest_train_sentence = get_closest_train_sentence(test_sentence_embedding, train_sentence_to_embedding)
    predicted_label = train_sentence_to_label[closest_train_sentence]
    if predicted_label == val_training_sentences[sentence]:
        num_correct = 1
    #acc = num_correct / len(test_sentence_to_label)
    return num_correct

def train(model, train_data, val_data, learning_rate, epochs, base_model, val_sentence_to_label, train_sentence_to_label):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = TripletLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0
        model.train()
        for anchor, pos, neg in tqdm(train_dataloader):
            anchor_mask = anchor['attention_mask'].to(device)
            anchor_input_id = anchor['input_ids'].squeeze(1).to(device)
            pos_mask = pos['attention_mask'].to(device)
            pos_input_id = pos['input_ids'].squeeze(1).to(device)
            neg_mask = neg['attention_mask'].to(device)
            neg_input_id = neg['input_ids'].squeeze(1).to(device)

            output = model(anchor_input_id, anchor_mask, pos_input_id, pos_mask, neg_input_id, neg_mask)

            #batch_loss = criterion(output, train_label.long())
            train_loss, percent_activated = criterion(*output)
            total_loss_train += train_loss.item()

            #acc = (output.argmax(dim=1) == 0).sum().item()
            #total_acc_train += acc

            model.zero_grad()
            train_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            train_sentence_to_embedding = {}
            print('computing training sentence embeddings...')
            #for train_sentence in train_sentence_to_label:
            for train_sentence in train_sentence_to_label:

                sentence = tokenizer(train_sentence,
                                     padding='max_length', max_length=512, truncation=True,
                                     return_tensors="pt").to(device)
                #embedding = get_sentence_embedding(model, train_sentence)
                train_sentence_to_embedding[train_sentence] = model.get_embedding(sentence).detach().cpu().numpy()[0]
            print('validating...')
            validation_predictions = [[], []]
            for index, row in df_val.iterrows():
                anchor = row['anchors']
                pos = row['pos']
                neg = row['neg']
                num_correct = eval_model(model, device, val_sentence_to_label, anchor, train_sentence_to_embedding)
                total_acc_val += num_correct
                validation_predictions.append(num_correct)

                #acc = eval_model(model, device, val_sentence_to_label, pos, train_sentence_to_embedding)
                #total_acc_val += acc
                anchor = tokenizer(anchor, padding='max_length', max_length=512, truncation=True,
                                   return_tensors="pt").to(device)
                pos = tokenizer(pos, padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt").to(device)
                neg = tokenizer(neg, padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt").to(device)

                anchor_mask = anchor['attention_mask']
                anchor_input_id = anchor['input_ids']
                pos_mask = pos['attention_mask']
                pos_input_id = pos['input_ids']
                neg_mask = neg['attention_mask']
                neg_input_id = neg['input_ids']

                output = model(anchor_input_id, anchor_mask, pos_input_id, pos_mask, neg_input_id, neg_mask)
                batch_loss, percent_activated  = criterion(*output)
                total_loss_val += batch_loss.item()
                #val_label = val_label.to(device)
                #mask = val_input['attention_mask'].to(device)
                #input_id = val_input['input_ids'].squeeze(1).to(device)

                #output = model(input_id, mask)

                #batch_loss = criterion(output, val_label.long())
                #total_loss_val += batch_loss.item()

                #acc = (output.argmax(dim=1) == val_label).sum().item()
                #total_acc_val += acc
        print(validation_predictions)
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(df_val): .3f}')
        model.save_model(base_model)
    return model



#with open('./files/ft_training_responses.json', 'r') as f:
#    training_responses = json.load(f)

with open('./files/ft_responses.json', 'r') as f:
    training_responses = json.load(f)

#df = pd.DataFrame.from_dict({'text':data_list, 'category': labels})

base_model_to_training_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                    "Multilingual-MiniLM-L12-H384": '8',
                    "gpt2-xl": '4', "gpt-neo-125M": '6', "opt-350m": '1', "xlnet-base-cased": '7',
                    "codegen-350M-multi": '9'}

random.seed(42)
np.random.seed(42)
df_latex = pd.DataFrame(columns = ['base model'])


for base_model in base_model_to_training_ft.keys():
    #if 'codegen' not in base_model:
    #    continue
    gt_ft_model = base_model_to_training_ft[base_model]
    EPOCHS = 5
    model = TripletNet('cuda:0' if torch.cuda.is_available() else 'cpu')
    LR = 1e-6
    print(f'training classifier for {base_model}')
    labels = []
    #prompt_responses = []
    anchors = []
    pos = []
    negs = []
    data_model_name = []
    sentence_to_label = {}
    for  dataset, prompts in training_responses.items():
        for prompt in prompts:
            #labels.append(1)
            #prompt_responses.append(testing_responses[dataset][prompt][base_model])
            #data_model_name.append(base_model)

            for model_name, response in prompts[prompt].items():
                #if base_model == model_name:
                #    labels.append(1)
                #    prompt_responses.append(response)
                #    data_model_name.append(model_name)
                if len(model_name) < 3:
                    if gt_ft_model != model_name:
                        pos.append(training_responses[dataset][prompt][base_model])
                        anchors.append(training_responses[dataset][prompt][base_model_to_training_ft[base_model]])
                        negs.append(training_responses[dataset][prompt][model_name])
                        sentence_to_label[training_responses[dataset][prompt][base_model_to_training_ft[base_model]]] = 1
                        sentence_to_label[training_responses[dataset][prompt][model_name]] = 0
                        sentence_to_label[training_responses[dataset][prompt][base_model]] = 1
                        data_model_name.append(model_name)
                    #data_model_name.append(model_name)
                    #prompt_responses.append(response)

    df = pd.DataFrame.from_dict({'anchors': anchors, 'pos':pos, 'neg':negs, 'model': data_model_name})
    arr_3D = df.values#.reshape(-1,11, df.shape[1])
    shuffle_idx = np.random.RandomState(seed=42).permutation(arr_3D.shape[0])
    df = pd.DataFrame(np.reshape(arr_3D[shuffle_idx], (df.values.shape)), columns=['anchors', 'pos', 'neg', 'model'])
    df_train, df_val, df_test = np.split(df,
                                         [int(.8 * len(df)), int(.9 * len(df))])

    val_sentence_to_label = {sentence: sentence_to_label[sentence] for sentence in df_val['anchors']}
    val_sentence_to_label = {**val_sentence_to_label, **{sentence: sentence_to_label[sentence] for sentence in df_val['pos']}}
    val_sentence_to_label = {**val_sentence_to_label, **{sentence: sentence_to_label[sentence] for sentence in df_val['neg']}}

    train_sentence_to_label = {sentence: sentence_to_label[sentence] for sentence in df_train['anchors']}
    train_sentence_to_label = {**train_sentence_to_label, **{sentence: sentence_to_label[sentence] for sentence in df_train['pos']}}
    train_sentence_to_label = {**train_sentence_to_label, **{sentence: sentence_to_label[sentence] for sentence in df_train['neg']}}


    test_sentence_to_label = {sentence: sentence_to_label[sentence] for sentence in df_test['anchors']}
    test_sentence_to_label = {**test_sentence_to_label, **{sentence: sentence_to_label[sentence] for sentence in df_test['pos']}}
    test_sentence_to_label = {**test_sentence_to_label, **{sentence: sentence_to_label[sentence] for sentence in df_test['neg']}}

    model = train(model, df_train, df_val, LR, EPOCHS, base_model, val_sentence_to_label, train_sentence_to_label)
    correct_predictions = 0
    model_correctness = {base_model_to_training_ft[base_model]:0 }
    model_correctness[base_model] = 0
    predictions = [[], []]
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_sentence_to_embedding = {}
    print('computing training sentence embeddings...')
    # for train_sentence in train_sentence_to_label:
    for train_sentence in train_sentence_to_label:
        sentence = tokenizer(train_sentence,
                             padding='max_length', max_length=512, truncation=True,
                             return_tensors="pt").to(device)
        # embedding = get_sentence_embedding(model, train_sentence)
        train_sentence_to_embedding[train_sentence] = model.get_embedding(sentence).detach().cpu().numpy()[0]
    for index, row in df_test.iterrows():
        anchor = row['anchors']
        pos = row['pos']
        neg = row['neg']
        acc = eval_model(model, device, test_sentence_to_label, anchor, train_sentence_to_embedding)
        #svm_prediction = model.predict([row['data']])[0]
        #gt = row['labels']
        if acc == 1:
            correct_predictions += 1
            model_correctness[base_model_to_training_ft[base_model]] += 1
            predictions[0].append(1)
        else:
            predictions[0].append(0)
        acc = eval_model(model, device, test_sentence_to_label, neg, train_sentence_to_embedding)
        #svm_prediction = model.predict([row['data']])[0]
        #gt = row['labels']
        if acc == 1:
            correct_predictions += 1
            model_correctness[base_model_to_training_ft[base_model]] += 1
            predictions[1].append(1)
        else:
            predictions[1].append(0)
    output_data = {'ground_truth': gt_ft_model, 'base_model': base_model, 'test_results':{}}
    df_acc = {'base model':base_model}
    print(f'ACCURACY OF {base_model} TRAINED : {correct_predictions/(2*len(df_test.index))}')
    print(f'Ground Truth: {gt_ft_model}')
    #for model, predictions in model_correctness.items():
    #num_model_samples = len(df_test[df_test['model'] == model].index)
    #if num_model_samples == 0:
    #    continue
    model = base_model_to_training_ft[base_model]
    num_model_samples = len(df_test)
    output_data['test_results'][model] = {
            'accuracy': model_correctness[model] / (2*num_model_samples),
            'raw_correct': model_correctness[model],
            'predictions':predictions,
            'total_prompts': 2*num_model_samples}

    if model != base_model:
        df_acc[model] = ["{:.2f}".format(model_correctness[model]/(2*num_model_samples))]
    if gt_ft_model == model:
            print(f'ACCURACY OF {base_model}  SVM for {model} (GT): {model_correctness[model]/(2*num_model_samples)} ({model_correctness[model]}/{2*num_model_samples})')
    else:
            print(f'ACCURACY OF {base_model}  SVM for {model}: {model_correctness[model]/(2*num_model_samples)} ({model_correctness[model]}/{2*num_model_samples})')
    df_latex = pd.concat([df_latex, pd.DataFrame.from_dict(df_acc)])

    if not os.path.exists(f"./files/triplet_loss_{base_model}_classifier"):
        os.makedirs(f"./files/triplet_loss_{base_model}_classifier")
    filename = f'./files/triplet_loss_{base_model}_classifier/test_results.json'
    with open(filename, 'w') as f:
        json.dump(output_data, f)
with open('./files/bert_pair_classifier_table.tex', 'w') as f:
    f.write(df_latex.to_latex())
