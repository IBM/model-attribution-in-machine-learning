from transformers import BertTokenizer
import torch
import numpy as np
import json
import pandas as pd
import torch.nn as nn
from transformers import BertModel
#from api.finetune_zoo.models import ft_models
ft_models = {str(i): 'a' for i in range(10)}
from torch.optim import Adam
from tqdm import tqdm
from itertools import chain, combinations
from scipy.spatial import distance

import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#bert_input = tokenizer(example_text,padding='max_length', max_length = 10,
#                       truncation=True, return_tensors="pt")

labels = {'other': 0,
          'base and finetune': 1,
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
        self.bert.save_pretrained(f'../files/triplet_loss_{base_model}_classifier')
        torch.save(self.embedding_net.state_dict(), f'../files/triplet_loss_{base_model}_classifier/ft_layer.pt')

    def load_model(self, path):
        self.bert.from_pretrained(path)
        self.embedding_net.load_state_dict(torch.load(f'{path}/ft_layer.pt'))

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

    #num_correct = 0  # probably should be refactored
    test_sentence_embedding = get_sentence_embedding(model, sentence)
    closest_train_sentence = get_closest_train_sentence(test_sentence_embedding, train_sentence_to_embedding)
    predicted_label = train_sentence_to_label[closest_train_sentence]
    #if predicted_label == val_training_sentences[sentence]:
    #    num_correct = 1
    #acc = num_correct / len(test_sentence_to_label)
    return predicted_label





with open('../files/ft_responses.json', 'r') as f:
    test_responses = json.load(f)

#df = pd.DataFrame.from_dict({'text':data_list, 'category': labels})

base_model_to_training_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                    "Multilingual-MiniLM-L12-H384": '8',
                    "gpt2-xl": '4', "gpt-neo-125M": '6', "opt-350m": '1', "xlnet-base-cased": '7',
                    "codegen-350M-multi": '9'}


df_latex = pd.DataFrame(columns = ['base model'])

for base_model in base_model_to_training_ft.keys():
    #if 'Dialo' not in base_model:
    #    continue
    gt_ft_model = base_model_to_training_ft[base_model]
    EPOCHS = 5
    model = TripletNet('cuda:0' if torch.cuda.is_available() else 'cpu')
    LR = 1e-6
    print(f'testing classifier for {base_model}')


    labels = []
    prompt_responses = []
    sentence_to_label = {}
    data_model_name = []
    for dataset, prompts in test_responses.items():
        for prompt in prompts:
            #labels.append(1)
            #prompt_responses.append(testing_responses[dataset][prompt][base_model])
            #data_model_name.append(base_model)
            for model_name, response in prompts[prompt].items():
                if base_model == model_name:
                    labels.append(1)
                    prompt_responses.append(response)
                    data_model_name.append(model_name)
                    sentence_to_label[test_responses[dataset][prompt][model_name]] = 1
                if len(model_name) < 3:
                    if gt_ft_model == model_name:
                        labels.append(1)
                        sentence_to_label[test_responses[dataset][prompt][model_name]] = 1
                    else:
                        labels.append(0)
                        sentence_to_label[test_responses[dataset][prompt][model_name]] = 0
                    data_model_name.append(model_name)
                    prompt_responses.append(response)

    df = pd.DataFrame.from_dict({'labels': labels, 'response': prompt_responses, 'model': data_model_name})
    #df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
    #                                     [int(.8 * len(df)), int(.9 * len(df))])
    arr_3D = df.values#.reshape(-1,11, df.shape[1])
    shuffle_idx = np.random.RandomState(seed=42).permutation(arr_3D.shape[0])
    df = pd.DataFrame(np.reshape(arr_3D[shuffle_idx], (df.values.shape)), columns=['labels', 'response', 'model'])
    df_train, df_val, df_test = np.split(df,
                                         [int(.8 * len(df)), int(.9 * len(df))])
    df_test = df

    test_sentence_to_label = {sentence: sentence_to_label[sentence] for sentence in df_test['response']}
    #test_sentence_to_label = {**test_sentence_to_label, **{sentence: sentence_to_label[sentence] for sentence in df_test['pos']}}
    #test_sentence_to_label = {**test_sentence_to_label, **{sentence: sentence_to_label[sentence] for sentence in df_test['neg']}}


    train_sentence_to_label = {sentence: sentence_to_label[sentence] for sentence in df_train['response']}
    #train_sentence_to_label = {**train_sentence_to_label, **{sentence: sentence_to_label[sentence] for sentence in df_train['pos']}}
    #train_sentence_to_label = {**train_sentence_to_label, **{sentence: sentence_to_label[sentence] for sentence in df_train['neg']}}

    model.load_model(f'../files/triplet_loss_{base_model}_classifier')
    #model = train(model, df_train, df_val, LR, EPOCHS, base_model)
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

    print('computing training sentence embeddings...')
    train_sentence_to_embedding = {}
    for train_sentence in train_sentence_to_label:
        sentence = tokenizer(train_sentence,
                             padding='max_length', max_length=512, truncation=True,
                             return_tensors="pt").to(device)
        train_sentence_to_embedding[train_sentence] = model.get_embedding(sentence).detach().cpu().numpy()[0]


    for index, row in df_test.iterrows():
        test_prompt = row['response']
        prediction = eval_model(model, device, test_sentence_to_label, test_prompt, train_sentence_to_embedding)
        gt = row['labels']
        model_predictions[row['model']].append(int(prediction))
        if gt == prediction:
            correct_predictions += 1
            model_correctness[row['model']] += 1


    '''for index, row in df_test.iterrows():
        prompt_set = tokenizer(row['response'],
                          padding='max_length', max_length=512, truncation=True,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            prediction = model(prompt_set['input_ids'], prompt_set['attention_mask']).argmax(dim=1).cpu().detach().numpy()[0]
        #svm_prediction = model.predict([row['data']])[0]
        model_predictions[row['model']].append(int(prediction))
        gt = row['labels']
        if gt == prediction:
            correct_predictions += 1
            model_correctness[row['model']] += 1
        if prediction > 1:
            print(f'prediction {prediction}')'''
    output_data = {'ground_truth': gt_ft_model, 'base_model': base_model, 'test_results':{}}

    print(f'ACCURACY OF {base_model} TRAINED SVM: {correct_predictions/len(df_test.index)}')
    print(f'Ground Truth: {gt_ft_model}')
    df_acc = {'base model':base_model}

    for model, predictions in model_correctness.items():
        num_model_samples = len(df_test[df_test['model'] == model].index)
        output_data['test_results'][model] = {
            'accuracy': model_correctness[model] / num_model_samples,
            'raw_correct': model_correctness[model],
            'predictions': model_predictions[model],
            'total_prompts': num_model_samples}

        if model != base_model:
            df_acc[model] = ["{:.2f}".format(model_correctness[model] / num_model_samples)]
        if gt_ft_model == model:
            print(f'ACCURACY OF {base_model}  SVM for {model} (GT): {model_correctness[model]/num_model_samples} ({model_correctness[model]}/{num_model_samples})')
            print('predictions:')
            print(model_predictions[model])
        else:
            print(f'ACCURACY OF {base_model}  SVM for {model}: {model_correctness[model]/num_model_samples} ({model_correctness[model]}/{num_model_samples})')
            print('predictions:')
            print(model_predictions[model])



with open('../files/triplet_test_classifier_table.tex', 'w') as f:
    f.write(df_latex.to_latex())
