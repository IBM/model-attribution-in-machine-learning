import pandas as pd
from transformers import BertTokenizer
import torch
import numpy as np
import json
import torch.nn as nn
from transformers import BertModel
from sklearn import metrics
import pickle as pkl
# from api.finetune_zoo.models import ft_models
ft_models = {str(i): 'a' for i in range(10)}
from torch.optim import Adam
from tqdm import tqdm
from itertools import chain, combinations
from matplotlib import pyplot as plt
from classifiers.attributor import Attributor

plt.rcParams.update({'font.size': 22})
plt.style.use('ggplot')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


base_model_to_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                             "Multilingual-MiniLM-L12-H384": '8',
                             "gpt2-xl": '4', "gpt-neo-125M": '6', "opt-350m": '1', "xlnet-base-cased": '7',
                             "codegen-350M-multi": '9'}

ft_to_base_model = {v:k for k, v in base_model_to_ft.items()}

with open('./files/ft_responses.json', 'r') as f:
    training_responses = json.load(f)

df_latex = pd.DataFrame(columns=['base model'])
for ft in base_model_to_ft.values():
    df_latex[ft] = None
models = ['k1_model', 'k1_tuple_model',  'k2_model', 'k2_tuple_model', 'k3_model']
all_model_plot_data =  {model: {} for model in models}
prec_recall_data = {}
for base_model in base_model_to_ft.keys():
    # if 'Dialo' not in base_model:
    #    continue
    gt_ft_model = base_model_to_ft[base_model]
    EPOCHS = 5
    models = {'k1_model':  Attributor(), 'k1_tuple_model':  Attributor(), 'k2_model':
        Attributor(), 'k2_tuple_model':  Attributor(), 'k3_model':  Attributor()}
    LR = 1e-6
    if torch.cuda.is_available():
        for model in models:
            models[model].cuda()
    print(f'training classifier for {base_model}')
    base_model_path = f'./files/bert_{base_model}_classifier'
    models['k1_model'].load_state_dict(torch.load(f'{base_model_path}/model.pth'))
    models['k1_tuple_model'].load_state_dict(torch.load(f'./files/bert_pair_{base_model}_classifier/model.pth'))
    models['k2_model'].load_state_dict(torch.load(f'./files/split_bert_{base_model}_classifier/model.pth'))
    models['k2_tuple_model'].load_state_dict(torch.load(f'./files/split_bert_tuple_{base_model}_classifier/model.pth'))
    models['k3_model'].load_state_dict(torch.load(f'./files/bert_base_{base_model}_classifier/model.pth'))
    labels = []
    prompt_responses = []
    tuple_prompt_responses = []
    data_model_name = []
    for dataset, prompts in training_responses.items():
        for prompt in prompts:
            model_responses = {}
            for model_name, response in prompts[prompt].items():
                if base_model == model_name:
                    #model_responses[model_name] = response[len(prompt):]
                    labels.append(1)
                    tuple_prompt_responses.append(response + '[SEP]' + response)
                    prompt_responses.append(response)
                    data_model_name.append(model_name)
                if len(model_name) < 3:
                    if model_name in ft_models.keys():
                        model_responses[model_name] = response[len(prompt):]
                        if gt_ft_model == model_name:
                            labels.append(1)
                        else:
                            labels.append(0)
                        data_model_name.append(model_name)
                        tuple_prompt_responses.append(
                            prompts[prompt][base_model] + '[SEP]' + response)
                        prompt_responses.append(response)

    df = pd.DataFrame.from_dict({'labels': labels, 'response': prompt_responses, 'model': data_model_name,
                                 'tuple': tuple_prompt_responses})
    arr_3D = df.values.reshape(-1, 11, df.shape[1])
    shuffle_idx = np.random.RandomState(seed=42).permutation(arr_3D.shape[0])
    df = pd.DataFrame(np.reshape(arr_3D[shuffle_idx], (df.values.shape)), columns=['labels', 'response', 'model', 'tuple'])
    df_train, df_val, df_test = np.split(df,
                                         [int(.8 * len(df)), int(.9 * len(df))])
    df_test = df_val.append(df_test, ignore_index=True)

    #df_test = df
    print(df_test)
    correct_predictions = 0
    model_correctness = {str(i): 0 for i in range(len(ft_models))}
    model_correctness[base_model] = 0
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    probs = {model: [] for model in models.keys()}
    labels = []

    tp = {model: 0 for model in models.keys()}
    fn = {model: 0 for model in models.keys()}
    fp = {model: 0 for model in models.keys()}
    for index, row in df_test.iterrows():
        prompt_set = tokenizer(row['response'],
                               padding='max_length', max_length=512, truncation=True,
                               return_tensors="pt").to(device)
        outs = {}
        pred = {}
        #probs = {}
        with torch.no_grad():
            for model in models.keys():
                if 'tuple' in model:
                    prompt_set = tokenizer(row['tuple'],
                                           padding='max_length', max_length=512, truncation=True,
                                           return_tensors="pt").to(device)
                else:
                    prompt_set = tokenizer(row['response'],
                                       padding='max_length', max_length=512, truncation=True,
                                       return_tensors="pt").to(device)
                outs[model] = models[model](prompt_set['input_ids'], prompt_set['attention_mask'])
                pred[model] = outs[model].argmax(dim=1).cpu().detach().numpy()[0]
                probs[model].append(torch.nn.functional.softmax(outs[model]).squeeze().cpu().detach().numpy()[1])
        # svm_prediction = model.predict([row['data']])[0]
        gt = row['labels']
        labels.append(gt)

        #labels.append(gt)
        for model in models.keys():
            if gt == 1:
                if pred[model] == 1:
                    tp[model] += 1
                else:
                    fn[model] += 1
            else:
                if pred[model] == 1:
                    fp[model] += 1

    precision = {}
    recall = {}
    for model in models.keys():
        if tp[model] == 0:
            tp[model] = 0.001
        elif fp[model] == 0:
            fp[model] = 0.001
        elif fn[model] == 0:
            fn[model] = 0.001
        precision[model] = tp[model]/(tp[model]+fp[model])
        recall[model] = tp[model]/(tp[model]+fn[model])
        fpr, tpr, _ = metrics.roc_curve(labels, probs[model])
        auc = metrics.roc_auc_score(labels, probs[model])
        auc = round(auc, 2)
        all_model_plot_data[model][base_model] = (fpr, tpr, auc)
    prec_recall_data[base_model] = (precision, recall)

average_auc_data = {model: (np.mgrid[0:1.01:.01, 0:len(base_model_to_ft):1][0].T, np.mgrid[0:1.01:.01, 0:len(base_model_to_ft):1][0].T,
                            np.zeros([len(base_model_to_ft), ])) for model in models}
xaxis_labels = ['bloom', 'DialoGPT-large', "distilgpt2", "gpt2", "ML-MiniLM", "gpt2-xl", "gpt-neo", "opt-350m",
                "xlnet", "codegen"]
#base_fpr = np.linspace(0, 1, 101)
for model in all_model_plot_data:
    for base_model in base_model_to_ft.keys():#models:
        fpr, tpr, auc = all_model_plot_data[model][base_model]
        tpr_new = np.interp(average_auc_data[model][0][list(all_model_plot_data).index(model)], fpr, tpr)
        tpr_new[0] = 0.0
        #average_auc_data[model][0][list(models).index(model)] = base_fpr
        average_auc_data[model][1][list(base_model_to_ft).index(base_model)] = tpr_new

        average_auc_data[model][2][list(base_model_to_ft).index(base_model)] = auc
        plt.plot(fpr, tpr, label=f"AUC {xaxis_labels[list(base_model_to_ft).index(base_model)]}:{auc}")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(list(all_model_plot_data))
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig(f'auc_roc_all_{model}.eps', format='eps')
    plt.clf()

for base_model in average_auc_data:
    fpr, tpr, auc = average_auc_data[base_model]
    plt.plot(fpr[0], tpr.mean(axis=0), label=f"AUC {base_model}: {round(np.average(auc), 3)}")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.legend(list(average_auc_data.keys()))
plt.legend(loc=4)
plt.tight_layout()
plt.savefig(f'auc_roc_all_average.eps', format='eps')
plt.clf()

x = np.arange(len(ft_to_base_model.keys()))  # the label locations
width = 0.15
p = []
r = []
for base_model in prec_recall_data:
    p_sub = []
    r_sub = []
    precision, recall = prec_recall_data[base_model]
    for model in models:
        p_sub.append(precision[model])
        r_sub.append(recall[model])
    p.append(p_sub)
    r.append(r_sub)
    #ax.bar(precision, width)
p = np.array(p)
plt.bar(x-width*2, p.T[0], width=width)
plt.bar(x-width, p.T[1], width=width)
plt.bar(x, p.T[2], width=width)
plt.bar(x+width, p.T[3], width=width)
plt.bar(x+width*2, p.T[4], width=width)
plt.ylabel('Precision')
plt.xlabel('Base model classifier')
plt.xticks(x, xaxis_labels, rotation=45)
#ax.set_yticks(np.arange(0, 81, 10))
plt.legend(labels=list(precision.keys()), loc=4, prop={'size': 8})
plt.tight_layout()
plt.savefig('precision.eps', format='eps')
plt.clf()

r = np.array(r)
plt.bar(x-width*2, r.T[0], width=width)
plt.bar(x-width, r.T[1], width=width)
plt.bar(x, r.T[2], width=width)
plt.bar(x+width, r.T[3], width=width)
plt.bar(x+width*2, r.T[4], width=width)
plt.ylabel('Recall')
plt.xlabel('Base model classifier')
plt.xticks(x, xaxis_labels, rotation=45)
#ax.set_yticks(np.arange(0, 81, 10))
plt.legend(labels=list(precision.keys()), loc=4, prop={'size': 8})
plt.savefig('recall.eps', format='eps')
plt.tight_layout()
plt.clf()

with open('auc_roc_data.pkl', 'wb') as f:
    pkl.dump(all_model_plot_data, f)
with open('prec_recall_data.pkl', 'wb') as f:
    pkl.dump(prec_recall_data, f)
with open('./files/bert_base_classifier.tex', 'w') as f:
    f.write(df_latex.to_latex())
