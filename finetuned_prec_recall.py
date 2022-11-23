import pandas as pd
from transformers import BertTokenizer
import torch
import numpy as np
import json
from sklearn import metrics
import pickle as pkl
from classifiers.attributor import Attributor
ft_models = {str(i): 'a' for i in range(10)}
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.style.use('ggplot')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

base_model_to_ft = {"bloom-350m": '0', "opt-350m": '1', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2-xl": '4',
                    "gpt2": '5', "gpt-neo-125M": '6', "xlnet-base-cased": '7',
                    "Multilingual-MiniLM-L12-H384": '8', "codegen-350M-multi": '9'}

ft_to_base_model = {v: k for k, v in base_model_to_ft.items()}

with open('./files/ft_responses.json', 'r') as f:
    custom = json.load(f)


df_latex = pd.DataFrame(columns=['base model'])
for ft in base_model_to_ft.values():
    df_latex[ft] = None
testing_datasets = ['pile150', 'pile500', 'pile1k', 'pile2k', 'pile4k', 'pile6k', 'pile8k',
                    'pile10k']  # ['pile2k', 'pile4k', 'pile6k', 'pile8k', 'pile10k']
all_model_plot_data = {model: {} for model in testing_datasets}
prec_recall_data = {model: {} for model in testing_datasets}
for base_model in base_model_to_ft.keys():
    gt_ft_model = base_model_to_ft[base_model]
    EPOCHS = 5
    testing_datasets = {'pile150': custom, 'pile500': custom, 'pile1k': custom, 'pile2k': custom,
                        'pile4k': custom, 'pile6k': custom, 'pile8k': custom,
                        'pile10k': custom}
    LR = 1e-6
    print(f'training classifier for {base_model}')
    for dataset in testing_datasets:
        model = Attributor()
        if torch.cuda.is_available():
            model.cuda()
        dataset_responses = testing_datasets[dataset]
        model.load_state_dict(torch.load(f'./files/{dataset}/finetuned_bert_base_ft_{base_model}_classifier/model.pth'))
        model.eval()
        labels = []
        prompt_responses = []
        data_model_name = []
        base_model_path = f'./files/bert_{base_model}_classifier'
        for prompts in dataset_responses.values():
            for prompt in prompts:
                for model_name, response in prompts[prompt].items():
                    if len(model_name) < 3:
                        if model_name in ft_models.keys():
                            response = response[len(prompt):]
                            if gt_ft_model == model_name:
                                labels.append(1)
                            else:
                                labels.append(0)
                            data_model_name.append(model_name)
                            prompt_responses.append(response)

        df = pd.DataFrame.from_dict({'labels': labels, 'response': prompt_responses, 'model': data_model_name})
        arr_3D = df.values.reshape(-1, 10, df.shape[1])
        shuffle_idx = np.random.RandomState(seed=42).permutation(arr_3D.shape[0])
        df = pd.DataFrame(np.reshape(arr_3D[shuffle_idx], (df.values.shape)), columns=['labels', 'response', 'model'])

        df_test = df
        print(df_test)
        correct_predictions = 0
        model_correctness = {str(i): 0 for i in range(len(ft_models))}
        # model_correctness[base_model] = 0
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        probs = []  # {model: [] for model in testing_datasets.keys()}
        labels = []

        tp = 0  # {model: 0 for model in testing_datasets.keys()}
        fn = 0  # {model: 0 for model in testing_datasets.keys()}
        fp = 0  # {model: 0 for model in testing_datasets.keys()}
        df_acc = {'base model': base_model}
        model_outs = []
        prev_start = 0
        for i in range(1, int((len(df_test)) / 50)):
            prompt_set = tokenizer(df_test['response'].tolist()[prev_start:i * 50],
                                   padding='max_length', max_length=512, truncation=True,
                                   return_tensors="pt").to(device)
            prev_start = i * 50
            with torch.no_grad():
                outs = model(prompt_set['input_ids'], prompt_set['attention_mask'])
            for out in outs:
                model_outs.append(out)

        prompt_set = tokenizer(df_test['response'].tolist()[prev_start:],
                               padding='max_length', max_length=512, truncation=True,
                               return_tensors="pt").to(device)
        with torch.no_grad():
            outs = model(prompt_set['input_ids'], prompt_set['attention_mask'])
        for out in outs:
            model_outs.append(out)

        print(len(model_outs))
        for index, row in df_test.iterrows():
            with torch.no_grad():
                out = model_outs[index]
                pred = out.argmax(dim=0).cpu().detach().numpy()
                probs.append(torch.nn.functional.softmax(out).cpu().squeeze().detach().numpy()[1])
            gt = row['labels']
            labels.append(gt)

            if gt == 1:
                if pred == 1:
                    model_correctness[row['model']] += 1
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == 1:
                    fp += 1
                else:
                    model_correctness[row['model']] += 1

        output_data = {'ground_truth': gt_ft_model, 'base_model': base_model, 'test_results': {}}
        for model, predictions in model_correctness.items():
            num_model_samples = len(df_test[df_test['model'] == model].index)
            output_data['test_results'][model] = {
                'accuracy': model_correctness[model] / num_model_samples,
                'raw_correct': model_correctness[model],
                'total_prompts': num_model_samples}
            if model != base_model:
                df_acc[model] = ["{:.2f}".format(model_correctness[model] / num_model_samples)]
                df_acc['dataset'] = dataset
            if gt_ft_model == model:
                print(
                    f'ACCURACY OF {base_model}  ATTRIBUTOR for {model} (GT): {model_correctness[model] / num_model_samples} ({model_correctness[model]}/{num_model_samples})')
            else:
                print(
                    f'ACCURACY OF {base_model}  ATTRIBUTOR for {model}: {model_correctness[model] / num_model_samples} ({model_correctness[model]}/{num_model_samples})')
        df_latex = pd.concat([df_latex, pd.DataFrame.from_dict(df_acc)])
        precision = {}
        recall = {}
        if fp == 0:
            fp = 0.001
        elif fn == 0:
            fn = 0.001
        precision = tp / (tp + fp) if tp != 0 else 0
        recall = tp / (tp + fn) if tp != 0 else 0
        fpr, tpr, _ = metrics.roc_curve(labels, probs)
        auc = metrics.roc_auc_score(labels, probs)
        auc = round(auc, 2)
        all_model_plot_data[dataset][base_model] = (fpr, tpr, auc)
        prec_recall_data[dataset][base_model] = (precision, recall)
with open('./files/pile_finetuned_classifications.tex', 'w') as f:
    f.write(df_latex.to_latex())

average_auc_data = {dataset: (
    np.mgrid[0:1.01:.01, 0:len(base_model_to_ft):1][0].T, np.mgrid[0:1.01:.01, 0:len(base_model_to_ft):1][0].T,
    np.zeros([len(base_model_to_ft), ])) for dataset in testing_datasets}
model_labels = ['bloom', 'DialoGPT-large', "distilgpt2", "gpt2", "ML-MiniLM", "gpt2-xl", "gpt-neo", "opt-350m",
                "xlnet", "codegen"]
xaxis_labels = [dataset.split('pile')[1] for dataset in testing_datasets.keys()]
# base_fpr = np.linspace(0, 1, 101)
for dataset in all_model_plot_data:
    for model in base_model_to_ft:
        fpr, tpr, auc = all_model_plot_data[dataset][model]
        tpr_new = np.interp(average_auc_data[dataset][0][list(all_model_plot_data).index(dataset)], fpr, tpr)
        tpr_new[0] = 0.0
        average_auc_data[dataset][1][list(base_model_to_ft).index(model)] = tpr_new
        average_auc_data[dataset][2][list(base_model_to_ft).index(model)] = auc
        plt.plot(fpr, tpr, label=f"AUC {model_labels[list(all_model_plot_data).index(dataset)]}:{auc}")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(list(all_model_plot_data))
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig(f'auc_roc_all_finetuned_{dataset}.eps', format='eps')
    plt.clf()

for base_model in average_auc_data:
    fpr, tpr, auc = average_auc_data[base_model]
    plt.plot(fpr[0], tpr.mean(axis=0), label=f"AUC {base_model}: {round(np.average(auc), 3)}")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(list(average_auc_data.keys()))
plt.legend(loc=4)
plt.tight_layout()
plt.savefig(f'auc_roc_all_average_finetuned_pile.eps', format='eps')
plt.clf()

x = np.arange(len(ft_to_base_model.keys()))  # the label locations
width = 0.15
p = []
r = []
for dataset in prec_recall_data:
    p_sub = []
    r_sub = []
    for model in base_model_to_ft:
        precision, recall = prec_recall_data[dataset][model]
        p_sub.append(precision)
        r_sub.append(recall)
    p.append(np.mean(p_sub))
    r.append(np.mean(r_sub))


plt.plot(list(range(len(p))), p, linestyle='--', marker='o')
plt.ylabel('Precision')
plt.xlabel('Number of Prompts')
plt.xticks(list(range(len(p))), xaxis_labels)
plt.tight_layout()
plt.savefig('precision_finetuned_pile.eps', format='eps')
plt.clf()

r = np.array(r)
plt.plot(list(range(len(r))), r, linestyle='--', marker='o')
plt.ylabel('Recall')
plt.xlabel('Number of Prompts')
plt.xticks(list(range(len(r))), xaxis_labels)
plt.tight_layout()
plt.savefig('recall_finetuned_pile.eps', format='eps')
plt.clf()

fig, ax = plt.subplots()
ax.plot([150, 500, 1000, 2000, 4000, 6000, 8000, 10000], p, linestyle='--', marker='o', color='blue')
ax.plot(150, 0.79150101, marker='^', color='blue')
ax.set_ylabel('Precision', color="blue")
ax.set_ylim((0, 1))
ax.set_xlabel('Number of Prompts', )
ax2 = ax.twinx()
ax2.plot([150, 500, 1000, 2000, 4000, 6000, 8000, 10000], r, linestyle='--', marker='o', color='red')
ax2.plot(150, 0.49, marker='^', color='red')
ax2.set_ylabel('Recall', color="red", )
ax2.set_ylim((0, 1))
plt.legend(labels=['The Pile', 'Manual'], loc=1)
plt.tight_layout()
plt.savefig('precision_recall_finetuned_pile.eps', format='eps')
plt.clf()
with open('auc_roc_data_finetuned_pile.pkl', 'wb') as f:
    pkl.dump(all_model_plot_data, f)
with open('prec_recall_data_finetuned_pile.pkl', 'wb') as f:
    pkl.dump(prec_recall_data, f)
with open('./files/bert_pile_finetuned_pile.tex', 'w') as f:
    f.write(df_latex.to_latex())

