import json
import torch
import torch.nn as nn
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig, BertModel

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# The first step is to fine-tune BERT model on SQUAD dataset. This can be easiy accomplished by following the steps described in hugging face's official web site: https://github.com/huggingface/transformers#run_squadpy-fine-tuning-on-squad-for-question-answering 
# 
# Note that the fine-tuning is done on a `bert-base-uncased` pre-trained model.

# After we pretrain the model, we can load the tokenizer and pre-trained BERT model using the commands described below. 

# In[3]:

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def forward(self, input_response, return_attentions=False, token_type_ids=None,
                   position_ids=None,
                   attention_mask=None):
        if return_attentions:
            prompts = self.tokenizer(list(input_response),
                               padding=True, max_length = 512, truncation=True,
                                return_tensors="pt")

            mask = prompts['attention_mask'].to(device)
            input_id = prompts['input_ids'].squeeze(1).to(device)
            _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        elif token_type_ids is not None:
            _, pooled_output = self.bert(input_response[:512], token_type_ids=token_type_ids[:512],
                    position_ids=position_ids[:512], attention_mask=attention_mask[:512], return_dict=False)
        else:
            '''prompts = self.tokenizer(input_response,
                                     padding='max_length', max_length=512, truncation=True,
                                     return_tensors="pt")
            mask = prompts['attention_mask'].to(device)
            input_id = prompts['input_ids'].squeeze(1).to(device)'''
            _, pooled_output = self.bert(input_ids=input_response[:512], attention_mask=None, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        if return_attentions:
            return final_layer.max(dim=1)[0].cpu().detach().numpy(), attention, prompts
        else:
            return final_layer

    def save_model(self, base_model):
        self.bert.save_pretrained(f'./files/bert_base{base_model}_classifier')

    def load_model(self, base_model):
        self.bert.from_pretrained(f'./files/bert_base{base_model}_classifier')


# replace <PATH-TO-SAVED-MODEL> with the real path of the saved model
#model_path = '<PATH-TO-SAVED-MODEL>'

# load model
#model = BertForQuestionAnswering.from_pretrained(model_path)
model = BertClassifier()
base_model = 'opt-350m'
model.load_state_dict(torch.load(f'../files/bert_base_{base_model}_classifier/model.pth'))
model.to(device)
model.cuda()
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = model.tokenizer


# A helper function to perform forward pass of the model and make predictions.

# In[4]:


def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    output = model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )
    return output


# Defining a custom forward function that will allow us to access the start and end postitions of our prediction using `position` input argument.

# In[5]:


def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred = predict(inputs,
                   token_type_ids=token_type_ids,
                   position_ids=position_ids,
                   attention_mask=attention_mask)
    #pred = pred[position]
    return pred.argmax(dim=1)


# Let's compute attributions with respect to the `BertEmbeddings` layer.
# 
# To do so, we need to define baselines / references, numericalize both the baselines and the inputs. We will define helper functions to achieve that.
# 
# The cell below defines numericalized special tokens that will be later used for constructing inputs and corresponding baselines/references.

# In[6]:


ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence


# Below we define a set of helper function for constructing references / baselines for word tokens, token types and position ids. We also provide separate helper functions that allow to construct attention masks and bert embeddings both for input and reference.

# In[7]:


def construct_input_ref_pair(question, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    #text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids[:511]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids[:511]) #+ [sep_token_id] +         [ref_token_id] #* len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def construct_whole_bert_embeddings(input_ids, ref_input_ids, token_type_ids=None, ref_token_type_ids=None,
                                position_ids=None, ref_position_ids=None):
    input_embeddings = model.bert.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = model.bert.embeddings(ref_input_ids, token_type_ids=ref_token_type_ids, position_ids=ref_position_ids)
    
    return input_embeddings, ref_input_embeddings


# Let's define the `question - text` pair that we'd like to use as an input for our Bert model and interpret what the model was forcusing on when predicting an answer to the question from given input text 

# In[8]:


#question, text = "What is important to us?", "It is important to us to include, empower and support humans of all kinds."
response = 'Aim: In this paper, we review the evidence publishable in the literature on the subject of the'

# Let's numericalize the question, the input text and generate corresponding baselines / references for all three sub-embeddings (word, token type and position embeddings) types using our helper functions defined above.

# In[9]:th for prediction's start and end positions.

# In[10]:



#ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)
#ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
#ground_truth_start_ind = ground_truth_end_ind - len(ground_truth_tokens) + 1


# Now let's make predictions using input, token type, position id and a default attention mask.


# There are two different ways of computing the attributions for emebdding layers. One option is to use `LayerIntegratedGradients` and compute the attributions with respect to `BertEmbedding`. The second option is to use `LayerIntegratedGradients` for each `word_embeddings`, `token_type_embeddings` and `position_embeddings` and compute the attributions w.r.t each embedding vector.
# 

# In[12]:



def custom_forward(inputs):
    preds = predict(inputs)
    return torch.softmax(preds, dim=1).max().unsqueeze(-1)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions



def compute_attribution_viz(response, label):
    lig = LayerIntegratedGradients(custom_forward, model.bert.embeddings)

    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(response, ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)
    print(input_ids.shape)
    output = predict(input_ids,
                     token_type_ids=token_type_ids,
                     position_ids=position_ids,
                     attention_mask=attention_mask)
    
    print(input_ids)
    print(input_ids[0])
    print(input_ids[0].detach())
    print(input_ids[0].detach().tolist())
    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)


    #pred = predict(input_ids)
    #torch.softmax(pred, dim = 1)




    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        n_steps=700,
                                        internal_batch_size=3,
                                        return_convergence_delta=True)

    attributions_sum = summarize_attributions(attributions)

    model_vis = viz.VisualizationDataRecord(
        attributions_sum,
        torch.softmax(output, dim=1)[0][0],
        torch.argmax(torch.softmax(output, dim=1)[0]),
        label,
        response,
        attributions_sum.sum(),
        all_tokens,
        delta)

    return viz.visualize_text([model_vis]).data


with open('../files/ft_responses.json', 'r') as f:
    testing_responses = json.load(f)

base_model_to_testing_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                    "Multilingual-MiniLM-L12-H384": '8',
                    "gpt2-xl": '4', "gpt-neo-125M": '6', "opt-350m": '1', "xlnet-base-cased": '7',
                    "codegen-350M-multi": '9'}

for base_model in base_model_to_testing_ft.keys():
    gt_ft_model = base_model_to_testing_ft[base_model]
    labels = []
    prompt_responses = []
    data_model_name = []
    # gather data
    for dataset, prompts in testing_responses.items():
        for prompt in prompts:
            for model_name, response in prompts[prompt].items():
                if base_model == model_name:
                    labels.append(1)
                    prompt_responses.append(response)
                    data_model_name.append(model_name)
                if len(model_name) < 3:
                    if gt_ft_model == model_name:
                        labels.append(1)
                    else:
                        labels.append(0)
                    data_model_name.append(model_name)
                    prompt_responses.append(response)

    # compute viz for response
    complete_visout = ''
    for idx in tqdm(range(len(prompt_responses[:22]))):
        response = prompt_responses[idx]
        label = labels[idx]
        #print(label)
        #print(response)
        visout = compute_attribution_viz(response, label)
        complete_visout += visout



    with open(f'../xai/{base_model}/{base_model}_attribution.html', 'w') as f:
        f.write(complete_visout)



