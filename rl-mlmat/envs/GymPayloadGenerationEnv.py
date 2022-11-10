import json
import random
import re
import gym
from datasets import Dataset
from transformers import BertModel, BertTokenizer, pipeline
import torch
import numpy as np
import json
import os
import time
import torch.nn as nn
from copy import deepcopy
from gym.spaces import Box, Discrete
from ray.rllib.env.env_context import EnvContext

from transformers import PreTrainedTokenizerFast

#tokeniser_file = os.path.abspath(os.path.join(os.path.abspath(__file__), '../bpe_tokenizer.json'))
#tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokeniser_file)
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5, device='cuda'):
        # add load base model classifier
        super(BertClassifier, self).__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, text):
        tokenised_response = self.tokenizer(text,
                                            padding='max_length', max_length=512, truncation=True,
                                            return_tensors="pt")
        input_id = tokenised_response.data['input_ids'].to(self.device)
        mask = tokenised_response.data['attention_mask'].to(self.device)

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer.argmax(dim=1).cpu().detach().numpy()[0]

    def save_model(self, base_model):
        save_file = os.path.abspath(os.path.join(__file__, '../../../files/ft_responses.json'))
        self.bert.save_pretrained(save_file)
        torch.save(self.linear.state_dict(), save_file + '/ft_layer.pt')

    def load_model(self, base_model):
        load_file = os.path.abspath(os.path.join(__file__, f'../../../files/bert_{base_model}_classifier'))
        self.bert.from_pretrained(load_file)
        self.linear.load_state_dict(torch.load(load_file + '/ft_layer.pt'))


base_model_names = ["bloom-350m", "DialoGPT-large", "distilgpt2", "gpt2", "Multilingual-MiniLM-L12-H384",
                    "gpt2-xl", "gpt-neo-125M", "opt-350m", "xlnet-base-cased", "codegen-350M-multi"]

def load_prompts() -> list:
    all_prompts = set()
    for root, dirs, files in os.walk(os.path.abspath(os.path.join(__file__, '../prompts'))):
        for file in files:
            if re.search('[^\d].csv', file) and 'ppl' not in file and file.split('.')[0] not in base_model_names:
                with open(os.path.join(root, file), 'r') as prompt_file:
                    for line in prompt_file.readlines():
                        all_prompts.add(line.split(' '))
                # dataset = os.path.join(root, file).split('/')[-2]
                # all_prompts[dataset] = list(prompts)
    return list(all_prompts)


class MLMATPayloadGeneration(gym.Env):
    def __init__(self, config):
        # self.action_space = spaces.Discrete(self.get_action_space(self.agent_name))
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(box_len,), dtype=np.float32)
        # self.reward_range = (float('-inf'), float('inf'))
        ft_models = config['ft_models']
        base_model = config['base_model']
        max_steps = config['max_steps']
        device = config['device']
        vocab_dir = config['vocab_dir']
        self.device = device

        # with open(os.path.abspath(os.path.join(__file__, '../../', vocab_dir)), 'r') as f:
        #    vocab = [word.strip() for word in f.read().split('\n')]
        self.ft_models = ft_models
        self.current_ft_name = random.choice(self.ft_models)
        device = 0 if torch.cuda.is_available() else -1
        self.ft_model_mapping = {
            '0': pipeline("text-generation", model="mrm8488/bloom-560m-finetuned-common_gen", device=device),
            '1': pipeline("text-generation", model="KoboldAI/OPT-350M-Nerys-v2", device=device),
            '2': pipeline("text-generation", model="LACAI/DialoGPT-large-PFG", device=device),
            '3': pipeline("text-generation", model="arminmehrabian/distilgpt2-finetuned-wikitext2-agu", device=device),
            '4': pipeline("text-generation", model="ethzanalytics/ai-msgbot-gpt2-XL", device=device),
            '5': pipeline("text-generation", model='dbmdz/german-gpt2', device=device),
            '6': pipeline("text-generation", model='wvangils/GPT-Neo-125m-Beatles-Lyrics-finetuned-newlyrics',
                          device=device),
            '7': pipeline("text-generation", model='textattack/xlnet-base-cased-imdb', device=device),
            '8': pipeline("text-generation",
                          model='veddm/paraphrase-multilingual-MiniLM-L12-v2-finetuned-DIT-10_epochs', device=device),
            '9': pipeline("text-generation", model="giulio98/codegen-350M-multi-xlcost", device=device),
        }
        self.ft = self.ft_model_mapping[self.current_ft_name]
        self.base_model = base_model
        self.max_steps = max_steps
        # self.classifier = BertClassifier() # add in base model name to load classifier
        with open(os.path.abspath(os.path.join(__file__, '../../', vocab_dir)), 'r') as f:
            vocab = [word.strip() for word in f.read().split('\n')]
        self.vocabulary = vocab + load_prompts()

        tokeniser_file = os.path.abspath(os.path.join(os.path.abspath(__file__), '../bpe_tokenizer.json'))
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokeniser_file)
        self.vocabulary = list(tokenizer.vocab.keys())

        # self.classifier.load_model(base_model)
        self.step_counter = 0
        # ft_response_file = os.path.abspath(os.path.join(__file__, '../../../files/ft_res.json'))
        # with open(ft_response_file) as f:
        #    prompt_set = json.load(f)
        self.prompt_responses = {}
        self.classifier = BertClassifier(device=self.device)
        #self.classifier.load_model(self.base_model)
        if torch.cuda.is_available():
            self.classifier = self.classifier.cuda()

        # for dataset, prompt in prompt_set.items():
        #    self.prompt_responses = {**prompt, **self.prompt_responses}
        #self.prompts = deepcopy(list(self.prompt_responses.keys()))
        #self.prompt = self.prompts[0]

        self.reward = 0
        # print(len(self.prompts))
        self.rewards = []
        self.feature_extracter = pipeline('feature-extraction', 'bert-base-multilingual-cased',
                                          device=0 if torch.cuda.is_available() else -1)
        self.base_model_to_training_ft = {"bloom-350m": '10', "DialoGPT-large": '12', "distilgpt2": '13', "gpt2": '15',
                                          "Multilingual-MiniLM-L12-H384": '18',
                                          "gpt2-xl": '14', "gpt-neo-125M": '16', "opt-350m": '11',
                                          "xlnet-base-cased": '17',
                                          "codegen-350M-multi": '19'}
        self.action_space = Discrete(len(self.vocabulary))
        # self.feature_extracter.model.config.
        self.observation_space = Box(-np.inf, np.inf, shape=(1, 770), dtype=np.float32)

    def step(self, action):
        # slice at index?
        self.step_counter += 1
        # self.prompt = self.prompts[action]
        self.prompt += ' ' + self.vocabulary[action]
        if self.step_counter == self.max_steps:
            st = time.time()
            ft_response = self.ft(self.prompt)[0]['generated_text']
            ft = time.time() - st
            print(f'ft inference: {ft}')
            st = time.time()
            classification = self.classifier(ft_response)
            ft = time.time() - st
            print(f'class inference: {ft}')
            if self.base_model_to_training_ft[self.base_model] == self.current_ft_name and int(classification) == 1:
                self.reward = 10
            elif self.base_model_to_training_ft[self.base_model] != self.current_ft_name and int(classification) == 0:
                self.reward = 10
            else:
                self.reward = -10
        else:
            self.reward = 0

        self.rewards.append(self.reward)

        # observation = {'prompt': self.prompt, 'base_obs': base_obs, 'ft_obs': ft_response, 'ft_model': self.ft_to_string()}
        # observation.extend([self.base(self.prompt)])
        st = time.time()
        prompt_features = self.feature_extracter(self.prompt)
        ft = time.time() - st
        print(f'extractor inference: {ft}')

        # prompt_features.extend(self.feature_extracter(self.prompt)[0][-1])
        prompt_features.append(int(self.current_ft_name))
        prompt_features.append(0)
        observation = np.array([prompt_features])

        if self.step_counter >= self.max_steps:
            done = True
        else:
            done = False

        info = {}
        return observation, self.reward, done, info

    def reset(self):
        self.prompt = random.choice(self.vocabulary)
        self.step_counter = 0
        # observation = {'prompt': self.prompt, 'base_obs': self.base(self.prompt)[0]['generated_text'], 'step_counter': str(self.step_counter)}
        # observation.extend([self.base(self.prompt)])
        # observation = {**observation, **{f"ft_obs_{list(self.ft).index(model)}": self.ft[model](self.prompt)[0]['generated_text'] for model in self.ft}}
        # observation.extend(ft_responses)
        self.current_ft_name = random.choice(self.ft_models)
        self.ft = self.ft_model_mapping[self.current_ft_name]
        # base_obs = self.prompt_responses[self.prompt][self.base_model]
        # ft_response = self.prompt_responses[self.prompt][self.ft]
        # observation = {'prompt': self.prompt, 'base_obs': base_obs, 'ft_obs': ft_response, 'ft_model': self.ft_to_string()}
        prompt_features = self.feature_extracter(self.prompt)[0][-1]

        # prompt_features.extend(self.feature_extracter(self.prompt)[0][-1])
        prompt_features.append(int(self.current_ft_name))  # ft model
        prompt_features.append(0)  # base model
        # obs_1 = np.array()
        observation = np.array([prompt_features])  # base model
        self.reward = 0
        # print(self.rewards)
        self.rewards = []
        return observation
