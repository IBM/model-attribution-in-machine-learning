import json
import random

import gym
from datasets import Dataset
from transformers import BertModel, BertTokenizer, pipeline
import torch
import numpy as np
import json
import os
import torch.nn as nn
from copy import deepcopy
from gym.spaces import Box, Discrete
from ray.rllib.env.env_context import EnvContext


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        # add load base model classifier
        super(BertClassifier, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer.argmax(dim=1).cpu().detach().numpy()[0]

    def save_model(self, base_model):
        save_file = os.path.abspath(os.path.join(__file__, '../../../files/ft_responses.json'))
        self.bert.save_pretrained(save_file)
        torch.save(self.linear.state_dict(), save_file+'/ft_layer.pt')

    def load_model(self, base_model):
        load_file = os.path.abspath(os.path.join(__file__, f'../../../files/split_bert_{base_model}_classifier'))
        self.bert.from_pretrained(load_file)
        self.linear.load_state_dict(torch.load(load_file + '/ft_layer.pt'))




class MLMATPayloadSelection(gym.Env):
    def __init__(self, config):
        # self.action_space = spaces.Discrete(self.get_action_space(self.agent_name))
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(box_len,), dtype=np.float32)
        # self.reward_range = (float('-inf'), float('inf'))
        ft_models = config['ft_models']
        base_model = config['base_model']
        max_steps = config['max_steps']
        device = config['device']
        if 'return_text' in config.keys():
            self.return_text = config['return_text']
        else:
            self.return_text = False
        self.device = device
        #with open(os.path.abspath(os.path.join(__file__, '../../', vocab_dir)), 'r') as f:
        #    vocab = [word.strip() for word in f.read().split('\n')]
        self.ft_models = ft_models
        self.ft = random.choice(self.ft_models)
        self.base_model = base_model
        self.max_steps = max_steps
        #self.classifier = BertClassifier() # add in base model name to load classifier

        #self.classifier.load_model(base_model)
        self.step_counter = 0
        if  config['0-9'] == False:
            ft_response_file = os.path.abspath(os.path.join(__file__, '../../../files/training_responses.json'))
            self.base_model_to_training_ft = {"bloom-350m": '10', "DialoGPT-large": '12', "distilgpt2": '13', "gpt2": '15',
                                          "Multilingual-MiniLM-L12-H384": '18',
                                          "gpt2-xl": '14', "gpt-neo-125M": '16', "opt-350m": '11', "xlnet-base-cased": '17',
                                          "codegen-350M-multi": '19'}
        else:
            self.base_model_to_training_ft = {"bloom-350m": '0', "DialoGPT-large": '2', "distilgpt2": '3', "gpt2": '5',
                                          "Multilingual-MiniLM-L12-H384": '8',
                                          "gpt2-xl": '4', "gpt-neo-125M": '6', "opt-350m": '1', "xlnet-base-cased": '7',
                                          "codegen-350M-multi": '9'}
            ft_response_file = os.path.abspath(os.path.join(__file__, '../../../files/ft_responses.json'))
        with open(ft_response_file) as f:
            prompt_set = json.load(f)
        self.prompt_responses = {}
        self.classifier = BertClassifier()
        #self.classifier.load_model(self.base_model)
        load_file = os.path.abspath(os.path.join(__file__, f'../../../files/bert_base_{base_model}_classifier'))
        self.classifier.load_state_dict(torch.load(f'{load_file}/model.pth'))

        if 'cuda' in self.device:
            self.classifier = self.classifier.cuda()
        for dataset, prompt in prompt_set.items():
            self.prompt_responses = {**prompt, **self.prompt_responses}
        self.prompts = deepcopy(list(self.prompt_responses.keys()))
        self.prompt = self.prompts[0]
        self.reward = 0
        #print(len(self.prompts))
        self.rewards = []
        self.feature_extracter = pipeline('feature-extraction', 'bert-base-multilingual-cased', device=0 if device == 'cuda' else -1)
        self.action_space = Discrete(len(self.prompts))
        #self.feature_extracter.model.config.
        self.observation_space = Box(-np.inf, np.inf, shape=(1, 770), dtype=np.float32)

    def step(self, action):
        # slice at index?
        self.step_counter += 1
        if self.return_text:
            self.prompt = action
        else:
            self.prompt = self.prompts[action]
        
        #base_obs = self.prompt_responses[self.prompt][self.base_model]
        #print(self.prompt)
        #print(self.ft)
        #print(len(self.prompt_responses))
        #print(list(self.prompt_responses.keys()))
        #print(type(list(self.prompt_responses.keys())[0]))
        ft_response = self.prompt_responses[self.prompt][str(self.ft)]
        classification = self.classifier(ft_response)
        if self.base_model == self.ft and int(classification) == 1:
            self.reward += 1
        elif self.base_model != self.ft and int(classification) == 0:
            self.reward += 1
        else:
            self.reward += -10
        self.rewards.append(self.reward)


        #observation = {'prompt': self.prompt, 'base_obs': base_obs, 'ft_obs': ft_response, 'ft_model': self.ft_to_string()}
        # observation.extend([self.base(self.prompt)])
        if not self.return_text:
            try:
                prompt_features = self.feature_extracter(ft_response[len(self.prompt):])[0][-1]
            except:
                prompt_features = self.feature_extracter(ft_response[len(self.prompt):400])[0][-1]
            #prompt_features.extend(self.feature_extracter(self.prompt)[0][-1])
            prompt_features.append(int(classification))
            prompt_features.append(0)
            #obs_1 = np.array()
            observation = np.array([prompt_features])
        #print(len(prompt_features))
        #print(len(ft_features))
        #print(observation)
        #print(type(observation[0][0]))
        #observation = np.array([[self.ft, self.feature_extracter(self.prompt)], [action, self.feature_extracter(ft_response)]])
        #observation = {**observation, **{f"ft_obs_{list(self.ft).index(model)}": self.ft[model](self.prompt)[0]['generated_text'] for model in self.ft}}
        # observation.extend(ft_responses)
        #self.compute_ppl
        #reward = self.compute_ppl(self.base.model, self.base.tokenizer, observation['ft_obs_0'][:388])
        #reward = np.absolute(reward)

        if self.step_counter >= self.max_steps:
            done = True
        else:
            done = False

        info = {'classifier_out': classification, 'ft': self.ft, 'gt': self.base_model_to_training_ft[self.base_model]}
        if self.return_text:
            response = ft_response[len(self.prompt):] if len(ft_response[len(self.prompt):]) > 0 else 'no response'
            return [{'prompt':self.prompt,'ft_obs':response}], [self.reward], [done], [info]
        else:
            return observation, self.reward, done, info


    def reset(self):
        self.prompt = self.prompts[0]
        self.step_counter = 0
        #observation = {'prompt': self.prompt, 'base_obs': self.base(self.prompt)[0]['generated_text'], 'step_counter': str(self.step_counter)}
        # observation.extend([self.base(self.prompt)])
        #observation = {**observation, **{f"ft_obs_{list(self.ft).index(model)}": self.ft[model](self.prompt)[0]['generated_text'] for model in self.ft}}
        # observation.extend(ft_responses)
        self.ft = random.choice(self.ft_models)
        #base_obs = self.prompt_responses[self.prompt][self.base_model]
        ft_response = self.prompt_responses[self.prompt][self.ft]
        #observation = {'prompt': self.prompt, 'base_obs': base_obs, 'ft_obs': ft_response, 'ft_model': self.ft_to_string()}
        prompt_features = self.feature_extracter(ft_response[len(self.prompt):])[0][-1]
        #prompt_features.extend(self.feature_extracter(self.prompt)[0][-1])
        classification = self.classifier(ft_response)
        prompt_features.append(int(classification))
        prompt_features.append(0)
        #obs_1 = np.array()
        observation = np.array([prompt_features])
        self.reward = 0
        #print(self.rewards)
        self.rewards = []
        if self.return_text:
            response = ft_response[len(self.prompt):] if len(ft_response[len(self.prompt):]) > 0 else 'no response'
            return [{'prompt':self.prompt,'ft_obs':response}]
        else:
            return observation
