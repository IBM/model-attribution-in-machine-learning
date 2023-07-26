from transformers import BertModel
import torch.nn as nn

class Attributor(nn.Module):

    def __init__(self, dropout=0.5):

        super(Attributor, self).__init__()

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

    def save_model(self, base_model):
        self.bert.save_pretrained(f'../files/bert_base_{base_model}_classifier')

    def load_model(self, base_model):
        self.bert.from_pretrained(f'../files/bert_base_{base_model}_classifier')

