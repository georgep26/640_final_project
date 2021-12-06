"""
BERT Model for Classification with additional hidden layers
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

class BERTTextClassifierBaseDeep(nn.Module):

  def __init__(self, n_classes, dropout):
    super(BERTTextClassifierBaseDeep, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased', return_dict=False)
    self.drop = nn.Dropout(p=dropout)
    self.hidden = nn.Linear(self.bert.config.hidden_size, 256)
    self.out = nn.Linear(256, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    output = self.hidden(output)
    output = self.drop(pooled_output)
    output = self.hidden(output)
    output = self.drop(pooled_output)
    output = self.hidden(output)
    output = F.relu(self.drop(output))
    return self.out(output)