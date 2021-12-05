"""
BERT Model for Classification from HW 5
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import transformers 
from transformers import BertModel, BertTokenizer, AdamW

class BERTTextClassifierBase(nn.Module):

  def __init__(self, n_classes):
    super(BERTTextClassifierBase, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased', return_dict=False)
    self.drop = nn.Dropout(p=0.5)
    self.hidden = nn.Linear(self.bert.config.hidden_size, 256)
    self.out = nn.Linear(256, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    output = self.hidden(output)
    output = F.relu(self.drop(output))
    return self.out(output)


