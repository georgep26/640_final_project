import model_exploration.models.bert_model_HW5_baseline as hw5_baseline_model
import model_exploration.datasets.bert_text_encoding_dataset as text_dataset
from torch import nn
from transformers import BertModel, BertTokenizer, AdamW

# Config for all BERT model runs

bert_baseline_model = {
    "model_name": "bert_baseline_hw5",
    "model_class": hw5_baseline_model.BERTTextClassifierBase,
    "optimizer": AdamW,
    "loss_fn": nn.CrossEntropyLoss(),
    "num_classes": 10,
    "num_epochs": 15,
    "dropout": 0.5
}

bert_baseline_data = {
    "train_data_loc": "data/preprocessed_data/data_train.csv",
    "test_data_loc": "data/preprocessed_data/data_test.csv",
    "dataset_type": text_dataset,
    "tokenizer": BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True),
    "text_col": ["headline"],
    "pred_col": "Q3 Theme1",
    "num_workers": 4,
    "max_len": 50,
    "batch_size": 128,
}