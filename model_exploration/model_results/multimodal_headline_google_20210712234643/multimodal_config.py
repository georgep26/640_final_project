from model_exploration.models.bert_model_HW5_baseline import BERTTextClassifierBase
from model_exploration.models.bert_model_deep import BERTTextClassifierBaseDeep
from model_exploration.models.multimodal import MultimodalClassifier

import model_exploration.datasets.bert_text_encoding_dataset as text_dataset
import model_exploration.datasets.multimodal_dataset as multimodal_dataset
from torch import nn
from transformers import BertModel, BertTokenizer, AdamW

# Config for all BERT model runs

model_config = {
    "model_name": "multimodal_headline_google",
    "model_class": MultimodalClassifier,
    "optimizer": AdamW,
    "loss_fn": nn.CrossEntropyLoss(),
    "num_classes": 10,
    "num_epochs": 15,
    "dropout": 0.5,
    "bert_model_path": "model_exploration/model_results/x_bert_baseline_hw5_headline_google_15_200_20210712222148/best_model_state.bin"
}

data_config = {
    "train_data_loc": "data/preprocessed_data/data_train.csv",
    "test_data_loc": "data/preprocessed_data/data_test.csv",
    "dataset_type": multimodal_dataset,
    "tokenizer": BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True),
    "text_col": ["headline", "google_visual_api_web_entities_detection_on_lead_image"],
    "pred_col": "Q3 Theme1",
    "num_workers": 4,
    "max_len": 200,
    "batch_size": 16,
}