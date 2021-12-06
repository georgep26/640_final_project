import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import transformers
from transformers import BertModel, BertTokenizer, AdamW

from model_exploration.image_classification import ImageModel
from model_exploration.models.bert_model_HW5_baseline import BERTTextClassifierBase


class BERTTextModelMultimodal(BERTTextClassifierBase):
    def __init__(self, n_classes, freeze_weights):
        super(BERTTextModelMultimodal, self).__init__(n_classes)

        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        output = self.hidden(output)
        # we are passing hidden layer weights
        # output = F.relu(self.drop(output))
        # output = self.out(output)
        return output


class ImageModelMultiMode(ImageModel):
    def __init__(self, state_dict_path, freeze_weights, base_model, num_labels, dropout=0.5):
        super(ImageModelMultiMode, self).__init__(base_model, num_labels, dropout)
        self.load_state_dict(torch.load(state_dict_path))

        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        # we return this layer representation - a 512 len vector
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


class MultimodalClassifier(nn.Module):
    def __init__(self, n_classes, base_image_model, image_model_path, dropout=0.5):
        super(MultimodalClassifier, self).__init__()
        self.image_model = ImageModelMultiMode(num_labels=n_classes,
                                               base_model=base_image_model,
                                               dropout=dropout,
                                               state_dict_path=image_model_path,
                                               freeze_weights=True)
        self.bert = BERTTextModelMultimodal(n_classes=n_classes,
                                            freeze_weights=True)

        # might be able to config these
        self.fc1 = nn.Linear(256 + 512, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, image):
        bert_output = self.bert(input_ids, attention_mask)
        img_output = self.image_model(image)

        concat = torch.cat((bert_output, img_output), 0)
        output = self.drop(concat)
        output = self.fc1(output)
        output = self.drop(output)
        output = F.relu(output)
        output = self.fc2(output)

        return output



