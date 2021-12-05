"""
Unimodal image classification - transfer learn using some
"""

import os
import sys

import numpy as np

sys.path.append(os.getcwd())
import config.constants as constants
import torch
import torch.nn as nn
import torchvision
import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from skimage import io
import matplotlib.pyplot as plt


class image_dataset(Dataset):
    # TODO: decide on transforms
    def __init__(self, data_path, image_dir, label_col, image_id_col, train_data):
        self.df = pd.read_csv(data_path)
        self.num_labels = len(self.df[label_col].unique())
        self.image_dir = image_dir
        self.label_col = label_col
        self.image_id_col = image_id_col

        if train_data:
            # train data transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(degrees=15),
                transforms.ToTensor()
                # transforms.Normalize()
            ])
            pass
        else:
            # inference transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            pass

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        label_raw = self.df.iloc[item, self.df.columns.get_loc(self.label_col)]
        image_id = self.df.iloc[item, self.df.columns.get_loc(self.image_id_col)]
        image = io.imread(os.path.join(self.image_dir, self.get_image_fname(image_id)))
        print(image.shape)
        image = self.transform(image)
        one_hot = torch.zeros(self.num_labels)
        one_hot[label_raw - 1] = 1

        return {
            "image": image,
            "label": one_hot
        }

    def get_image_fname(self, item):
        return f"{item}.jpg"


class ImageModel(nn.Module):
    def __init__(self, base_model, num_labels, dropout=0.5):
        super(ImageModel, self).__init__()
        self.base_model = base_model
        # I know the last layer in resnet is fc by looking at code, this may not generalize with other models
        # reset the last layer of resnet to a fresh fc layer to learn new classification
        # TODO can add extra layer here
        # TODO dropout? dont think I can add before last fc layer
        # TODO how about freezing layers?
        last_layer_in_size = base_model.fc.in_features
        self.base_model.fc = nn.Linear(last_layer_in_size, num_labels)
        self.drop = nn.Dropout(dropout)

    def forward(self, image):
        x = self.base_model(image)
        return x


def train_model():
    pass


def train_epoch(model, data_loader, loss_func, optimizer, divice, sheduler, n_examples):
    model = model.train()
    losses = []

    for entry in data_loader:
        # vars
        img = entry['image']

        outputs = model(img)

    pass



res_mod = models.resnet18(pretrained=True)

if __name__ == "__main__":
    ds = image_dataset(**constants.dataset_config['train_image_dataset'])

    print(len(ds))

    # testing dataset
    for image_num in range(5):
        plt.imshow(ds[image_num]['image'].permute(1, 2, 0))
        print(constants.q3_theme_labels[np.argmax(ds[image_num]['label']).item()+1])
        plt.show()
