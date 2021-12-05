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
import torch.optim as optim
import torchvision
import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io
from collections import defaultdict
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class image_dataset(Dataset):
    # TODO: decide on transforms
    def __init__(self, df, image_dir, label_col, image_id_col, train_data):
        self.df = df
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
        image = self.transform(image)
        # one_hot = torch.zeros(self.num_labels)
        # one_hot[label_raw - 1] = 1

        return {
            "image": image,
            # "label": one_hot
            "label": label_raw - 1 # TODO I think CEL uses raw value like this (-1 to move indexed from zero)
        }

    def get_image_fname(self, item):
        return f"{item}.jpg"


def get_data_loader(ds, batch_size):
    # TODO had another arg num_workers in docs see what that is
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


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


def eval_model(model, data_loader, loss_func, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  # just inference, no grad reduces memory load
  with torch.no_grad():
    # for batch in data loader
    for entry in data_loader:

      img_batch = entry['image'].to(device)
      label_batch = entry['label'].to(device)

      # train model
      outputs = model(img_batch)
      _, preds = torch.max(outputs, dim=1)
      loss = loss_func(outputs, label_batch)

      correct_predictions += torch.sum(preds == label_batch)
      losses.append(loss.item())

  # return acc
  return correct_predictions.double() / n_examples, np.mean(losses)


def train_model(base_model, train_df, val_df, num_epochs, learning_rate, train_ds_config, val_ds_config, model_config, loader_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    model = ImageModel(base_model, **model_config)
    train_ds = image_dataset(train_df, **train_ds_config)
    val_ds = image_dataset(val_df, **val_ds_config)
    train_data_loader = get_data_loader(train_ds, **loader_config)
    val_data_loader = get_data_loader(val_ds, **loader_config)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss().to(device)
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_function, optimizer, device, len(train_df))

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, val_data_loader, loss_function, device, len(val_df))

        print(f'Val   loss {val_loss} accuracy {val_acc}\n')

        # log history
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        # checkpoint best performing model so far
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc


def train_epoch(model, data_loader, loss_func, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_pred = 0

    for entry in data_loader:
        # vars
        img_batch = entry['image'].to(device)
        label_batch = entry['label'].to(device)

        optimizer.zero_grad()
        outputs = model(img_batch)
        loss = loss_func(outputs, label_batch) # TODO make sure we use the right output here sigmoid or no
        # nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0) # TODO what is this do we need it
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        x, preds = torch.max(outputs, dim=1)
        correct_pred += torch.sum(preds == label_batch) # TODO check that this is doing what I think
        losses.append(loss.item())

    return correct_pred.double() / n_examples, np.mean(losses)




if __name__ == "__main__":
    df = pd.read_csv(constants.data_paths['preprocessed_train_data'])
    train_df, val_df = train_test_split(df, test_size=.2)
    train_df = train_df.sample(64)
    train_ds = image_dataset(train_df, **constants.dataset_config['train_image_dataset'])
    val_ds = image_dataset(val_df, **constants.dataset_config['val_image_dataset'])
    res_mod = models.resnet18(pretrained=True)
    model = ImageModel(res_mod, 9)

    train_model(res_mod, train_df, val_df, **constants.train_config)

