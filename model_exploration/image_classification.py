"""
Unimodal image classification - transfer learn using some
"""

import os
import sys
import time
import json
from datetime import datetime

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
from sklearn.model_selection import KFold
from training_plots import create_training_plot


def build_transfroms(transform_config):
    tfm_list = []
    tfm_list.append(transforms.ToPILImage())
    tfm_list.append(transforms.Resize(transform_config['image_shape']))

    if "horizontal_flip" in transform_config.keys():
        tfm_list.append(transforms.RandomHorizontalFlip(**transform_config['horizontal_flip']))

    if "rotation" in transform_config.keys():
        tfm_list.append(transforms.RandomRotation(**transform_config['rotation']))

    tfm_list.append(transforms.ToTensor())

    return transforms.Compose(tfm_list)


class image_dataset(Dataset):
    # TODO: decide on transforms
    def __init__(self, df, transforms, image_dir, label_col, image_id_col):
        self.df = df
        self.num_labels = len(self.df[label_col].unique())
        self.image_dir = image_dir
        self.label_col = label_col
        self.image_id_col = image_id_col
        self.transform = transforms

        # if train_data:
        #     # train data transforms
        #     self.transform = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.Resize((224, 224)),
        #         # transforms.RandomHorizontalFlip(p=0.5),
        #         # transforms.RandomRotation(degrees=15),
        #         transforms.ToTensor()
        #         # transforms.Normalize()
        #     ])
        #     pass
        # else:
        #     # inference transforms
        #     self.transform = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor()
        #     ])
        #     pass

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


def train_model(base_model, train_df, val_df, output_dir, transform_config, num_epochs, learning_rate, train_ds_config, val_ds_config, model_config,
                loader_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_writer.print(f"DEVICE: {device}")

    model = ImageModel(base_model, **model_config)
    model.to(device)

    train_ds = image_dataset(train_df, build_transfroms(transform_config['train']), **train_ds_config)
    val_ds = image_dataset(val_df, build_transfroms(transform_config['inference']), **val_ds_config)
    train_data_loader = get_data_loader(train_ds, **loader_config)
    val_data_loader = get_data_loader(val_ds, **loader_config)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss().to(device)
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        config_writer.print(f"Epoch {epoch+1} of {num_epochs}")

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_function, optimizer, device, len(train_df))

        config_writer.print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, val_data_loader, loss_function, device, len(val_df))

        config_writer.print(f'Val   loss {val_loss} accuracy {val_acc}')

        # log history
        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss.item())
        history['val_acc'].append(val_acc.item())
        history['val_loss'].append(val_loss.item())

        # checkpoint best performing model so far
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_state.bin'))
            best_accuracy = val_acc

        config_writer.print(f"Epoch elapsed time: {time.time() - start_time}\n")

    return history


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


class ConfigWriter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.config = {}
        self.out_string = ""

    def add(self, title, dictionary):
        self.config[title] = dictionary

    def print(self, print_string):
        self.out_string = self.out_string + "\n" + print_string
        self.write()

    def write(self):
        with open(os.path.join(self.output_path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        with open(os.path.join(self.output_path, "output.txt"), "w") as f:
            f.write(self.out_string)


def k_fold_cross_val(df, train_func, output_dir, constants):
    # train_df, val_df = train_test_split(df, test_size=.2)
    master_history = defaultdict(list)
    kf = KFold(n_splits=4, random_state=None, shuffle=False)
    count = 0
    for train_index, val_index in kf.split(df):
        count += 1
        config_writer.print(f"Fold number {count} of 4")
        train_df = df.iloc[train_index]
        train_df.sample(frac=constants.dataset_config['train_downsample_frac'])
        val_df = df.iloc[val_index]
        train_df = train_df.sample(frac=constants.dataset_config['train_downsample_frac'])

        if constants.model_base["model"] == "resnet18":
            res_mod = models.resnet18(pretrained=True)
        elif constants.model_base["model"] == "resnet50":
            res_mod = models.resnet50(pretrained=True)
        elif constants.model_base["model"] == "resnet101":
            res_mod = models.resnet50(pretrained=True)
        else:
            raise("invalid model selection")

        history = train_func(res_mod, train_df, val_df, output_dir, constants.transform_config, **constants.train_config)
        create_training_plot(history, output_dir, title_modifier=count)
        master_history['train_acc'].append(history['train_acc'])
        master_history['train_loss'].append(history['train_loss'])
        master_history['val_acc'].append(max(history['val_acc']))
        master_history['val_loss'].append(history['val_loss'])

    return np.mean(master_history['val_acc'])


if __name__ == "__main__":
    # TODO
    # * completely config driven - transforms, layers etc, freeze layers?
    # * done! save config into dir
    # * done! informative prints for output on scc time elapsed etc
    """
    validation, kfold, transforms for images
    csv going
    """

    csv_output = defaultdict(list)
    csv_output['description'].append(constants.description)

    timestamp = datetime.now().strftime("%Y%d%m%H%M%S")
    output_dir_name = f"unimodal_image_{timestamp}"
    csv_output['execution'].append(output_dir_name)
    output_dir = os.path.join(constants.data_dirs['model_results'], output_dir_name)
    os.mkdir(output_dir)
    config_writer = ConfigWriter(output_dir)

    df = pd.read_csv(constants.data_paths['preprocessed_train_data'])

    acc = k_fold_cross_val(df, train_model, output_dir, constants)
    csv_output['4-fold accuracy'].append(acc)

    config_writer.add("dataset_config", constants.dataset_config)
    config_writer.add("model_config", constants.model_config)
    config_writer.add("loader_config", constants.loader_config)
    config_writer.add("train_config", constants.train_config)
    config_writer.add("transform_config", constants.transform_config)
    config_writer.write()

    pd.DataFrame.from_dict(csv_output).to_csv(os.path.join(output_dir, "summary_csv.csv"), index=False)

