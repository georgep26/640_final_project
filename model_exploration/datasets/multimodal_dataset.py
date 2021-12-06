import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import io
import os
import transforms


class MultimodalDataset(Dataset):

    def __init__(self, df, transforms, image_dir, label_col, image_id_col, text, classification, tokenizer, max_len):
        super().__init__()
        # Text data
        self.text = text.to_numpy()
        self.classification = classification.to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df = df
        self.num_labels = len(self.df[label_col].unique())
        # Image data 
        self.image_dir = image_dir
        self.label_col = label_col
        self.image_id_col = image_id_col
        self.transform = transforms

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        X = str(self.text[item])
        Y = self.classification[item]

        encoding = self.tokenizer.encode_plus(
        X,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding='max_length', # Changed this from 'pad_to_max_length=True'
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
        )

        if torch.is_tensor(item):
            item = item.tolist()

        # Image data
        label_raw = self.df.iloc[item, self.df.columns.get_loc(self.label_col)]
        image_id = self.df.iloc[item, self.df.columns.get_loc(self.image_id_col)]
        image = io.imread(os.path.join(self.image_dir, self.get_image_fname(image_id)))
        image = self.transform(image)

        return {
        'text': X,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        "image": image,
        "label": label_raw - 1,
        'classification': torch.tensor(Y, dtype=torch.long)
        }

def create_data_loader(df, transform_config, image_config, x_cols, y_col, tokenizer, max_len, batch_size, num_workers):
    '''
    Creates a DataLoader object
    '''
    # Additional logic for handling multiple input colums - all strings are concatinated
    df["X"] = ""
    for col in x_cols:
        df["X"] = df["X"] + " " + df[col]
    
    # Create DataSet
    ds = MultimodalDataset(text=df["X"], transformers=build_transfroms(transform_config['train']), image_dir=image_config["image_dir"], label_col=image_config["label_col"], image_id_col=image_config["image_id_col"], classification=df[y_col], tokenizer=tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

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