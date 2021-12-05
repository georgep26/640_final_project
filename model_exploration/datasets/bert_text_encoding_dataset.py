import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TestDataset(Dataset):

    def __init__(self, text, classification, tokenizer, max_len):
        super().__init__()
        self.text = text.to_numpy()
        self.classification = classification.to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len

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

        return {
        'text': X,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'classification': torch.tensor(Y, dtype=torch.long)
        }

def create_data_loader(data_loc, x_col, y_col, tokenizer, max_len, batch_size, num_workers):
    df = pd.read_csv(data_loc)
    ds = TestDataset(text=df[x_col], classification=df[y_col], tokenizer=tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
