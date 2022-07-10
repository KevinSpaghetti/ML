import os
import pandas as pd
from torch.utils.data import Dataset

class CLDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.pairs = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, idx):
        text = self.pairs['text'].iloc[idx]
        positive = self.pairs['positive'].iloc[idx]

        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            positive = self.target_transform(positive)
        return text, positive
