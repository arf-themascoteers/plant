import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class PlantDataset(Dataset):
    def __init__(self, is_train):
        self.is_train = 0
        if is_train:
            self.is_train = 1
        df = pd.read_csv("data/info_modified.csv")
        df = df[df["is_train"] == self.is_train]
        self.image_files = list(df["File.Name"])
        self.group = list(df["Group"])

    def __len__(self):
        return len(self.group)

    def __getitem__(self, idx):
        return torch.zeros((100,200))