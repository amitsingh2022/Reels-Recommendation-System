import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class TwoTowerDataset(Dataset):
    def __init__(self, interactions_path):
        self.df = pd.read_csv(interactions_path)

        self.users = self.df["user_id"].values
        self.reels = self.df["reel_id"].values

        self.num_reels = self.df["reel_id"].nunique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.users[idx]
        positive_reel = self.reels[idx]

        # Negative sampling
        negative_reel = np.random.randint(1, self.num_reels + 1)

        return {
            "user_id": torch.tensor(user),
            "pos_reel_id": torch.tensor(positive_reel),
            "neg_reel_id": torch.tensor(negative_reel),
        }
