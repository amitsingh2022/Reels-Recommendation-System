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

class RankingDataset(Dataset):
    def __init__(self, interactions_path):
        self.df = pd.read_csv(interactions_path)

        # Build label score
        self.df["label"] = (
            0.2 * (self.df["watch_time"] > 3).astype(int)
            + 1.0 * self.df["liked"]
            + 2.0 * self.df["shared"]
            + 3.0 * self.df["commented"]
        )

        self.users = self.df["user_id"].values
        self.reels = self.df["reel_id"].values
        self.labels = self.df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.users[idx]),
            "reel_id": torch.tensor(self.reels[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }
