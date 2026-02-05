import pandas as pd
import torch
from torch.utils.data import Dataset

class ReelsDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # interaction score (label)
        self.df["label"] = (
            0.2 * (self.df["watch_time"] > 3).astype(int) +
            1.0 * self.df["liked"] +
            2.0 * self.df["shared"] +
            3.0 * self.df["commented"]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            "user_id": torch.tensor(row.user_id),
            "reel_id": torch.tensor(row.reel_id),
            "watch_time": torch.tensor(row.watch_time, dtype=torch.float),
            "label": torch.tensor(row.label, dtype=torch.float),
        }
