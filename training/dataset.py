from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DataStats:
    num_users: int
    num_reels: int


def load_interactions(interactions_path: str) -> pd.DataFrame:
    df = pd.read_csv(interactions_path)

    required_cols = {
        "user_id",
        "reel_id",
        "watch_time",
        "liked",
        "shared",
        "commented",
        "timestamp",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep all ids as int for stable embedding/index lookup.
    df["user_id"] = df["user_id"].astype(int)
    df["reel_id"] = df["reel_id"].astype(int)
    return df


def compute_label(df: pd.DataFrame) -> pd.Series:
    return (
        0.2 * (df["watch_time"] > 3).astype(float)
        + 1.0 * df["liked"].astype(float)
        + 2.0 * df["shared"].astype(float)
        + 3.0 * df["commented"].astype(float)
    )


def get_data_stats(df: pd.DataFrame) -> DataStats:
    return DataStats(
        num_users=int(df["user_id"].max()),
        num_reels=int(df["reel_id"].max()),
    )


def build_user_positive_items(df: pd.DataFrame) -> Dict[int, Set[int]]:
    return df.groupby("user_id")["reel_id"].apply(lambda x: set(x.tolist())).to_dict()


def split_interactions(
    interactions_path: str,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal per-user split:
    - Sort by timestamp for each user.
    - Hold out the most recent interactions as validation.
    - Keep at least one interaction in train for users with >=2 interactions.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1)")

    df = load_interactions(interactions_path)
    rng = np.random.default_rng(random_state)

    train_parts = []
    val_parts = []

    for _, group in df.groupby("user_id", sort=False):
        g = group.sort_values("timestamp")
        n = len(g)

        if n < 2:
            train_parts.append(g)
            continue

        val_count = max(1, int(round(n * val_ratio)))
        val_count = min(val_count, n - 1)

        train_parts.append(g.iloc[:-val_count])
        val_parts.append(g.iloc[-val_count:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = (
        pd.concat(val_parts, ignore_index=True)
        if val_parts
        else df.sample(0, random_state=int(rng.integers(0, 1_000_000)))
    )

    return train_df, val_df


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        all_reel_ids: Optional[Iterable[int]] = None,
        user_positive_items: Optional[Dict[int, Set[int]]] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.users = self.df["user_id"].astype(int).values
        self.reels = self.df["reel_id"].astype(int).values

        if all_reel_ids is None:
            all_reel_ids = sorted(self.df["reel_id"].astype(int).unique().tolist())
        self.all_reel_ids = np.array(list(all_reel_ids), dtype=np.int64)

        if self.all_reel_ids.size == 0:
            raise ValueError("all_reel_ids cannot be empty")

        self.user_positive_items = (
            user_positive_items
            if user_positive_items is not None
            else build_user_positive_items(self.df)
        )

    def __len__(self) -> int:
        return len(self.df)

    def _sample_negative(self, user_id: int, pos_reel_id: int) -> int:
        positives = self.user_positive_items.get(int(user_id), set())

        # Fast random tries first, then deterministic fallback when user positives are dense.
        for _ in range(10):
            candidate = int(np.random.choice(self.all_reel_ids))
            if candidate not in positives:
                return candidate

        allowed = np.setdiff1d(self.all_reel_ids, np.array(list(positives), dtype=np.int64))
        if allowed.size == 0:
            # Degenerate case: user interacted with every known reel.
            # Use padding id 0 as a guaranteed non-positive negative.
            return 0
        return int(np.random.choice(allowed))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user = int(self.users[idx])
        positive_reel = int(self.reels[idx])
        negative_reel = self._sample_negative(user, positive_reel)

        return {
            "user_id": torch.tensor(user, dtype=torch.long),
            "pos_reel_id": torch.tensor(positive_reel, dtype=torch.long),
            "neg_reel_id": torch.tensor(negative_reel, dtype=torch.long),
        }


class RankingDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        self.df["label"] = compute_label(self.df)

        self.users = self.df["user_id"].astype(int).values
        self.reels = self.df["reel_id"].astype(int).values
        self.labels = self.df["label"].astype(float).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "user_id": torch.tensor(int(self.users[idx]), dtype=torch.long),
            "reel_id": torch.tensor(int(self.reels[idx]), dtype=torch.long),
            "label": torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        }
