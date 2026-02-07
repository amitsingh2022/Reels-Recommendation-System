import torch
import torch.nn as nn
import torch.nn.functional as F


class UserEncoder(nn.Module):
    def __init__(self, num_users, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_users + 1, embedding_dim)

    def forward(self, user_ids):
        return self.embedding(user_ids)


class ReelEncoder(nn.Module):
    def __init__(self, num_reels, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_reels + 1, embedding_dim)

    def forward(self, reel_ids):
        return self.embedding(reel_ids)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_reels, embedding_dim=64):
        super().__init__()
        self.user_encoder = UserEncoder(num_users, embedding_dim)
        self.reel_encoder = ReelEncoder(num_reels, embedding_dim)

    def forward(self, user_ids, reel_ids):
        user_emb = self.user_encoder(user_ids)
        reel_emb = self.reel_encoder(reel_ids)

        similarity = F.cosine_similarity(user_emb, reel_emb)
        return similarity
