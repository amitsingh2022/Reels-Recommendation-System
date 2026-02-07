import torch
import numpy as np
from app.models import TwoTowerModel
import pandas as pd

EMBED_DIM = 64

df = pd.read_csv("data/interactions.csv")
num_users = df["user_id"].nunique()
num_reels = df["reel_id"].nunique()

model = TwoTowerModel(num_users, num_reels, EMBED_DIM)
model.load_state_dict(torch.load("training/two_tower_model.pth"))
model.eval()

user_embeddings = model.user_encoder.embedding.weight.detach().numpy()
reel_embeddings = model.reel_encoder.embedding.weight.detach().numpy()

np.save("data/processed/user_embeddings.npy", user_embeddings)
np.save("data/processed/reel_embeddings.npy", reel_embeddings)

print("✅ Embeddings exported")
