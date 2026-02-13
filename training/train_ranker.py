import torch
from torch.utils.data import DataLoader
from training.dataset import RankingDataset
from app.models import TwoTowerModel, Ranker
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
EPOCHS = 5
EMBED_DIM = 64

dataset = RankingDataset("data/interactions.csv")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

df = pd.read_csv("data/interactions.csv")
num_users = df["user_id"].nunique()
num_reels = df["reel_id"].nunique()

# Load trained two-tower model
two_tower = TwoTowerModel(num_users, num_reels, EMBED_DIM)
two_tower.load_state_dict(torch.load("training/two_tower_model.pth"))
two_tower.to(DEVICE)
two_tower.eval()

ranker = Ranker(EMBED_DIM).to(DEVICE)
optimizer = torch.optim.Adam(ranker.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(EPOCHS):
    total_loss = 0

    for batch in loader:
        user = batch["user_id"].to(DEVICE)
        reel = batch["reel_id"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        with torch.no_grad():
            user_emb = two_tower.user_encoder(user)
            reel_emb = two_tower.reel_encoder(reel)

        preds = ranker(user_emb, reel_emb)

        loss = loss_fn(preds, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(ranker.state_dict(), "training/ranker_model.pth")
print("✅ Ranker saved")
