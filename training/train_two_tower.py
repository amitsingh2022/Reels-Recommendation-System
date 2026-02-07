import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from training.dataset import TwoTowerDataset
from app.models import TwoTowerModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 512
EMBED_DIM = 64

dataset = TwoTowerDataset("data/interactions.csv")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

num_users = dataset.df["user_id"].nunique()
num_reels = dataset.df["reel_id"].nunique()

model = TwoTowerModel(num_users, num_reels, EMBED_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    total_loss = 0

    for batch in loader:
        user = batch["user_id"].to(DEVICE)
        pos = batch["pos_reel_id"].to(DEVICE)
        neg = batch["neg_reel_id"].to(DEVICE)

        pos_score = model(user, pos)
        neg_score = model(user, neg)

        loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "training/two_tower_model.pth")
print("✅ Model saved")
