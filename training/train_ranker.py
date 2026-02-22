import torch
from torch.utils.data import DataLoader

from app.models import Ranker, TwoTowerModel
from training.dataset import RankingDataset, get_data_stats, load_interactions, split_interactions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
EPOCHS = 5
EMBED_DIM = 64
INTERACTIONS_PATH = "data/interactions.csv"
TWO_TOWER_PATH = "training/two_tower_model.pth"
RANKER_PATH = "training/ranker_model.pth"


def main() -> None:
    full_df = load_interactions(INTERACTIONS_PATH)
    train_df, val_df = split_interactions(INTERACTIONS_PATH, val_ratio=0.2)
    stats = get_data_stats(full_df)

    train_dataset = RankingDataset(train_df)
    val_dataset = RankingDataset(val_df if len(val_df) else train_df.iloc[:1])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    two_tower = TwoTowerModel(stats.num_users, stats.num_reels, EMBED_DIM)
    two_tower.load_state_dict(torch.load(TWO_TOWER_PATH, map_location="cpu"))
    two_tower.to(DEVICE)
    two_tower.eval()

    ranker = Ranker(EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(ranker.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        ranker.train()
        train_loss = 0.0

        for batch in train_loader:
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

            train_loss += loss.item()

        ranker.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                user = batch["user_id"].to(DEVICE)
                reel = batch["reel_id"].to(DEVICE)
                label = batch["label"].to(DEVICE)

                user_emb = two_tower.user_encoder(user)
                reel_emb = two_tower.reel_encoder(reel)
                preds = ranker(user_emb, reel_emb)
                val_loss += loss_fn(preds, label).item()

        print(
            f"Epoch {epoch + 1} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    torch.save(ranker.state_dict(), RANKER_PATH)
    print(f"Saved ranker model to {RANKER_PATH}")


if __name__ == "__main__":
    main()
