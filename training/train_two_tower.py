import torch
from torch.utils.data import DataLoader

from app.models import TwoTowerModel
from training.dataset import (
    TwoTowerDataset,
    build_user_positive_items,
    get_data_stats,
    load_interactions,
    split_interactions,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 512
EMBED_DIM = 64
INTERACTIONS_PATH = "data/interactions.csv"
MODEL_PATH = "training/two_tower_model.pth"


def _bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()


def main() -> None:
    full_df = load_interactions(INTERACTIONS_PATH)
    train_df, val_df = split_interactions(INTERACTIONS_PATH, val_ratio=0.2)

    stats = get_data_stats(full_df)
    all_reels = sorted(full_df["reel_id"].astype(int).unique().tolist())

    train_dataset = TwoTowerDataset(
        train_df,
        all_reel_ids=all_reels,
        user_positive_items=build_user_positive_items(train_df),
    )
    val_dataset = TwoTowerDataset(
        val_df if len(val_df) else train_df.iloc[:1],
        all_reel_ids=all_reels,
        user_positive_items=build_user_positive_items(train_df),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TwoTowerModel(stats.num_users, stats.num_reels, EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            user = batch["user_id"].to(DEVICE)
            pos = batch["pos_reel_id"].to(DEVICE)
            neg = batch["neg_reel_id"].to(DEVICE)

            pos_score = model(user, pos)
            neg_score = model(user, neg)

            loss = _bpr_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                user = batch["user_id"].to(DEVICE)
                pos = batch["pos_reel_id"].to(DEVICE)
                neg = batch["neg_reel_id"].to(DEVICE)

                pos_score = model(user, pos)
                neg_score = model(user, neg)
                val_loss += _bpr_loss(pos_score, neg_score).item()

        print(
            f"Epoch {epoch + 1} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved two-tower model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
