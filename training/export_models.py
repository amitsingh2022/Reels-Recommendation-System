import numpy as np
import torch

from app.models import TwoTowerModel
from training.dataset import get_data_stats, load_interactions

EMBED_DIM = 64
INTERACTIONS_PATH = "data/interactions.csv"
TWO_TOWER_PATH = "training/two_tower_model.pth"
USER_EMB_PATH = "data/processed/user_embeddings.npy"
REEL_EMB_PATH = "data/processed/reel_embeddings.npy"
REEL_IDS_PATH = "data/processed/reel_ids.npy"


def main() -> None:
    df = load_interactions(INTERACTIONS_PATH)
    stats = get_data_stats(df)

    model = TwoTowerModel(stats.num_users, stats.num_reels, EMBED_DIM)
    model.load_state_dict(torch.load(TWO_TOWER_PATH, map_location="cpu"))
    model.eval()

    user_weights = model.user_encoder.embedding.weight.detach().cpu().numpy()
    reel_weights = model.reel_encoder.embedding.weight.detach().cpu().numpy()

    # Embedding index 0 is reserved padding and is not a real id.
    user_embeddings = user_weights[1:]
    reel_embeddings = reel_weights[1:]

    user_ids = np.arange(1, stats.num_users + 1, dtype=np.int64)
    reel_ids = np.arange(1, stats.num_reels + 1, dtype=np.int64)

    np.save(USER_EMB_PATH, user_embeddings)
    np.save(REEL_EMB_PATH, reel_embeddings)
    np.save(REEL_IDS_PATH, reel_ids)
    np.save("data/processed/user_ids.npy", user_ids)

    print("Exported user/reel embeddings and id mappings")


if __name__ == "__main__":
    main()
