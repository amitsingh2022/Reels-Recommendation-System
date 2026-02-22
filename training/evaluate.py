from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch

from app.faiss_index import FaissRetriever
from app.models import Ranker, TwoTowerModel
from training.dataset import get_data_stats, load_interactions, split_interactions
from training.metrics import mean_metrics_per_user

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 64
K = 10
RETRIEVAL_CANDIDATES = 100
INTERACTIONS_PATH = "data/interactions.csv"
TWO_TOWER_PATH = "training/two_tower_model.pth"
RANKER_PATH = "training/ranker_model.pth"
REEL_EMBEDDINGS_PATH = "data/processed/reel_embeddings.npy"
REEL_IDS_PATH = "data/processed/reel_ids.npy"


def _group_truth(val_df) -> Dict[int, List[int]]:
    return (
        val_df.groupby("user_id")["reel_id"]
        .apply(lambda x: sorted(set(int(v) for v in x.tolist())))
        .to_dict()
    )


def evaluate(k: int = K) -> Dict[str, Dict[str, float]]:
    full_df = load_interactions(INTERACTIONS_PATH)
    _, val_df = split_interactions(INTERACTIONS_PATH, val_ratio=0.2)
    truth = _group_truth(val_df)

    stats = get_data_stats(full_df)

    two_tower = TwoTowerModel(stats.num_users, stats.num_reels, EMBED_DIM)
    two_tower.load_state_dict(torch.load(TWO_TOWER_PATH, map_location="cpu"))
    two_tower.to(DEVICE)
    two_tower.eval()

    ranker = Ranker(EMBED_DIM)
    ranker.load_state_dict(torch.load(RANKER_PATH, map_location="cpu"))
    ranker.to(DEVICE)
    ranker.eval()

    reel_ids_path = REEL_IDS_PATH if os.path.exists(REEL_IDS_PATH) else None
    retriever = FaissRetriever(
        reel_embeddings_path=REEL_EMBEDDINGS_PATH,
        reel_ids_path=reel_ids_path,
    )

    retrieval_preds: Dict[int, List[int]] = {}
    reranked_preds: Dict[int, List[int]] = {}

    for user_id in truth.keys():
        if user_id <= 0 or user_id > stats.num_users:
            continue

        with torch.no_grad():
            user_tensor = torch.tensor([user_id], dtype=torch.long, device=DEVICE)
            user_emb = two_tower.user_encoder(user_tensor).squeeze(0).cpu().numpy()

        candidate_ids, retrieval_scores = retriever.search(
            user_embedding=user_emb,
            k=max(k, RETRIEVAL_CANDIDATES),
        )
        valid = np.where(candidate_ids > 0)[0]
        candidate_ids = candidate_ids[valid]
        retrieval_scores = retrieval_scores[valid]
        if len(candidate_ids) == 0:
            continue
        retrieval_preds[user_id] = [int(v) for v in candidate_ids[:k]]

        with torch.no_grad():
            user_batch = torch.tensor([user_id] * len(candidate_ids), dtype=torch.long, device=DEVICE)
            reel_batch = torch.tensor(candidate_ids, dtype=torch.long, device=DEVICE)
            user_batch_emb = two_tower.user_encoder(user_batch)
            reel_batch_emb = two_tower.reel_encoder(reel_batch)
            rank_scores = ranker(user_batch_emb, reel_batch_emb).cpu().numpy()

        # Keep retrieval score as tie-breaker for deterministic ordering.
        order = np.argsort(rank_scores + 1e-4 * retrieval_scores)[::-1]
        reranked = [int(candidate_ids[i]) for i in order[:k]]
        reranked_preds[user_id] = reranked

    retrieval_metrics = mean_metrics_per_user(retrieval_preds, truth, k=k)
    rerank_metrics = mean_metrics_per_user(reranked_preds, truth, k=k)

    results = {"retrieval": retrieval_metrics, "retrieval_plus_ranker": rerank_metrics}

    print("Offline evaluation:")
    print(results)
    return results


if __name__ == "__main__":
    evaluate()
