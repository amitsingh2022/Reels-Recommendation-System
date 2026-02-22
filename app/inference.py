from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from app.faiss_index import FaissRetriever
from app.models import Ranker, TwoTowerModel
from training.dataset import compute_label, get_data_stats, load_interactions

LOGGER = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ArtifactPaths:
    interactions_path: str = "data/interactions.csv"
    two_tower_path: str = "training/two_tower_model.pth"
    ranker_path: str = "training/ranker_model.pth"
    reel_embeddings_path: str = "data/processed/reel_embeddings.npy"
    reel_ids_path: str = "data/processed/reel_ids.npy"


class RecommendationService:
    def __init__(self, embed_dim: int = 64, artifact_paths: ArtifactPaths | None = None):
        self.embed_dim = embed_dim
        self.paths = artifact_paths or ArtifactPaths()

        self.two_tower: TwoTowerModel | None = None
        self.ranker: Ranker | None = None
        self.retriever: FaissRetriever | None = None

        self.known_users: set[int] = set()
        self.popular_reels: List[int] = []
        self.reel_count: int = 0

    def reload(self) -> None:
        interactions = load_interactions(self.paths.interactions_path)
        stats = get_data_stats(interactions)

        interactions = interactions.copy()
        interactions["label"] = compute_label(interactions)

        popularity = (
            interactions.groupby("reel_id")["label"].mean().sort_values(ascending=False)
        )

        self.known_users = set(interactions["user_id"].astype(int).unique().tolist())
        self.popular_reels = [int(v) for v in popularity.index.tolist()]
        self.reel_count = int(stats.num_reels)

        two_tower = TwoTowerModel(stats.num_users, stats.num_reels, self.embed_dim)
        two_tower.load_state_dict(torch.load(self.paths.two_tower_path, map_location="cpu"))
        two_tower.to(DEVICE)
        two_tower.eval()

        ranker = Ranker(self.embed_dim)
        ranker.load_state_dict(torch.load(self.paths.ranker_path, map_location="cpu"))
        ranker.to(DEVICE)
        ranker.eval()

        reel_ids_path = (
            self.paths.reel_ids_path if os.path.exists(self.paths.reel_ids_path) else None
        )
        retriever = FaissRetriever(
            reel_embeddings_path=self.paths.reel_embeddings_path,
            reel_ids_path=reel_ids_path,
        )

        self.two_tower = two_tower
        self.ranker = ranker
        self.retriever = retriever

        LOGGER.info(
            "Artifacts loaded | users=%d | reels=%d | popular_fallback=%d",
            len(self.known_users),
            self.reel_count,
            len(self.popular_reels),
        )

    def _ensure_ready(self) -> None:
        if self.two_tower is None or self.ranker is None or self.retriever is None:
            raise RuntimeError("Recommendation service is not initialized")

    def _cold_start_response(self, user_id: int, top_k: int) -> List[Dict[str, float | int | str]]:
        fallback = self.popular_reels[:top_k]
        return [
            {
                "reel_id": int(reel_id),
                "rank_score": 0.0,
                "retrieval_score": 0.0,
                "source": "popular_fallback",
            }
            for reel_id in fallback
        ]

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        retrieval_k: int = 200,
    ) -> Dict[str, object]:
        self._ensure_ready()

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if user_id not in self.known_users:
            return {
                "user_id": int(user_id),
                "is_cold_start": True,
                "recommendations": self._cold_start_response(user_id, top_k),
            }

        assert self.two_tower is not None
        assert self.ranker is not None
        assert self.retriever is not None

        with torch.no_grad():
            user_tensor = torch.tensor([user_id], dtype=torch.long, device=DEVICE)
            user_emb = self.two_tower.user_encoder(user_tensor).squeeze(0).cpu().numpy()

        candidate_k = max(top_k, retrieval_k)
        candidate_ids, retrieval_scores = self.retriever.search(user_emb, k=candidate_k)
        valid = np.where(candidate_ids > 0)[0]
        candidate_ids = candidate_ids[valid]
        retrieval_scores = retrieval_scores[valid]

        if len(candidate_ids) == 0:
            return {
                "user_id": int(user_id),
                "is_cold_start": False,
                "recommendations": [],
            }

        with torch.no_grad():
            user_batch = torch.tensor([user_id] * len(candidate_ids), dtype=torch.long, device=DEVICE)
            reel_batch = torch.tensor(candidate_ids, dtype=torch.long, device=DEVICE)

            user_batch_emb = self.two_tower.user_encoder(user_batch)
            reel_batch_emb = self.two_tower.reel_encoder(reel_batch)
            rank_scores = self.ranker(user_batch_emb, reel_batch_emb).cpu().numpy()

        combined = rank_scores + (1e-4 * retrieval_scores)
        ranked_indices = np.argsort(combined)[::-1][:top_k]

        recommendations = [
            {
                "reel_id": int(candidate_ids[i]),
                "rank_score": float(rank_scores[i]),
                "retrieval_score": float(retrieval_scores[i]),
                "source": "two_tower_plus_ranker",
            }
            for i in ranked_indices
        ]

        return {
            "user_id": int(user_id),
            "is_cold_start": False,
            "recommendations": recommendations,
        }
