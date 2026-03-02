from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Tuple

import numpy as np
import torch

from app.config import Settings
from app.faiss_index import FaissRetriever
from app.models import Ranker, TwoTowerModel
from training.dataset import compute_label, get_data_stats, load_interactions

LOGGER = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ArtifactPaths:
    interactions_path: str
    two_tower_path: str
    ranker_path: str
    reel_embeddings_path: str
    reel_ids_path: str


class RecommendationService:
    def __init__(self, settings: Settings, embed_dim: int = 64):
        self.settings = settings
        self.embed_dim = embed_dim
        self.paths = ArtifactPaths(
            interactions_path=settings.interactions_path,
            two_tower_path=settings.two_tower_path,
            ranker_path=settings.ranker_path,
            reel_embeddings_path=settings.reel_embeddings_path,
            reel_ids_path=settings.reel_ids_path,
        )

        self.two_tower: TwoTowerModel | None = None
        self.ranker: Ranker | None = None
        self.retriever: FaissRetriever | None = None

        self.known_users: set[int] = set()
        self.popular_reels: List[int] = []
        self.reel_count: int = 0
        self.last_reload_ts: float = 0.0
        self.artifact_metadata: Dict[str, object] = {}

        self._cache: Dict[Tuple[int, int, int], Tuple[float, Dict[str, object]]] = {}
        self._cache_lock = Lock()

    def _artifact_info(self, path: str) -> Dict[str, object]:
        stat = os.stat(path)
        return {
            "path": path,
            "size_bytes": int(stat.st_size),
            "modified_ts": float(stat.st_mtime),
        }

    def _validate_artifacts(self, stats_num_reels: int) -> None:
        required = [
            self.paths.interactions_path,
            self.paths.two_tower_path,
            self.paths.ranker_path,
            self.paths.reel_embeddings_path,
        ]
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"missing artifacts: {missing}")

        reel_embeddings = np.load(self.paths.reel_embeddings_path)
        if reel_embeddings.ndim != 2:
            raise ValueError("reel_embeddings.npy must be 2D")
        if int(reel_embeddings.shape[1]) != self.embed_dim:
            raise ValueError(
                f"embedding dim mismatch: expected {self.embed_dim}, got {reel_embeddings.shape[1]}"
            )

        if os.path.exists(self.paths.reel_ids_path):
            reel_ids = np.load(self.paths.reel_ids_path)
            if reel_ids.shape[0] != reel_embeddings.shape[0]:
                raise ValueError("reel_ids and reel_embeddings size mismatch")
            if reel_ids.max(initial=0) > stats_num_reels:
                raise ValueError("reel_ids contain ids not present in training stats")

    def _evict_cache_if_needed(self) -> None:
        max_items = self.settings.request_cache_max_items
        if len(self._cache) <= max_items:
            return
        oldest_key = min(self._cache.items(), key=lambda x: x[1][0])[0]
        self._cache.pop(oldest_key, None)

    def _cache_get(self, key: Tuple[int, int, int]) -> Dict[str, object] | None:
        ttl = self.settings.request_cache_ttl_seconds
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            ts, payload = cached
            if time.time() - ts > ttl:
                self._cache.pop(key, None)
                return None
            return payload

    def _cache_set(self, key: Tuple[int, int, int], payload: Dict[str, object]) -> None:
        with self._cache_lock:
            self._cache[key] = (time.time(), payload)
            self._evict_cache_if_needed()

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def metadata(self) -> Dict[str, object]:
        return {
            "device": DEVICE,
            "embed_dim": self.embed_dim,
            "known_users": len(self.known_users),
            "reels_indexed": self.reel_count,
            "last_reload_ts": self.last_reload_ts,
            "cache_size": len(self._cache),
            "cache_ttl_seconds": self.settings.request_cache_ttl_seconds,
            "artifact_metadata": self.artifact_metadata,
        }

    def reload(self) -> None:
        interactions = load_interactions(self.paths.interactions_path)
        stats = get_data_stats(interactions)
        self._validate_artifacts(stats_num_reels=stats.num_reels)

        interactions = interactions.copy()
        interactions["label"] = compute_label(interactions)

        popularity = interactions.groupby("reel_id")["label"].mean().sort_values(ascending=False)

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

        reel_ids_path = self.paths.reel_ids_path if os.path.exists(self.paths.reel_ids_path) else None
        retriever = FaissRetriever(
            reel_embeddings_path=self.paths.reel_embeddings_path,
            reel_ids_path=reel_ids_path,
        )

        self.two_tower = two_tower
        self.ranker = ranker
        self.retriever = retriever
        self.last_reload_ts = time.time()
        self.artifact_metadata = {
            "interactions": self._artifact_info(self.paths.interactions_path),
            "two_tower": self._artifact_info(self.paths.two_tower_path),
            "ranker": self._artifact_info(self.paths.ranker_path),
            "reel_embeddings": self._artifact_info(self.paths.reel_embeddings_path),
        }
        if os.path.exists(self.paths.reel_ids_path):
            self.artifact_metadata["reel_ids"] = self._artifact_info(self.paths.reel_ids_path)
        self.clear_cache()

        LOGGER.info(
            "Artifacts loaded | users=%d reels=%d popular_fallback=%d",
            len(self.known_users),
            self.reel_count,
            len(self.popular_reels),
        )

    def _ensure_ready(self) -> None:
        if self.two_tower is None or self.ranker is None or self.retriever is None:
            raise RuntimeError("Recommendation service is not initialized")

    def _cold_start_response(self, top_k: int) -> List[Dict[str, float | int | str]]:
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

    def recommend(self, user_id: int, top_k: int = 10, retrieval_k: int = 200) -> Dict[str, object]:
        self._ensure_ready()
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        cache_key = (int(user_id), int(top_k), int(retrieval_k))
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if user_id not in self.known_users:
            payload = {
                "user_id": int(user_id),
                "is_cold_start": True,
                "recommendations": self._cold_start_response(top_k),
            }
            self._cache_set(cache_key, payload)
            return payload

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
            payload = {
                "user_id": int(user_id),
                "is_cold_start": False,
                "recommendations": [],
            }
            self._cache_set(cache_key, payload)
            return payload

        with torch.no_grad():
            user_batch = torch.tensor([user_id] * len(candidate_ids), dtype=torch.long, device=DEVICE)
            reel_batch = torch.tensor(candidate_ids, dtype=torch.long, device=DEVICE)
            user_batch_emb = self.two_tower.user_encoder(user_batch)
            reel_batch_emb = self.two_tower.reel_encoder(reel_batch)
            rank_scores = self.ranker(user_batch_emb, reel_batch_emb).cpu().numpy()

        combined = rank_scores + (1e-4 * retrieval_scores)
        ranked_indices = np.argsort(combined)[::-1][:top_k]

        payload = {
            "user_id": int(user_id),
            "is_cold_start": False,
            "recommendations": [
                {
                    "reel_id": int(candidate_ids[i]),
                    "rank_score": float(rank_scores[i]),
                    "retrieval_score": float(retrieval_scores[i]),
                    "source": "two_tower_plus_ranker",
                }
                for i in ranked_indices
            ],
        }
        self._cache_set(cache_key, payload)
        return payload
