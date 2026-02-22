import numpy as np
import torch

from app.inference import RecommendationService


class _FakeEncoder:
    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.stack((ids.float(), ids.float()), dim=1)


class _FakeTwoTower:
    def __init__(self):
        self.user_encoder = _FakeEncoder()
        self.reel_encoder = _FakeEncoder()


class _FakeRanker:
    def __call__(self, user_emb: torch.Tensor, reel_emb: torch.Tensor) -> torch.Tensor:
        # Higher reel id -> higher score.
        return reel_emb[:, 0]


class _FakeRetriever:
    def search(self, user_embedding: np.ndarray, k: int = 10):
        reel_ids = np.array([4, 2, 7, 1], dtype=np.int64)
        scores = np.array([0.2, 0.1, 0.4, 0.3], dtype=np.float32)
        return reel_ids[:k], scores[:k]


def _build_service() -> RecommendationService:
    svc = RecommendationService()
    svc.two_tower = _FakeTwoTower()
    svc.ranker = _FakeRanker()
    svc.retriever = _FakeRetriever()
    svc.known_users = {123}
    svc.popular_reels = [11, 10, 9]
    svc.reel_count = 100
    return svc


def test_cold_start_fallback_for_unknown_user():
    svc = _build_service()
    result = svc.recommend(user_id=999, top_k=2)

    assert result["is_cold_start"] is True
    assert [row["reel_id"] for row in result["recommendations"]] == [11, 10]


def test_ranked_recommendations_for_known_user():
    svc = _build_service()
    result = svc.recommend(user_id=123, top_k=3)

    assert result["is_cold_start"] is False
    assert [row["reel_id"] for row in result["recommendations"]] == [7, 4, 2]
