from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main as app_main


class FakeRecommendationService:
    def __init__(self):
        self.known_users = {123}
        self.reel_count = 2000
        self.reload_count = 0

    def reload(self) -> None:
        self.reload_count += 1

    def recommend(self, user_id: int, top_k: int = 10):
        if user_id == 999999:
            return {
                "user_id": user_id,
                "is_cold_start": True,
                "recommendations": [
                    {
                        "reel_id": 1,
                        "rank_score": 0.0,
                        "retrieval_score": 0.0,
                        "source": "popular_fallback",
                    }
                ],
            }

        return {
            "user_id": user_id,
            "is_cold_start": False,
            "recommendations": [
                {
                    "reel_id": 7,
                    "rank_score": 1.23,
                    "retrieval_score": 0.45,
                    "source": "two_tower_plus_ranker",
                }
            ][:top_k],
        }



@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_main, "RecommendationService", FakeRecommendationService)
    with TestClient(app_main.app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "reels-recommendation-api"
    assert "GET /health" in data["routes"]
    assert "X-Request-ID" in response.headers


def test_health_endpoint(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "X-Request-ID" in response.headers


def test_recommend_endpoint_for_known_user(client: TestClient):
    response = client.get("/recommend", params={"user_id": 123, "top_k": 1})

    assert response.status_code == 200
    data = response.json()
    assert data["is_cold_start"] is False
    assert data["recommendations"][0]["source"] == "two_tower_plus_ranker"


def test_recommend_endpoint_for_cold_start_user(client: TestClient):
    response = client.get("/recommend", params={"user_id": 999999, "top_k": 10})

    assert response.status_code == 200
    data = response.json()
    assert data["is_cold_start"] is True
    assert data["recommendations"][0]["source"] == "popular_fallback"


def test_request_id_is_echoed_when_provided(client: TestClient):
    response = client.get("/health", headers={"X-Request-ID": "req-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-123"


def test_recommend_endpoint_query_validation(client: TestClient):
    response = client.get("/recommend", params={"user_id": 0, "top_k": 10})

    assert response.status_code == 422


def test_metrics_endpoint(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "reels_api_requests_total" in response.text


def test_reload_models_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_ADMIN_API_KEY", "secret")
    response = client.post("/reload-models", headers={"X-API-Key": "secret"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_reload_models_unauthorized(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_ADMIN_API_KEY", "secret")
    response = client.post("/reload-models", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_reload_models_disabled_without_config(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MODEL_ADMIN_API_KEY", raising=False)
    response = client.post("/reload-models", headers={"X-API-Key": "secret"})
    assert response.status_code == 503


def test_reload_models_wrong_method_not_allowed(client: TestClient):
    response = client.get("/reload-models")

    assert response.status_code == 405
