from __future__ import annotations

import pytest
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient

from app import main as app_main


class FakeRecommendationService:
    def __init__(self, settings):
        self.settings = settings
        self.known_users = {123}
        self.reel_count = 2000
        self.reload_count = 0
        self.two_tower = object()
        self.ranker = object()
        self.retriever = object()

    def reload(self) -> None:
        self.reload_count += 1

    def metadata(self):
        return {
            "known_users": len(self.known_users),
            "reels_indexed": self.reel_count,
            "last_reload_ts": 1.0,
        }

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
    @asynccontextmanager
    async def test_lifespan(app):
        app.state.recommendation_service = FakeRecommendationService(settings=app_main.SETTINGS)
        yield

    monkeypatch.setattr(app_main.app.router, "lifespan_context", test_lifespan)
    app_main.RATE_LIMITER._store.clear()
    app_main.SETTINGS.reload_api_key = "secret"
    app_main.SETTINGS.enable_reload_endpoint = True
    app_main.RATE_LIMITER.requests_per_minute = 1000
    with TestClient(app_main.app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == app_main.SETTINGS.service_name
    assert "GET /health/ready" in data["routes"]
    assert "X-Request-ID" in response.headers


def test_health_endpoints(client: TestClient):
    health = client.get("/health")
    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert health.status_code == 200
    assert live.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_model_metadata_endpoint(client: TestClient):
    response = client.get("/model-metadata")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_endpoint_for_known_user(client: TestClient):
    response = client.get("/recommend", params={"user_id": 123, "top_k": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["is_cold_start"] is False


def test_recommend_endpoint_for_cold_start_user(client: TestClient):
    response = client.get("/recommend", params={"user_id": 999999, "top_k": 10})
    assert response.status_code == 200
    assert response.json()["is_cold_start"] is True


def test_request_id_echo(client: TestClient):
    response = client.get("/health", headers={"X-Request-ID": "req-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-123"


def test_recommend_query_validation(client: TestClient):
    response = client.get("/recommend", params={"user_id": 0, "top_k": 10})
    assert response.status_code == 422


def test_metrics_endpoint(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "reels_api_requests_total" in response.text


def test_reload_models_success(client: TestClient):
    response = client.post("/reload-models", headers={"X-API-Key": "secret"})
    assert response.status_code == 200


def test_reload_models_unauthorized(client: TestClient):
    response = client.post("/reload-models", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_reload_models_disabled(client: TestClient):
    app_main.SETTINGS.enable_reload_endpoint = False
    response = client.post("/reload-models", headers={"X-API-Key": "secret"})
    assert response.status_code == 503


def test_reload_models_wrong_method_not_allowed(client: TestClient):
    response = client.get("/reload-models")
    assert response.status_code == 405


def test_recommend_rate_limit(client: TestClient):
    app_main.RATE_LIMITER.requests_per_minute = 1
    app_main.RATE_LIMITER._store.clear()

    first = client.get("/recommend", params={"user_id": 123, "top_k": 1})
    second = client.get("/recommend", params={"user_id": 123, "top_k": 1})

    assert first.status_code == 200
    assert second.status_code == 429
