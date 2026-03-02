from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    service_name: str = "reels-recommendation-api"
    environment: str = "dev"
    log_level: str = "INFO"

    host: str = "0.0.0.0"
    port: int = 8000

    reload_api_key: str | None = None
    enable_reload_endpoint: bool = True

    cors_allow_origins: str = "*"

    request_cache_ttl_seconds: int = 30
    request_cache_max_items: int = 5000

    rate_limit_requests_per_minute: int = 120

    interactions_path: str = "data/interactions.csv"
    two_tower_path: str = "training/two_tower_model.pth"
    ranker_path: str = "training/ranker_model.pth"
    reel_embeddings_path: str = "data/processed/reel_embeddings.npy"
    reel_ids_path: str = "data/processed/reel_ids.npy"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        service_name=os.getenv("SERVICE_NAME", "reels-recommendation-api"),
        environment=os.getenv("ENVIRONMENT", "dev"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload_api_key=os.getenv("MODEL_ADMIN_API_KEY"),
        enable_reload_endpoint=_as_bool(os.getenv("ENABLE_RELOAD_ENDPOINT"), True),
        cors_allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*"),
        request_cache_ttl_seconds=int(os.getenv("REQUEST_CACHE_TTL_SECONDS", "30")),
        request_cache_max_items=int(os.getenv("REQUEST_CACHE_MAX_ITEMS", "5000")),
        rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "120")),
        interactions_path=os.getenv("INTERACTIONS_PATH", "data/interactions.csv"),
        two_tower_path=os.getenv("TWO_TOWER_PATH", "training/two_tower_model.pth"),
        ranker_path=os.getenv("RANKER_PATH", "training/ranker_model.pth"),
        reel_embeddings_path=os.getenv("REEL_EMBEDDINGS_PATH", "data/processed/reel_embeddings.npy"),
        reel_ids_path=os.getenv("REEL_IDS_PATH", "data/processed/reel_ids.npy"),
    )
