from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    reel_id: int = Field(..., description="Recommended reel identifier")
    rank_score: float = Field(..., description="Neural ranker score")
    retrieval_score: float = Field(..., description="FAISS retrieval similarity score")
    source: Literal["two_tower_plus_ranker", "popular_fallback"] = Field(
        ..., description="Recommendation source"
    )


class RecommendResponse(BaseModel):
    user_id: int = Field(..., description="User id requested")
    is_cold_start: bool = Field(..., description="Whether cold-start fallback was used")
    recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description="Top reels returned by the recommender",
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    users_loaded: int
    reels_indexed: int


class LiveResponse(BaseModel):
    status: Literal["alive"]
    service: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    reason: str | None = None


class RootResponse(BaseModel):
    service: str
    status: Literal["ok"]
    users_loaded: int
    reels_indexed: int
    routes: List[str]


class ReloadResponse(BaseModel):
    status: Literal["ok"]
    message: str


class ModelMetadataResponse(BaseModel):
    status: Literal["ok"]
    metadata: Dict[str, object]


class ErrorResponse(BaseModel):
    detail: str
