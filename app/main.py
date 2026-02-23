from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request

from app.inference import RecommendationService
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    RecommendResponse,
    ReloadResponse,
    RootResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("reels-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RecommendationService()
    service.reload()
    app.state.recommendation_service = service
    yield


app = FastAPI(
    title="Reels Recommendation Service",
    description=(
        "Production-ready Reels recommendation API backed by a Two-Tower retriever, "
        "FAISS ANN index, and neural ranker."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "System", "description": "Service health and metadata endpoints"},
        {"name": "Recommendation", "description": "User recommendation endpoints"},
        {"name": "ModelOps", "description": "Model/index lifecycle operations"},
    ],
)


@app.middleware("http")
async def latency_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        LOGGER.info(
            "request path=%s method=%s latency_ms=%.2f",
            request.url.path,
            request.method,
            duration_ms,
        )
    return response


def _service(request: Request) -> RecommendationService:
    service = getattr(request.app.state, "recommendation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="service not initialized")
    return service


@app.get(
    "/",
    response_model=RootResponse,
    tags=["System"],
    summary="Service metadata",
    description="Returns API metadata and currently available routes.",
    responses={503: {"model": ErrorResponse, "description": "Service not initialized"}},
)
def root(request: Request) -> RootResponse:
    service = _service(request)
    return {
        "service": "reels-recommendation-api",
        "status": "ok",
        "users_loaded": len(service.known_users),
        "reels_indexed": service.reel_count,
        "routes": [
            "GET /",
            "GET /health",
            "GET /recommend?user_id=<id>&top_k=10",
            "POST /reload-models",
        ],
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns service status and loaded artifact counts.",
    responses={503: {"model": ErrorResponse, "description": "Service not initialized"}},
)
def health(request: Request) -> HealthResponse:
    service = _service(request)
    return {
        "status": "ok",
        "users_loaded": len(service.known_users),
        "reels_indexed": service.reel_count,
    }


@app.get(
    "/recommend",
    response_model=RecommendResponse,
    tags=["Recommendation"],
    summary="Get top reel recommendations",
    description=(
        "Returns top-K recommendations for a user. Unknown users automatically use "
        "cold-start popular fallback."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid recommendation request"},
        422: {"model": ErrorResponse, "description": "Query parameter validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
def recommend(
    request: Request,
    user_id: int = Query(
        ...,
        ge=1,
        description="User identifier to generate recommendations for",
        example=123,
    ),
    top_k: int = Query(
        10,
        ge=1,
        le=100,
        description="Number of recommendations to return",
        example=10,
    ),
) -> RecommendResponse:
    service = _service(request)

    try:
        return service.recommend(user_id=user_id, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"artifact not found: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("recommend failed")
        raise HTTPException(status_code=500, detail="internal server error") from exc


@app.post(
    "/reload-models",
    response_model=ReloadResponse,
    tags=["ModelOps"],
    summary="Reload models and index",
    description="Reloads Two-Tower model, ranker model, and FAISS index without restarting the server.",
    responses={500: {"model": ErrorResponse, "description": "Reload failed"}},
)
def reload_models(request: Request) -> ReloadResponse:
    service = _service(request)

    try:
        service.reload()
        return {"status": "ok", "message": "models and index reloaded"}
    except Exception as exc:
        LOGGER.exception("reload failed")
        raise HTTPException(status_code=500, detail=f"reload failed: {exc}") from exc
