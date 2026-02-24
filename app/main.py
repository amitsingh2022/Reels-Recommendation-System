from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

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
RELOAD_API_KEY_ENV = "MODEL_ADMIN_API_KEY"

REQUEST_COUNT = Counter(
    "reels_api_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "reels_api_request_latency_seconds",
    "Request latency in seconds",
    ["method", "path"],
)
COLD_START_COUNT = Counter(
    "reels_api_cold_start_recommendations_total",
    "Count of recommendation responses served from cold-start fallback",
)


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
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        status_code = response.status_code if response is not None else 500
        REQUEST_COUNT.labels(
            method=request.method,
            path=request.url.path,
            status_code=str(status_code),
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, path=request.url.path).observe(
            duration_ms / 1000.0
        )
        LOGGER.info(
            "request_id=%s path=%s method=%s status_code=%s latency_ms=%.2f",
            request_id,
            request.url.path,
            request.method,
            status_code,
            duration_ms,
        )
    response.headers["X-Request-ID"] = request_id
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
            "GET /metrics",
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
    "/metrics",
    tags=["System"],
    summary="Prometheus metrics",
    description="Exposes service metrics in Prometheus text format.",
)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
        result = service.recommend(user_id=user_id, top_k=top_k)
        if result.get("is_cold_start"):
            COLD_START_COUNT.inc()
        return result
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
    description=(
        "Reloads Two-Tower model, ranker model, and FAISS index without restarting the "
        "server. Requires `X-API-Key` header matching `MODEL_ADMIN_API_KEY`."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Reload failed"},
        503: {"model": ErrorResponse, "description": "Reload disabled / service unavailable"},
    },
)
def reload_models(
    request: Request,
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
        description="Admin API key for model reload operations",
    ),
) -> ReloadResponse:
    service = _service(request)
    configured_api_key = os.getenv(RELOAD_API_KEY_ENV)

    if not configured_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                f"reload endpoint disabled: set {RELOAD_API_KEY_ENV} environment variable"
            ),
        )
    if x_api_key != configured_api_key:
        raise HTTPException(status_code=401, detail="unauthorized")

    try:
        service.reload()
        return {"status": "ok", "message": "models and index reloaded"}
    except Exception as exc:
        LOGGER.exception("reload failed")
        raise HTTPException(status_code=500, detail=f"reload failed: {exc}") from exc
