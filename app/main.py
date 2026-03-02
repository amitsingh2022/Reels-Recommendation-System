from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from threading import Lock
from typing import Deque, Dict

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.config import Settings, get_settings
from app.inference import RecommendationService
from app.logging_utils import configure_logging
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    LiveResponse,
    ModelMetadataResponse,
    ReadyResponse,
    RecommendResponse,
    ReloadResponse,
    RootResponse,
)

SETTINGS = get_settings()
configure_logging(SETTINGS.log_level)
LOGGER = logging.getLogger("reels-api")

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
RATE_LIMIT_REJECTED = Counter(
    "reels_api_rate_limited_total",
    "Count of requests rejected due to rate limits",
)


class FixedWindowRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self._store: Dict[str, Deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            queue = self._store.setdefault(key, deque())
            while queue and queue[0] < cutoff:
                queue.popleft()
            if len(queue) >= self.requests_per_minute:
                return False
            queue.append(now)
            return True


RATE_LIMITER = FixedWindowRateLimiter(SETTINGS.rate_limit_requests_per_minute)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RecommendationService(settings=SETTINGS)
    service.reload()
    app.state.recommendation_service = service
    yield


app = FastAPI(
    title="Reels Recommendation Service",
    description=(
        "Production-ready Reels recommendation API backed by a Two-Tower retriever, "
        "FAISS ANN index, and neural ranker."
    ),
    version="1.1.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "System", "description": "Service health and metadata endpoints"},
        {"name": "Recommendation", "description": "User recommendation endpoints"},
        {"name": "ModelOps", "description": "Model/index lifecycle operations"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[s.strip() for s in SETTINGS.cors_allow_origins.split(",") if s.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    if request.url.path == "/recommend":
        ip = request.client.host if request.client else "unknown"
        if not RATE_LIMITER.allow(ip):
            RATE_LIMIT_REJECTED.inc()
            return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})

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
            "request completed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "latency_ms": round(duration_ms, 2),
            },
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
)
def root(request: Request) -> RootResponse:
    service = _service(request)
    return {
        "service": SETTINGS.service_name,
        "status": "ok",
        "users_loaded": len(service.known_users),
        "reels_indexed": service.reel_count,
        "routes": [
            "GET /",
            "GET /health",
            "GET /health/live",
            "GET /health/ready",
            "GET /metrics",
            "GET /model-metadata",
            "GET /recommend?user_id=<id>&top_k=10",
            "POST /reload-models",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Health check")
def health(request: Request) -> HealthResponse:
    service = _service(request)
    return {
        "status": "ok",
        "users_loaded": len(service.known_users),
        "reels_indexed": service.reel_count,
    }


@app.get("/health/live", response_model=LiveResponse, tags=["System"], summary="Liveness")
def health_live() -> LiveResponse:
    return {"status": "alive", "service": SETTINGS.service_name}


@app.get(
    "/health/ready",
    response_model=ReadyResponse,
    tags=["System"],
    summary="Readiness",
    responses={503: {"model": ErrorResponse, "description": "Service not ready"}},
)
def health_ready(request: Request) -> ReadyResponse:
    service = _service(request)
    if service.two_tower is None or service.ranker is None or service.retriever is None:
        raise HTTPException(status_code=503, detail="service is not ready")
    return {"status": "ready", "reason": None}


@app.get(
    "/metrics",
    tags=["System"],
    summary="Prometheus metrics",
    description="Exposes service metrics in Prometheus text format.",
)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get(
    "/model-metadata",
    response_model=ModelMetadataResponse,
    tags=["System"],
    summary="Current model metadata",
)
def model_metadata(request: Request) -> ModelMetadataResponse:
    service = _service(request)
    return {"status": "ok", "metadata": service.metadata()}


@app.get(
    "/recommend",
    response_model=RecommendResponse,
    tags=["Recommendation"],
    summary="Get top reel recommendations",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid recommendation request"},
        422: {"model": ErrorResponse, "description": "Query parameter validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
def recommend(
    request: Request,
    user_id: int = Query(..., ge=1, description="User identifier", examples=[123]),
    top_k: int = Query(10, ge=1, le=100, description="Number of recommendations", examples=[10]),
) -> RecommendResponse:
    service = _service(request)

    try:
        result = service.recommend(user_id=user_id, top_k=top_k)
        if result.get("is_cold_start"):
            COLD_START_COUNT.inc()
        LOGGER.info(
            "recommendation served",
            extra={
                "request_id": getattr(request.state, "request_id", None),
                "user_id": user_id,
                "is_cold_start": bool(result.get("is_cold_start")),
            },
        )
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
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Reload failed"},
        503: {"model": ErrorResponse, "description": "Reload disabled / service unavailable"},
    },
)
def reload_models(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ReloadResponse:
    service = _service(request)

    if not SETTINGS.enable_reload_endpoint:
        raise HTTPException(status_code=503, detail="reload endpoint disabled by configuration")

    if not SETTINGS.reload_api_key:
        raise HTTPException(status_code=503, detail="reload endpoint disabled: missing API key config")

    if x_api_key != SETTINGS.reload_api_key:
        raise HTTPException(status_code=401, detail="unauthorized")

    try:
        service.reload()
        return {"status": "ok", "message": "models and index reloaded"}
    except Exception as exc:
        LOGGER.exception("reload failed")
        raise HTTPException(status_code=500, detail=f"reload failed: {exc}") from exc
