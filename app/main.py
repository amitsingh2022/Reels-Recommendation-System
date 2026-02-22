from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query, Request

from app.inference import RecommendationService

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


app = FastAPI(title="Reels Recommendation Service", lifespan=lifespan)


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


@app.get("/")
def root(request: Request) -> Dict[str, Any]:
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


@app.get("/health")
def health(request: Request) -> Dict[str, Any]:
    service = _service(request)
    return {
        "status": "ok",
        "users_loaded": len(service.known_users),
        "reels_indexed": service.reel_count,
    }


@app.get("/recommend")
def recommend(
    request: Request,
    user_id: int = Query(..., ge=1),
    top_k: int = Query(10, ge=1, le=100),
) -> Dict[str, Any]:
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


@app.post("/reload-models")
def reload_models(request: Request) -> Dict[str, str]:
    service = _service(request)

    try:
        service.reload()
        return {"status": "ok", "message": "models and index reloaded"}
    except Exception as exc:
        LOGGER.exception("reload failed")
        raise HTTPException(status_code=500, detail=f"reload failed: {exc}") from exc
