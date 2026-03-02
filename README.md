# Reels Recommendation System

Production-grade recommendation service for short-form reels using:
- Two-Tower retrieval model (PyTorch)
- FAISS ANN candidate search
- Neural ranker re-scoring
- FastAPI online serving
- Dockerized deployment

## System Architecture

```text
User ID
 -> Two-Tower user embedding
 -> FAISS ANN retrieval over reel embeddings
 -> Top-N candidates
 -> Neural ranker scores (user_emb + reel_emb)
 -> Top-K final recommendations
```

## Repository Layout

```text
app/
  main.py            # API routes, middleware, observability, auth, rate limit
  inference.py       # RecommendationService, artifact validation, cache
  config.py          # env-driven settings (pydantic-settings)
  logging_utils.py   # structured JSON logging formatter
  models.py          # TwoTower + Ranker model definitions
  faiss_index.py     # FAISS wrapper
  schemas.py         # Pydantic API schemas
training/
  dataset.py         # schema checks, temporal split, safe negatives
  train_two_tower.py
  train_ranker.py
  export_models.py
  evaluate.py        # overall + segment metrics + reranker uplift
  metrics.py
tests/
  test_api_endpoints.py
  test_inference.py
docker/
  Dockerfile
.github/workflows/
  ci.yml
```

## What Was Implemented

### Core ML pipeline
- Temporal user-level train/validation split.
- Negative sampling that avoids known positives.
- Two-stage retrieval + rerank pipeline preserved.
- Offline metrics for Recall@10 and NDCG@10.
- Evaluation now reports:
  - retrieval vs retrieval+ranker
  - uplift
  - user segments (`new_or_sparse`, `active`)

### Inference hardening
- Models and FAISS load once at startup.
- Artifact compatibility checks before serving:
  - required files exist
  - embedding shape and ID mapping consistency
- Metadata snapshot captured on reload (size + modified time).
- In-memory TTL cache for hot recommendation calls.
- Cold-start fallback for unknown users.

### API and operations
- Endpoints:
  - `GET /`
  - `GET /health`
  - `GET /health/live`
  - `GET /health/ready`
  - `GET /metrics`
  - `GET /model-metadata`
  - `GET /recommend`
  - `POST /reload-models`
- Structured JSON logs with request metadata.
- Request ID propagation via `X-Request-ID`.
- Prometheus metrics:
  - request count
  - request latency histogram
  - cold-start count
  - rate-limit reject count
- Reload endpoint protected with `X-API-Key` (`MODEL_ADMIN_API_KEY`).
- Per-IP rate limiting on `/recommend`.
- CORS configurable from environment.

### CI/CD and quality
- GitHub Actions pipeline added (`.github/workflows/ci.yml`) running:
  - `ruff check`
  - `mypy`
  - `pytest`
  - API smoke checks (`/health/live`, `/health/ready`, `/recommend`)
- Tooling config added in `pyproject.toml`.

### Deployment
- Dockerfile hardened with:
  - non-root runtime user
  - healthcheck (`/health/ready`)
- `docker-compose.yml` added for local container orchestration.

## Configuration

Use environment variables (example in `.env.example`):
- `MODEL_ADMIN_API_KEY`
- `ENABLE_RELOAD_ENDPOINT`
- `REQUEST_CACHE_TTL_SECONDS`
- `REQUEST_CACHE_MAX_ITEMS`
- `RATE_LIMIT_REQUESTS_PER_MINUTE`
- `CORS_ALLOW_ORIGINS`
- artifact paths (`INTERACTIONS_PATH`, `TWO_TOWER_PATH`, etc.)

## Quick Start

### 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Train and export artifacts

```bash
python -m training.train_two_tower
python -m training.train_ranker
python -m training.export_models
python -m training.evaluate
```

### 3) Run API

```bash
export MODEL_ADMIN_API_KEY="change-me"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Docs and checks
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- Metrics: `http://127.0.0.1:8000/metrics`

## API Examples

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
curl "http://127.0.0.1:8000/recommend?user_id=123&top_k=10"
curl http://127.0.0.1:8000/model-metadata
curl -X POST http://127.0.0.1:8000/reload-models -H "X-API-Key: change-me"
```

## Testing and Quality

```bash
ruff check .
mypy app training tests
pytest -q
```

## Docker

### Build and run

```bash
docker build -f docker/Dockerfile -t reels-recommendation .
docker run --rm -p 8000:8000 -e MODEL_ADMIN_API_KEY=change-me reels-recommendation
```

### Compose

```bash
docker compose up --build
```

## Notes
- CPU-compatible by design (`faiss-cpu`).
- `POST /reload-models` is intentionally protected and disabled when `MODEL_ADMIN_API_KEY` is unset.
- If you run into missing packages, reinstall from `requirements.txt` in the active venv.

## Roadmap
- Feature store integration for richer real-time features.
- A/B testing hooks and business-rule re-ranking.
- Vector index persistence/versioning policy.
- CI pipeline for lint, type checks, tests, and container smoke tests.

## Current Constraints
- ANN index is in-memory and rebuilt from exported embeddings.
- Rate limiter and cache are per-process (not shared across replicas).
- For multi-instance production, move cache/rate-limit state to Redis.

## 👨‍💻 Author  

**Amit Singh**  
Machine Learning Engineer | Data Enthusiast  
🌐 [LinkedIn](https://www.linkedin.com/in/amit-singh101/) | [GitHub](https://github.com/amitsingh2022)