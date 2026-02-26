# Reels Recommendation System

Production-ready recommendation system for short-video reels using a **Two-Tower Retrieval model**, **FAISS ANN search**, and a **Neural Ranker**, served via **FastAPI**.

## Overview
This project implements a two-stage recommendation pipeline:

1. **Candidate Retrieval (Two-Tower + FAISS)**
- Learns user and reel embeddings with a Two-Tower model (PyTorch).
- Exports reel embeddings and serves fast nearest-neighbor retrieval with FAISS.

2. **Candidate Re-Ranking (Neural Ranker)**
- Scores retrieved candidates with a ranker MLP for better top-k ordering.

3. **Online Serving (FastAPI)**
- Startup-loaded models and FAISS index (no heavy per-request initialization).
- Cold-start fallback for unknown users.
- Latency logging, request-id tracing, model reload endpoint, and Prometheus metrics.

## Project Structure

```text
reels-recommendation-system/
├── app/
│   ├── main.py             # FastAPI app, middleware, endpoints, metrics, auth
│   ├── inference.py        # RecommendationService (load/reload/recommend)
│   ├── models.py           # TwoTowerModel + Ranker definitions
│   ├── faiss_index.py      # FAISS retriever wrapper
│   ├── features.py         # User/reel aggregate feature utilities
│   └── schemas.py          # Pydantic request/response models
├── training/
│   ├── dataset.py          # Data loading, temporal split, safe negative sampling
│   ├── train_two_tower.py  # Two-tower training
│   ├── train_ranker.py     # Ranker training
│   ├── metrics.py          # Recall@K, NDCG@K, mean user-level metrics
│   ├── evaluate.py         # Offline retrieval + reranker evaluation
│   └── export_models.py    # Embedding export + id mappings
├── data/
│   ├── interactions.csv
│   └── processed/
├── tests/
│   ├── test_inference.py
│   └── test_api_endpoints.py
├── docker/
│   └── Dockerfile
├── requirements.txt
└── README.md
```

## What We Implemented

### 1) Data pipeline improvements
- Added strict interaction schema validation.
- Implemented **temporal per-user train/validation split**.
- Implemented safe negative sampling that avoids sampling positive user interactions.

### 2) Training improvements
- Refactored training scripts to use train/validation splits.
- Added cleaner script entrypoints (`main()`) and consistent artifact paths.
- Preserved original Two-Tower + Ranker architecture.

### 3) Evaluation improvements
- Built offline evaluation for:
  - **Retrieval-only**
  - **Retrieval + Ranker**
- Computes average:
  - **Recall@10**
  - **NDCG@10**

### 4) Inference & serving hardening
- Added `RecommendationService` with global model/index lifecycle.
- Ensured heavy objects are loaded once at startup (and reloadable).
- Added cold-start fallback based on popular reels.
- Added robust API error handling.
- Added latency logging.

### 5) API and OpenAPI
- Implemented endpoints:
  - `GET /`
  - `GET /health`
  - `GET /metrics`
  - `GET /recommend?user_id=...&top_k=...`
  - `POST /reload-models`
- Added typed response schemas and detailed Swagger docs.

### 6) MLOps/observability add-ons
- Request-id propagation (`X-Request-ID`) for tracing.
- Prometheus metrics endpoint (`/metrics`) with counters/histogram.
- API-key protection for model reload endpoint (`X-API-Key`).

### 7) Deployment readiness
- Added production-oriented Dockerfile (Python 3.11, non-root user, uvicorn entrypoint).
- Updated dependencies for training, serving, testing, and monitoring.

## Architecture

```text
User ID
  -> Two-Tower user encoder
  -> User embedding
  -> FAISS ANN over reel embeddings (candidate retrieval)
  -> Top-N candidate reel IDs
  -> Neural Ranker (user emb + reel emb)
  -> Final Top-K recommendations
```

## Offline Training Flow

1. Train retriever:
```bash
python -m training.train_two_tower
```

2. Train ranker:
```bash
python -m training.train_ranker
```

3. Export embeddings and id maps:
```bash
python -m training.export_models
```

4. Evaluate retrieval and retrieval+ranker:
```bash
python -m training.evaluate
```

## Run Locally

### Prerequisites
- Python **3.11** recommended
- Virtual environment

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Start API
```bash
export MODEL_ADMIN_API_KEY="change-me"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Docs
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## API Usage Examples

### Service metadata
```bash
curl http://127.0.0.1:8000/
```

### Health
```bash
curl http://127.0.0.1:8000/health
```

### Recommend for known user
```bash
curl "http://127.0.0.1:8000/recommend?user_id=123&top_k=10"
```

### Recommend for unknown user (cold start fallback)
```bash
curl "http://127.0.0.1:8000/recommend?user_id=999999&top_k=10"
```

### Reload models/index (secured)
```bash
curl -X POST http://127.0.0.1:8000/reload-models \
  -H "X-API-Key: change-me"
```

### Prometheus metrics
```bash
curl http://127.0.0.1:8000/metrics
```

## Testing

```bash
python -m pytest -q
```

Current tests cover:
- inference behavior (known user + cold start)
- endpoint behavior for all routes
- method/validation/auth edge cases

## Docker

### Build
```bash
docker build -f docker/Dockerfile -t reels-recommendation .
```

### Run
```bash
docker run --rm -p 8000:8000 \
  -e MODEL_ADMIN_API_KEY=change-me \
  reels-recommendation
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

## 👨‍💻 Author  

**Amit Singh**  
Machine Learning Engineer | Data Enthusiast  
🌐 [LinkedIn](https://www.linkedin.com/in/amit-singh101/) | [GitHub](https://github.com/amitsingh2022)
