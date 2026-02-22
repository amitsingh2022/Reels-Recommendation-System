from __future__ import annotations

import os
from typing import Optional, Tuple

import faiss
import numpy as np


class FaissRetriever:
    def __init__(self, reel_embeddings_path: str, reel_ids_path: Optional[str] = None):
        self.reel_embeddings = np.load(reel_embeddings_path).astype("float32")
        if reel_ids_path and os.path.exists(reel_ids_path):
            self.reel_ids = np.load(reel_ids_path).astype(np.int64)
        else:
            # Backward compatibility for old exports without explicit id mapping.
            self.reel_ids = np.arange(self.reel_embeddings.shape[0], dtype=np.int64)

        if self.reel_embeddings.shape[0] != self.reel_ids.shape[0]:
            raise ValueError("reel embeddings and reel ids must have same length")

        self.embedding_dim = int(self.reel_embeddings.shape[1])

        # Use inner product on L2-normalized vectors as cosine similarity.
        faiss.normalize_L2(self.reel_embeddings)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.reel_embeddings)

    def search(self, user_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        user_embedding = user_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(user_embedding)

        k = min(k, self.reel_ids.shape[0])
        scores, indices = self.index.search(user_embedding, k)
        reel_ids = self.reel_ids[indices[0]]
        return reel_ids, scores[0]
