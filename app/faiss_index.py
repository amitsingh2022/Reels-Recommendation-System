import faiss
import numpy as np


class FaissRetriever:
    def __init__(self, reel_embeddings_path):
        self.reel_embeddings = np.load(reel_embeddings_path).astype("float32")

        self.embedding_dim = self.reel_embeddings.shape[1]

        # Cosine similarity → normalize vectors
        faiss.normalize_L2(self.reel_embeddings)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.reel_embeddings)

        print("✅ FAISS index built")

    def search(self, user_embedding, k=10):
        user_embedding = user_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(user_embedding)

        scores, indices = self.index.search(user_embedding, k)

        return indices[0], scores[0]
