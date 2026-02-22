import numpy as np

from app.faiss_index import FaissRetriever

user_embeddings = np.load("data/processed/user_embeddings.npy")
retriever = FaissRetriever(
    reel_embeddings_path="data/processed/reel_embeddings.npy",
    reel_ids_path="data/processed/reel_ids.npy",
)

user_id = 1
user_vector = user_embeddings[user_id - 1]

reel_ids, scores = retriever.search(user_vector, k=10)

print("Top 10 reel ids:", reel_ids)
print("Scores:", scores)
