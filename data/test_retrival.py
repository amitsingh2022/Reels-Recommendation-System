import numpy as np
from app.faiss_index import FaissRetriever

user_embeddings = np.load("data/processed/user_embeddings.npy")

retriever = FaissRetriever("data/processed/reel_embeddings.npy")

user_id = 1
user_vector = user_embeddings[user_id]

indices, scores = retriever.search(user_vector, k=10)

print("Top 10 reel indices:", indices)
print("Scores:", scores)
