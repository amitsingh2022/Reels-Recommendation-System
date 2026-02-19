from training.metrics import recall_at_k, ndcg_at_k

def evaluate(predicted_reels, true_reels):

    recall = recall_at_k(predicted_reels, true_reels, k=10)
    ndcg = ndcg_at_k(predicted_reels, true_reels, k=10)

    print("Recall@10:", recall)
    print("NDCG@10:", ndcg)
