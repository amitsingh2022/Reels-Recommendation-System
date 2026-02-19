import numpy as np

def recall_at_k(pred_items, true_items, k):
    pred_k = pred_items[:k]
    return len(set(pred_k) & set(true_items)) / len(true_items)


def dcg_at_k(relevance_scores, k):
    relevance_scores = np.array(relevance_scores)[:k]
    return np.sum(
        relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2))
    )


def ndcg_at_k(pred_items, true_items, k):
    relevance = [1 if item in true_items else 0 for item in pred_items]
    dcg = dcg_at_k(relevance, k)

    ideal_relevance = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg
