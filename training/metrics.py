from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def recall_at_k(pred_items: Iterable[int], true_items: Iterable[int], k: int) -> float:
    pred_k = list(pred_items)[:k]
    true_set = set(true_items)
    if not true_set:
        return 0.0
    return len(set(pred_k) & true_set) / len(true_set)


def dcg_at_k(relevance_scores: Iterable[float], k: int) -> float:
    relevance = np.array(list(relevance_scores), dtype=np.float32)[:k]
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance / discounts))


def ndcg_at_k(pred_items: Iterable[int], true_items: Iterable[int], k: int) -> float:
    pred = list(pred_items)[:k]
    true_set = set(true_items)
    relevance = [1.0 if item in true_set else 0.0 for item in pred]

    dcg = dcg_at_k(relevance, k)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def mean_metrics_per_user(
    predictions: Dict[int, List[int]],
    ground_truth: Dict[int, List[int]],
    k: int = 10,
) -> Dict[str, float]:
    recalls = []
    ndcgs = []

    for user_id, true_items in ground_truth.items():
        if not true_items:
            continue
        pred_items = predictions.get(user_id, [])
        recalls.append(recall_at_k(pred_items, true_items, k))
        ndcgs.append(ndcg_at_k(pred_items, true_items, k))

    if not recalls:
        return {f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, "users_evaluated": 0}

    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "users_evaluated": len(recalls),
    }
