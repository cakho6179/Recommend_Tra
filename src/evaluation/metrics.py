"""
Định nghĩa các hàm độ đo chuẩn mực để đánh giá Recommender System.
"""
from typing import List, Set, Sequence
import numpy as np


def recall_at_k(actual: Sequence[str], predicted: Sequence[str], k: int = 5) -> float:
    """
    Recall@K = (Số lượng item thực tế nằm trong Top K) / (Tổng số item thực tế của user)
    """
    if not actual:
        return 0.0
    actual_set = set(actual)
    pred_k = predicted[:k]
    hits = sum(1 for item in pred_k if item in actual_set)
    return float(hits / len(actual_set))


def precision_at_k(actual: Sequence[str], predicted: Sequence[str], k: int = 5) -> float:
    """
    Precision@K = (Số lượng item thực tế nằm trong Top K) / K
    """
    if k <= 0:
        return 0.0
    actual_set = set(actual)
    pred_k = predicted[:k]
    hits = sum(1 for item in pred_k if item in actual_set)
    return float(hits / k)


def ndcg_at_k(actual: Sequence[str], predicted: Sequence[str], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain tại vị trí K.
    """
    if not actual or k <= 0:
        return 0.0

    actual_set = set(actual)
    pred_k = predicted[:k]

    # Tính DCG
    dcg = 0.0
    for idx, item in enumerate(pred_k):
        if item in actual_set:
            # log2(rank + 1), rank bắt đầu từ 1 -> idx + 2
            dcg += 1.0 / np.log2(idx + 2)

    # Tính IDCG (Ideal DCG)
    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def map_at_k(actual: Sequence[str], predicted: Sequence[str], k: int = 5) -> float:
    """
    Mean Average Precision tại vị trí K cho một user đơn lẻ (AP@K).
    """
    if not actual or k <= 0:
        return 0.0

    actual_set = set(actual)
    pred_k = predicted[:k]

    running_hits = 0
    score_sum = 0.0

    for idx, item in enumerate(pred_k):
        if item in actual_set:
            running_hits += 1
            precision_at_i = running_hits / (idx + 1)
            score_sum += precision_at_i

    num_relevant = min(len(actual_set), k)
    if num_relevant == 0:
        return 0.0
    return float(score_sum / num_relevant)


def catalog_coverage(all_recommendations: List[List[str]], all_items: Sequence[str]) -> float:
    """
    Tỷ lệ % sản phẩm trong danh mục được đề xuất ít nhất một lần.
    """
    if not all_items:
        return 0.0
    unique_recs = set()
    for rec_list in all_recommendations:
        unique_recs.update(rec_list)
    return float(len(unique_recs) / len(set(all_items)))
