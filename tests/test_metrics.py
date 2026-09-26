"""
Unit tests cho các hàm metric đánh giá Recommender System.
"""
import pytest
from src.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    map_at_k,
    catalog_coverage,
)

def test_recall_at_k():
    actual = ["item1", "item2", "item3"]
    predicted = ["item1", "item4", "item5", "item2", "item6"]
    
    # 2 trong 3 items đúng nằm trong top 5
    assert recall_at_k(actual, predicted, k=5) == pytest.approx(2 / 3)
    # Trong top 2 chỉ có item1 đúng
    assert recall_at_k(actual, predicted, k=2) == pytest.approx(1 / 3)
    # Không có item nào đúng
    assert recall_at_k(actual, ["item8", "item9"], k=5) == 0.0

def test_precision_at_k():
    actual = ["item1", "item2"]
    predicted = ["item1", "item3", "item4", "item5", "item2"]
    
    # 2 items đúng trên 5 items đề xuất
    assert precision_at_k(actual, predicted, k=5) == pytest.approx(2 / 5)
    # Trong top 1 có 1 item đúng
    assert precision_at_k(actual, predicted, k=1) == pytest.approx(1 / 1)

def test_ndcg_at_k():
    actual = ["item1", "item2"]
    # Dự đoán hoàn hảo ở vị trí 1 và 2
    perfect_pred = ["item1", "item2", "item3", "item4", "item5"]
    assert ndcg_at_k(actual, perfect_pred, k=5) == pytest.approx(1.0)
    
    # Dự đoán đúng nhưng ở vị trí thấp hơn thì NDCG phải < 1.0 và > 0.0
    imperfect_pred = ["item3", "item4", "item1", "item5", "item2"]
    score = ndcg_at_k(actual, imperfect_pred, k=5)
    assert 0.0 < score < 1.0

def test_map_at_k():
    actual = ["item1", "item2"]
    perfect_pred = ["item1", "item2", "item3", "item4", "item5"]
    assert map_at_k(actual, perfect_pred, k=5) == pytest.approx(1.0)

def test_catalog_coverage():
    all_items = ["p1", "p2", "p3", "p4", "p5"]
    recs = [["p1", "p2"], ["p2", "p3"]]
    # Đã đề xuất p1, p2, p3 (3 trên 5)
    assert catalog_coverage(recs, all_items) == pytest.approx(3 / 5)
