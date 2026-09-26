"""
Popularity Recommender (Mô hình cơ sở dựa trên mức độ phổ biến).
"""
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np

from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """
    Đề xuất sản phẩm dựa trên số lượng mua và tần suất giao dịch toàn hệ thống.
    Được dùng làm Baseline và làm Fallback cho khách hàng mới (Cold-Start).
    """

    def __init__(self, name: str = "PopularityRecommender"):
        super().__init__(name=name)
        self.item_scores: Dict[str, float] = {}
        self.category_item_scores: Dict[str, List[Tuple[str, float]]] = {}
        self.ranked_items: List[Tuple[str, float]] = []

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame) -> "PopularityRecommender":
        """
        Tính toán điểm phổ biến cho từng sản phẩm:
        score = tổng (confidence_score) của sản phẩm / max_score
        """
        self._record_user_history(train_interactions)
        
        # Chỉ xét các sản phẩm ACTIVE
        active_items = set(item_features[item_features["status"] == "ACTIVE"]["product_id"])

        # Tính tổng điểm confidence_score cho từng sản phẩm
        item_grouped = (
            train_interactions[train_interactions["product_id"].isin(active_items)]
            .groupby("product_id")["confidence_score"]
            .sum()
        )

        # Chuẩn hóa về [0, 1]
        max_score = item_grouped.max() if not item_grouped.empty and item_grouped.max() > 0 else 1.0
        norm_scores = (item_grouped / max_score).to_dict()

        # Đảm bảo các sản phẩm active chưa có tương tác cũng có mặt với điểm nhỏ
        for pid in active_items:
            if pid not in norm_scores:
                norm_scores[pid] = 0.001

        self.item_scores = norm_scores
        self.ranked_items = sorted(self.item_scores.items(), key=lambda x: x[1], reverse=True)

        # Tính xếp hạng theo từng danh mục
        prod_cat_map = dict(zip(item_features["product_id"], item_features["category_id"]))
        cat_items_dict: Dict[str, List[Tuple[str, float]]] = {}
        for pid, score in self.ranked_items:
            cat_id = prod_cat_map.get(pid)
            if cat_id:
                if cat_id not in cat_items_dict:
                    cat_items_dict[cat_id] = []
                cat_items_dict[cat_id].append((pid, score))

        self.category_item_scores = cat_items_dict
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: str,
        top_k: int = 5,
        exclude_purchased: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Gợi ý Top K sản phẩm phổ biến nhất.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit(). Vui lòng gọi fit() trước.")

        purchased = self.get_user_purchased_items(user_id) if exclude_purchased else set()
        
        results: List[Tuple[str, float]] = []
        for pid, score in self.ranked_items:
            if pid not in purchased:
                results.append((pid, score))
            if len(results) >= top_k:
                break
        return results

    def recommend_by_category(
        self,
        category_id: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Gợi ý sản phẩm phổ biến nhất trong một danh mục cụ thể.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit().")
        items = self.category_item_scores.get(category_id, [])
        return items[:top_k]
