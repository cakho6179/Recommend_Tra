"""
Hybrid Recommender & Multi-Objective Ranking System.
"""
from typing import List, Tuple, Dict, Set, Optional, Any
import pandas as pd
import numpy as np

from src.models.base import BaseRecommender
from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.config import (
    HYBRID_CF_WEIGHT,
    HYBRID_CONTENT_WEIGHT,
    HYBRID_POPULARITY_WEIGHT,
    DEFAULT_TOP_K,
    DEFAULT_NEW_DISCOVERY_COUNT,
    DEFAULT_REPEAT_PURCHASE_COUNT,
)


class HybridRecommender(BaseRecommender):
    """
    Hệ thống đề xuất kết hợp (Hybrid Recommender):
    - Tích hợp 3 mô hình: Collaborative Filtering, Content-Based, và Popularity.
    - Xếp hạng đa mục tiêu: Tách bạch giữa Khám Phá Mới (New Discovery) và Mua Lại (Repeat Purchase).
    - Cung cấp mã lý do (reason_code) phục vụ Explainability.
    """

    def __init__(
        self,
        cf_weight: float = HYBRID_CF_WEIGHT,
        content_weight: float = HYBRID_CONTENT_WEIGHT,
        pop_weight: float = HYBRID_POPULARITY_WEIGHT,
        cf_algorithm: str = "item_item",
        name: str = "HybridRecommender",
    ):
        super().__init__(name=name)
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.pop_weight = pop_weight

        self.pop_model = PopularityRecommender()
        self.content_model = ContentBasedRecommender()
        self.cf_model = CollaborativeRecommender(algorithm=cf_algorithm)

        self.item_features_df: Optional[pd.DataFrame] = None
        self.active_item_ids: List[str] = []
        self.user_purchase_details: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame) -> "HybridRecommender":
        self._record_user_history(train_interactions)
        self.item_features_df = item_features.copy()
        
        active_df = item_features[item_features["status"] == "ACTIVE"]
        self.active_item_ids = sorted(active_df["product_id"].unique().tolist())

        # Fit các mô hình thành phần
        self.pop_model.fit(train_interactions, item_features)
        self.content_model.fit(train_interactions, item_features)
        self.cf_model.fit(train_interactions, item_features)

        # Lưu thông tin chi tiết các lần mua của từng user để hỗ trợ reorder score
        purch_details: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for _, row in train_interactions.iterrows():
            u = row["customer_id"]
            p = row["product_id"]
            if u not in purch_details:
                purch_details[u] = {}
            purch_details[u][p] = {
                "purchase_count": int(row["total_purchase_count"]),
                "purchase_qty": int(row["total_purchase_quantity"]),
                "return_count": int(row["total_return_count"]),
                "confidence_score": float(row["confidence_score"]),
            }
        self.user_purchase_details = purch_details

        self.is_fitted = True
        return self

    def predict_all_scores(self, user_id: str) -> Dict[str, float]:
        """
        Dự đoán điểm tổng hợp cho tất cả các active items đối với user_id.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit().")

        content_scores_arr = self.content_model.predict_user_scores(user_id)
        cf_scores_arr = self.cf_model.predict_user_scores(user_id)

        combined_scores: Dict[str, float] = {}
        for idx, pid in enumerate(self.active_item_ids):
            # Điểm từ content-based
            cb_idx = self.content_model.prod2idx.get(pid)
            cb_score = content_scores_arr[cb_idx] if cb_idx is not None else 0.0

            # Điểm từ collaborative filtering
            cf_idx = self.cf_model.item2idx.get(pid)
            cf_score = cf_scores_arr[cf_idx] if cf_idx is not None else 0.0

            # Điểm từ popularity
            pop_score = self.pop_model.item_scores.get(pid, 0.0)

            # Tính điểm tổ hợp có trọng số
            hybrid_score = (
                self.cf_weight * cf_score
                + self.content_weight * cb_score
                + self.pop_weight * pop_score
            )
            combined_scores[pid] = float(np.clip(hybrid_score, 0.0, 1.0))

        return combined_scores

    def recommend(
        self,
        user_id: str,
        top_k: int = DEFAULT_TOP_K,
        exclude_purchased: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Phương thức recommend chuẩn trả về [(product_id, score), ...]
        """
        scores_dict = self.predict_all_scores(user_id)
        purchased = self.get_user_purchased_items(user_id) if exclude_purchased else set()

        ranked = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        results = [(pid, score) for pid, score in ranked if pid not in purchased]
        return results[:top_k]

    def recommend_structured(
        self,
        user_id: str,
        top_k: int = DEFAULT_TOP_K,
        target_new: int = DEFAULT_NEW_DISCOVERY_COUNT,
        target_reorder: int = DEFAULT_REPEAT_PURCHASE_COUNT,
    ) -> List[Dict[str, Any]]:
        """
        Gợi ý có cấu trúc phân chia Khám Phá Mới và Mua Lại, kèm reason_code.
        """
        scores_dict = self.predict_all_scores(user_id)
        purchased_set = self.get_user_purchased_items(user_id)
        user_purchases = self.user_purchase_details.get(user_id, {})

        # Tách danh sách ứng viên New vs Reorder
        new_candidates: List[Tuple[str, float]] = []
        reorder_candidates: List[Tuple[str, float]] = []

        for pid, score in scores_dict.items():
            if pid in purchased_set:
                # Tính boost cho reorder dựa trên tần suất mua và trừ tỉ lệ trả hàng
                details = user_purchases.get(pid, {})
                p_cnt = details.get("purchase_count", 1)
                r_cnt = details.get("return_count", 0)
                satisfaction_ratio = max(0.1, 1.0 - (r_cnt / max(1, p_cnt)))
                reorder_score = float(np.clip(score * (1.0 + 0.15 * min(p_cnt, 5)) * satisfaction_ratio, 0.0, 1.0))
                reorder_candidates.append((pid, reorder_score))
            else:
                new_candidates.append((pid, score))

        new_candidates.sort(key=lambda x: x[1], reverse=True)
        reorder_candidates.sort(key=lambda x: x[1], reverse=True)

        selected_new = new_candidates[:target_new]
        selected_reorder = reorder_candidates[:target_reorder]

        # Nếu không đủ reorder (ví dụ khách mới), bù thêm từ new candidates
        shortage = (target_new + target_reorder) - (len(selected_new) + len(selected_reorder))
        if shortage > 0:
            remaining_new = new_candidates[len(selected_new): len(selected_new) + shortage]
            selected_new.extend(remaining_new)

        # Tạo output có cấu trúc
        output: List[Dict[str, Any]] = []

        for pid, score in selected_reorder:
            details = user_purchases.get(pid, {})
            p_cnt = details.get("purchase_count", 1)
            reason_code = "HIGH_PURCHASE_FREQUENCY" if p_cnt >= 2 else "PREVIOUS_FAVORITE"
            output.append({
                "product_id": pid,
                "score": round(score, 4),
                "recommendation_type": "REPEAT_PURCHASE",
                "reason_code": reason_code,
            })

        for pid, score in selected_new:
            # Xác định lý do khám phá mới
            is_cold = len(purchased_set) == 0
            if is_cold:
                reason_code = "TOP_POPULAR_CHOICE"
            else:
                reason_code = "SIMILAR_TASTE_PROFILE"
            
            output.append({
                "product_id": pid,
                "score": round(score, 4),
                "recommendation_type": "NEW_DISCOVERY",
                "reason_code": reason_code,
            })

        # Sắp xếp tổng thể theo score giảm dần
        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:top_k]
