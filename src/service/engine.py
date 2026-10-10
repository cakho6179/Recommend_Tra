"""
Recommendation Engine chuẩn hóa v1.1:
- Cung cấp suy luận Top-K với source: "MODEL" hoặc "FALLBACK".
- Hỗ trợ tham số exclude_purchased: True (chỉ hàng mới) hoặc False (bao gồm mua lại).
- Tích hợp Fallback Popularity tự động khi gặp khách mới.
"""
from typing import Dict, List, Optional, Any, Set
import pandas as pd
import numpy as np

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.feedback_store import FeedbackStore
from src.models.hybrid import HybridRecommender
from src.models.popularity import PopularityRecommender
from src.service.schemas import (
    RecommendationItem,
    CustomerRecommendationResponse,
    GuestRecommendationRequest,
    SystemHealthResponse,
)


class RecommendationEngine:
    """
    Engine dịch vụ trung tâm tuân thủ chuẩn v1.1.
    Tích hợp Feedback Loop: In-Memory Taste Boost & Cool-down Suppression.
    """

    _instance: Optional["RecommendationEngine"] = None

    def __init__(self):
        self.is_initialized = False
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.interaction_df: Optional[pd.DataFrame] = None
        self.item_features: Optional[pd.DataFrame] = None
        self.hybrid_model: Optional[HybridRecommender] = None
        self.pop_model: Optional[PopularityRecommender] = None
        self.item_dict: Dict[str, Dict[str, Any]] = {}
        self.user_history_map: Dict[str, List[str]] = {}
        self.active_pids: List[str] = []

        # Tầng lưu trữ và bộ đệm Feedback Loop tức thì
        self.feedback_store: FeedbackStore = FeedbackStore()
        self.user_dynamic_boost: Dict[str, Dict[str, float]] = {}
        self.user_suppressed_items: Dict[str, Set[str]] = {}

    @classmethod
    def get_instance(cls) -> "RecommendationEngine":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.initialize()
        return cls._instance

    def initialize(self) -> "RecommendationEngine":
        if self.is_initialized:
            return self

        loader = DataLoader()
        self.raw_data = loader.load_all()

        prep = DataPreprocessor(self.raw_data)
        self.interaction_df, _ = prep.build_interaction_matrix()
        self.item_features = prep.build_item_features()

        # Lưu active items
        active_df = self.item_features[self.item_features["status"] == "ACTIVE"]
        self.active_pids = active_df["product_id"].tolist()

        # Metadata dictionary tra cứu nhanh
        for _, row in self.item_features.iterrows():
            pid = row["product_id"]
            self.item_dict[pid] = {
                "product_id": pid,
                "product_code": row["product_code"],
                "product_name": row["product_name"],
                "category_id": row["category_id"],
                "category_name": row["category_name"],
                "min_price": row["min_price"],
                "max_price": row["max_price"],
            }

        # Lưu user history
        user_hist: Dict[str, List[str]] = {}
        for _, row in self.interaction_df.iterrows():
            uid = row["customer_id"]
            pid = row["product_id"]
            if uid not in user_hist:
                user_hist[uid] = []
            user_hist[uid].append(pid)
        self.user_history_map = user_hist

        # Huấn luyện mô hình Hybrid và mô hình Fallback Popularity
        self.pop_model = PopularityRecommender()
        self.pop_model.fit(self.interaction_df, self.item_features)

        self.hybrid_model = HybridRecommender()
        self.hybrid_model.fit(self.interaction_df, self.item_features)

        self.is_initialized = True
        return self

    def get_recommendations(
        self,
        customer_id: str,
        limit: int = 5,
        exclude_purchased: bool = False,
    ) -> CustomerRecommendationResponse:
        """
        Lấy Top-K sản phẩm cho khách hàng:
        - Nếu khách hàng có trong lịch sử: chạy hybrid_model -> source = "MODEL"
        - Nếu khách mới tinh (Cold-start): chạy pop_model -> source = "FALLBACK"
        - Tự động áp dụng Real-time Taste Boost & Cool-down Suppression từ Feedback Loop.
        """
        if not self.is_initialized or self.hybrid_model is None or self.pop_model is None:
            self.initialize()

        is_known_user = customer_id in self.user_history_map
        fetch_k = max(limit * 2, 10)

        if is_known_user:
            source = "MODEL"
            raw_recs = self.hybrid_model.recommend(
                user_id=customer_id,
                top_k=fetch_k,
                exclude_purchased=exclude_purchased,
            )
        else:
            source = "FALLBACK"
            raw_recs = self.pop_model.recommend(
                user_id=customer_id,
                top_k=fetch_k,
                exclude_purchased=exclude_purchased,
            )

        # Áp dụng cơ chế Feedback Loop tức thì (Real-time In-memory Boost & Cool-down)
        suppressed = self.user_suppressed_items.get(customer_id, set())
        boosts = self.user_dynamic_boost.get(customer_id, {})

        adjusted_recs = []
        for pid, score in raw_recs:
            if pid in suppressed:
                adj_score = max(0.0, score - 0.50)
            else:
                boost = boosts.get(pid, 0.0)
                adj_score = min(1.0, max(0.0, score + boost))
            adjusted_recs.append((pid, adj_score))

        adjusted_recs.sort(key=lambda x: x[1], reverse=True)
        top_recs = adjusted_recs[:limit]

        items: List[RecommendationItem] = []
        for rank, (pid, score) in enumerate(top_recs, 1):
            info = self.item_dict.get(pid, {})
            item = RecommendationItem(
                rank=rank,
                product_id=pid,
                product_code=info.get("product_code", ""),
                score=round(float(score), 4),
                product_name=info.get("product_name"),
                category_name=info.get("category_name"),
                min_price=info.get("min_price"),
                max_price=info.get("max_price"),
            )
            items.append(item)

        return CustomerRecommendationResponse(
            customer_id=customer_id,
            source=source,
            top_k=len(items),
            recommendations=items,
        )

    def record_feedback(
        self,
        customer_id: str,
        product_id: str,
        action: str,
        rank: int = 1,
        recommendation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ghi nhận phản hồi và cập nhật gu vị giác tức thì (Real-time In-Memory Boost).
        """
        action_clean = action.upper().strip()
        event = self.feedback_store.log_event(
            customer_id=customer_id,
            product_id=product_id,
            action=action_clean,
            rank=rank,
            recommendation_id=recommendation_id,
        )

        if customer_id not in self.user_dynamic_boost:
            self.user_dynamic_boost[customer_id] = {}
        if customer_id not in self.user_suppressed_items:
            self.user_suppressed_items[customer_id] = set()

        if action_clean == "PURCHASE":
            # Cập nhật lịch sử mua của user
            if customer_id not in self.user_history_map:
                self.user_history_map[customer_id] = []
            if product_id not in self.user_history_map[customer_id]:
                self.user_history_map[customer_id].append(product_id)

            # Tăng điểm ưu tiên cho sản phẩm vừa mua
            self.user_dynamic_boost[customer_id][product_id] = (
                self.user_dynamic_boost[customer_id].get(product_id, 0.0) + 0.35
            )
            self.user_suppressed_items[customer_id].discard(product_id)

        elif action_clean == "REJECT":
            # Tạm ẩn để không lặp lại món khách không thích
            self.user_suppressed_items[customer_id].add(product_id)
            self.user_dynamic_boost[customer_id][product_id] = (
                self.user_dynamic_boost[customer_id].get(product_id, 0.0) - 0.50
            )

        elif action_clean == "CLICK":
            self.user_dynamic_boost[customer_id][product_id] = (
                self.user_dynamic_boost[customer_id].get(product_id, 0.0) + 0.10
            )

        elif action_clean == "RETURN":
            self.user_suppressed_items[customer_id].add(product_id)
            self.user_dynamic_boost[customer_id][product_id] = (
                self.user_dynamic_boost[customer_id].get(product_id, 0.0) - 1.00
            )

        return event

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Lấy thống kê dữ liệu phản hồi đã tích lũy."""
        return self.feedback_store.get_stats()


    def get_guest_recommendations(
        self,
        request: GuestRecommendationRequest,
    ) -> CustomerRecommendationResponse:
        """
        Đề xuất theo độ phổ biến hoặc lọc theo danh mục cho khách vãng lai.
        """
        if not self.is_initialized or self.pop_model is None:
            self.initialize()

        if request.category_id:
            raw_recs = self.pop_model.recommend_by_category(request.category_id, top_k=request.limit)
        else:
            raw_recs = self.pop_model.recommend("guest", top_k=request.limit)

        items: List[RecommendationItem] = []
        for rank, (pid, score) in enumerate(raw_recs, 1):
            info = self.item_dict.get(pid, {})
            item = RecommendationItem(
                rank=rank,
                product_id=pid,
                product_code=info.get("product_code", ""),
                score=round(float(score), 4),
                product_name=info.get("product_name"),
                category_name=info.get("category_name"),
                min_price=info.get("min_price"),
                max_price=info.get("max_price"),
            )
            items.append(item)

        return CustomerRecommendationResponse(
            customer_id="guest",
            source="FALLBACK",
            top_k=len(items),
            recommendations=items,
        )

    def get_system_health(self) -> SystemHealthResponse:
        return SystemHealthResponse(
            status="healthy",
            version="1.1.0",
            model_loaded=self.is_initialized and self.hybrid_model is not None,
            total_active_products=len(self.active_pids),
            total_trained_customers=len(self.user_history_map),
        )
