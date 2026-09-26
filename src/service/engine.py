"""
Recommendation Engine chuẩn hóa v1.1:
- Cung cấp suy luận Top-K với source: "MODEL" hoặc "FALLBACK".
- Hỗ trợ tham số exclude_purchased: True (chỉ hàng mới) hoặc False (bao gồm mua lại).
- Tích hợp Fallback Popularity tự động khi gặp khách mới.
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
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
        """
        if not self.is_initialized or self.hybrid_model is None or self.pop_model is None:
            self.initialize()

        is_known_user = customer_id in self.user_history_map

        if is_known_user:
            source = "MODEL"
            raw_recs = self.hybrid_model.recommend(
                user_id=customer_id,
                top_k=limit,
                exclude_purchased=exclude_purchased,
            )
        else:
            source = "FALLBACK"
            raw_recs = self.pop_model.recommend(
                user_id=customer_id,
                top_k=limit,
                exclude_purchased=exclude_purchased,
            )

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
            customer_id=customer_id,
            source=source,
            top_k=len(items),
            recommendations=items,
        )

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
