"""
Lớp cơ sở trừu tượng cho tất cả các mô hình Recommender.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set, Optional
import pandas as pd


class BaseRecommender(ABC):
    """
    Interface chung cho các mô hình đề xuất sản phẩm trà.
    """

    def __init__(self, name: str = "BaseRecommender"):
        self.name = name
        self.is_fitted = False
        self.user_history: Dict[str, Set[str]] = {}

    @abstractmethod
    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame) -> "BaseRecommender":
        """
        Huấn luyện mô hình từ dữ liệu tương tác và đặc trưng sản phẩm.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        top_k: int = 5,
        exclude_purchased: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Sinh danh sách Top-K sản phẩm đề xuất kèm điểm số.
        Trả về danh sách các tuple: [(product_id, score), ...] sắp xếp giảm dần theo điểm.
        """
        pass

    def _record_user_history(self, interactions_df: pd.DataFrame) -> None:
        """
        Ghi nhận lịch sử sản phẩm từng user đã mua.
        """
        self.user_history = (
            interactions_df.groupby("customer_id")["product_id"]
            .apply(set)
            .to_dict()
        )

    def get_user_purchased_items(self, user_id: str) -> Set[str]:
        """
        Lấy danh sách các product_id mà user đã từng mua.
        """
        return self.user_history.get(user_id, set())
