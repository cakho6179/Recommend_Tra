"""
Collaborative Filtering Recommender (Lọc cộng tác dựa trên hành vi tương tác).
"""
from typing import List, Tuple, Dict, Set, Optional
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from src.models.base import BaseRecommender


class CollaborativeRecommender(BaseRecommender):
    """
    Mô hình Collaborative Filtering:
    - Thuật toán 1: Item-Item Collaborative Filtering (Cosine similarity giữa các cột sản phẩm).
    - Thuật toán 2: Matrix Factorization dùng TruncatedSVD cho implicit interaction matrix.
    """

    def __init__(
        self,
        algorithm: str = "item_item",
        n_components: int = 15,
        random_state: int = 42,
        name: Optional[str] = None,
    ):
        model_name = name or f"Collaborative_{algorithm.upper()}"
        super().__init__(name=model_name)
        self.algorithm = algorithm
        self.n_components = n_components
        self.random_state = random_state

        self.user2idx: Dict[str, int] = {}
        self.idx2user: Dict[int, str] = {}
        self.item2idx: Dict[str, int] = {}
        self.idx2item: Dict[int, str] = {}

        self.interaction_matrix: Optional[csr_matrix] = None
        self.item_item_sim: Optional[np.ndarray] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame) -> "CollaborativeRecommender":
        self._record_user_history(train_interactions)

        # Lọc danh sách unique users và active items
        active_items = set(item_features[item_features["status"] == "ACTIVE"]["product_id"])
        train_filtered = train_interactions[train_interactions["product_id"].isin(active_items)].copy()

        unique_users = sorted(train_filtered["customer_id"].unique())
        unique_items = sorted(active_items)

        self.user2idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx2user = {idx: u for u, idx in self.user2idx.items()}
        self.item2idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx2item = {idx: i for i, idx in self.item2idx.items()}

        u_indices = train_filtered["customer_id"].map(self.user2idx).values
        i_indices = train_filtered["product_id"].map(self.item2idx).values
        conf_scores = train_filtered["confidence_score"].values

        num_u = len(unique_users)
        num_i = len(unique_items)

        self.interaction_matrix = csr_matrix((conf_scores, (u_indices, i_indices)), shape=(num_u, num_i))

        if self.algorithm == "item_item":
            # Item similarity: Cosine giữa các cột (items)
            item_matrix = self.interaction_matrix.tocsc()
            # Tính cosine similarity giữa các cột
            self.item_item_sim = cosine_similarity(item_matrix.T, dense_output=True)
            # Khử self-similarity trên đường chéo
            np.fill_diagonal(self.item_item_sim, 0.0)

        elif self.algorithm == "svd":
            # Phân rã TruncatedSVD trên ma trận User-Item
            k = min(self.n_components, num_i - 1, num_u - 1)
            svd = TruncatedSVD(n_components=k, random_state=self.random_state)
            self.user_factors = svd.fit_transform(self.interaction_matrix)  # Shape (U, k)
            self.item_factors = svd.components_.T  # Shape (I, k)

        self.is_fitted = True
        return self

    def predict_user_scores(self, user_id: str) -> np.ndarray:
        """
        Dự đoán điểm sở thích cho tất cả các item đối với user_id.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit().")

        num_items = len(self.item2idx)
        if user_id not in self.user2idx:
            # Cold-start user
            return np.zeros(num_items)

        u_idx = self.user2idx[user_id]

        if self.algorithm == "item_item":
            user_ratings = self.interaction_matrix[u_idx, :].toarray().flatten()
            scores = np.dot(user_ratings, self.item_item_sim)
        elif self.algorithm == "svd":
            scores = np.dot(self.user_factors[u_idx, :], self.item_factors.T)
        else:
            scores = np.zeros(num_items)

        # Chuẩn hóa về [0, 1]
        max_score = np.max(scores)
        if max_score > 0:
            scores = scores / max_score
        return np.maximum(0.0, scores)

    def recommend(
        self,
        user_id: str,
        top_k: int = 5,
        exclude_purchased: bool = False,
    ) -> List[Tuple[str, float]]:
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit().")

        scores = self.predict_user_scores(user_id)
        purchased = self.get_user_purchased_items(user_id) if exclude_purchased else set()

        ranked_indices = np.argsort(scores)[::-1]
        results: List[Tuple[str, float]] = []

        for idx in ranked_indices:
            pid = self.idx2item[idx]
            if pid not in purchased:
                results.append((pid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results
