"""
Content-Based Recommender (Đề xuất dựa trên nội dung & thành phần hương vị trà).
"""
from typing import List, Tuple, Dict, Set, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """
    Mô hình Content-Based Filtering:
    - Xây dựng profile biểu diễn nội dung (Tên trà, Danh mục, Nguyên liệu BOM type 1).
    - Tính ma trận Cosine Similarity giữa tất cả sản phẩm.
    - Dự đoán sở thích người dùng bằng cách tính trung bình có trọng số của các sản phẩm khách đã mua.
    """

    def __init__(self, name: str = "ContentBasedRecommender"):
        super().__init__(name=name)
        self.item_features_df: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.prod2idx: Dict[str, int] = {}
        self.idx2prod: Dict[int, str] = {}
        self.user_item_weights: Dict[str, Dict[str, float]] = {}
        self.active_item_ids: List[str] = []

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame) -> "ContentBasedRecommender":
        self._record_user_history(train_interactions)
        self.item_features_df = item_features.copy()

        # Lọc danh sách active products
        active_df = item_features[item_features["status"] == "ACTIVE"].copy().reset_index(drop=True)
        self.active_item_ids = active_df["product_id"].tolist()

        self.prod2idx = {pid: idx for idx, pid in enumerate(self.active_item_ids)}
        self.idx2prod = {idx: pid for idx, pid in enumerate(self.active_item_ids)}

        # Tính TF-IDF trên text_content (Tên sản phẩm, Danh mục, Nguyên liệu)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        tfidf_mat = vectorizer.fit_transform(active_df["text_content"])
        self.similarity_matrix = cosine_similarity(tfidf_mat)

        # Lưu trọng số tương tác của từng user với các sản phẩm đã mua
        user_weights: Dict[str, Dict[str, float]] = {}
        for _, row in train_interactions.iterrows():
            u = row["customer_id"]
            p = row["product_id"]
            score = float(row["confidence_score"])
            if u not in user_weights:
                user_weights[u] = {}
            user_weights[u][p] = score
        self.user_item_weights = user_weights

        self.is_fitted = True
        return self

    def predict_user_scores(self, user_id: str) -> np.ndarray:
        """
        Dự đoán điểm cho tất cả các active items đối với user_id.
        """
        if not self.is_fitted or self.similarity_matrix is None:
            raise RuntimeError("Mô hình chưa được fit().")

        user_history = self.user_item_weights.get(user_id, {})
        num_items = len(self.active_item_ids)

        if not user_history:
            # Cold-start: trả về mảng điểm bằng nhau
            return np.ones(num_items) / num_items

        # Lấy index và trọng số của các item user đã tương tác
        interacted_indices = []
        weights = []
        for pid, w in user_history.items():
            if pid in self.prod2idx:
                interacted_indices.append(self.prod2idx[pid])
                weights.append(w)

        if not interacted_indices:
            return np.ones(num_items) / num_items

        weights_arr = np.array(weights)
        sub_sim = self.similarity_matrix[interacted_indices, :]  # Shape: (len_history, num_items)
        user_scores = np.dot(weights_arr, sub_sim) / (np.sum(weights_arr) + 1e-9)

        # Chuẩn hóa về [0, 1]
        max_val = np.max(user_scores)
        if max_val > 0:
            user_scores = user_scores / max_val
        return user_scores

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
            pid = self.idx2prod[idx]
            if pid not in purchased:
                results.append((pid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def explain_recommendation(self, user_id: str, candidate_pid: str) -> str:
        """
        Giải thích vì sao candidate_pid được đề xuất dựa trên tương đồng thành phần với sản phẩm cũ.
        """
        if candidate_pid not in self.prod2idx:
            return "Sản phẩm trà cao cấp được tuyển chọn."

        cand_idx = self.prod2idx[candidate_pid]
        user_history = self.user_item_weights.get(user_id, {})

        if not user_history:
            return "Sản phẩm trà có hương vị cân bằng, phù hợp khách hàng mới."

        # Tìm sản phẩm trong lịch sử có độ tương đồng cao nhất với candidate
        best_sim = -1.0
        best_history_pid = None
        for pid in user_history.keys():
            if pid in self.prod2idx and pid != candidate_pid:
                sim = self.similarity_matrix[cand_idx, self.prod2idx[pid]]
                if sim > best_sim:
                    best_sim = sim
                    best_history_pid = pid

        if best_history_pid and self.item_features_df is not None:
            cand_row = self.item_features_df[self.item_features_df["product_id"] == candidate_pid].iloc[0]
            hist_row = self.item_features_df[self.item_features_df["product_id"] == best_history_pid].iloc[0]
            
            # Kiểm tra nguyên liệu chung
            common_ings = set(cand_row["ingredient_names"]).intersection(set(hist_row["ingredient_names"]))
            if common_ings:
                ing_str = ", ".join(list(common_ings)[:2])
                return f"Có thành phần ({ing_str}) tương đồng với '{hist_row['product_name']}' bạn từng thưởng thức."
            if cand_row["category_id"] == hist_row["category_id"]:
                return f"Cùng dòng {cand_row['category_name']} với '{hist_row['product_name']}' bạn yêu thích."

        return "Phù hợp với khẩu vị thưởng trà dựa trên lịch sử mua hàng của bạn."
