"""
Module thực hiện đánh giá benchmark và so sánh các mô hình Recommender.
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.models.base import BaseRecommender
from src.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    map_at_k,
    catalog_coverage,
)


class ModelEvaluator:
    """
    Đánh giá độc lập trên tập test đã chia:
    - Với mỗi user trong test:
      - Lấy ground-truth actual items trong test_interactions.
      - Sinh Top K gợi ý từ mô hình (loại trừ các item user đã mua trong train để đánh giá khả năng khám phá/dự đoán).
      - Tính các độ đo Recall@K, Precision@K, NDCG@K, MAP@K.
    - Tính Catalog Coverage trên toàn bộ các gợi ý.
    """

    def __init__(self, k: int = 5):
        self.k = k

    def evaluate_model(
        self,
        model: BaseRecommender,
        test_interactions: pd.DataFrame,
        all_active_items: List[str],
        exclude_train_purchased: bool = True,
    ) -> Dict[str, float]:
        """
        Đánh giá một mô hình đơn lẻ.
        """
        # Nhóm actual items theo customer_id trong tập test
        test_ground_truth = (
            test_interactions.groupby("customer_id")["product_id"]
            .apply(list)
            .to_dict()
        )

        recall_list = []
        precision_list = []
        ndcg_list = []
        map_list = []
        all_recs: List[List[str]] = []

        for user_id, actual_items in test_ground_truth.items():
            # Sinh Top K gợi ý
            recs = model.recommend(
                user_id=user_id,
                top_k=self.k,
                exclude_purchased=exclude_train_purchased,
            )
            pred_items = [pid for pid, _ in recs]
            all_recs.append(pred_items)

            recall_list.append(recall_at_k(actual_items, pred_items, k=self.k))
            precision_list.append(precision_at_k(actual_items, pred_items, k=self.k))
            ndcg_list.append(ndcg_at_k(actual_items, pred_items, k=self.k))
            map_list.append(map_at_k(actual_items, pred_items, k=self.k))

        coverage = catalog_coverage(all_recs, all_active_items)

        metrics = {
            f"Recall@{self.k}": float(np.mean(recall_list)),
            f"Precision@{self.k}": float(np.mean(precision_list)),
            f"NDCG@{self.k}": float(np.mean(ndcg_list)),
            f"MAP@{self.k}": float(np.mean(map_list)),
            "Coverage": float(coverage),
        }
        return metrics

    def compare_models(
        self,
        models: Dict[str, BaseRecommender],
        test_interactions: pd.DataFrame,
        all_active_items: List[str],
    ) -> pd.DataFrame:
        """
        So sánh đồng thời nhiều mô hình và trả về bảng tổng kết.
        """
        records = []
        for name, model in models.items():
            res = self.evaluate_model(model, test_interactions, all_active_items)
            res["Model"] = name
            records.append(res)

        df = pd.DataFrame(records)
        cols = ["Model", f"Recall@{self.k}", f"Precision@{self.k}", f"NDCG@{self.k}", f"MAP@{self.k}", "Coverage"]
        return df[cols]
