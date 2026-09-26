"""
Module chia tập dữ liệu train/test cho Recommender System.
"""
from typing import Tuple
import pandas as pd
import numpy as np


def train_test_split_interactions(
    interaction_df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_interactions_for_test: int = 2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chia tập train/test theo phương pháp Leave-K-Out / Stratified theo User:
    - Với các user có >= min_interactions_for_test tương tác, lấy một phần tương tác đưa vào test_df.
    - Đảm bảo toàn bộ user trong test_df đều có ít nhất 1 tương tác trong train_df.
    - Trả về (train_df, test_df).
    """
    rng = np.random.default_rng(random_state)
    
    train_indices = []
    test_indices = []

    grouped = interaction_df.groupby("customer_id")
    for customer_id, group in grouped:
        indices = group.index.tolist()
        n = len(indices)
        if n >= min_interactions_for_test:
            # Chọn số lượng test item cho user này
            n_test = max(1, int(round(n * test_ratio)))
            # Đảm bảo vẫn còn ít nhất 1 item trong train
            if n_test >= n:
                n_test = n - 1
            
            # Chọn ngẫu nhiên n_test item làm test
            shuffled = rng.permutation(indices)
            u_test = shuffled[:n_test].tolist()
            u_train = shuffled[n_test:].tolist()
            
            test_indices.extend(u_test)
            train_indices.extend(u_train)
        else:
            # User chỉ có 1 tương tác thì đưa vào train để huấn luyện
            train_indices.extend(indices)

    train_df = interaction_df.loc[train_indices].copy()
    test_df = interaction_df.loc[test_indices].copy()

    return train_df, test_df
