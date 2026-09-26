"""
Unit tests cho DataPreprocessor và train_test_split
"""
import pytest
import numpy as np
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.split import train_test_split_interactions

@pytest.fixture(scope="module")
def raw_data():
    loader = DataLoader()
    return loader.load_all()

def test_preprocessor_build_interaction_matrix(raw_data):
    prep = DataPreprocessor(raw_data)
    interaction_df, user_item_matrix = prep.build_interaction_matrix()
    
    assert "confidence_score" in interaction_df.columns
    assert (interaction_df["confidence_score"] >= 0).all()
    assert len(prep.user2idx) == 980
    assert len(prep.item2idx) == 60
    assert user_item_matrix.shape == (980, 60)

def test_preprocessor_build_item_features(raw_data):
    prep = DataPreprocessor(raw_data)
    item_features = prep.build_item_features()
    
    assert len(item_features) > 0
    assert "product_id" in item_features.columns
    assert "product_name" in item_features.columns
    assert "category_name" in item_features.columns
    assert "ingredient_names" in item_features.columns
    assert "text_content" in item_features.columns
    assert "min_price" in item_features.columns
    assert "max_price" in item_features.columns
    
    # Kiểm tra một sản phẩm cụ thể
    sample = item_features[item_features["product_code"] == "LFM_TRA_NHAI"].iloc[0]
    assert "Trà" in sample["product_name"]
    assert len(sample["ingredient_names"]) > 0

def test_train_test_split(raw_data):
    prep = DataPreprocessor(raw_data)
    interaction_df, _ = prep.build_interaction_matrix()
    train_df, test_df = train_test_split_interactions(interaction_df, test_ratio=0.2, random_state=42)
    
    assert len(train_df) + len(test_df) == len(interaction_df)
    assert len(test_df) > 0
    # Đảm bảo các user trong test đều có trong train để đánh giá ranking
    train_users = set(train_df["customer_id"])
    test_users = set(test_df["customer_id"])
    assert test_users.issubset(train_users)
