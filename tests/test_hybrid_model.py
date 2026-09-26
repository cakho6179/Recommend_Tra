"""
Unit tests cho HybridRecommender
"""
import pytest
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.hybrid import HybridRecommender

@pytest.fixture(scope="module")
def prepared_data():
    loader = DataLoader()
    raw = loader.load_all()
    prep = DataPreprocessor(raw)
    interaction_df, _ = prep.build_interaction_matrix()
    item_features = prep.build_item_features()
    return interaction_df, item_features

def test_hybrid_recommender_allocation(prepared_data):
    interaction_df, item_features = prepared_data
    model = HybridRecommender()
    model.fit(interaction_df, item_features)
    
    # Lấy một user có mua nhiều sản phẩm (>= 3)
    user_counts = interaction_df.groupby("customer_id")["product_id"].nunique()
    regular_user = user_counts[user_counts >= 3].index[0]
    
    recs = model.recommend_structured(
        user_id=regular_user,
        top_k=5,
        target_new=3,
        target_reorder=2,
    )
    
    assert len(recs) == 5
    types = [r["recommendation_type"] for r in recs]
    assert "NEW_DISCOVERY" in types
    assert "REPEAT_PURCHASE" in types
    
    # Kiểm tra các trường thông tin bắt buộc
    for r in recs:
        assert "product_id" in r
        assert "score" in r
        assert "recommendation_type" in r
        assert "reason_code" in r
        assert 0.0 <= r["score"] <= 1.0

def test_hybrid_recommender_cold_start(prepared_data):
    interaction_df, item_features = prepared_data
    model = HybridRecommender()
    model.fit(interaction_df, item_features)
    
    fake_user = "brand_new_customer_123"
    recs = model.recommend_structured(user_id=fake_user, top_k=5)
    
    assert len(recs) == 5
    # Khách mới thì 100% là NEW_DISCOVERY
    types = [r["recommendation_type"] for r in recs]
    assert all(t == "NEW_DISCOVERY" for t in types)
