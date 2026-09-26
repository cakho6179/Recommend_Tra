"""
Unit tests cho ContentBasedRecommender và CollaborativeRecommender
"""
import pytest
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender

@pytest.fixture(scope="module")
def prepared_data():
    loader = DataLoader()
    raw = loader.load_all()
    prep = DataPreprocessor(raw)
    interaction_df, _ = prep.build_interaction_matrix()
    item_features = prep.build_item_features()
    return interaction_df, item_features

def test_content_based_recommender(prepared_data):
    interaction_df, item_features = prepared_data
    model = ContentBasedRecommender()
    model.fit(interaction_df, item_features)
    
    # User có lịch sử mua hàng
    user_id = interaction_df["customer_id"].iloc[0]
    recs = model.recommend(user_id=user_id, top_k=5, exclude_purchased=True)
    
    assert len(recs) == 5
    # Các sản phẩm gợi ý không được nằm trong tập đã mua
    purchased = set(interaction_df[interaction_df["customer_id"] == user_id]["product_id"])
    for pid, _ in recs:
        assert pid not in purchased
    
    # Kiểm tra giải thích
    cand_id = recs[0][0]
    explanation = model.explain_recommendation(user_id, cand_id)
    assert explanation is not None
    assert len(explanation) > 0

def test_collaborative_recommender_item_based(prepared_data):
    interaction_df, item_features = prepared_data
    model = CollaborativeRecommender(algorithm="item_item")
    model.fit(interaction_df, item_features)
    
    user_id = interaction_df["customer_id"].iloc[0]
    recs = model.recommend(user_id=user_id, top_k=5, exclude_purchased=False)
    
    assert len(recs) == 5
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)

def test_collaborative_recommender_svd(prepared_data):
    interaction_df, item_features = prepared_data
    model = CollaborativeRecommender(algorithm="svd", n_components=10)
    model.fit(interaction_df, item_features)
    
    user_id = interaction_df["customer_id"].iloc[0]
    recs = model.recommend(user_id=user_id, top_k=5, exclude_purchased=True)
    
    assert len(recs) == 5
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)
