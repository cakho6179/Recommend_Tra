"""
Unit tests cho PopularityRecommender
"""
import pytest
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.popularity import PopularityRecommender

@pytest.fixture(scope="module")
def prepared_data():
    loader = DataLoader()
    raw = loader.load_all()
    prep = DataPreprocessor(raw)
    interaction_df, _ = prep.build_interaction_matrix()
    item_features = prep.build_item_features()
    return interaction_df, item_features

def test_popularity_recommender(prepared_data):
    interaction_df, item_features = prepared_data
    model = PopularityRecommender()
    model.fit(interaction_df, item_features)
    
    # Gợi ý cho một user bất kỳ
    user_id = interaction_df["customer_id"].iloc[0]
    recs = model.recommend(user_id=user_id, top_k=5, exclude_purchased=False)
    
    assert len(recs) == 5
    # Điểm số phải giảm dần
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] <= 1.0 and scores[-1] >= 0.0

def test_popularity_recommender_cold_start(prepared_data):
    interaction_df, item_features = prepared_data
    model = PopularityRecommender()
    model.fit(interaction_df, item_features)
    
    # User hoàn toàn mới
    fake_user = "non_existent_customer_xyz"
    recs = model.recommend(user_id=fake_user, top_k=5)
    assert len(recs) == 5
    
    # Gợi ý theo category
    cat_id = item_features["category_id"].iloc[0]
    cat_recs = model.recommend_by_category(category_id=cat_id, top_k=3)
    assert len(cat_recs) <= 3
