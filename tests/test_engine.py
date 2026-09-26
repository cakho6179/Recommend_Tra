"""
Unit tests cho RecommendationEngine và Schemas chuẩn v1.1
"""
import pytest
from src.service.schemas import CustomerRecommendationResponse, GuestRecommendationRequest
from src.service.engine import RecommendationEngine

@pytest.fixture(scope="module")
def engine():
    rec_engine = RecommendationEngine()
    rec_engine.initialize()
    return rec_engine

def test_engine_recommend_existing_customer(engine):
    sample_user = engine.interaction_df["customer_id"].iloc[0]
    res: CustomerRecommendationResponse = engine.get_recommendations(sample_user, limit=5, exclude_purchased=False)
    
    assert res.customer_id == sample_user
    assert res.source == "MODEL"
    assert res.top_k == 5
    assert len(res.recommendations) == 5
    
    for i, item in enumerate(res.recommendations, 1):
        assert item.rank == i
        assert item.product_id != ""
        assert item.product_code != ""
        assert 0.0 <= item.score <= 1.0

def test_engine_recommend_exclude_purchased(engine):
    sample_user = engine.interaction_df["customer_id"].iloc[0]
    purchased_set = set(engine.user_history_map.get(sample_user, []))
    
    res = engine.get_recommendations(sample_user, limit=5, exclude_purchased=True)
    assert res.source == "MODEL"
    for item in res.recommendations:
        assert item.product_id not in purchased_set

def test_engine_recommend_cold_start_customer(engine):
    fake_user = "brand_new_customer_xyz"
    res: CustomerRecommendationResponse = engine.get_recommendations(fake_user, limit=5)
    
    assert res.customer_id == fake_user
    assert res.source == "FALLBACK"
    assert res.top_k == 5
    assert len(res.recommendations) == 5

def test_engine_guest_recommendations(engine):
    req = GuestRecommendationRequest(limit=3)
    res = engine.get_guest_recommendations(req)
    assert res.source == "FALLBACK"
    assert len(res.recommendations) == 3

def test_engine_system_health(engine):
    health = engine.get_system_health()
    assert health.status == "healthy"
    assert health.model_loaded is True
    assert health.total_active_products > 0
