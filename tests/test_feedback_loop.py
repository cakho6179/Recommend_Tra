"""
Unit & Integration Tests cho hệ thống Feedback Loop & Continuous Learning.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.data.feedback_store import FeedbackStore
from src.service.engine import RecommendationEngine
from src.service.api import app
from scripts.retrain_pipeline import run_retrain_pipeline, load_baseline_config


@pytest.fixture
def temp_feedback_store(tmp_path):
    csv_file = tmp_path / "test_feedback.csv"
    return FeedbackStore(file_path=csv_file)


def test_feedback_store_logging_and_stats(temp_feedback_store):
    store = temp_feedback_store
    
    # 1. Ban đầu rỗng
    stats_init = store.get_stats()
    assert stats_init["total_events"] == 0
    assert stats_init["purchases"] == 0

    # 2. Ghi nhận sự kiện
    store.log_event(customer_id="cust_01", product_id="prod_A", action="PURCHASE", rank=1)
    store.log_event(customer_id="cust_01", product_id="prod_B", action="REJECT", rank=2)
    store.log_event(customer_id="cust_02", product_id="prod_A", action="PURCHASE", rank=1)

    # 3. Kiểm tra thống kê
    stats = store.get_stats()
    assert stats["total_events"] == 3
    assert stats["purchases"] == 2
    assert stats["rejects"] == 1
    assert stats["unique_customers"] == 2
    assert stats["unique_products"] == 2
    assert stats["conversion_rate"] == round(2 / 3, 4)

    # 4. Kiểm tra tổng hợp tương tác
    agg = store.aggregate_feedback_to_interactions()
    assert len(agg) == 2  # (cust_01, prod_A) và (cust_02, prod_A)
    row_01 = agg[agg["customer_id"] == "cust_01"].iloc[0]
    assert row_01["additional_purchases"] == 1


def test_engine_realtime_feedback_boost():
    engine = RecommendationEngine.get_instance()
    engine.initialize()

    test_customer = list(engine.user_history_map.keys())[0]

    # Lấy danh sách ban đầu
    initial_recs = engine.get_recommendations(customer_id=test_customer, limit=5)
    assert len(initial_recs.recommendations) > 0
    top_pid = initial_recs.recommendations[0].product_id
    second_pid = initial_recs.recommendations[1].product_id

    # Khách hàng từ chối món top_pid
    engine.record_feedback(customer_id=test_customer, product_id=top_pid, action="REJECT", rank=1)
    
    # Khách hàng chọn mua món second_pid
    engine.record_feedback(customer_id=test_customer, product_id=second_pid, action="PURCHASE", rank=2)

    # Lấy danh sách mới sau khi phản hồi
    updated_recs = engine.get_recommendations(customer_id=test_customer, limit=5)
    updated_pids = [item.product_id for item in updated_recs.recommendations]

    # Món mua được boost điểm lên đầu
    assert second_pid == updated_pids[0]
    
    # Điểm món mua được tăng
    second_item = [it for it in updated_recs.recommendations if it.product_id == second_pid][0]
    assert second_item.score > 0


def test_api_feedback_endpoints():
    client = TestClient(app)

    # 1. Gửi phản hồi PURCHASE qua API
    payload = {
        "customer_id": "test_cust_999",
        "product_id": "test_prod_888",
        "action": "PURCHASE",
        "rank": 1,
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["recorded_event"]["customer_id"] == "test_cust_999"

    # 2. Lấy thống kê phản hồi qua API
    stats_res = client.get("/api/v1/feedback/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert stats_data["total_events"] >= 1
    assert stats_data["purchases"] >= 1


def test_retrain_pipeline_gatekeeper():
    baseline_cfg = load_baseline_config()
    assert "baseline_metrics" in baseline_cfg
    assert baseline_cfg["baseline_metrics"]["recall_at_5"] > 0

    # Chạy pipeline sát hạch
    result = run_retrain_pipeline(force_promote=False)
    assert "status" in result
    assert "is_promoted" in result
    assert "new_metrics" in result
    assert result["new_metrics"]["Recall@5"] > 0
