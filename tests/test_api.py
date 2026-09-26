"""
Integration tests cho FastAPI REST API
"""
import pytest
from fastapi.testclient import TestClient
from src.service.api import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_api_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.1.0"
    assert data["model_loaded"] is True
    assert data["total_active_products"] > 0

def test_api_customer_recommendations_known_user(client):
    # Lấy một ID khách hàng có sẵn
    customer_id = "014f08dc-741f-ab40-4eb2-fcc081858e0a"
    response = client.get(f"/api/v1/recommendations/{customer_id}?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == customer_id
    assert data["source"] in ["MODEL", "FALLBACK"]
    assert len(data["recommendations"]) == 5
    assert data["recommendations"][0]["rank"] == 1

def test_api_customer_recommendations_cold_start(client):
    fake_id = "non_existent_guest_123"
    response = client.get(f"/api/v1/recommendations/{fake_id}?limit=3")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == fake_id
    assert data["source"] == "FALLBACK"
    assert len(data["recommendations"]) == 3

def test_api_guest_recommendations(client):
    payload = {"limit": 4}
    response = client.post("/api/v1/recommendations/guest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "FALLBACK"
    assert len(data["recommendations"]) == 4

def test_api_products_list(client):
    response = client.get("/api/v1/products")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
