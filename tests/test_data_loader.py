"""
Unit tests cho DataLoader
"""
import pytest
from src.data.loader import DataLoader

def test_data_loader_loads_all_tables():
    loader = DataLoader()
    data = loader.load_all()
    
    expected_tables = [
        "categories",
        "products",
        "product_variants",
        "ingredients",
        "product_ingredients",
        "customer_interactions"
    ]
    for table in expected_tables:
        assert table in data, f"Thiếu bảng {table}"
        assert len(data[table]) > 0, f"Bảng {table} bị rỗng"

    # Kiểm tra số lượng bản ghi chính xác theo dataset thực tế
    assert len(data["categories"]) == 7
    assert len(data["products"]) == 67
    assert len(data["ingredients"]) == 61
    assert len(data["product_variants"]) == 127
    assert len(data["product_ingredients"]) == 745
    assert len(data["customer_interactions"]) == 9350

def test_data_loader_data_types_and_columns():
    loader = DataLoader()
    data = loader.load_all()
    
    assert "id" in data["products"].columns
    assert "product_code" in data["products"].columns
    assert "name" in data["products"].columns
    
    assert "customer_id" in data["customer_interactions"].columns
    assert "product_id" in data["customer_interactions"].columns
    assert "total_purchase_count" in data["customer_interactions"].columns
