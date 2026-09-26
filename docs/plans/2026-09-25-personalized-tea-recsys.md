# Personalized Tea Product Recommendation System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Xây dựng hoàn chỉnh hệ thống AI đề xuất sản phẩm trà cá nhân hóa (Top 5) hỗ trợ Seller tư vấn, bao gồm data pipeline, các thuật toán recommender (Popularity, Content-based, Collaborative Filtering, Hybrid), benchmark offline, REST API FastAPI và giao diện Streamlit.

**Architecture:** Kiến trúc đa tầng dạng Clean Architecture gồm ETL & Feature Engineering (tính implicit feedback score và trích xuất BOM/category feature), Tầng Recommender Model Zoo (Item-Item CF, SVD, TF-IDF Content-Based, Multi-objective Hybrid), Tầng Inference Engine kèm Explainability, và Tầng Phục Vụ (FastAPI REST service + Streamlit Consultation UI).

**Tech Stack:** Python 3.13, Pandas, NumPy, Scipy Sparse, Scikit-learn, FastAPI, Pydantic v2, Uvicorn, Streamlit, Pytest, Openpyxl.

---

### Task 1: Thiết lập Project Configuration & Data Loader

**Files:**
- Create: `src/config.py`
- Create: `src/data/loader.py`
- Test: `tests/test_data_loader.py`

**Step 1: Write the failing test**
Viết unit test cho data loader kiểm tra đọc đúng 6 file excel, kiểm tra dữ liệu không rỗng và các cột cốt lõi.

```python
# tests/test_data_loader.py
import pytest
from src.data.loader import DataLoader

def test_data_loader_loads_all_tables():
    loader = DataLoader()
    data = loader.load_all()
    assert "categories" in data
    assert "products" in data
    assert "product_variants" in data
    assert "ingredients" in data
    assert "product_ingredients" in data
    assert "customer_interactions" in data
    assert len(data["categories"]) == 7
    assert len(data["products"]) == 67
    assert len(data["customer_interactions"]) == 9350
```

**Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_data_loader.py -v`
Expected: FAIL with ModuleNotFoundError or NameError.

**Step 3: Write minimal implementation**
Tạo `src/config.py` chứa đường dẫn dữ liệu và cấu hình hệ số. Tạo `src/data/loader.py` nạp và chuẩn hóa dữ liệu từ `data/*.xlsx`.

**Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_data_loader.py -v`
Expected: PASS.

---

### Task 2: Data Preprocessor & Feature Engineering

**Files:**
- Create: `src/data/preprocessor.py`
- Create: `src/data/split.py`
- Test: `tests/test_preprocessor.py`

**Step 1: Write the failing test**
Viết unit test kiểm tra tính toán ma trận tương tác implicit feedback score ($r_{u, i} \ge 0$), trích xuất đặc trưng item từ Category và Ingredient BOM type 1, và chia train/test split.

```python
# tests/test_preprocessor.py
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.split import train_test_split_interactions

def test_preprocessor_matrix_and_features():
    loader = DataLoader()
    raw = loader.load_all()
    prep = DataPreprocessor(raw)
    interaction_df, user_item_matrix = prep.build_interaction_matrix()
    assert interaction_df["confidence_score"].min() >= 0
    assert user_item_matrix.shape[0] == 980
    
    item_features = prep.build_item_features()
    assert len(item_features) > 0
    assert "LFM_TRA_NHAI" in item_features["product_code"].values
```

**Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_preprocessor.py -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
Cài đặt `DataPreprocessor` trong `src/data/preprocessor.py` và hàm chia tập `train_test_split_interactions` trong `src/data/split.py`.

**Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_preprocessor.py -v`
Expected: PASS.

---

### Task 3: Base Recommender & Baseline (Popularity Model)

**Files:**
- Create: `src/models/base.py`
- Create: `src/models/popularity.py`
- Test: `tests/test_popularity_model.py`

**Step 1: Write the failing test**
Kiểm tra mô hình Popularity trả về danh sách Top 5 sản phẩm bán chạy nhất toàn sàn hoặc theo từng category.

```python
# tests/test_popularity_model.py
from src.models.popularity import PopularityRecommender

def test_popularity_recommender():
    model = PopularityRecommender()
    # Mock data fit
    # Assert recommend trả về đúng 5 sản phẩm có điểm số giảm dần
```

**Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_popularity_model.py -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
Cài đặt `BaseRecommender` và `PopularityRecommender` (hỗ trợ cả global popularity và category popularity, kèm filter sản phẩm active).

**Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_popularity_model.py -v`
Expected: PASS.

---

### Task 4: Content-Based Recommender & Collaborative Filtering

**Files:**
- Create: `src/models/content_based.py`
- Create: `src/models/collaborative.py`
- Test: `tests/test_recommender_models.py`

**Step 1: Write the failing test**
Kiểm tra `ContentBasedRecommender` (tính độ tương đồng Cosine từ BOM + Category) và `CollaborativeRecommender` (Item-Item CF & TruncatedSVD) sinh dự đoán điểm phù hợp cho user.

**Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_recommender_models.py -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
Cài đặt `ContentBasedRecommender` bằng TfidfVectorizer & Cosine Similarity.
Cài đặt `CollaborativeRecommender` bằng Cosine Item Similarity & TruncatedSVD Matrix Factorization.

**Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_recommender_models.py -v`
Expected: PASS.

---

### Task 5: Hybrid Recommender & Multi-Objective Ranking Layer

**Files:**
- Create: `src/models/hybrid.py`
- Test: `tests/test_hybrid_model.py`

**Step 1: Write the failing test**
Kiểm tra `HybridRecommender` kết hợp điểm số CF + CB + Popularity, và thực hiện phân bổ hợp lý giữa New Discovery và Repeat Purchase (ví dụ 3 New + 2 Reorder cho khách quen, 5 Discovery cho khách mới).

**Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_hybrid_model.py -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
Cài đặt `HybridRecommender` với trọng số tùy biến, cơ chế xếp hạng phân tầng và fallback an toàn.

**Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_hybrid_model.py -v`
Expected: PASS.

---

### Task 6: Evaluation Metrics & Offline Benchmark Suite

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `src/evaluation/evaluator.py`
- Create: `scripts/train_and_evaluate.py`
- Test: `tests/test_metrics.py`

**Step 1: Write the failing test**
Kiểm tra các hàm tính toán `recall_at_k`, `precision_at_k`, `ndcg_at_k`, `map_at_k`.

**Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
Cài đặt metric và pipeline benchmark so sánh đồng thời 4 mô hình: Popularity, Content-based, Collaborative Filtering, Hybrid Recommender.
Tạo script `scripts/train_and_evaluate.py` in bảng kết quả so sánh chi tiết.

**Step 4: Run test to verify it passes & Run benchmark**
Run: `python -m pytest tests/test_metrics.py -v`
Run: `python scripts/train_and_evaluate.py`
Expected: PASS và xuất bảng so sánh số liệu thực tế.

---

### Task 7: Inference Engine & Explainability Generator

**Files:**
- Create: `src/service/schemas.py`
- Create: `src/service/engine.py`
- Test: `tests/test_engine.py`

**Step 1: Write the failing test**
Kiểm tra `RecommendationEngine`: nhận `customer_id`, trả về Pydantic schema `CustomerRecommendationResponse` đầy đủ thông tin sản phẩm, biến thể giá min-max, nguyên liệu chính, phân loại New/Reorder và lời giải thích bằng tiếng Việt.

**Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_engine.py -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
Cài đặt Pydantic schemas và engine tổng hợp thông tin sản phẩm, giá bán, biến thể và sinh lời giải thích (*Explainability*).

**Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_engine.py -v`
Expected: PASS.

---

### Task 8: FastAPI REST Service

**Files:**
- Create: `src/service/api.py`
- Create: `scripts/run_api.py`
- Test: `tests/test_api.py`

**Step 1: Write the failing test**
Viết integration test dùng FastAPI `TestClient` gọi endpoint:
- `GET /api/v1/health`
- `GET /api/v1/recommendations/{customer_id}`
- `POST /api/v1/recommendations/guest`
- `GET /api/v1/customers`
- `GET /api/v1/products`

**Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
Cài đặt FastAPI app với CORS, dependency injection nạp model engine khi startup, và các route chuẩn RESTful.

**Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_api.py -v`
Expected: PASS.

---

### Task 9: Streamlit Seller Consultation Dashboard

**Files:**
- Create: `src/app/streamlit_app.py`
- Create: `scripts/run_demo.py`

**Step 1: Implement UI features**
Xây dựng dashboard hiện đại bằng Streamlit:
- Xem hồ sơ khách hàng, số lượng đơn, các loại trà hay mua.
- Bảng Top 5 gợi ý nổi bật trực quan (Huy hiệu Mua lại / Khám phá mới, điểm số, thẻ thành phần nguyên liệu, khoảng giá).
- Hộp thoại giải thích lý do gợi ý ("Vì sao nên tư vấn món này").
- Chế độ tư vấn khách mới (Guest cold-start) với bộ lọc gu thưởng trà (Hương hoa, Đậm đà, Thanh mát...).

**Step 2: Smoke test app loading**
Kiểm tra chạy file app không lỗi import.

---

### Task 10: Toàn bộ Unit Tests, Hoàn thiện Tài liệu Hướng Dẫn & README

**Files:**
- Create: `README.md`
- Create: `docs/api_spec.md`

**Step 1: Run entire test suite**
Run: `python -m pytest tests/ -v`
Expected: 100% tests PASS.

**Step 2: Cập nhật tài liệu hoàn chỉnh**
Viết `README.md` hướng dẫn chạy script train, khởi chạy REST API và khởi chạy Streamlit demo.
