"""
FastAPI REST API cho Recommendation Service (Hương Vân Trà) chuẩn v1.1.
"""
from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.service.schemas import (
    CustomerRecommendationResponse,
    GuestRecommendationRequest,
    SystemHealthResponse,
)
from src.service.engine import RecommendationEngine

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo và nạp mô hình vào RAM khi service khởi động."""
    engine = RecommendationEngine.get_instance()
    engine.initialize()
    yield

app = FastAPI(
    title="Hương Vân Trà - Recommendation API",
    description="Dịch vụ AI gợi ý sản phẩm trà cá nhân hóa (Personalized Tea Recommender Service)",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> RecommendationEngine:
    return RecommendationEngine.get_instance()


@app.get("/api/v1/health", response_model=SystemHealthResponse, tags=["System"])
def health_check(engine: RecommendationEngine = Depends(get_engine)):
    """Kiểm tra tình trạng hoạt động của service và model."""
    return engine.get_system_health()


@app.get(
    "/api/v1/recommendations/{customer_id}",
    response_model=CustomerRecommendationResponse,
    tags=["Recommendations"],
)
def get_customer_recommendations(
    customer_id: str,
    limit: int = Query(default=5, ge=1, le=20, description="Số lượng sản phẩm đề xuất (Top-K)"),
    exclude_purchased: bool = Query(
        default=False,
        description="True: Chỉ gợi ý sản phẩm mới chưa từng mua; False: Cho phép gợi ý cả sản phẩm mua lại",
    ),
    engine: RecommendationEngine = Depends(get_engine),
):
    """
    Lấy danh sách Top-K sản phẩm trà đề xuất cho một khách hàng cụ thể.
    - Nếu khách hàng có trong lịch sử: Trả về kết quả từ AI Model (source = 'MODEL').
    - Nếu khách hàng mới tinh (Cold-start): Tự động Fallback sang hàng bán chạy nhất (source = 'FALLBACK').
    """
    return engine.get_recommendations(
        customer_id=customer_id,
        limit=limit,
        exclude_purchased=exclude_purchased,
    )


@app.post(
    "/api/v1/recommendations/guest",
    response_model=CustomerRecommendationResponse,
    tags=["Recommendations"],
)
def get_guest_recommendations(
    request: GuestRecommendationRequest,
    engine: RecommendationEngine = Depends(get_engine),
):
    """
    Đề xuất sản phẩm cho khách vãng lai hoặc lọc theo danh mục quan tâm.
    """
    return engine.get_guest_recommendations(request)


@app.get("/api/v1/products", tags=["Metadata"])
def get_active_products(engine: RecommendationEngine = Depends(get_engine)):
    """Danh sách tất cả sản phẩm trà đang hoạt động (ACTIVE)."""
    return list(engine.item_dict.values())
