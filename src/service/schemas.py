"""
Pydantic Schemas chuẩn hóa theo Requirement v1.1 cho Recommendation Service.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Chi tiết sản phẩm được xếp hạng trong Top-K."""
    rank: int = Field(..., description="Thứ hạng từ 1 đến K")
    product_id: str = Field(..., description="UUID sản phẩm")
    product_code: str = Field(..., description="Mã sản phẩm (ví dụ: LFM_TRA_NHAI)")
    score: float = Field(..., description="Điểm số phù hợp đã chuẩn hóa (0.0 - 1.0)")
    # Metadata bổ trợ (không bắt buộc, phục vụ hiển thị trực quan)
    product_name: Optional[str] = Field(None, description="Tên sản phẩm trà")
    category_name: Optional[str] = Field(None, description="Tên danh mục trà")
    min_price: Optional[int] = Field(None, description="Giá thấp nhất (VND)")
    max_price: Optional[int] = Field(None, description="Giá cao nhất (VND)")


class CustomerRecommendationResponse(BaseModel):
    """Hợp đồng phản hồi đề xuất chuẩn v1.1."""
    customer_id: str = Field(..., description="Định danh khách hàng")
    source: str = Field(..., description="'MODEL' (AI tính toán) hoặc 'FALLBACK' (phổ biến toàn sàn)")
    top_k: int = Field(..., description="Số lượng sản phẩm đề xuất")
    recommendations: List[RecommendationItem] = Field(..., description="Danh sách sản phẩm được xếp hạng")


class GuestRecommendationRequest(BaseModel):
    """Yêu cầu đề xuất cho khách vãng lai hoặc lọc theo danh mục."""
    category_id: Optional[str] = Field(None, description="ID danh mục quan tâm")
    limit: int = Field(default=5, ge=1, le=20, description="Số lượng sản phẩm muốn lấy")


class SystemHealthResponse(BaseModel):
    """Thông tin trạng thái dịch vụ."""
    status: str
    version: str
    model_loaded: bool
    total_active_products: int
    total_trained_customers: int


class FeedbackEventRequest(BaseModel):
    """Yêu cầu ghi nhận sự kiện phản hồi từ quầy hoặc web."""
    customer_id: str = Field(..., description="Mã định danh khách hàng")
    product_id: str = Field(..., description="Mã định danh sản phẩm")
    action: str = Field(..., description="Hành động: 'PURCHASE', 'CLICK', 'REJECT', 'RETURN'")
    rank: Optional[int] = Field(default=1, ge=1, le=20, description="Vị trí hiển thị của sản phẩm (Top 1-20)")
    recommendation_id: Optional[str] = Field(default=None, description="Mã phiên đề xuất (nếu có)")


class FeedbackEventResponse(BaseModel):
    """Phản hồi sau khi lưu vết sự kiện."""
    status: str = "success"
    message: str
    recorded_event: dict


class FeedbackStatsResponse(BaseModel):
    """Thống kê tổng hợp về dữ liệu phản hồi tích lũy."""
    total_events: int
    purchases: int
    clicks: int
    rejects: int
    returns: int
    unique_customers: int
    unique_products: int
    conversion_rate: float

