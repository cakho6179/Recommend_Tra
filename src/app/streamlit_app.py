"""
Giao diện Trợ Lý Tư Vấn Bán Trà (Hương Vân Trà - Seller Dashboard)
Xây dựng trên Streamlit, tuân thủ chuẩn v1.1.
"""
import sys
from pathlib import Path

# Thêm root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
from src.service.engine import RecommendationEngine

st.set_page_config(
    page_title="Hương Vân Trà - AI Recommender",
    page_icon="🍵",
    layout="wide",
)

@st.cache_resource
def load_engine():
    engine = RecommendationEngine.get_instance()
    engine.initialize()
    return engine

engine = load_engine()

st.title("🍵 Hương Vân Trà — Trợ Lý Đề Xuất Bán Hàng (v1.1)")
st.markdown("*Hệ thống AI đề xuất sản phẩm trà cá nhân hóa hỗ trợ Seller tư vấn tại quầy & online.*")

# Sidebar
st.sidebar.header("⚙️ Cấu Hình Tư Vấn")
mode = st.sidebar.radio("Chế độ khách hàng:", ["Khách Hàng Quen (Đã có mã)", "Khách Hàng Mới / Vãng Lai"])
top_k = st.sidebar.slider("Số lượng gợi ý (Top-K):", min_value=3, max_value=10, value=5)
exclude_purchased = st.sidebar.checkbox(
    "Chỉ gợi ý món mới (Khám phá)",
    value=False,
    help="Bật tùy chọn này để loại bỏ các món khách đã từng mua, tập trung khám phá sản phẩm mới."
)

if mode == "Khách Hàng Quen (Đã có mã)":
    # Lấy danh sách khách hàng mẫu
    sample_customers = list(engine.user_history_map.keys())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_customer = st.selectbox(
            "Chọn mã khách hàng (Customer ID):",
            options=sample_customers,
            index=0,
        )
    with col2:
        st.write("")
        st.write("")
        custom_input = st.text_input("Hoặc nhập Customer ID khác:", placeholder="UUID khách...")
        if custom_input.strip():
            selected_customer = custom_input.strip()

    # Lấy kết quả gợi ý
    res = engine.get_recommendations(
        customer_id=selected_customer,
        limit=top_k,
        exclude_purchased=exclude_purchased,
    )

    # Hiển thị thông tin tổng quan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Khách hàng", selected_customer[:8] + "...")
    
    source_badge = "🧠 AI MODEL (Hybrid)" if res.source == "MODEL" else "⭐ FALLBACK (Hàng bán chạy)"
    c2.metric("Nguồn tính toán", source_badge)
    
    hist_items = engine.user_history_map.get(selected_customer, [])
    c3.metric("Số món đã từng mua", len(hist_items))
    c4.metric("Chế độ lọc", "Chỉ món mới" if exclude_purchased else "Bao gồm mua lại")

    st.subheader(f"📋 Top {top_k} Sản Phẩm Được Đề Xuất Cho Khách Hàng")
    
    # Hiển thị dạng bảng trực quan
    rec_data = []
    for item in res.recommendations:
        price_range = f"{item.min_price:,.0f} - {item.max_price:,.0f} đ" if item.min_price else "Liên hệ"
        rec_data.append({
            "Hạng": f"#{item.rank}",
            "Tên Sản Phẩm": item.product_name or item.product_code,
            "Danh Mục": item.category_name or "Khác",
            "Mã SKU": item.product_code,
            "Độ Phù Hợp (Score)": f"{item.score * 100:.1f}%",
            "Khoảng Giá": price_range,
        })
    st.table(pd.DataFrame(rec_data))

    # Danh sách các sản phẩm đã mua trước đây
    if hist_items:
        with st.expander("📜 Xem lịch sử các sản phẩm khách đã từng mua"):
            hist_records = []
            for pid in set(hist_items):
                p_info = engine.item_dict.get(pid, {})
                hist_records.append({
                    "Mã sản phẩm": p_info.get("product_code", pid),
                    "Tên sản phẩm": p_info.get("product_name", "N/A"),
                    "Danh mục": p_info.get("category_name", "N/A"),
                })
            st.dataframe(pd.DataFrame(hist_records), use_container_width=True)

else:
    # Chế độ khách vãng lai
    st.info("💡 Với khách hàng mới chưa từng mua hàng, hệ thống kích hoạt cơ chế **Popularity Fallback** để gợi ý các dòng trà bán chạy nhất theo danh mục.")
    
    categories = engine.raw_data.get("categories", pd.DataFrame())
    cat_options = {"Tất cả danh mục": None}
    for _, row in categories.iterrows():
        cat_options[row["name"]] = row["id"]

    selected_cat_name = st.selectbox("Chọn dòng trà khách quan tâm:", list(cat_options.keys()))
    selected_cat_id = cat_options[selected_cat_name]

    from src.service.schemas import GuestRecommendationRequest
    guest_req = GuestRecommendationRequest(category_id=selected_cat_id, limit=top_k)
    res = engine.get_guest_recommendations(guest_req)

    st.subheader(f"⭐ Top {top_k} Sản Phẩm Bán Chạy Nhất ({selected_cat_name})")
    guest_data = []
    for item in res.recommendations:
        price_range = f"{item.min_price:,.0f} - {item.max_price:,.0f} đ" if item.min_price else "Liên hệ"
        guest_data.append({
            "Hạng": f"#{item.rank}",
            "Tên Sản Phẩm": item.product_name or item.product_code,
            "Danh Mục": item.category_name or "Khác",
            "Mã SKU": item.product_code,
            "Độ Phổ Biến": f"{item.score * 100:.1f}%",
            "Khoảng Giá": price_range,
        })
    st.table(pd.DataFrame(guest_data))
