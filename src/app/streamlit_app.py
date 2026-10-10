"""
Giao diện Trợ Lý Tư Vấn Bán Trà (Hương Vân Trà - Seller Dashboard)
Xây dựng trên Streamlit, tuân thủ chuẩn v1.1.
Tích hợp Feedback Loop thời gian thực & Cổng sát hạch Baseline.
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
st.markdown("*Hệ thống AI đề xuất sản phẩm trà cá nhân hóa & Vòng lặp phản hồi tự học (Continuous Learning).*")

# Sidebar
st.sidebar.header("⚙️ Cấu Hình Tư Vấn")
mode = st.sidebar.radio("Chế độ khách hàng:", ["Khách Hàng Quen (Đã có mã)", "Khách Hàng Mới / Vãng Lai"])
top_k = st.sidebar.slider("Số lượng gợi ý (Top-K):", min_value=3, max_value=10, value=5)
exclude_purchased = st.sidebar.checkbox(
    "Chỉ gợi ý món mới (Khám phá)",
    value=False,
    help="Bật tùy chọn này để loại bỏ các món khách đã từng mua, tập trung khám phá sản phẩm mới.",
)

# Sidebar: Thống kê Feedback Loop
st.sidebar.markdown("---")
st.sidebar.header("📈 Vòng Lặp Phản Hồi (Feedback Loop)")
fb_stats = engine.get_feedback_stats()
st.sidebar.metric("Tổng tương tác thu thập", fb_stats["total_events"])
col_sb1, col_sb2 = st.sidebar.columns(2)
col_sb1.metric("🛒 Chốt mua", fb_stats["purchases"])
col_sb2.metric("❌ Từ chối", fb_stats["rejects"])
if fb_stats["total_events"] > 0:
    st.sidebar.caption(f"Tỷ lệ chuyển đổi: **{fb_stats['conversion_rate'] * 100:.1f}%**")

# Nút kích hoạt Retrain kiểm thử ngay trên UI
if st.sidebar.button("🔄 Chạy Retrain & Sát Hạch Baseline"):
    from scripts.retrain_pipeline import run_retrain_pipeline
    with st.sidebar.status("Đang chạy sát hạch mô hình..."):
        retrain_res = run_retrain_pipeline()
    if retrain_res["is_promoted"]:
        st.sidebar.success(f"✅ Đạt chuẩn! Recall@5: {retrain_res['new_metrics']['recall'] * 100:.2f}%")
    else:
        st.sidebar.warning("🛡️ Giữ nguyên Baseline hiện hành (Mô hình mới chưa vượt chuẩn an toàn).")

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
    st.caption("👉 **Chỉ cần tích chọn các món khách chốt mua**, sau đó bấm nút **[Xác Nhận Khách Mua]** bên dưới. Các món không tích chọn sẽ tự động được ghi nhận là không mua để tối giản thao tác bán hàng.")

    # Form phản hồi một chạm cho nhân viên bán hàng
    with st.form(key=f"order_feedback_form_{selected_customer}"):
        selected_pids = []
        displayed_pids = []

        for item in res.recommendations:
            displayed_pids.append(item.product_id)
            price_range = f"{item.min_price:,.0f} - {item.max_price:,.0f} đ" if item.min_price else "Liên hệ"
            
            with st.container():
                col_check, col_rank, col_info, col_score = st.columns([1, 1, 5, 2])
                
                with col_check:
                    st.write("")
                    is_checked = st.checkbox(
                        "Mua",
                        key=f"chk_{selected_customer}_{item.product_id}",
                        label_visibility="collapsed",
                    )
                    if is_checked:
                        selected_pids.append(item.product_id)
                
                with col_rank:
                    st.markdown(f"### **#{item.rank}**")
                
                with col_info:
                    st.markdown(f"**{item.product_name or item.product_code}**")
                    st.caption(f"Danh mục: {item.category_name or 'Khác'} | SKU: `{item.product_code}` | {price_range}")
                
                with col_score:
                    st.metric("Độ phù hợp", f"{item.score * 100:.1f}%")
                
                st.divider()

        # NÚT BẤM DUY NHẤT
        submit_btn = st.form_submit_button(
            "🛒 Xác Nhận Khách Mua (Các món không tích tự động coi là không mua)",
            type="primary",
            use_container_width=True,
        )

        if submit_btn:
            batch_result = engine.record_batch_feedback(
                customer_id=selected_customer,
                displayed_product_ids=displayed_pids,
                purchased_product_ids=selected_pids,
            )
            purchased_cnt = batch_result["purchased_count"]
            rejected_cnt = batch_result["rejected_count"]
            
            if purchased_cnt > 0:
                st.success(f"✅ Đã ghi nhận: Khách mua {purchased_cnt} món. {rejected_cnt} món còn lại tự động ghi nhận là không mua!")
            else:
                st.info(f"ℹ️ Đã ghi nhận: Khách không mua món nào trong đợt tư vấn này ({rejected_cnt} món tự động ghi nhận là không mua).")
            st.rerun()


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
