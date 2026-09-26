# Nghiên Cứu SOTA & Thiết Kế Feedback Loop Cho Hệ Thống Đề Xuất Trà
## (Recommendation Feedback Loop, Continual Learning & SOTA Open-Source Research)

---

## 1. Đặt Vấn Đề: Tại Sao Hệ Thống Gợi Ý Bắt Buộc Cần Feedback Loop?

Trong các hệ thống đề xuất truyền thống chạy theo mô hình một chiều (*Open-loop System*):
1. Dữ liệu lịch sử $\rightarrow$ Huấn luyện mô hình $\rightarrow$ Triển khai Serving $\rightarrow$ Gợi ý Top-5 cho người dùng.
2. Hệ thống dừng lại ở bước gợi ý mà không có cơ chế thu thập, phản hồi và học hỏi liên tục từ kết quả của chính các gợi ý đó.

### Những hệ quả tiêu cực khi thiếu Feedback Loop:
- **Thiên lệch hiển thị (Exposure Bias / Position Bias):** Người dùng chỉ có cơ hội mua những món mà hệ thống hoặc nhân viên đem ra giới thiệu. Những món không được gợi ý sẽ mãi mãi không có lượt mua, dẫn đến việc mô hình kết luận sai lầm rằng *"khách hàng không thích những món này"*.
- **Hiệu ứng Matthew (The Rich Get Richer / Popularity Echo-Chamber):** Các sản phẩm vốn đã bán chạy tiếp tục được gợi ý nhiều hơn, đè bẹp các dòng trà mới ra mắt, làm giảm tính đa dạng (*Diversity*) và tính mới mẻ (*Novelty*) của danh mục trà.
- **Phản hồi chậm trễ và nhiễu (Delayed & Noisy Feedback):** Hành vi mua trà không diễn ra tức thì như click xem video. Khách có thể mua về uống thử sau 3-5 ngày, thậm chí 1 tháng sau mới quay lại mua tiếp hoặc yêu cầu đổi trả. Nếu không có cơ chế ghi nhận vòng lặp, hệ thống không thể phân biệt giữa *"khách đã thử và rất thích"* với *"khách mua một lần rồi thôi"*.
- **Mô hình bị suy thoái theo thời gian (Model Drift & Concept Drift):** Gu thưởng trà thay đổi theo mùa (mùa hè chuộng trà xanh, thảo mộc thanh mát; mùa đông chuộng hồng trà, ô long, trà gừng, quế). Không có feedback loop, mô hình sẽ không kịp thích ứng với sự dịch chuyển khẩu vị này.

---

## 2. Nghiên Cứu Các Phương Pháp SOTA & Thư Viện Open-Source Hàng Đầu

### 2.1. Counterfactual Learning & Off-Policy Evaluation (OPE)
Khi hệ thống đề xuất tạo ra dữ liệu, dữ liệu thu thập được là **Logged Bandit Feedback** (chỉ quan sát được kết quả của hành động đã chọn, không biết nếu gợi ý món khác thì khách có mua không).

* **Công nghệ SOTA:**
  - **Open Bandit Pipeline (OBP)** phát triển bởi **ZOZO Technologies**: Thư viện mã nguồn mở chuẩn mực học thuật quốc tế (NeurIPS / KDD) cho bài toán đánh giá chính sách đề xuất ngoại tuyến (*Off-Policy Evaluation*).
  - Thuật toán **Inverse Propensity Scoring (IPS)** và **Doubly Robust (DR)**: Điều chỉnh trọng số của các lượt tương tác bằng xác suất hiển thị (*propensity score*) nhằm triệt tiêu exposure bias:
    $$w_{u, i} = \frac{1}{P(\text{gợi ý } i \mid u)}$$

### 2.2. Contextual Bandits & Cân Bằng Exploration vs. Exploitation
Để vừa tối ưu hóa doanh số ngắn hạn vừa liên tục học hỏi sở thích mới của khách hàng:

* **Mô hình SOTA:**
  - **LinUCB (Linear Upper Confidence Bound):** Mô hình hóa sở thích người dùng bằng hàm tuyến tính trên vector đặc trưng, cộng thêm phần thưởng bất định (confidence bound). Với sản phẩm mới hoặc khách ít tương tác, độ bất định cao $\rightarrow$ điểm số UCB cao $\rightarrow$ hệ thống chủ động thử nghiệm gợi ý (*Exploration*).
  - **Thompson Sampling:** Tiếp cận theo xác suất Bayes, lấy mẫu ngẫu nhiên từ phân phối hậu nghiệm của tham số mô hình. Rất hiệu quả trong việc khám phá các món trà mới ra mắt mà không gây sụt giảm doanh thu.
* **Thư viện Open-Source tiêu biểu:**
  - **Vowpal Wabbit (VW - Microsoft Research):** Framework tối ưu hóa nhanh nhất cho Contextual Bandits, hỗ trợ CB exploration, Cost-Sensitive One-Against-All và Reductions.
  - **MABWiser (Fidelity Investments):** Thư viện Python chuyên biệt cho Multi-Armed & Contextual Bandits, dễ dàng nhúng vào kiến trúc microservice.

### 2.3. Continual Learning & Streaming Recommendation
* **Tiếp cận:**
  - Thay vì phải chạy lại toàn bộ quá trình train tốn kém hàng giờ, hệ thống cập nhật ma trận trọng số (*Online Matrix Factorization*) hoặc cập nhật vector người dùng ngay khi có giao dịch mới phát sinh.
* **Thư viện Open-Source tiêu biểu:**
  - **River (Python Online ML):** Cung cấp các thuật toán incremental learning cho collaborative filtering và ranking theo luồng stream.
  - **RecBole (AI Box):** Thư viện nghiên cứu RecSys toàn diện với hơn 80+ mô hình SOTA hỗ trợ cả Sequential Recommendation và Context-aware Recommendation.

### 2.4. Khảo Sát Chi Tiết Các Framework Recommendation Open-Source Phổ Biến

| Thư Viện Mã Nguồn Mở | Tác Giả / Tổ Chức | Đặc Trưng Thuật Toán | Đánh Giá Khả Năng Ứng Dụng |
| :--- | :--- | :--- | :--- |
| **Implicit** | Ben Frederickson | Tối ưu hóa phản hồi ngầm: Implicit ALS, BPR, Logistic Matrix Factorization. Cực nhanh bằng C++/Cython. | Rất phù hợp cho ma trận tương tác mua hàng. Tuy nhiên phụ thuộc build tool C++ trên Windows. |
| **RecBole** | AI Box (ĐH RUC) | Framework học thuật toàn diện nhất với hơn 80+ mô hình SOTA (Sequential, GNN, Knowledge-aware, Auto-encoders). | Lý tưởng cho nghiên cứu nâng cao, nhưng khá nặng nề để triển khai microservice siêu nhẹ. |
| **Cornac** | Preferred Networks | Chuyên biệt cho hệ thống đa phương thức (Multimodal RecSys), kết hợp CF với thông tin phụ trợ (BOM, text, category). | **Rất tương đồng với bài toán của chúng ta**: Cho phép nhúng định mức nguyên liệu BOM vào latent space. |
| **Microsoft Recommenders** | Microsoft | Kho mã nguồn chuẩn enterprise cung cấp trọn vẹn pipeline: ETL, Model Zoo, Evaluation, và Deployment. | Cung cấp các best practices chuẩn công nghiệp về đánh giá ngoại tuyến (Recall, NDCG, MAP, Novelty). |
| **Open Bandit Pipeline (OBP)** | ZOZO Technologies | Framework số 1 về Counterfactual Learning, OPE và Contextual Bandits. | Cung cấp cơ sở lý thuyết chuẩn để khử thiên lệch hiển thị bằng trọng số Inverse Propensity Scoring. |
| **SciPy + Scikit-Learn Custom** | Đội ngũ Hương Vân Trà | Kết hợp Sparse CSR Matrix + Item-Item Cosine + BOM TF-IDF + Multi-Objective Allocator (100% Native Python). | **Lựa chọn tối ưu thực tế**: Ổn định tuyệt đối, không lỗi C-build, độ trễ &lt; 20ms, kiểm soát hoàn toàn mã nguồn. |

---

## 3. Kiến Trúc Feedback Loop Đề Xuất Cho Hệ Thống Hương Vân Trà

Chúng tôi thiết kế kiến trúc **Vòng Lặp Phản Hồi Khép Kín (Closed-Loop Recommendation Architecture)** gồm 4 thành phần chặt chẽ:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          1. TELEMETRY & LOGGING LAYER                  │
│  - Ghi nhận Lượt hiển thị (Impression Event): recommendation_id, item  │
│  - Ghi nhận Hành vi lựa chọn: Khách chọn / Seller tư vấn thành công    │
│  - Ghi nhận Giao dịch: Mua hàng (Purchase), Hoàn trả (Return)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Kafka / Async Event Queue
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        2. ATTRIBUTION & REWARD ENGINE                  │
│  - Khớp nối Recommendation ID với Order ID                             │
│  - Tính toán hàm thưởng Reward:                                        │
│      R = +2.0 (Mua hàng) + 1.0 (Mua lặp) - 3.0 (Hoàn trả)              │
│  - Khử thiên lệch vị trí bằng Inverse Propensity Weighting (IPW)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Batch / Stream Sync
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   3. EXPLORATION CONTROLLER (BANDIT LAYER)             │
│  - Thuật toán Epsilon-Greedy / LinUCB:                                 │
│      85% Khai thác gu quen thuộc (Exploitation)                        │
│      15% Khám phá dòng trà mới tiềm năng (Exploration)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Drift Alert / Scheduled
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                4. CONTINUAL RETRAINING & SHADOW EVALUATOR              │
│  - Lập lịch tự động cập nhật trọng số (Weekly Batch Retrain)          │
│  - Cập nhật trực tiếp User Profile Vector khi có đơn hàng mới          │
│  - Kiểm thử mô hình mới ở chế độ Shadow (Shadow Serving)               │
│  - Chỉ thăng cấp mô hình (Promote) khi Recall@5 vượt Baseline hiện tại │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Chi Tiết Thực Thi Kỹ Thuật (Implementation Specification)

### 4.1. Cấu trúc Bản Ghi Sự Kiện Phản Hồi (Feedback Schema)
Mỗi tương tác trong hệ thống sinh ra một bản ghi chuẩn hóa:

```json
{
  "event_id": "evt_987654321",
  "recommendation_id": "rec_123456789",
  "customer_id": "014f08dc-741f-ab40-4eb2-fcc081858e0a",
  "product_id": "016a1a81-27c5-268f-f943-4fa3132822a6",
  "display_rank": 2,
  "action_type": "PURCHASE", 
  "reward_value": 2.0,
  "timestamp": "2026-09-25T23:50:00Z"
}
```

### 4.2. Cơ chế Điều Phối Khám Phá (Exploration Strategy)
Trong quá trình xếp hạng Top-K tại tầng suy luận (*Online Inference*):
1. **Slot 1 - 3 (Core Exploitation):** Dành cho các sản phẩm có điểm số dự đoán cao nhất từ mô hình Hybrid (đảm bảo độ tin cậy và sự hài lòng ngay tức thì).
2. **Slot 4 (Bandit Exploration):** Dành cho một sản phẩm mới ra mắt hoặc sản phẩm có độ bất định cao nhưng mang các thành phần hương vị gần với gu của khách (giúp hệ thống liên tục mở rộng hiểu biết về khách).
3. **Slot 5 (Seasonal / Category Diversity):** Dành cho sản phẩm nổi bật của mùa hoặc danh mục mà khách chưa từng trải nghiệm nhằm kích thích nhu cầu mới.

### 4.3. Tiêu chí Kích hoạt Huấn luyện lại (Retraining Triggers)
Hệ thống không train lại vô tội vạ mà dựa trên 3 điều kiện kích hoạt:
1. **Theo thời gian (Time-based Trigger):** Định kỳ vào lúc 02:00 sáng Chủ Nhật hàng tuần.
2. **Theo khối lượng dữ liệu (Data Volume Trigger):** Khi số lượng giao dịch mới tích lũy vượt quá 500 đơn hàng.
3. **Theo độ lệch phân phối (Drift Trigger):** Khi tỷ lệ chấp nhận đề xuất (*Acceptance Rate*) giảm quá 15% so với đường cơ sở trung bình 30 ngày.
