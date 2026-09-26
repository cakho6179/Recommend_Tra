"""
Script chạy huấn luyện, benchmark so sánh toàn bộ các mô hình và lưu artifact.
"""
import sys
from pathlib import Path

# Thêm root dir vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.split import train_test_split_interactions
from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender
from src.evaluation.evaluator import ModelEvaluator


def main():
    print("=" * 65)
    print("  PERSONALIZED TEA RECOMMENDER SYSTEM - BENCHMARK & EVALUATION  ")
    print("=" * 65)

    # 1. Nạp dữ liệu
    print("\n[1/4] Đang nạp dữ liệu từ thư mục data/ ...")
    loader = DataLoader()
    raw = loader.load_all()
    print(" -> Đã nạp thành công 6 bảng dữ liệu.")

    # 2. Tiền xử lý & Trích xuất đặc trưng
    print("\n[2/4] Tiền xử lý dữ liệu và tạo ma trận tương tác...")
    prep = DataPreprocessor(raw)
    interaction_df, user_item_mat = prep.build_interaction_matrix()
    item_features = prep.build_item_features()
    active_items = item_features[item_features["status"] == "ACTIVE"]["product_id"].tolist()
    print(f" -> Tổng số tương tác: {len(interaction_df)}")
    print(f" -> Khách hàng: {len(prep.user2idx)}, Sản phẩm tương tác: {len(prep.item2idx)}")
    print(f" -> Số sản phẩm đang Active: {len(active_items)}")

    # 3. Chia tập Train/Test
    print("\n[3/4] Phân chia tập Train/Test theo phương pháp Leave-K-Out...")
    train_df, test_df = train_test_split_interactions(interaction_df, test_ratio=0.2, random_state=42)
    print(f" -> Train interactions: {len(train_df)}")
    print(f" -> Test interactions: {len(test_df)}")

    # 4. Khởi tạo và huấn luyện các mô hình
    print("\n[4/4] Huấn luyện các mô hình Recommender...")
    models = {
        "1. Popularity (Baseline)": PopularityRecommender().fit(train_df, item_features),
        "2. Content-Based (TF-IDF/Cosine)": ContentBasedRecommender().fit(train_df, item_features),
        "3. Collaborative (Item-Item CF)": CollaborativeRecommender(algorithm="item_item").fit(train_df, item_features),
        "4. Collaborative (SVD Matrix Fact)": CollaborativeRecommender(algorithm="svd", n_components=12).fit(train_df, item_features),
        "5. Hybrid Recommender (Ensemble)": HybridRecommender(
            cf_weight=0.50,
            content_weight=0.35,
            pop_weight=0.15,
        ).fit(train_df, item_features),
    }

    # Đánh giá so sánh
    print("\n" + "=" * 65)
    print("  KẾT QUẢ ĐÁNH GIÁ OFFLINE BENCHMARK (Top K = 5)  ")
    print("=" * 65)
    evaluator = ModelEvaluator(k=5)
    results_df = evaluator.compare_models(models, test_df, active_items)

    # In bảng định dạng markdown đẹp
    print("\n" + results_df.to_markdown(index=False))
    print("=" * 65)

    # Kiểm tra thử một đề xuất mẫu
    sample_user = test_df["customer_id"].iloc[0]
    hybrid_model: HybridRecommender = models["5. Hybrid Recommender (Ensemble)"]
    print(f"\n[DEMO] Gợi ý mẫu cho khách hàng '{sample_user[:8]}...':")
    sample_recs = hybrid_model.recommend_structured(sample_user, top_k=5)
    
    prod_info = dict(zip(item_features["product_id"], item_features["product_name"]))
    cat_info = dict(zip(item_features["product_id"], item_features["category_name"]))

    for i, r in enumerate(sample_recs, 1):
        pid = r["product_id"]
        pname = prod_info.get(pid, pid)
        cat = cat_info.get(pid, "")
        rtype = r["recommendation_type"]
        score = r["score"]
        code = r["reason_code"]
        print(f" {i}. [{rtype:15}] {pname} ({cat}) - Score: {score:.4f} - Reason: {code}")

    print("\nBenchmark hoàn tất thành công!\n")


if __name__ == "__main__":
    main()
