"""
Kịch bản Tự động Huấn luyện lại & Sát hạch Baseline (Automated Retraining & Baseline Gatekeeper).
- Tích hợp dữ liệu phản hồi mới từ data/feedback_events.csv.
- Cập nhật ma trận tương tác ngầm định (Implicit Confidence Matrix).
- Huấn luyện lại cỗ máy chủ lực Collaborative (Item-Item CF).
- Sát hạch đối đầu với chuẩn Baseline hiện tại trước khi thăng cấp lên Production.
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Đảm bảo UTF-8 an toàn trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.split import train_test_split_interactions
from src.data.feedback_store import FeedbackStore
from src.models.collaborative import CollaborativeRecommender
from src.evaluation.evaluator import ModelEvaluator

BASELINE_CONFIG_PATH = Path("src/config_baseline.json")


def load_baseline_config() -> dict:
    if BASELINE_CONFIG_PATH.exists():
        try:
            with open(BASELINE_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "system": "Hương Vân Trà - Personalized Tea Recommender Service",
        "version": "1.1.0",
        "primary_model": "ItemItemRecommender",
        "baseline_metrics": {
            "recall_at_5": 0.4895,
            "precision_at_5": 0.1951,
            "ndcg_at_5": 0.4050,
            "map_at_5": 0.3239,
            "catalog_coverage": 0.7812,
        },
        "retrain_history": [],
    }


def save_baseline_config(config: dict) -> None:
    with open(BASELINE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def run_retrain_pipeline(force_promote: bool = False, min_feedback: int = 1) -> dict:
    print("=" * 70)
    print("🍵 HƯƠNG VÂN TRÀ — TIẾN TRÌNH TỰ ĐỘNG RETRAIN & SÁT HẠCH BASELINE")
    print("=" * 70)

    # 1. Nạp dữ liệu gốc
    loader = DataLoader()
    raw_data = loader.load_all()
    interactions_df = raw_data["customer_interactions"].copy()

    # 2. Nạp dữ liệu phản hồi tích lũy
    feedback_store = FeedbackStore()
    feedback_agg = feedback_store.aggregate_feedback_to_interactions()
    feedback_stats = feedback_store.get_stats()

    print(f"\n[1/5] Thống kê phản hồi mới từ Feedback Loop:")
    print(f"  • Tổng số sự kiện ghi nhận: {feedback_stats['total_events']}")
    print(f"  • Số lượt chốt mua (PURCHASE): {feedback_stats['purchases']}")
    print(f"  • Số lượt bỏ qua (REJECT): {feedback_stats['rejects']}")
    print(f"  • Khách hàng tương tác: {feedback_stats['unique_customers']}")

    # 3. Gộp phản hồi vào ma trận giao dịch
    if not feedback_agg.empty:
        print(f"\n[2/5] Đang tích hợp {len(feedback_agg)} cặp tương tác mới vào tập dữ liệu...")
        merged = pd.merge(
            interactions_df,
            feedback_agg,
            on=["customer_id", "product_id"],
            how="outer",
        )
        merged["total_purchase_count"] = merged["total_purchase_count"].fillna(0) + merged["additional_purchases"].fillna(0)
        merged["total_return_count"] = merged["total_return_count"].fillna(0) + merged["additional_returns"].fillna(0)
        if "total_purchase_quantity" in merged.columns:
            merged["total_purchase_quantity"] = merged["total_purchase_quantity"].fillna(merged["total_purchase_count"])
        merged.drop(columns=["additional_purchases", "additional_returns"], inplace=True)
        raw_data["customer_interactions"] = merged
    else:
        print("\n[2/5] Chưa có phản hồi mới tích lũy. Sử dụng tập dữ liệu giao dịch hiện có.")

    # 4. Tiền xử lý & Xây dựng ma trận tương tác mới
    print("\n[3/5] Tiền xử lý & Tính toán lại Implicit Confidence Score...")
    prep = DataPreprocessor(raw_data)
    interaction_matrix, _ = prep.build_interaction_matrix()
    item_features = prep.build_item_features()

    # 5. Chia tập Train/Test theo tỷ lệ 80/20 (Leave-K-Out)
    train_df, test_df = train_test_split_interactions(interaction_matrix, test_ratio=0.2, random_state=42)

    # 6. Huấn luyện ứng viên mới (Candidate Item-Item CF)
    print("\n[4/5] Tái huấn luyện mô hình chủ lực Collaborative (Item-Item CF)...")
    candidate_model = CollaborativeRecommender(algorithm="item_item", name="Candidate_ItemItemCF")
    candidate_model.fit(train_df, item_features)

    # 7. Đánh giá chất lượng mô hình ứng viên
    all_active_pids = item_features[item_features["status"] == "ACTIVE"]["product_id"].tolist()
    evaluator = ModelEvaluator(k=5)
    candidate_metrics = evaluator.evaluate_model(candidate_model, test_df, all_active_pids)

    new_recall = candidate_metrics["Recall@5"]
    new_ndcg = candidate_metrics["NDCG@5"]
    new_prec = candidate_metrics["Precision@5"]
    new_map = candidate_metrics["MAP@5"]
    new_cov = candidate_metrics["Coverage"]

    # 8. SÁT HẠCH VỚI BASELINE (EVALUATION GATEKEEPER)
    print("\n[5/5] Sát hạch đối đầu với chuẩn Baseline hiện hành:")
    config = load_baseline_config()
    current_baseline = config["baseline_metrics"]["recall_at_5"]

    print(f"  • Chuẩn sàn Baseline hiện tại (Recall@5) : {current_baseline * 100:.2f}%")
    print(f"  • Điểm của Ứng viên mới (Recall@5)        : {new_recall * 100:.2f}%")
    print(f"  • NDCG@5 mới                             : {new_ndcg:.4f}")
    print(f"  • Precision@5 mới                        : {new_prec * 100:.2f}%")
    print(f"  • Catalog Coverage mới                   : {new_cov * 100:.2f}%")

    is_promoted = (new_recall >= current_baseline) or force_promote
    status = "PROMOTED_NEW_BASELINE" if is_promoted else "REJECTED_BELOW_BASELINE"

    history_entry = {
        "iteration": len(config["retrain_history"]) + 1,
        "date": datetime.now().isoformat(),
        "trigger": "FEEDBACK_LOOP_WEEKLY",
        "events_count": feedback_stats["total_events"],
        "recall_at_5": round(new_recall, 4),
        "ndcg_at_5": round(new_ndcg, 4),
        "precision_at_5": round(new_prec, 4),
        "map_at_5": round(new_map, 4),
        "catalog_coverage": round(new_cov, 4),
        "status": status,
    }

    print("-" * 70)
    if is_promoted:
        print("🎉 [KẾT QUẢ: ĐẠT CHUẨN] Ứng viên đáp ứng yêu cầu chất lượng!")
        if new_recall > current_baseline:
            print(f"🚀 THĂNG CẤP BASELINE: Nâng chuẩn sàn từ {current_baseline * 100:.2f}% lên {new_recall * 100:.2f}%")
            config["baseline_metrics"]["recall_at_5"] = round(new_recall, 4)
            config["baseline_metrics"]["ndcg_at_5"] = round(new_ndcg, 4)
            config["baseline_metrics"]["precision_at_5"] = round(new_prec, 4)
            config["baseline_metrics"]["map_at_5"] = round(new_map, 4)
            config["baseline_metrics"]["catalog_coverage"] = round(new_cov, 4)
        else:
            print("✅ Ứng viên giữ vững phong độ chuẩn sàn hiện tại.")
    else:
        print("⚠️ [KẾT QUẢ: KHÔNG ĐẠT] Điểm số ứng viên thấp hơn mức Baseline an toàn.")
        print("🛡️ Cơ chế Gatekeeper từ chối cập nhật để bảo vệ trải nghiệm khách hàng.")

    config["retrain_history"].append(history_entry)
    save_baseline_config(config)
    print("=" * 70)

    return {
        "status": status,
        "is_promoted": is_promoted,
        "new_metrics": candidate_metrics,
        "baseline_recall": current_baseline,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Retrain Pipeline & Sát hạch Baseline cho Hương Vân Trà")
    parser.add_argument("--force", action="store_true", help="Bắt buộc thăng cấp mô hình bỏ qua chuẩn Baseline")
    args = parser.parse_args()

    run_retrain_pipeline(force_promote=args.force)
