"""
Module quản lý lưu trữ và tổng hợp vết phản hồi (Feedback Store)
phục vụ Continuous Learning và Retraining Loop cho Hương Vân Trà.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np


class FeedbackStore:
    """
    Quản lý việc ghi nhận, đọc và chuyển đổi các sự kiện phản hồi của người dùng.
    Dữ liệu được lưu trữ dạng CSV an toàn, dễ đọc và tích hợp trực tiếp vào pipeline.
    """

    DEFAULT_PATH = Path("data/feedback_events.csv")

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path or self.DEFAULT_PATH
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        customer_id: str,
        product_id: str,
        action: str,
        rank: int = 1,
        recommendation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ghi nhận một sự kiện phản hồi.
        action có thể là: 'PURCHASE', 'CLICK', 'REJECT', 'RETURN'
        """
        action_clean = action.upper().strip()
        timestamp = datetime.now().isoformat()
        
        event = {
            "timestamp": timestamp,
            "customer_id": customer_id,
            "product_id": product_id,
            "action": action_clean,
            "rank": int(rank),
            "recommendation_id": recommendation_id or "",
        }

        df_row = pd.DataFrame([event])
        
        if not self.file_path.exists():
            df_row.to_csv(self.file_path, index=False, encoding="utf-8")
        else:
            df_row.to_csv(self.file_path, mode="a", header=False, index=False, encoding="utf-8")

        return event

    def load_events(self) -> pd.DataFrame:
        """Đọc toàn bộ sự kiện phản hồi đã tích lũy."""
        if not self.file_path.exists():
            return pd.DataFrame(columns=[
                "timestamp", "customer_id", "product_id", "action", "rank", "recommendation_id"
            ])
        try:
            return pd.read_csv(self.file_path, encoding="utf-8")
        except Exception:
            return pd.DataFrame(columns=[
                "timestamp", "customer_id", "product_id", "action", "rank", "recommendation_id"
            ])

    def get_stats(self) -> Dict[str, Any]:
        """Thống kê tổng hợp các chỉ số phản hồi."""
        df = self.load_events()
        if df.empty:
            return {
                "total_events": 0,
                "purchases": 0,
                "clicks": 0,
                "rejects": 0,
                "returns": 0,
                "unique_customers": 0,
                "unique_products": 0,
                "conversion_rate": 0.0,
            }

        counts = df["action"].value_counts().to_dict()
        purchases = int(counts.get("PURCHASE", 0))
        clicks = int(counts.get("CLICK", 0))
        rejects = int(counts.get("REJECT", 0))
        returns = int(counts.get("RETURN", 0))
        total = len(df)

        # Tỷ lệ chuyển đổi = số lượt mua / tổng số lượt hiển thị được phản hồi
        conv_rate = round(float(purchases / total), 4) if total > 0 else 0.0

        return {
            "total_events": total,
            "purchases": purchases,
            "clicks": clicks,
            "rejects": rejects,
            "returns": returns,
            "unique_customers": int(df["customer_id"].nunique()),
            "unique_products": int(df["product_id"].nunique()),
            "conversion_rate": conv_rate,
        }

    def aggregate_feedback_to_interactions(self) -> pd.DataFrame:
        """
        Gộp các sự kiện phản hồi thành bản ghi tương tác tương thích với customer_interactions.
        Trả về DataFrame với các cột: customer_id, product_id, additional_purchases, additional_returns.
        """
        df = self.load_events()
        if df.empty:
            return pd.DataFrame(columns=["customer_id", "product_id", "additional_purchases", "additional_returns"])

        # Lọc các sự kiện có tác động trực tiếp đến giao dịch
        purchase_mask = df["action"] == "PURCHASE"
        return_mask = df["action"] == "RETURN"

        purchases = df[purchase_mask].groupby(["customer_id", "product_id"]).size().rename("additional_purchases")
        returns = df[return_mask].groupby(["customer_id", "product_id"]).size().rename("additional_returns")

        merged = pd.concat([purchases, returns], axis=1).fillna(0).reset_index()
        merged["additional_purchases"] = merged["additional_purchases"].astype(int)
        merged["additional_returns"] = merged["additional_returns"].astype(int)
        return merged
