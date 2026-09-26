"""
Module tiền xử lý dữ liệu và trích xuất đặc trưng cho Recommender System.
"""
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    PURCHASE_COUNT_WEIGHT,
    PURCHASE_QTY_WEIGHT,
    RETURN_COUNT_PENALTY,
    RETURN_QTY_PENALTY,
)


class DataPreprocessor:
    """
    Tiền xử lý dữ liệu thô:
    - Tính điểm tin cậy tương tác ngầm (Implicit Feedback Score).
    - Tạo ma trận User-Item và từ điển ánh xạ ID <-> Index.
    - Trích xuất đặc trưng sản phẩm (Item Features) từ Category, BOM và Variants.
    - Tính ma trận tương đồng nội dung (Content Similarity Matrix).
    """

    def __init__(self, raw_data: Dict[str, pd.DataFrame]):
        self.raw_data = raw_data
        self.categories_df = raw_data["categories"].copy()
        self.products_df = raw_data["products"].copy()
        self.variants_df = raw_data["product_variants"].copy()
        self.ingredients_df = raw_data["ingredients"].copy()
        self.prod_ingredients_df = raw_data["product_ingredients"].copy()
        self.interactions_df = raw_data["customer_interactions"].copy()

        # Từ điển ánh xạ
        self.user2idx: Dict[str, int] = {}
        self.idx2user: Dict[int, str] = {}
        self.item2idx: Dict[str, int] = {}
        self.idx2item: Dict[int, str] = {}

    def calculate_implicit_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Tính điểm tin cậy tương tác (Implicit Confidence Score) r_{u,i}.
        Công thức kết hợp purchase_count, purchase_quantity và trừ phạt return.
        """
        raw_val = (
            PURCHASE_COUNT_WEIGHT * df["total_purchase_count"].astype(float)
            + PURCHASE_QTY_WEIGHT * df["total_purchase_quantity"].astype(float)
            - RETURN_COUNT_PENALTY * df["total_return_count"].astype(float)
            - RETURN_QTY_PENALTY * df["total_return_quantity"].astype(float)
        )
        # Điểm số log để giảm độ lệch của các đơn hàng mua sỉ quá lớn
        scores = np.log2(1.0 + np.maximum(0.0, raw_val))
        # Nếu đã có mua hàng, đảm bảo điểm tối thiểu 0.1
        scores = np.where((df["total_purchase_count"] > 0) & (scores < 0.1), 0.1, scores)
        return pd.Series(scores, index=df.index, name="confidence_score")

    def build_interaction_matrix(self) -> Tuple[pd.DataFrame, csr_matrix]:
        """
        Xây dựng bảng tương tác có gán điểm số và ma trận thưa User-Item (CSR).
        """
        df = self.interactions_df.copy()
        df["confidence_score"] = self.calculate_implicit_score(df)

        unique_users = sorted(df["customer_id"].unique())
        unique_items = sorted(df["product_id"].unique())

        self.user2idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx2user = {idx: u for u, idx in self.user2idx.items()}
        self.item2idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx2item = {idx: i for i, idx in self.item2idx.items()}

        df["user_idx"] = df["customer_id"].map(self.user2idx)
        df["item_idx"] = df["product_id"].map(self.item2idx)

        num_users = len(unique_users)
        num_items = len(unique_items)

        row_indices = df["user_idx"].values
        col_indices = df["item_idx"].values
        data_values = df["confidence_score"].values

        matrix = csr_matrix((data_values, (row_indices, col_indices)), shape=(num_users, num_items))
        return df, matrix

    def build_item_features(self) -> pd.DataFrame:
        """
        Trích xuất đặc trưng toàn diện cho từng sản phẩm:
        - Tên sản phẩm, mã sản phẩm.
        - Tên danh mục (Category).
        - Danh sách nguyên liệu hương vị (type == 1).
        - Khoảng giá bán (min_price, max_price) từ variants.
        - Text biểu diễn tổng hợp phục vụ TF-IDF / Content-based.
        """
        # Map Category
        cat_map = dict(zip(self.categories_df["id"], self.categories_df["name"]))

        # Lọc nguyên liệu tạo vị/hương (type == 1: nguyên liệu, type == 0: bao bì)
        flavor_ingredients = self.ingredients_df[self.ingredients_df["type"] == 1]
        flavor_ing_ids = set(flavor_ingredients["id"])
        ing_name_map = dict(zip(flavor_ingredients["id"], flavor_ingredients["name"]))

        # Lọc BOM hợp lệ
        valid_bom = self.prod_ingredients_df[
            self.prod_ingredients_df["ingredient_id"].isin(flavor_ing_ids)
        ]
        
        # Nhóm nguyên liệu theo product_id
        prod_ing_dict: Dict[str, List[str]] = {}
        for _, row in valid_bom.iterrows():
            pid = row["product_id"]
            ing_id = row["ingredient_id"]
            if pid not in prod_ing_dict:
                prod_ing_dict[pid] = []
            ing_name = ing_name_map.get(ing_id)
            if ing_name and ing_name not in prod_ing_dict[pid]:
                prod_ing_dict[pid].append(ing_name)

        # Tính min_price, max_price từ variants
        price_stats = self.variants_df.groupby("product_id")["selling_price"].agg(["min", "max"]).reset_index()
        price_min_map = dict(zip(price_stats["product_id"], price_stats["min"]))
        price_max_map = dict(zip(price_stats["product_id"], price_stats["max"]))

        items = []
        for _, row in self.products_df.iterrows():
            pid = row["id"]
            pcode = row["product_code"]
            pname = row["name"]
            cat_name = cat_map.get(row["category_id"], "Khác")
            ing_list = prod_ing_dict.get(pid, [])
            min_p = price_min_map.get(pid, 0)
            max_p = price_max_map.get(pid, min_p)

            # Tạo text content có trọng số
            # Lặp tên danh mục và sản phẩm để tăng trọng số trong TF-IDF
            text = f"{pname} {pname} {cat_name} {' '.join(ing_list)}"

            items.append({
                "product_id": pid,
                "product_code": pcode,
                "product_name": pname,
                "category_id": row["category_id"],
                "category_name": cat_name,
                "status": row.get("status", "ACTIVE"),
                "ingredient_names": ing_list,
                "min_price": int(min_p),
                "max_price": int(max_p),
                "text_content": text,
            })

        item_features_df = pd.DataFrame(items)
        return item_features_df

    def compute_content_similarity_matrix(self, item_features_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Tính ma trận tương đồng nội dung Cosine giữa các sản phẩm dựa trên TF-IDF.
        """
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(item_features_df["text_content"])
        sim_matrix = cosine_similarity(tfidf_matrix)

        prod2idx = {pid: idx for idx, pid in enumerate(item_features_df["product_id"])}
        return sim_matrix, prod2idx
