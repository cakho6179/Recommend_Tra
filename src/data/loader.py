"""
Module đọc và nạp dữ liệu từ các file Excel.
"""
from pathlib import Path
from typing import Dict
import pandas as pd

from src.config import (
    CATEGORIES_FILE,
    PRODUCTS_FILE,
    PRODUCT_VARIANTS_FILE,
    INGREDIENTS_FILE,
    PRODUCT_INGREDIENTS_FILE,
    CUSTOMER_INTERACTIONS_FILE,
)


class DataLoader:
    """
    DataLoader chịu trách nhiệm đọc dữ liệu từ các file excel đầu vào,
    kiểm tra tính toàn vẹn và trả về dictionary các pandas DataFrame.
    """

    def __init__(self, data_files: Dict[str, Path] = None):
        if data_files is None:
            self.data_files = {
                "categories": CATEGORIES_FILE,
                "products": PRODUCTS_FILE,
                "product_variants": PRODUCT_VARIANTS_FILE,
                "ingredients": INGREDIENTS_FILE,
                "product_ingredients": PRODUCT_INGREDIENTS_FILE,
                "customer_interactions": CUSTOMER_INTERACTIONS_FILE,
            }
        else:
            self.data_files = data_files

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Nạp tất cả các bảng dữ liệu.
        """
        data: Dict[str, pd.DataFrame] = {}
        for name, path in self.data_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {path}")
            df = pd.read_excel(path)
            # Làm sạch chuỗi cơ bản cho các cột text
            for col in df.select_dtypes(include=["object", "string"]).columns:
                df[col] = df[col].astype(str).str.strip()
            data[name] = df
        return data

    def load_table(self, table_name: str) -> pd.DataFrame:
        """
        Nạp riêng một bảng cụ thể.
        """
        if table_name not in self.data_files:
            raise KeyError(f"Tên bảng '{table_name}' không hợp lệ. Chọn từ {list(self.data_files.keys())}")
        return pd.read_excel(self.data_files[table_name])
