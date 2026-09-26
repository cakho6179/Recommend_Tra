"""
Cấu hình tập trung cho hệ thống Personalized Tea Recommender System.
"""
from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Thư mục chứa dữ liệu đầu vào
DATA_DIR = BASE_DIR / "data"

# Đường dẫn các tệp Excel
CATEGORIES_FILE = DATA_DIR / "categories.xlsx"
PRODUCTS_FILE = DATA_DIR / "products.xlsx"
PRODUCT_VARIANTS_FILE = DATA_DIR / "product_variants.xlsx"
INGREDIENTS_FILE = DATA_DIR / "ingredients.xlsx"
PRODUCT_INGREDIENTS_FILE = DATA_DIR / "product_ingredients.xlsx"
CUSTOMER_INTERACTIONS_FILE = DATA_DIR / "customer_interactions.xlsx"

# Thư mục lưu trữ artifact mô hình
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Trọng số tính Implicit Interaction Confidence Score
PURCHASE_COUNT_WEIGHT = 2.0
PURCHASE_QTY_WEIGHT = 1.0
RETURN_COUNT_PENALTY = 3.0
RETURN_QTY_PENALTY = 1.5
RECENCY_DECAY_RATE = 0.05  # Suy giảm theo năm

# Cấu hình Mô hình Đề Xuất
DEFAULT_TOP_K = 5
HYBRID_CF_WEIGHT = 0.50
HYBRID_CONTENT_WEIGHT = 0.35
HYBRID_POPULARITY_WEIGHT = 0.15

# Tỷ lệ phân bổ mặc định giữa Khám Phá Mới và Mua Lại
DEFAULT_NEW_DISCOVERY_COUNT = 3
DEFAULT_REPEAT_PURCHASE_COUNT = 2

# Random seed cho tính tái lập
RANDOM_STATE = 42
