"""
Script chạy giao diện demo Streamlit cho Seller
"""
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
app_path = BASE_DIR / "src" / "app" / "streamlit_app.py"

if __name__ == "__main__":
    print("=" * 60)
    print(" Khởi chạy Giao diện Trợ Lý Bán Trà (Hương Vân Trà Dashboard) ")
    print(f" File: {app_path}")
    print("=" * 60)
    subprocess.run(["streamlit", "run", str(app_path)], check=True)
