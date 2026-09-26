"""
Script khởi chạy FastAPI Recommendation Service (Hương Vân Trà)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print(" Khởi động Hương Vân Trà - Recommendation API Service v1.1 ")
    print(" Swagger UI: http://127.0.0.1:8000/docs")
    print(" ReDoc:      http://127.0.0.1:8000/redoc")
    print("=" * 60)
    uvicorn.run("src.service.api:app", host="0.0.0.0", port=8000, reload=False)
